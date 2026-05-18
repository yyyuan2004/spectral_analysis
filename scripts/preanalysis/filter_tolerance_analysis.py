"""A3 -- Filter-shift tolerance (rewritten).

Goal
----
Quantify how robust the chosen 3-band combo is to +/-1 hardware-realistic
filter shifts.  Each of the ~27 neighbour combos is **really retrained**
(no lookup) for ``--epochs`` epochs using msi's ``train_and_eval``.

Why a rewrite was needed
------------------------
The previous version reported empty neighbourhoods because its
upstream adapter passed keyword arguments that ``train_and_eval`` does
not accept (``config_path``, ``epochs`` instead of ``cfg``, ``num_epochs``),
so every call threw silently and the script wrote NaN.  This script
calls ``train_and_eval`` **directly** with the documented signature:

    iou = train_and_eval(
        cfg=cfg, seed=seed, band_indices=combo,
        num_epochs=epochs, device=device,
        verbose=True, combo_tag=combo_tag,
    )

If your upstream signature differs, edit the single call below.

CLI
---
  --base_combo 57,62,70           # 91-band cropped HSI index
  --offset_range 1                # locked to 1 to keep training budget sane
  --epochs 30                     # 30-epoch quick eval (NOT directly
                                  # comparable to 80-epoch baseline IoU)
  --seed 42
  --data_dir /root/autodl-tmp/datasets/full_HSI_dataset
  --base_config ../msi/configs/baseline.yaml
  --output_dir outputs/preanalysis/filter_tolerance_analysis
  --msi_root ../msi

Outputs
-------
* filter_tolerance_partial.json   (deleted on success)
* filter_tolerance_results.json
* filter_tolerance_report.txt
* tolerance_distance_curve.png
* tolerance_offset_heatmap.png
* tolerance_distribution.png

Estimated runtime
-----------------
~27 combos * 30 epochs ~= 27 * ~10 min ~= 4.5 h on a single GPU
(worst case 6 h).  Every combo is flushed to partial JSON immediately.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml
from tqdm import tqdm

from _common import (
    DEFAULT_MSI_ROOT,
    SEED,
    save_json,
    set_seed,
    setup_logging,
    setup_msi_path,
)

SCRIPT_NAME = "filter_tolerance_analysis"
DEFAULT_BASE = [57, 62, 70]
DEFAULT_EPOCHS = 30
DEFAULT_DATA_DIR = "/root/autodl-tmp/datasets/full_HSI_dataset"
DEFAULT_OUTPUT_DIR = "outputs/preanalysis/filter_tolerance_analysis"
DEFAULT_BASE_CONFIG = "../msi/configs/baseline.yaml"
MAX_BAND = 90  # 91-band cropped HSI -> valid index range [0, 90]
OFFSET_RANGE = 1  # locked
REFERENCE_BASELINE_IOU_80EP = 0.7293  # 80-epoch baseline for reference (NOT for comparison)
ORIGINAL_HSI_OFFSET = 20  # cropped index + 20 -> original HSI index


def _parse_combo(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected exactly 3 comma-separated band indices, got {len(parts)}: {s!r}"
        )
    return [int(p) for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_combo", type=_parse_combo, default=list(DEFAULT_BASE),
                   help="Comma-separated 3-band base combo on the 91-band cropped HSI grid")
    p.add_argument("--offset_range", type=int, default=OFFSET_RANGE, choices=[1],
                   help="Per-band offset range (locked to 1 in this version)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Epochs per combo (default 30 -- a 'fast eval', NOT directly "
                        "comparable to the 80-epoch baseline 0.7293)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str,
                   help="Full output directory for this run (used as-is)")
    p.add_argument("--base_config", default=DEFAULT_BASE_CONFIG, type=str)
    p.add_argument("--msi_root", default=DEFAULT_MSI_ROOT, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--no_resume", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Combo generation

def generate_combos(base: list[int], R: int = 1, max_band: int = MAX_BAND) -> list[dict]:
    """27 -> after constraints, ~27 unique combos around base.

    Constraints:
      * each band in [0, max_band]
      * three distinct bands
      * deduplicate by sorted tuple
    Base combo (offsets [0,0,0]) is always first.
    """
    import itertools

    offsets = list(range(-R, R + 1))
    out: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    base_entry = {"offsets": [0, 0, 0], "bands": [int(b) for b in base]}
    seen.add(tuple(sorted(base_entry["bands"])))
    out.append(base_entry)

    for o1, o2, o3 in itertools.product(offsets, offsets, offsets):
        if o1 == 0 and o2 == 0 and o3 == 0:
            continue
        b = (int(base[0] + o1), int(base[1] + o2), int(base[2] + o3))
        if any(x < 0 or x > max_band for x in b):
            continue
        if len(set(b)) != 3:
            continue
        bs = tuple(sorted(b))
        if bs in seen:
            continue
        seen.add(bs)
        out.append({"offsets": [int(o1), int(o2), int(o3)], "bands": list(b)})
    return out


def combo_key(bands) -> str:
    return ",".join(str(int(x)) for x in bands)


def manhattan(offsets) -> int:
    return int(sum(abs(int(o)) for o in offsets))


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    setup_msi_path(args.msi_root)
    import torch  # type: ignore
    from scripts.band_range_search import train_and_eval  # type: ignore

    # Load cfg from yaml; override data_dir if user specified
    cfg_path = Path(args.base_config)
    if not cfg_path.exists():
        log.error("base_config not found: %s", cfg_path)
        return 1
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    log.info("cfg keys: %s", list(cfg.keys()))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    base = list(args.base_combo)
    combos = generate_combos(base, args.offset_range, MAX_BAND)
    n_combos = len(combos)
    log.info("generated %d unique in-range combos (raw=%d)",
             n_combos, (2 * args.offset_range + 1) ** 3)

    # Resume support -------------------------------------------------------
    partial_path = out_dir / "filter_tolerance_partial.json"
    final_path = out_dir / "filter_tolerance_results.json"
    done: dict[str, dict] = {}
    if not args.no_resume and partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text())
            for entry in prev.get("results", []):
                done[combo_key(entry["bands"])] = entry
            log.info("resuming: %d combos already done", len(done))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse partial (%s); starting fresh", e)
            done = {}

    # Training loop --------------------------------------------------------
    for i, entry in enumerate(tqdm(combos, desc="combos")):
        key = combo_key(entry["bands"])
        if key in done:
            continue

        offset_tuple = tuple(entry["offsets"])
        bands = entry["bands"]
        combo_tag = f"[tol {i + 1}/{n_combos} offset={offset_tuple} bands={bands}]"
        log.info("training %s", combo_tag)

        set_seed(args.seed)
        t0 = time.time()
        try:
            iou = train_and_eval(
                cfg=cfg,
                seed=args.seed,
                band_indices=bands,
                num_epochs=args.epochs,
                device=device,
                verbose=True,
                combo_tag=combo_tag,
            )
            iou = float(iou)
        except Exception as e:  # noqa: BLE001
            log.exception("train_and_eval failed for %s: %s", bands, e)
            iou = float("nan")
        dt = time.time() - t0

        record = {
            "offsets": entry["offsets"],
            "bands": entry["bands"],
            "iou": iou,
            "duration_s": float(dt),
        }
        done[key] = record
        log.info("%s -> IoU=%.4f (%.1fs)", combo_tag, iou, dt)

        # Free GPU memory between combos
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()

        _write_partial(partial_path, args, base, [done[combo_key(c["bands"])]
                                                  for c in combos
                                                  if combo_key(c["bands"]) in done],
                       n_combos)

    # Assemble final results in the canonical combo order
    results = [done[combo_key(c["bands"])] for c in combos if combo_key(c["bands"]) in done]
    base_record = next((r for r in results if r["offsets"] == [0, 0, 0]), None)
    base_iou_30ep = float(base_record["iou"]) if base_record else float("nan")
    log.info("base_iou_30ep = %.4f", base_iou_30ep)

    summary = _summarise(results, base_iou_30ep, base)

    final_payload = {
        "config": {
            "base_combo": base,
            "base_combo_original_hsi": [b + ORIGINAL_HSI_OFFSET for b in base],
            "offset_range": args.offset_range,
            "epochs": args.epochs,
            "seed": args.seed,
            "data_dir": args.data_dir,
            "base_config": args.base_config,
            "max_band": MAX_BAND,
            "note": (f"IoU values are the best class-1 IoU from {args.epochs}-epoch "
                     f"training; NOT directly comparable to the full 80-epoch baseline "
                     f"({REFERENCE_BASELINE_IOU_80EP:.4f})."),
        },
        "base_iou_30ep": base_iou_30ep,
        "reference_baseline_iou_80ep": REFERENCE_BASELINE_IOU_80EP,
        "n_combos": len(results),
        "summary": summary,
        "results": results,
    }
    save_json(final_payload, final_path)
    if partial_path.exists():
        try:
            partial_path.unlink()
            log.info("removed partial file (final results saved to %s)", final_path)
        except Exception as e:  # noqa: BLE001
            log.warning("could not remove partial: %s", e)

    # Visualisations -------------------------------------------------------
    _plot_distance_curve(out_dir / "tolerance_distance_curve.png", results, base_iou_30ep)
    _plot_offset_heatmap(out_dir / "tolerance_offset_heatmap.png", results, base)
    _plot_distribution(out_dir / "tolerance_distribution.png", results, base_iou_30ep)

    _write_report(out_dir / "filter_tolerance_report.txt", results, base, base_iou_30ep,
                  args, cfg, summary)

    # Headline
    valid = [r["iou"] for r in results if math.isfinite(r["iou"])]
    if math.isfinite(base_iou_30ep):
        drops = [base_iou_30ep - v for v in valid]
        n_within = sum(1 for d in drops if d <= 0.01)
        mean_drop = float(np.mean(drops)) if drops else float("nan")
        headline = (f"Tolerance: {n_within}/{len(results)} combos within 0.01 IoU drop "
                    f"from base, mean drop = {mean_drop:.4f}")
    else:
        headline = "Tolerance: base IoU is not finite, cannot compute drops"
    log.info(headline)
    print(headline)
    return 0


# ---------------------------------------------------------------------------
# Persistence

def _write_partial(path: Path, args, base: list[int], results: list[dict], n_total: int) -> None:
    save_json({
        "base_combo": base,
        "offset_range": args.offset_range,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_total": n_total,
        "n_completed": len(results),
        "results": results,
    }, path)


# ---------------------------------------------------------------------------
# Summary stats

def _summarise(results: list[dict], base_iou: float, base: list[int]) -> dict:
    valid = [r for r in results if math.isfinite(r["iou"])]
    ious = np.array([r["iou"] for r in valid], dtype=np.float64) if valid else np.empty(0)
    out: dict = {
        "n": int(len(valid)),
        "iou_min": float(ious.min()) if len(ious) else float("nan"),
        "iou_max": float(ious.max()) if len(ious) else float("nan"),
        "iou_mean": float(ious.mean()) if len(ious) else float("nan"),
        "iou_std": float(ious.std()) if len(ious) else float("nan"),
    }
    if math.isfinite(base_iou) and len(ious):
        drops = base_iou - ious
        out["drop_le_0.005"] = int((drops <= 0.005).sum())
        out["drop_le_0.010"] = int((drops <= 0.010).sum())
        out["drop_le_0.020"] = int((drops <= 0.020).sum())
        out["mean_drop"] = float(drops.mean())

    # Per-band sensitivity: mean IoU drop when only this band shifts by +/-1
    sens: dict[int, float] = {}
    for pos in range(3):
        shifted_ious = []
        for r in valid:
            offs = r["offsets"]
            if offs == [0, 0, 0]:
                continue
            # exactly one non-zero offset, at position `pos`
            if abs(offs[pos]) == 1 and all(offs[p] == 0 for p in range(3) if p != pos):
                shifted_ious.append(r["iou"])
        if shifted_ious and math.isfinite(base_iou):
            sens[pos] = float(base_iou - np.mean(shifted_ious))
        else:
            sens[pos] = float("nan")
    out["per_band_sensitivity"] = {f"band_{pos}_({base[pos]})": v for pos, v in sens.items()}

    # by-distance stats
    by_d: dict[int, dict] = {}
    max_d = max((manhattan(r["offsets"]) for r in valid), default=0)
    for d in range(0, max_d + 1):
        vals = [r["iou"] for r in valid if manhattan(r["offsets"]) == d]
        if not vals:
            by_d[d] = {"n": 0}
            continue
        arr = np.asarray(vals)
        by_d[d] = {"n": int(len(arr)), "mean": float(arr.mean()),
                   "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}
    out["by_distance"] = by_d
    return out


# ---------------------------------------------------------------------------
# Plots

def _plot_distance_curve(path: Path, results: list[dict], base_iou: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    valid = [r for r in results if math.isfinite(r["iou"])]
    if not valid:
        return
    xs = np.array([manhattan(r["offsets"]) for r in valid])
    ys = np.array([r["iou"] for r in valid])
    is_base = np.array([r["offsets"] == [0, 0, 0] for r in valid])

    # Jitter x for visibility
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, size=len(xs))
    xs_j = xs + jitter

    # Per-distance mean + std for error bars
    max_d = int(xs.max()) if len(xs) else 0
    d_grid = np.arange(0, max_d + 1)
    means = []
    stds = []
    counts = []
    for d in d_grid:
        v = ys[xs == d]
        means.append(float(v.mean()) if len(v) else np.nan)
        stds.append(float(v.std()) if len(v) else np.nan)
        counts.append(int(len(v)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Color individual points by IoU (RdYlGn)
    sc = ax.scatter(xs_j[~is_base], ys[~is_base], c=ys[~is_base], cmap="RdYlGn",
                    s=55, edgecolor="black", linewidth=0.4, zorder=3,
                    vmin=float(ys.min()), vmax=float(ys.max()))
    # Base = red star
    if is_base.any():
        ax.scatter(xs_j[is_base], ys[is_base], marker="*", s=320, color="red",
                   edgecolor="black", linewidth=0.8, zorder=5, label="base combo")

    # Mean +/- std connector
    valid_d = [(d, m, s, c) for d, m, s, c in zip(d_grid, means, stds, counts) if c > 0]
    if valid_d:
        d_arr = np.array([d for d, *_ in valid_d])
        m_arr = np.array([m for _, m, *_ in valid_d])
        s_arr = np.array([s for _, _, s, _ in valid_d])
        ax.errorbar(d_arr, m_arr, yerr=s_arr, fmt="o-", color="black", lw=1.5,
                    capsize=4, markersize=7, zorder=4, label="mean +/- std")
        for d, m, _, c in valid_d:
            ax.annotate(f"n={c}", (d, m), xytext=(6, -4),
                        textcoords="offset points", fontsize=9, color="dimgray")

    # Base reference line
    if math.isfinite(base_iou):
        ax.axhline(base_iou, color="grey", linestyle="--", lw=1,
                   label=f"base IoU (30 ep) = {base_iou:.4f}")

    cbar = fig.colorbar(sc, ax=ax, label="defect IoU")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(d_grid)
    ax.set_xlabel("Manhattan distance to base combo (sum of |offsets|)")
    ax.set_ylabel("defect IoU (30-epoch eval)")
    ax.set_title("Filter tolerance: IoU vs distance from base combo")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_offset_heatmap(path: Path, results: list[dict], base: list[int]) -> None:
    """3 subplots: for each pair of bands, fix the third band's offset to 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    valid = [r for r in results if math.isfinite(r["iou"])]
    if not valid:
        return
    all_iou = np.array([r["iou"] for r in valid])
    vmin, vmax = float(all_iou.min()), float(all_iou.max())

    offsets = [-1, 0, 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    last_im = None

    # Definitions: (fixed_pos, x_pos, y_pos, label)
    layouts = [
        (2, 0, 1, f"fix b3={base[2]} (offset 0)",
         f"offset b1 (base={base[0]})", f"offset b2 (base={base[1]})"),
        (1, 0, 2, f"fix b2={base[1]} (offset 0)",
         f"offset b1 (base={base[0]})", f"offset b3 (base={base[2]})"),
        (0, 1, 2, f"fix b1={base[0]} (offset 0)",
         f"offset b2 (base={base[1]})", f"offset b3 (base={base[2]})"),
    ]

    for ax, (fix, x_pos, y_pos, title, xlabel, ylabel) in zip(axes, layouts):
        grid = np.full((3, 3), np.nan)
        for r in valid:
            offs = r["offsets"]
            if offs[fix] != 0:
                continue
            grid[offsets.index(offs[y_pos]), offsets.index(offs[x_pos])] = r["iou"]
        last_im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=vmin, vmax=vmax,
                            extent=[-1.5, 1.5, -1.5, 1.5])
        ax.set_xticks(offsets)
        ax.set_yticks(offsets)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for ox in offsets:
            for oy in offsets:
                v = grid[offsets.index(oy), offsets.index(ox)]
                if math.isnan(v):
                    continue
                color = "white" if v < (vmin + vmax) / 2 else "black"
                ax.text(ox, oy, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, color=color)
        # Mark base cell (0, 0) with a thick black border
        ax.plot(0, 0, marker="s", markersize=44, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=2.5, zorder=10)

    fig.suptitle(f"Filter tolerance heatmaps  (base = {base})", y=1.02, fontsize=12)
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, label="defect IoU",
                        location="right", pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_distribution(path: Path, results: list[dict], base_iou: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    valid = [r for r in results if math.isfinite(r["iou"])]
    if not valid:
        return
    ious = np.array([r["iou"] for r in valid])
    is_base = np.array([r["offsets"] == [0, 0, 0] for r in valid])

    fig, ax = plt.subplots(figsize=(6.5, 6))
    parts = ax.violinplot([ious], positions=[1], vert=True, widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.35)
        body.set_edgecolor("black")

    # Swarm-style scatter with jitter
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.15, 0.15, size=len(ious))
    ax.scatter(np.full(len(ious), 1.0)[~is_base] + jitter[~is_base], ious[~is_base],
               s=45, color="tab:blue", alpha=0.8, edgecolor="black", linewidth=0.3,
               zorder=3, label="combo")
    if is_base.any():
        ax.scatter(np.full(len(ious), 1.0)[is_base] + jitter[is_base], ious[is_base],
                   s=200, color="red", marker="*", edgecolor="black", linewidth=0.6,
                   zorder=4, label="base combo")

    mean_iou = float(ious.mean())
    median_iou = float(np.median(ious))
    min_iou = float(ious.min())
    max_iou = float(ious.max())
    ax.axhline(mean_iou, color="tab:purple", linestyle="--", lw=1, label=f"mean = {mean_iou:.4f}")
    ax.axhline(median_iou, color="tab:green", linestyle=":", lw=1,
               label=f"median = {median_iou:.4f}")

    drop_text = ""
    if math.isfinite(base_iou):
        n_close = int(((base_iou - ious) < 0.01).sum())
        drop_text = f"IoU drop < 0.01 from base: {n_close}/{len(ious)} combos"

    ax.set_xticks([1])
    ax.set_xticklabels(["all combos"])
    ax.set_xlim(0.3, 1.7)
    ax.set_ylabel("defect IoU (30-epoch eval)")
    title = f"Tolerance distribution  (n={len(ious)})\nmin={min_iou:.4f}, max={max_iou:.4f}"
    if drop_text:
        title += f"\n{drop_text}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report

def _wavelengths_from_cfg(cfg: dict, bands: list[int]) -> list[float] | None:
    """Best-effort: pull a band->nm mapping from cfg, in cropped-index space.

    Expected key candidates: ``band_wavelengths``, ``wavelengths``.
    If the cfg list is on the original HSI grid (length 91+20 = 111+), we
    assume the cropped index corresponds to the first element of the
    cropped slice, but the caller can adjust if needed.
    """
    wls = cfg.get("band_wavelengths") or cfg.get("wavelengths")
    if not wls:
        return None
    try:
        out = []
        for b in bands:
            if b < 0 or b >= len(wls):
                return None
            out.append(float(wls[b]))
        return out
    except Exception:  # noqa: BLE001
        return None


def _write_report(path: Path, results: list[dict], base: list[int],
                  base_iou: float, args, cfg: dict, summary: dict) -> None:
    lines = ["Filter-shift tolerance report (30-epoch quick evaluation)",
             "=" * 60, ""]
    base_original = [b + ORIGINAL_HSI_OFFSET for b in base]
    lines.append(f"Base combo (91-band cropped index):  {base}")
    lines.append(f"Base combo (original HSI index):      {base_original}  (cropped + {ORIGINAL_HSI_OFFSET})")
    wls = _wavelengths_from_cfg(cfg, base)
    if wls is not None:
        lines.append(f"Base combo wavelength (nm):           {[round(w, 1) for w in wls]}")
    lines.append(f"Base IoU ({args.epochs} epochs):                {base_iou:.4f}"
                 if math.isfinite(base_iou) else
                 f"Base IoU ({args.epochs} epochs):                not finite")
    lines.append(f"Reference IoU (80 epochs, exhaustive): {REFERENCE_BASELINE_IOU_80EP:.4f}  "
                 f"(NOT directly comparable -- 30 vs 80 epoch)")
    lines.append("")
    lines.append(f"Neighborhood: +/-{args.offset_range} band offset, {summary['n']} unique combos (incl. base)")
    lines.append("")
    lines.append("Tolerance summary:")
    lines.append(f"  IoU range across {summary['n']} combos: "
                 f"[{summary['iou_min']:.4f}, {summary['iou_max']:.4f}]")
    lines.append(f"  IoU mean +/- std: {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f}")
    if "drop_le_0.005" in summary:
        n = summary["n"]
        lines.append(f"  Drop from base <= 0.005: {summary['drop_le_0.005']}/{n} combos "
                     f"({summary['drop_le_0.005'] / n:.1%})")
        lines.append(f"  Drop from base <= 0.010: {summary['drop_le_0.010']}/{n} combos "
                     f"({summary['drop_le_0.010'] / n:.1%})")
        lines.append(f"  Drop from base <= 0.020: {summary['drop_le_0.020']}/{n} combos "
                     f"({summary['drop_le_0.020'] / n:.1%})")
    lines.append("")

    # Top-5 / worst-5
    valid = [r for r in results if math.isfinite(r["iou"])]
    valid_sorted = sorted(valid, key=lambda r: -r["iou"])
    lines.append("Top-5 combos in neighborhood:")
    for rank, r in enumerate(valid_sorted[:5], 1):
        marker = "  (BASE)" if r["offsets"] == [0, 0, 0] else ""
        lines.append(f"  {rank}. bands={r['bands']}  offset={tuple(r['offsets'])}  "
                     f"IoU={r['iou']:.4f}{marker}")
    lines.append("")
    lines.append("Worst-5 combos in neighborhood:")
    for rank, r in enumerate(valid_sorted[-5:][::-1], 1):
        lines.append(f"  {rank}. bands={r['bands']}  offset={tuple(r['offsets'])}  "
                     f"IoU={r['iou']:.4f}")
    lines.append("")

    # Per-band sensitivity
    sens = summary.get("per_band_sensitivity", {})
    if sens:
        lines.append("Per-band sensitivity (mean IoU drop when only this band shifts +/-1):")
        for pos, b in enumerate(base):
            key = f"band_{pos}_({b})"
            v = sens.get(key, float("nan"))
            v_str = f"{v:+.4f}" if math.isfinite(v) else "n/a"
            lines.append(f"  band b{pos + 1} (cropped={b}, original={b + ORIGINAL_HSI_OFFSET}): "
                         f"mean drop = {v_str}")
        lines.append("")

    # By-distance
    by_d = summary.get("by_distance", {})
    if by_d:
        lines.append("IoU by Manhattan distance to base:")
        for d in sorted(int(k) for k in by_d.keys()):
            s = by_d[d] if d in by_d else by_d[str(d)]
            if s.get("n", 0):
                lines.append(f"  d={d}: n={s['n']:2d}  mean={s['mean']:.4f}  "
                             f"std={s['std']:.4f}  range=[{s['min']:.4f}, {s['max']:.4f}]")
        lines.append("")

    # Engineering interpretation
    interp = _engineering_interpretation(summary, base_iou)
    lines.append("Engineering interpretation:")
    lines.append("  " + interp)
    path.write_text("\n".join(lines) + "\n")


def _engineering_interpretation(summary: dict, base_iou: float) -> str:
    if not math.isfinite(base_iou):
        return ("Base IoU is not finite, cannot interpret tolerance.  "
                "Verify train_and_eval is actually training.")
    n = summary.get("n", 0)
    if n == 0:
        return "No finite IoU values to interpret."
    n_010 = summary.get("drop_le_0.010", 0)
    frac_010 = n_010 / n
    mean_drop = summary.get("mean_drop", float("nan"))
    if frac_010 >= 0.9:
        verdict = ("Filter tolerance is excellent: nearly all +/-1-band perturbations "
                   "stay within 0.01 IoU of the base combo, so typical hardware "
                   "filter-centre tolerances should be acceptable.")
    elif frac_010 >= 0.7:
        verdict = ("Filter tolerance is good: most +/-1-band perturbations stay within "
                   "0.01 IoU drop.  Standard filter selection should suffice.")
    elif frac_010 >= 0.4:
        verdict = ("Filter tolerance is moderate: a meaningful fraction of "
                   "neighbour combos lose >0.01 IoU.  Prefer tighter filter "
                   "specifications and verify per-batch.")
    else:
        verdict = ("Filter tolerance is limited: most +/-1-band perturbations cost "
                   ">0.01 IoU.  Tighter filter centre wavelengths or recomputed "
                   "band selection may be required for production.")
    return verdict + f"  (mean drop across {n} combos: {mean_drop:+.4f})"


if __name__ == "__main__":
    raise SystemExit(main())
