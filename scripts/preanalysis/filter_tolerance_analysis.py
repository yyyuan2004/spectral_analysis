"""A3 -- Filter-shift tolerance for the chosen 3-band combination.

Purpose
-------
Quantify how robust the chosen 3-band combo is to small, hardware-realistic
shifts of each filter centre wavelength (+/-1 and +/-2 HSI bands on the
91-band cropped HSI grid).  Each neighbour combo is **re-trained from
scratch** with the same baseline config; no lookup-table shortcut.

Critical assumption
-------------------
``_adapters.run_training`` -> ``scripts.band_range_search.train_and_eval``
must actually train and return the best class-1 IoU for the given
``band_indices``.  As a guard against silent caching/lookup behaviour
upstream, this script trains the base combo first and aborts (exit 2)
if the returned IoU is non-finite or zero.

CLI (per the spec)
------------------
  --base_combo 57,62,70           # comma-separated, 91-band cropped HSI index
  --offset_range {1, 2}           # default 2 -> up to 5^3 = 125 raw candidates
  --epochs 30                     # default 30 (NOT 80)
  --data_dir /root/autodl-tmp/hsi # default
  --output_dir outputs/preanalysis/filter_tolerance_analysis

Dash-style aliases (``--base-combo``, ``--data-dir``, ``--output-dir``,
``--offset-range``) are accepted for backward compatibility.

Outputs (under ``--output_dir``)
--------------------------------
* ``filter_tolerance_results.json`` : config + base_iou + per-combo
                                       (offsets, bands, iou, duration_s)
* ``filter_tolerance_heatmaps.png`` : 1x5 figure, one heatmap per offset of
                                       the third band
* ``filter_tolerance_distance.png`` : Manhattan distance vs IoU (mean+/-std)
* ``filter_tolerance_3d.png``       : 3D scatter (offsets vs IoU)
* ``filter_tolerance_report.txt``   : base IoU, +/-1 and +/-2 neighbourhood
                                       stats, top-5, worst-5
* ``filter_tolerance_analysis.log``

Estimated runtime
-----------------
Per spec: 80 epoch baseline ~= 30 min on a single GPU, so 30 epoch ~= 12 min.
~125 combos * 12 min ~= 25 h.  If that exceeds your budget, drop to
``--offset_range 1`` (27 raw candidates).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
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
DEFAULT_DATA_DIR = "/root/autodl-tmp/hsi"
DEFAULT_OUTPUT_DIR = "outputs/preanalysis/filter_tolerance_analysis"
DEFAULT_CONFIG = "../msi/configs/baseline.yaml"
DEFAULT_MAX_BAND = 90  # 91-band cropped HSI => valid range [0, 90]


def _parse_combo(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected exactly 3 comma-separated band indices, got {len(parts)}: {s!r}"
        )
    return [int(p) for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_combo", "--base-combo", type=_parse_combo,
                   default=list(DEFAULT_BASE),
                   help="Comma-separated 3-band base combo (default 57,62,70)")
    p.add_argument("--offset_range", "--offset-range", type=int, default=2, choices=[1, 2],
                   help="Offset range per filter (default 2 -> up to 5^3 = 125 raw candidates)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Epochs per combo (default 30)")
    p.add_argument("--data_dir", "--data-dir", "--data-root", default=DEFAULT_DATA_DIR, type=str,
                   help="HSI dataset directory passed to train_and_eval")
    p.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR, type=str,
                   help="Full output directory for this run (not a parent root)")
    p.add_argument("--msi-root", "--msi_root", default=DEFAULT_MSI_ROOT, type=str)
    p.add_argument("--config", default=DEFAULT_CONFIG, type=str,
                   help="Baseline training config (default ../msi/configs/baseline.yaml)")
    p.add_argument("--max-band", "--max_band", type=int, default=DEFAULT_MAX_BAND,
                   help="Inclusive upper bound on valid band index (91-band cropped HSI -> 90)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-resume", "--no_resume", action="store_true")
    p.add_argument("--dry-run", "--dry_run", action="store_true",
                   help="Skip training, fill iou=NaN; useful for plumbing check")
    p.add_argument("--skip-sanity", "--skip_sanity", action="store_true",
                   help="Skip the base-combo sanity check (NOT recommended)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Combo generation

def generate_combos(base: list[int], R: int, max_band: int) -> list[dict]:
    """Generate de-duplicated, in-range offset combos.

    Constraints:
      * each band in [0, max_band]
      * three distinct bands
      * deduplicate by sorted band tuple ([57,62,70] == [70,62,57])
    The base combo (offsets=[0,0,0]) is always first in the returned list.
    """
    offsets = list(range(-R, R + 1))
    out: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    base_entry = {
        "offsets": [0, 0, 0],
        "bands": [int(b) for b in base],
    }
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
        out.append({
            "offsets": [int(o1), int(o2), int(o3)],
            "bands": [int(b[0]), int(b[1]), int(b[2])],
        })
    return out


def combo_key(bands) -> str:
    return ",".join(str(int(x)) for x in bands)


def manhattan(offsets) -> int:
    return int(sum(abs(int(o)) for o in offsets))


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    setup_msi_path(args.msi_root)
    from _adapters import run_training  # noqa: E402

    base = list(args.base_combo)
    combos = generate_combos(base, args.offset_range, args.max_band)
    log.info("base combo: %s", base)
    log.info("generated %d unique in-range combos (raw=%d)",
             len(combos), (2 * args.offset_range + 1) ** 3)

    results_path = out_dir / "filter_tolerance_results.json"
    existing: dict[str, dict] = {}
    if results_path.exists() and not args.no_resume:
        try:
            prev = json.loads(results_path.read_text())
            for entry in prev.get("results", []):
                existing[combo_key(entry["bands"])] = entry
            log.info("resuming: %d combos already done", len(existing))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse existing results (%s); starting fresh", e)
            existing = {}

    results: dict[str, dict] = dict(existing)

    # ------------------------------------------------------------------
    # Sanity check: actually train the base combo first.
    # ------------------------------------------------------------------
    base_key = combo_key(base)
    base_done = base_key in results and results[base_key].get("iou") is not None and \
        math.isfinite(float(results[base_key]["iou"])) and float(results[base_key]["iou"]) > 0
    if not args.dry_run and not args.skip_sanity and not base_done:
        log.info("=== sanity check: training base combo %s ===", base)
        t0 = time.time()
        try:
            iou = run_training(
                band_indices=base,
                config_path=args.config,
                epochs=args.epochs,
                seed=args.seed,
                data_dir=args.data_dir,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("base combo training raised: %s", e)
            log.error("aborting: train_and_eval is not callable with the current signature; "
                      "edit scripts/preanalysis/_adapters.py::run_training")
            return 2
        dt = time.time() - t0
        log.info("base combo IoU=%.4f (%.1fs)", iou, dt)
        if not math.isfinite(iou) or iou <= 0:
            log.error("base combo IoU is %.4f (non-finite or <=0). "
                      "This suggests train_and_eval is returning a cached/lookup "
                      "result instead of training; fix the upstream function or "
                      "edit _adapters.py.", iou)
            return 2
        if dt < 5.0:
            log.warning("base combo training took only %.1fs -- suspiciously fast; "
                        "verify train_and_eval really trained instead of pulling "
                        "from a precomputed table.", dt)
        results[base_key] = {
            "offsets": [0, 0, 0], "bands": base,
            "iou": float(iou), "duration_s": float(dt),
        }
        _write_results(results_path, args, base, results)

    # ------------------------------------------------------------------
    # Train each combo
    # ------------------------------------------------------------------
    for entry in tqdm(combos, desc="combos"):
        key = combo_key(entry["bands"])
        if key in results:
            continue
        if args.dry_run:
            results[key] = {**entry, "iou": float("nan"), "duration_s": 0.0, "dry_run": True}
            _write_results(results_path, args, base, results)
            continue

        t0 = time.time()
        try:
            iou = run_training(
                band_indices=entry["bands"],
                config_path=args.config,
                epochs=args.epochs,
                seed=args.seed,
                data_dir=args.data_dir,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("training failed for combo %s: %s", entry["bands"], e)
            iou = float("nan")
        dt = time.time() - t0
        results[key] = {**entry, "iou": float(iou), "duration_s": float(dt)}
        log.info("combo=%s offsets=%s iou=%.4f (%.1fs)",
                 entry["bands"], entry["offsets"], iou, dt)
        _write_results(results_path, args, base, results)

    # ------------------------------------------------------------------
    # Aggregate + plots + report
    # ------------------------------------------------------------------
    summary = _summary_stats(results, base)
    log.info("summary: %s", summary)
    _write_results(results_path, args, base, results, summary=summary)

    _plot_heatmaps(out_dir / "filter_tolerance_heatmaps.png", results, base, args.offset_range)
    _plot_distance_curve(out_dir / "filter_tolerance_distance.png", results, summary)
    _plot_3d(out_dir / "filter_tolerance_3d.png", results, base)
    _write_report(out_dir / "filter_tolerance_report.txt", results, base, summary, args)

    log.info("done. outputs in %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Persistence

def _write_results(path: Path, args, base: list[int], results: dict,
                   summary: dict | None = None) -> None:
    payload = {
        "config": {
            "base_combo": list(base),
            "offset_range": args.offset_range,
            "epochs": args.epochs,
            "seed": args.seed,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "config_path": args.config,
            "max_band": args.max_band,
            "note": (f"IoU values are 'best class-1 IoU' from {args.epochs}-epoch training; "
                     "NOT directly comparable to the full 80-epoch baseline numbers."),
        },
        "base_iou": float(results.get(combo_key(base), {}).get("iou", float("nan"))),
        "results": list(results.values()),
    }
    if summary is not None:
        payload["summary"] = summary
    save_json(payload, path)


# ---------------------------------------------------------------------------
# Summary stats

def _summary_stats(results: dict, base: list[int]) -> dict:
    base_iou = float(results.get(combo_key(base), {}).get("iou", float("nan")))

    def stats_for_radius(R: int) -> dict:
        ious = []
        for v in results.values():
            offs = v["offsets"]
            if max(abs(o) for o in offs) <= R:
                iou = v.get("iou")
                if iou is None or not math.isfinite(float(iou)):
                    continue
                ious.append(float(iou))
        if not ious:
            return {"n": 0}
        arr = np.array(ious)
        out = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        if math.isfinite(base_iou):
            drops = base_iou - arr
            out["frac_drop_lt_0.01"] = float((drops < 0.01).mean())
            out["frac_drop_lt_0.02"] = float((drops < 0.02).mean())
            out["frac_drop_lt_0.05"] = float((drops < 0.05).mean())
        return out

    def stats_at_distance(d: int) -> dict:
        ious = []
        for v in results.values():
            if manhattan(v["offsets"]) != d:
                continue
            iou = v.get("iou")
            if iou is None or not math.isfinite(float(iou)):
                continue
            ious.append(float(iou))
        if not ious:
            return {"n": 0}
        arr = np.array(ious)
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    max_d = max((manhattan(v["offsets"]) for v in results.values()), default=0)
    return {
        "base_iou": base_iou,
        "neighborhood_R1": stats_for_radius(1),
        "neighborhood_R2": stats_for_radius(2),
        "by_distance": {int(d): stats_at_distance(d) for d in range(0, max_d + 1)},
    }


# ---------------------------------------------------------------------------
# Plots

def _finite_iou_range(results: dict) -> tuple[float, float] | None:
    vals = [float(v["iou"]) for v in results.values()
            if v.get("iou") is not None and math.isfinite(float(v["iou"]))]
    if not vals:
        return None
    return float(min(vals)), float(max(vals))


def _plot_heatmaps(path: Path, results: dict, base: list[int], R: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = _finite_iou_range(results)
    if rng is None:
        return
    vmin, vmax = rng

    side = 2 * R + 1
    offsets = list(range(-R, R + 1))
    fig, axes = plt.subplots(1, side, figsize=(3.2 * side, 3.6), sharey=True)
    if side == 1:
        axes = [axes]

    last_im = None
    for k, o3 in enumerate(offsets):
        grid = np.full((side, side), np.nan)
        for v in results.values():
            o1, o2, o3v = v["offsets"]
            if o3v != o3:
                continue
            iou = v.get("iou")
            if iou is None or not math.isfinite(float(iou)):
                continue
            grid[offsets.index(o2), offsets.index(o1)] = float(iou)
        ax = axes[k]
        last_im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                            extent=[-R - 0.5, R + 0.5, -R - 0.5, R + 0.5])
        ax.set_xticks(offsets)
        ax.set_yticks(offsets)
        ax.set_xlabel(f"offset b1 (base={base[0]})")
        if k == 0:
            ax.set_ylabel(f"offset b2 (base={base[1]})")
        ax.set_title(f"offset b3 = {o3:+d}\n(b3 base={base[2]})")
        for o1 in offsets:
            for o2 in offsets:
                v = grid[offsets.index(o2), offsets.index(o1)]
                if not math.isnan(v):
                    ax.text(o1, o2, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v < (vmin + vmax) / 2 else "black")
        # Mark base cell (0,0) with a red box when this is the base slice (o3=0)
        if o3 == 0:
            ax.plot(0, 0, marker="s", markersize=24, markerfacecolor="none",
                    markeredgecolor="red", markeredgewidth=2.0)

    fig.colorbar(last_im, ax=axes, shrink=0.85, label="defect IoU",
                 location="right", pad=0.02)
    fig.suptitle(f"Filter-shift tolerance heatmaps (base combo = {base})", y=1.03)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_distance_curve(path: Path, results: dict, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_d = summary.get("by_distance", {})
    if not by_d:
        return
    distances = sorted(int(d) for d, s in by_d.items() if s.get("n", 0) > 0)
    if not distances:
        return

    means = [by_d[str(d) if isinstance(list(by_d.keys())[0], str) else d]["mean"] for d in distances]
    stds = [by_d[str(d) if isinstance(list(by_d.keys())[0], str) else d]["std"] for d in distances]
    ns = [by_d[str(d) if isinstance(list(by_d.keys())[0], str) else d]["n"] for d in distances]

    # All individual points
    scatter_x: list[int] = []
    scatter_y: list[float] = []
    for v in results.values():
        iou = v.get("iou")
        if iou is None or not math.isfinite(float(iou)):
            continue
        scatter_x.append(manhattan(v["offsets"]))
        scatter_y.append(float(iou))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(scatter_x, scatter_y, s=30, color="tab:blue", alpha=0.45, label="individual combos")
    ax.errorbar(distances, means, yerr=stds, fmt="o-", color="tab:red", lw=2,
                capsize=4, label="mean +/- std", markersize=8, zorder=5)
    base_iou = summary.get("base_iou")
    if base_iou is not None and math.isfinite(base_iou):
        ax.axhline(base_iou, color="grey", linestyle="--", lw=1,
                   label=f"base IoU = {base_iou:.4f}")
    for d, n, m in zip(distances, ns, means):
        ax.annotate(f"n={n}", (d, m), xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color="tab:red")
    ax.set_xlabel("Manhattan distance to base combo (sum of |offsets|)")
    ax.set_ylabel("defect IoU")
    ax.set_title("Tolerance vs distance from base combo")
    ax.set_xticks(distances)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_3d(path: Path, results: dict, base: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    offs = []
    ious = []
    for v in results.values():
        iou = v.get("iou")
        if iou is None or not math.isfinite(float(iou)):
            continue
        offs.append(v["offsets"])
        ious.append(float(iou))
    if not offs:
        return
    offs = np.asarray(offs)
    ious = np.asarray(ious)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(offs[:, 0], offs[:, 1], offs[:, 2], c=ious, cmap="viridis",
                    s=80, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"offset to band {base[0]}")
    ax.set_ylabel(f"offset to band {base[1]}")
    ax.set_zlabel(f"offset to band {base[2]}")
    ax.set_title(f"Filter tolerance: defect IoU vs filter offsets (base={base})")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="defect IoU")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report

def _write_report(path: Path, results: dict, base: list[int], summary: dict, args) -> None:
    lines = ["Filter-shift tolerance report", "=" * 60, ""]
    lines.append(f"Base combo (91-band cropped HSI index): {base}")
    lines.append(f"Epochs per combo: {args.epochs}  (seed={args.seed})")
    lines.append(f"Note: {args.epochs}-epoch IoU is NOT directly comparable to the "
                 "80-epoch baseline IoU (e.g. 0.7293).  Treat the base IoU below as "
                 "this run's reference point.")
    lines.append("")
    base_iou = summary["base_iou"]
    lines.append(f"Base IoU (this run, {args.epochs} ep): {base_iou:.4f}"
                 if math.isfinite(base_iou) else "Base IoU (this run): not finite")
    lines.append("")

    for r_name, r_stats in [("R = +/-1", summary["neighborhood_R1"]),
                            ("R = +/-2", summary["neighborhood_R2"])]:
        lines.append(f"[{r_name}] n = {r_stats.get('n', 0)}")
        if r_stats.get("n", 0):
            lines.append(f"  IoU mean = {r_stats['mean']:.4f}  std = {r_stats['std']:.4f}  "
                         f"min = {r_stats['min']:.4f}  max = {r_stats['max']:.4f}")
            if "frac_drop_lt_0.01" in r_stats:
                lines.append(f"  IoU drop < 0.01 vs base: {r_stats['frac_drop_lt_0.01']:.1%}")
                lines.append(f"  IoU drop < 0.02 vs base: {r_stats['frac_drop_lt_0.02']:.1%}")
                lines.append(f"  IoU drop < 0.05 vs base: {r_stats['frac_drop_lt_0.05']:.1%}")
        lines.append("")

    by_d = summary.get("by_distance", {})
    if by_d:
        lines.append("IoU by Manhattan distance to base:")
        for d in sorted(int(k) for k in by_d.keys()):
            s = by_d[d] if d in by_d else by_d[str(d)]
            if s.get("n", 0):
                lines.append(f"  d={d}: n={s['n']:3d}  mean={s['mean']:.4f}  "
                             f"std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}")
        lines.append("")

    # Top-5 best (including base), worst-5
    rows = []
    for v in results.values():
        iou = v.get("iou")
        if iou is None or not math.isfinite(float(iou)):
            continue
        rows.append((float(iou), v["bands"], v["offsets"]))
    rows.sort(key=lambda r: -r[0])
    lines.append("Top-5 best combos:")
    for r, (iou, bands, offs) in enumerate(rows[:5], 1):
        marker = " (BASE)" if bands == list(base) else ""
        lines.append(f"  {r}. bands={bands}  offsets={offs}  IoU={iou:.4f}{marker}")
    lines.append("")
    lines.append("Worst-5 combos:")
    for r, (iou, bands, offs) in enumerate(rows[-5:][::-1], 1):
        lines.append(f"  {r}. bands={bands}  offsets={offs}  IoU={iou:.4f}")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
