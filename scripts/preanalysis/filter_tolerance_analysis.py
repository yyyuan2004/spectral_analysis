"""A3 -- Filter-shift tolerance for the chosen 3-band combination.

Purpose
-------
Quantify how robust the chosen 3-band combo is to small, hardware-realistic
shifts of each filter centre wavelength (+/-1 and +/-2 HSI bands).  Each
neighbour combo is re-trained from scratch (same baseline config).

Inputs
------
* base combination (default ``77 82 90``)
* baseline config (default ``../msi/configs/baseline.yaml``)
* train_and_eval entry point in msi (via ``_adapters.run_training``)

Outputs (under ``outputs/preanalysis/filter_tolerance_analysis/``)
-----------------------------------------------------------------
* ``filter_tolerance_results.json``    : per-combo IoU, offsets, ordering
* ``filter_tolerance_3d.png``          : 3D scatter (axes = offsets, colour = IoU)
* ``filter_tolerance_heatmaps.png``    : 5 heatmaps (one per offset of band 3)
* ``filter_tolerance_report.txt``      : neighborhood stats, tolerance fraction
* ``filter_tolerance_analysis.log``

Typical runtime
---------------
Up to 125 trainings, but the cartesian product is de-duplicated and
clipped to the valid HSI range.  Budget per spec: < 8 h.  Run is
resumable -- every combo is flushed to disk immediately.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

from _common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MSI_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SEED,
    make_output_dir,
    save_json,
    set_seed,
    setup_logging,
    setup_msi_path,
)

SCRIPT_NAME = "filter_tolerance_analysis"
DEFAULT_CONFIG = "../msi/configs/baseline.yaml"
DEFAULT_BASE = [77, 82, 90]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    p.add_argument("--msi-root", default=DEFAULT_MSI_ROOT, type=str)
    p.add_argument("--config", default=DEFAULT_CONFIG, type=str)
    p.add_argument("--base-combo", type=int, nargs=3, default=DEFAULT_BASE,
                   help="Three HSI band indices used as the base combo")
    p.add_argument("--offset-range", type=int, default=2,
                   help="Generate offsets in [-R, +R] per band (default 2 -> 5^3 = 125)")
    p.add_argument("--max-band", type=int, default=None,
                   help="Inclusive upper bound on valid HSI band index (auto-detect if omitted)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_output_dir(SCRIPT_NAME, args.output_root)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    setup_msi_path(args.msi_root)
    from _adapters import run_training  # noqa: E402

    max_band = args.max_band if args.max_band is not None else _auto_max_band(args.data_root, log)
    log.info("HSI band range: [0, %d]", max_band)

    base = tuple(int(b) for b in args.base_combo)
    log.info("base combo: %s", base)

    combos = _generate_combos(base, args.offset_range, max_band)
    log.info("generated %d unique in-range combos", len(combos))

    results_path = out_dir / "filter_tolerance_results.json"
    existing: dict[str, dict] = {}
    if results_path.exists() and not args.no_resume:
        try:
            existing = {k: v for k, v in json.loads(results_path.read_text()).get("results", {}).items()}
            log.info("resuming: %d combos already done", len(existing))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse existing results (%s); starting fresh", e)

    results: dict[str, dict] = dict(existing)
    timings: dict[str, float] = {}

    for entry in tqdm(combos, desc="combos"):
        key = _combo_key(entry["hsi_bands"])
        if key in results:
            continue
        if args.dry_run:
            results[key] = {**entry, "iou": float("nan"), "dry_run": True}
            continue

        t0 = time.time()
        try:
            iou = run_training(
                band_indices=entry["hsi_bands"],
                config_path=args.config,
                epochs=args.epochs,
                seed=args.seed,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("training failed for combo %s: %s", entry["hsi_bands"], e)
            iou = float("nan")
        dt = time.time() - t0
        timings[key] = dt
        results[key] = {**entry, "iou": float(iou)}
        log.info("combo=%s iou=%.4f (%.1fs)", entry["hsi_bands"], iou, dt)

        _write_results(results_path, args, base, results, timings)

    _write_results(results_path, args, base, results, timings)
    summary = _summary_stats(results, base)
    log.info("summary: %s", summary)

    _plot_3d(out_dir / "filter_tolerance_3d.png", results, base)
    _plot_heatmaps(out_dir / "filter_tolerance_heatmaps.png", results, base, args.offset_range)
    _write_report(out_dir / "filter_tolerance_report.txt", results, base, summary)

    log.info("done. outputs in %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Helpers

def _auto_max_band(data_root: str, log) -> int:
    from _common import list_stems, load_hsi

    stems = list_stems(Path(data_root))
    img = load_hsi(Path(data_root) / "images" / f"{stems[0]}.npy")
    B = img.shape[-1]
    log.info("auto-detected HSI band count from %s: B=%d", stems[0], B)
    return B - 1


def _generate_combos(base: tuple[int, int, int], R: int, max_band: int) -> list[dict]:
    offsets = list(range(-R, R + 1))
    out: list[dict] = []
    seen: set[tuple[int, int, int]] = set()
    for o1, o2, o3 in itertools.product(offsets, offsets, offsets):
        b = (base[0] + o1, base[1] + o2, base[2] + o3)
        if any(x < 0 or x > max_band for x in b):
            continue
        if len(set(b)) != 3:
            continue  # filter collisions (two filters at same wavelength is pointless)
        bs = tuple(sorted(b))
        if bs in seen:
            continue  # different orderings of the same 3 bands count once
        seen.add(bs)
        out.append({
            "offsets": [int(o1), int(o2), int(o3)],
            "hsi_bands": [int(b[0]), int(b[1]), int(b[2])],
        })
    return out


def _combo_key(hsi_bands) -> str:
    return ",".join(str(int(x)) for x in hsi_bands)


def _write_results(path: Path, args, base, results: dict, timings: dict) -> None:
    save_json({
        "config": {
            "data_root": args.data_root,
            "msi_root": args.msi_root,
            "config_path": args.config,
            "base_combo": list(base),
            "offset_range": args.offset_range,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "results": results,
        "timings_seconds": timings,
    }, path)


def _summary_stats(results: dict, base) -> dict:
    base_key = _combo_key(base)
    base_iou = float(results.get(base_key, {}).get("iou", float("nan")))

    def neighborhood_stats(r: int) -> dict:
        ious = []
        for v in results.values():
            offs = v["offsets"]
            if max(abs(o) for o in offs) <= r:
                iou = v.get("iou")
                if iou is None or np.isnan(iou):
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
        if not np.isnan(base_iou):
            drops = base_iou - arr
            out["frac_drop_lt_0.01"] = float((drops < 0.01).mean())
            out["frac_drop_lt_0.02"] = float((drops < 0.02).mean())
        return out

    return {
        "base_iou": base_iou,
        "neighborhood_R1": neighborhood_stats(1),
        "neighborhood_R2": neighborhood_stats(2),
    }


def _plot_3d(path: Path, results: dict, base) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    offs = np.array([v["offsets"] for v in results.values()])
    ious = np.array([float(v.get("iou", np.nan)) for v in results.values()])
    finite = ~np.isnan(ious)
    if not finite.any():
        return
    offs = offs[finite]
    ious = ious[finite]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(offs[:, 0], offs[:, 1], offs[:, 2], c=ious, cmap="viridis",
                    s=80, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"offset to band {base[0]}")
    ax.set_ylabel(f"offset to band {base[1]}")
    ax.set_zlabel(f"offset to band {base[2]}")
    ax.set_title(f"Filter tolerance: defect IoU vs filter offsets (base={list(base)})")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="defect IoU")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_heatmaps(path: Path, results: dict, base, R: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    side = 2 * R + 1
    offsets = list(range(-R, R + 1))
    all_iou = [v.get("iou") for v in results.values() if v.get("iou") is not None]
    finite_iou = [v for v in all_iou if not np.isnan(v)]
    if not finite_iou:
        return
    vmin, vmax = float(np.min(finite_iou)), float(np.max(finite_iou))

    n_cols = side
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5), sharey=True)
    if n_cols == 1:
        axes = [axes]
    for k, o3 in enumerate(offsets):
        grid = np.full((side, side), np.nan)
        for v in results.values():
            o1, o2, o3v = v["offsets"]
            if o3v != o3:
                continue
            iou = v.get("iou")
            if iou is None or np.isnan(iou):
                continue
            grid[offsets.index(o2), offsets.index(o1)] = float(iou)
        ax = axes[k]
        im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                       extent=[-R - 0.5, R + 0.5, -R - 0.5, R + 0.5])
        ax.set_xticks(offsets)
        ax.set_yticks(offsets)
        ax.set_xlabel(f"offset b1 ({base[0]})")
        if k == 0:
            ax.set_ylabel(f"offset b2 ({base[1]})")
        ax.set_title(f"offset b3={o3:+d}")
        for o1 in offsets:
            for o2 in offsets:
                v = grid[offsets.index(o2), offsets.index(o1)]
                if not np.isnan(v):
                    ax.text(o1, o2, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v < (vmin + vmax) / 2 else "black")

    fig.colorbar(im, ax=axes, shrink=0.85, label="defect IoU", location="right", pad=0.02)
    fig.suptitle(f"Filter-shift tolerance heatmaps, base combo = {list(base)}", y=1.02)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, results: dict, base, summary: dict) -> None:
    lines = ["Filter-shift tolerance report", "=" * 50, ""]
    lines.append(f"Base combo: {list(base)}  (IoU={summary['base_iou']:.4f})")
    lines.append("")
    for r_name, r_stats in [("R=+/-1", summary["neighborhood_R1"]),
                            ("R=+/-2", summary["neighborhood_R2"])]:
        lines.append(f"[{r_name}] n={r_stats.get('n', 0)}")
        if r_stats.get("n", 0):
            lines.append(f"  IoU mean={r_stats['mean']:.4f}  std={r_stats['std']:.4f}  "
                         f"min={r_stats['min']:.4f}  max={r_stats['max']:.4f}")
            if "frac_drop_lt_0.01" in r_stats:
                lines.append(f"  fraction of combos with IoU drop < 0.01 vs base: "
                             f"{r_stats['frac_drop_lt_0.01']:.1%}")
                lines.append(f"  fraction of combos with IoU drop < 0.02 vs base: "
                             f"{r_stats['frac_drop_lt_0.02']:.1%}")
        lines.append("")

    # Top-10 best combos
    rows = []
    for v in results.values():
        iou = v.get("iou")
        if iou is None or np.isnan(iou):
            continue
        rows.append((float(iou), v["hsi_bands"], v["offsets"]))
    rows.sort(key=lambda r: -r[0])
    lines.append("Top-10 best combos in neighborhood:")
    for r, (iou, bands, offs) in enumerate(rows[:10], 1):
        lines.append(f"  {r:2d}. bands={bands}  offsets={offs}  IoU={iou:.4f}")
    lines.append("")
    lines.append("Worst-5 combos in neighborhood:")
    for r, (iou, bands, offs) in enumerate(rows[-5:][::-1], 1):
        lines.append(f"  {r:2d}. bands={bands}  offsets={offs}  IoU={iou:.4f}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
