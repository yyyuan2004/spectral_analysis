"""A5 -- Single-band IoU ablation.

Purpose
-------
Answer "if I could only mount one filter, which band gives the best IoU?"
by training a UNet with ``num_channels=1`` on each HSI band in the
search range and recording the defect IoU.

Inputs
------
* baseline config (default ``../msi/configs/baseline.yaml``)
* train_and_eval entry point in msi (called via ``_adapters.run_training``)
* optional: A1's ``per_band_stats.json`` to overlay Fisher/AUC curves

Outputs (under ``outputs/preanalysis/single_band_ablation/``)
------------------------------------------------------------
* ``single_band_results.json``           : {band: iou} plus run metadata
* ``single_band_iou_curve.png``          : IoU vs band, with Fisher/AUC overlay
* ``single_band_ablation_report.txt``    : top-5 by IoU and by Fisher
* ``single_band_ablation.log``

Resume
------
Results are flushed to disk after every band, and a partial run can be
resumed by re-invoking the script -- already-trained bands are skipped
unless ``--no-resume`` is given.

Typical runtime
---------------
51 bands * 80 epochs.  Budget per spec: < 12 h alongside A1 + A2.
That is ~12 min per band; tune ``--epochs`` if your GPU is slower.
"""
from __future__ import annotations

import argparse
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
    top_k_indices,
)

SCRIPT_NAME = "single_band_ablation"
DEFAULT_CONFIG = "../msi/configs/baseline.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    p.add_argument("--msi-root", default=DEFAULT_MSI_ROOT, type=str)
    p.add_argument("--config", default=DEFAULT_CONFIG, type=str)
    p.add_argument("--band-start", type=int, default=60)
    p.add_argument("--band-end", type=int, default=110, help="Inclusive end")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-resume", action="store_true",
                   help="Restart from scratch even if previous results exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate adapter and skip actual training")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_output_dir(SCRIPT_NAME, args.output_root)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    setup_msi_path(args.msi_root)
    from _adapters import run_training  # noqa: E402

    bands = list(range(int(args.band_start), int(args.band_end) + 1))
    log.info("scanning %d bands: %d..%d", len(bands), bands[0], bands[-1])

    results_path = out_dir / "single_band_results.json"
    existing: dict[str, float] = {}
    if results_path.exists() and not args.no_resume:
        try:
            existing = json.loads(results_path.read_text()).get("results", {})
            existing = {str(k): float(v) for k, v in existing.items()}
            log.info("resuming: %d bands already done", len(existing))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse existing results (%s); starting fresh", e)
            existing = {}

    results: dict[str, float] = dict(existing)
    timings: dict[str, float] = {}

    for band in tqdm(bands, desc="bands"):
        key = str(band)
        if key in results:
            continue
        if args.dry_run:
            log.info("[dry-run] band=%d", band)
            results[key] = float("nan")
            continue

        t0 = time.time()
        try:
            iou = run_training(
                band_indices=[int(band)],
                config_path=args.config,
                epochs=int(args.epochs),
                seed=int(args.seed),
                num_channels=1,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("training failed for band=%d: %s", band, e)
            iou = float("nan")
        dt = time.time() - t0
        timings[key] = dt
        results[key] = float(iou)
        log.info("band=%d iou=%.4f (%.1fs)", band, iou, dt)

        # Persist every iter to support resume
        _write_results(results_path, args, results, timings)

    fisher, auc = _load_a1_overlay(args.output_root, log)
    _plot_curve(out_dir / "single_band_iou_curve.png", results, fisher, auc)
    _write_report(out_dir / "single_band_ablation_report.txt", results, fisher)

    log.info("done. outputs in %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Helpers

def _write_results(path: Path, args, results: dict, timings: dict) -> None:
    save_json({
        "config": {
            "data_root": args.data_root,
            "msi_root": args.msi_root,
            "config_path": args.config,
            "band_start": args.band_start,
            "band_end": args.band_end,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "results": results,
        "timings_seconds": timings,
    }, path)


def _load_a1_overlay(output_root: str, log) -> tuple[dict | None, dict | None]:
    p = Path(output_root) / "spectral_curves_with_separability" / "per_band_stats.json"
    if not p.exists():
        log.info("A1 results not found at %s; skipping overlay", p)
        return None, None
    try:
        data = json.loads(p.read_text())
        fisher = {int(d["band"]): float(d["fisher"]) for d in data["per_band"]}
        auc = {int(d["band"]): float(d["auc"]) for d in data["per_band"]}
        return fisher, auc
    except Exception as e:  # noqa: BLE001
        log.warning("could not parse A1 results: %s", e)
        return None, None


def _plot_curve(path: Path, results: dict, fisher: dict | None, auc: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = sorted(int(b) for b in results.keys())
    iou = np.array([results[str(b)] for b in bands], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(bands, iou, "-o", color="tab:blue", lw=1.5, label="defect IoU")
    ax1.set_xlabel("HSI band index")
    ax1.set_ylabel("defect IoU", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    if fisher or auc:
        ax2 = ax1.twinx()
        if fisher:
            f = np.array([fisher.get(b, np.nan) for b in bands])
            ax2.plot(bands, f, "--", color="tab:green", lw=1.2, label="Fisher (A1)")
        if auc:
            a = np.array([abs(auc.get(b, 0.5) - 0.5) for b in bands])
            ax2.plot(bands, a, ":", color="tab:orange", lw=1.2, label="|AUC-0.5| (A1)")
        ax2.set_ylabel("separability (A1)", color="tab:gray")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    ax1.set_title("Single-band ablation: defect IoU vs HSI band")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_report(path: Path, results: dict, fisher: dict | None) -> None:
    bands = sorted(int(b) for b in results.keys())
    iou = np.array([results[str(b)] for b in bands], dtype=np.float64)
    finite = ~np.isnan(iou)
    if not finite.any():
        path.write_text("No finite IoU values recorded.\n")
        return
    top5_iou_idx = np.argsort(np.where(finite, iou, -np.inf))[::-1][:5]
    top5_iou = [(bands[i], float(iou[i])) for i in top5_iou_idx]

    lines = ["Single-band ablation report", "=" * 50, ""]
    lines.append("Top-5 single bands by IoU")
    for r, (b, v) in enumerate(top5_iou, 1):
        lines.append(f"  {r}. band {b:3d}  IoU={v:.4f}")
    lines.append("")
    if fisher:
        top5_fisher = sorted(fisher.items(), key=lambda kv: -kv[1])[:5]
        lines.append("Top-5 single bands by Fisher (from A1)")
        for r, (b, f) in enumerate(top5_fisher, 1):
            lines.append(f"  {r}. band {b:3d}  Fisher={f:.4f}  IoU={results.get(str(b), float('nan')):.4f}")
        lines.append("")
        top_iou_set = {b for b, _ in top5_iou}
        top_fisher_set = {b for b, _ in top5_fisher}
        overlap = top_iou_set & top_fisher_set
        lines.append(f"Overlap of top-5 sets: {sorted(overlap)} ({len(overlap)}/5)")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
