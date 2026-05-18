"""A4 -- Multi-seed error distance to GT boundary + publication samples.

What changed vs the previous version
------------------------------------
* Single-seed -> 3-seed aggregation (--checkpoints + --seeds).
* Distance buckets [3, 5, 10] -> [0, 5, 10, 20, 50, 100, 200, inf].
* Adds a 6-row publication-quality sample grid:
    pseudo-RGB | GT | pred (baseline seed) | error overlay (FP=red->blue,
    FN=viridis) | distance histogram.
* Saves per-sample standalone PNGs alongside the grid.
* Specular overlap is dropped from this revision -- the previous addition
  was kept simple and is now superseded by the per-sample visualisations
  which make highlight overlap visually obvious in the overlay panel.

CLI
---
  --checkpoints PATH PATH PATH    # baseline seed 42 / 123 / 456 best_model.pth
  --seeds 42 123 456
  --data_dir /root/autodl-tmp/datasets/full_HSI_dataset
  --output_dir outputs/preanalysis/error_distance_to_boundary
  --msi_root ../msi
  --device cuda
  --n_samples 6
  --batch_size 1

Outputs
-------
* error_distance_per_seed.json    : raw per-seed stats
* error_distance_aggregate.json   : mean +/- std across seeds + per-bin frac
* error_distance_report.txt       : per-seed cumulative table + mean +/- std row
* distribution_summary.png        : FP/FN density curves (log x), 3 seeds + mean
* error_samples_grid.png          : 6 samples x 5 panels
* samples/<stem>_panels.png       : each sample as a standalone PNG
* error_distance_to_boundary.log

Typical runtime
---------------
3 seeds * ~55 val images inference + distance transforms + plotting.
Expect 15-40 minutes on a single GPU.
"""
from __future__ import annotations

import argparse
import math
import sys
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

SCRIPT_NAME = "error_distance_to_boundary"
DEFAULT_DATA_DIR = "/root/autodl-tmp/datasets/full_HSI_dataset"
DEFAULT_OUTPUT_DIR = "outputs/preanalysis/error_distance_to_boundary"
DEFAULT_CHECKPOINTS = [
    "../msi/outputs/baseline_seed42/checkpoints/best_model.pth",
    "../msi/outputs/baseline_seed123/checkpoints/best_model.pth",
    "../msi/outputs/baseline_seed456/checkpoints/best_model.pth",
]
DEFAULT_SEEDS = [42, 123, 456]
DISTANCE_BIN_EDGES = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, float("inf")]
DISTANCE_BIN_LABELS = ["0-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200+"]
CUMULATIVE_THRESHOLDS = [5, 10, 20, 50, 100, 200]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS, type=str,
                   help="Per-seed best_model.pth paths (length must match --seeds)")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--data_dir", "--data-dir", "--data-root", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR, type=str,
                   help="Full output directory for this run (used as-is)")
    p.add_argument("--msi_root", "--msi-root", default=DEFAULT_MSI_ROOT, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--n_samples", "--n-samples", type=int, default=6,
                   help="Number of sample images to render (2 high, 2 mid, 2 low IoU)")
    p.add_argument("--batch_size", "--batch-size", type=int, default=1)
    p.add_argument("--num_workers", "--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if len(args.checkpoints) != len(args.seeds):
        print(f"ERROR: --checkpoints ({len(args.checkpoints)}) and --seeds "
              f"({len(args.seeds)}) lengths must match", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    setup_msi_path(args.msi_root)
    from _adapters import get_val_loader, load_trained_model, unpack_batch  # noqa: E402
    import torch  # type: ignore

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    pairs = list(zip(args.seeds, args.checkpoints))
    baseline_seed = pairs[0][0]
    band_indices_cache: list[int] | None = None

    per_seed: dict[int, list[dict]] = {}
    input_cache: dict[str, np.ndarray] = {}
    gt_cache: dict[str, np.ndarray] = {}
    pred_cache: dict[str, np.ndarray] = {}
    dist_cache: dict[str, np.ndarray] = {}

    for seed, ckpt in pairs:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            log.warning("checkpoint missing for seed %d: %s; skipping", seed, ckpt_path)
            continue
        log.info("=== seed %d: loading %s ===", seed, ckpt_path)

        model, in_channels, band_indices, num_classes = load_trained_model(ckpt_path)
        model = model.to(device).eval()
        if band_indices_cache is None and band_indices:
            band_indices_cache = list(band_indices)

        loader = get_val_loader(
            args.data_dir,
            band_indices=list(band_indices_cache or band_indices or []),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        records: list[dict] = []
        for idx, batch in enumerate(tqdm(loader, desc=f"seed {seed}")):
            img, mask, _whole, stem = unpack_batch(batch)
            if img is None or mask is None:
                log.warning("seed %d: skipping batch %d (missing fields)", seed, idx)
                continue
            stem_str = _stem_str(stem, idx)

            with torch.no_grad():
                logits = model(img.to(device))
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                if logits.dim() == 4 and logits.shape[1] == 1:
                    pred = (torch.sigmoid(logits) > 0.5).long()
                else:
                    pred = logits.argmax(dim=1, keepdim=True)
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
            gt_np = _to_hw(mask)
            if pred_np.shape != gt_np.shape:
                log.warning("shape mismatch %s pred=%s gt=%s", stem_str, pred_np.shape, gt_np.shape)
                continue

            if stem_str not in dist_cache:
                dist_cache[stem_str] = _distance_to_gt_boundary(gt_np)
                gt_cache[stem_str] = gt_np
            dist_map = dist_cache[stem_str]

            fp_mask = (pred_np == 1) & (gt_np == 0)
            fn_mask = (pred_np == 0) & (gt_np == 1)
            fp_d = dist_map[fp_mask]
            fn_d = dist_map[fn_mask]

            inter = int((pred_np & gt_np).sum())
            union = int((pred_np | gt_np).sum())
            iou = inter / union if union > 0 else float("nan")

            records.append({
                "stem": stem_str,
                "iou": float(iou),
                "n_fp": int(fp_mask.sum()),
                "n_fn": int(fn_mask.sum()),
                "fp_distances": fp_d.astype(np.float32),
                "fn_distances": fn_d.astype(np.float32),
            })

            # Cache baseline-seed prediction + once-per-image input
            if seed == baseline_seed:
                if stem_str not in input_cache:
                    input_cache[stem_str] = _to_chw(img)
                pred_cache[stem_str] = pred_np

        per_seed[seed] = records
        log.info("seed %d: collected %d records", seed, len(records))

        del model
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if not per_seed:
        log.error("no seeds completed successfully; aborting")
        return 1

    # ------------------------------------------------------------------
    # Per-seed aggregate distance stats + cumulative tables
    per_seed_summary = _per_seed_summary(per_seed)
    aggregate = _aggregate_across_seeds(per_seed_summary)
    log.info("aggregate summary: %s", aggregate)

    # ------------------------------------------------------------------
    # Per-image mean IoU and sample selection
    all_stems = sorted({r["stem"] for records in per_seed.values() for r in records})
    mean_iou_per_stem = _mean_iou_per_stem(per_seed, all_stems)
    samples = _pick_samples(all_stems, mean_iou_per_stem, args.n_samples)
    log.info("selected %d samples: %s", len(samples), samples)

    # ------------------------------------------------------------------
    # Persist
    _save_per_seed_json(out_dir / "error_distance_per_seed.json",
                         per_seed, per_seed_summary, mean_iou_per_stem)
    _save_aggregate_json(out_dir / "error_distance_aggregate.json",
                          per_seed_summary, aggregate, list(per_seed.keys()),
                          band_indices_cache)

    # ------------------------------------------------------------------
    # Plots
    _plot_distribution_summary(out_dir / "distribution_summary.png", per_seed)
    if samples:
        _plot_error_samples(
            out_dir / "error_samples_grid.png", samples_dir,
            samples, input_cache, gt_cache, pred_cache, dist_cache,
            per_seed, baseline_seed, mean_iou_per_stem, band_indices_cache,
        )

    # ------------------------------------------------------------------
    # Report
    _write_report(out_dir / "error_distance_report.txt",
                  per_seed_summary, aggregate, list(per_seed.keys()),
                  samples, mean_iou_per_stem)

    # ------------------------------------------------------------------
    # Headline
    fn_median = aggregate.get("fn", {}).get("median_mean")
    fp_median = aggregate.get("fp", {}).get("median_mean")
    n_seeds = len(per_seed)
    if fn_median is not None and fp_median is not None:
        headline = (f"Across {n_seeds} seeds: FN concentrates within "
                    f"{fn_median:.1f}px (median), FP scatters at "
                    f"{fp_median:.1f}px (median)")
    else:
        headline = f"Across {n_seeds} seeds: no FP/FN pixels recorded"
    log.info(headline)
    print(headline)
    return 0


# ---------------------------------------------------------------------------
# Numeric helpers

def _stem_str(stem, idx: int) -> str:
    if stem is None:
        return f"img_{idx:04d}"
    if isinstance(stem, (list, tuple)) and stem:
        s = stem[0]
    else:
        s = stem
    try:
        if hasattr(s, "item"):
            s = s.item()
    except Exception:  # noqa: BLE001
        pass
    return str(s)


def _to_hw(t) -> np.ndarray:
    try:
        import torch  # type: ignore
        is_tensor = isinstance(t, torch.Tensor)
    except ImportError:
        is_tensor = False
    arr = t.detach().cpu().numpy() if is_tensor else np.asarray(t)
    arr = arr.squeeze()
    return (arr > 0).astype(np.uint8)


def _to_chw(t) -> np.ndarray:
    try:
        import torch  # type: ignore
        is_tensor = isinstance(t, torch.Tensor)
    except ImportError:
        is_tensor = False
    arr = t.detach().cpu().numpy() if is_tensor else np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] >= arr.shape[-1]:
        # already (H, W, C); transpose to (C, H, W) for consistency
        arr = np.moveaxis(arr, -1, 0)
    return arr.astype(np.float32)


def _distance_to_gt_boundary(gt: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    if gt.sum() == 0 or gt.sum() == gt.size:
        return np.full(gt.shape, 1e6, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
    src = ((~boundary).astype(np.uint8)) * 255
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def _bin_fractions(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_counts, bin_fracs, cumulative_fracs).

    cumulative_fracs aligns with CUMULATIVE_THRESHOLDS.
    """
    if len(d) == 0:
        return (np.zeros(len(DISTANCE_BIN_LABELS)),
                np.zeros(len(DISTANCE_BIN_LABELS)),
                np.zeros(len(CUMULATIVE_THRESHOLDS)))
    counts, _ = np.histogram(d, bins=DISTANCE_BIN_EDGES)
    fracs = counts.astype(np.float64) / max(int(counts.sum()), 1)
    cum = np.array([float((d <= t).mean()) for t in CUMULATIVE_THRESHOLDS])
    return counts, fracs, cum


def _per_seed_summary(per_seed: dict[int, list[dict]]) -> dict:
    out: dict = {}
    for seed, records in per_seed.items():
        if records:
            fp = np.concatenate([r["fp_distances"] for r in records if len(r["fp_distances"])]) \
                if any(len(r["fp_distances"]) for r in records) else np.empty(0)
            fn = np.concatenate([r["fn_distances"] for r in records if len(r["fn_distances"])]) \
                if any(len(r["fn_distances"]) for r in records) else np.empty(0)
        else:
            fp = np.empty(0); fn = np.empty(0)

        def stats(d: np.ndarray) -> dict:
            counts, fracs, cum = _bin_fractions(d)
            return {
                "n": int(len(d)),
                "mean": float(d.mean()) if len(d) else float("nan"),
                "median": float(np.median(d)) if len(d) else float("nan"),
                "p90": float(np.percentile(d, 90)) if len(d) else float("nan"),
                "bin_counts": counts.astype(int).tolist(),
                "bin_fractions": fracs.tolist(),
                "cumulative": {str(int(t)): float(c)
                               for t, c in zip(CUMULATIVE_THRESHOLDS, cum)},
            }

        out[seed] = {"fp": stats(fp), "fn": stats(fn)}
    return out


def _aggregate_across_seeds(per_seed_summary: dict) -> dict:
    """Mean +/- std across seeds of each metric."""
    if not per_seed_summary:
        return {}
    seeds = sorted(per_seed_summary.keys())

    def agg(side: str) -> dict:
        n = [per_seed_summary[s][side]["n"] for s in seeds]
        means = [per_seed_summary[s][side]["mean"] for s in seeds]
        medians = [per_seed_summary[s][side]["median"] for s in seeds]
        p90s = [per_seed_summary[s][side]["p90"] for s in seeds]
        cum = {str(t): [per_seed_summary[s][side]["cumulative"][str(t)] for s in seeds]
               for t in CUMULATIVE_THRESHOLDS}
        bin_fracs = np.stack([np.array(per_seed_summary[s][side]["bin_fractions"])
                              for s in seeds], axis=0)  # (n_seeds, n_bins)

        def m_s(arr):
            a = np.asarray([x for x in arr if not (isinstance(x, float) and math.isnan(x))],
                            dtype=np.float64)
            if not len(a):
                return (float("nan"), float("nan"))
            return (float(a.mean()), float(a.std()))

        return {
            "n_seeds": len(seeds),
            "n_total": int(sum(n)),
            "mean_mean": m_s(means)[0], "mean_std": m_s(means)[1],
            "median_mean": m_s(medians)[0], "median_std": m_s(medians)[1],
            "p90_mean": m_s(p90s)[0], "p90_std": m_s(p90s)[1],
            "cumulative_mean": {t: float(np.mean(v)) for t, v in cum.items()},
            "cumulative_std": {t: float(np.std(v)) for t, v in cum.items()},
            "bin_fractions_mean": bin_fracs.mean(axis=0).tolist(),
            "bin_fractions_std": bin_fracs.std(axis=0).tolist(),
        }

    return {"fp": agg("fp"), "fn": agg("fn")}


def _mean_iou_per_stem(per_seed: dict[int, list[dict]], all_stems: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for stem in all_stems:
        ious = []
        for records in per_seed.values():
            for r in records:
                if r["stem"] == stem and not math.isnan(r["iou"]):
                    ious.append(r["iou"])
        out[stem] = float(np.mean(ious)) if ious else float("nan")
    return out


def _pick_samples(all_stems: list[str], mean_iou: dict[str, float], n: int) -> list[str]:
    finite = [s for s in all_stems if not math.isnan(mean_iou[s])]
    finite.sort(key=lambda s: -mean_iou[s])
    if len(finite) <= n:
        return finite

    # 2 highest, 2 mid, 2 lowest (or pro-rated)
    per_group = max(1, n // 3)
    high = finite[:per_group]
    low = finite[-per_group:]
    mid_count = n - 2 * per_group
    if mid_count > 0:
        mid_start = max(per_group, len(finite) // 2 - mid_count // 2)
        mid = finite[mid_start: mid_start + mid_count]
    else:
        mid = []
    picked = list(dict.fromkeys(high + mid + low))  # dedup preserve order
    # If dedup left holes, top up from remaining sorted list
    remaining = [s for s in finite if s not in picked]
    while len(picked) < n and remaining:
        picked.append(remaining.pop(0))
    return picked[:n]


# ---------------------------------------------------------------------------
# Persistence

def _save_per_seed_json(path: Path, per_seed: dict, per_seed_summary: dict,
                        mean_iou_per_stem: dict) -> None:
    payload = {
        "seeds": sorted(per_seed.keys()),
        "distance_bin_edges": DISTANCE_BIN_EDGES,
        "distance_bin_labels": DISTANCE_BIN_LABELS,
        "cumulative_thresholds": CUMULATIVE_THRESHOLDS,
        "mean_iou_per_stem": mean_iou_per_stem,
        "summary": {str(s): per_seed_summary[s] for s in per_seed_summary},
        "per_image": {
            str(seed): [
                {"stem": r["stem"], "iou": r["iou"],
                 "n_fp": r["n_fp"], "n_fn": r["n_fn"]}
                for r in records
            ]
            for seed, records in per_seed.items()
        },
    }
    save_json(payload, path)


def _save_aggregate_json(path: Path, per_seed_summary: dict, aggregate: dict,
                          seeds: list[int], band_indices: list[int] | None) -> None:
    save_json({
        "seeds": sorted(seeds),
        "band_indices": list(band_indices) if band_indices else [],
        "distance_bin_edges": DISTANCE_BIN_EDGES,
        "distance_bin_labels": DISTANCE_BIN_LABELS,
        "cumulative_thresholds": CUMULATIVE_THRESHOLDS,
        "aggregate": aggregate,
        "per_seed": {str(s): per_seed_summary[s] for s in per_seed_summary},
    }, path)


# ---------------------------------------------------------------------------
# Distribution summary plot

def _plot_distribution_summary(path: Path, per_seed: dict[int, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    seeds = sorted(per_seed.keys())
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple",
              "tab:orange", "tab:cyan"][: len(seeds)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    edges = np.geomspace(0.5, 250.0, 40)

    for ax, side, label in [(axes[0], "fp", "FP"), (axes[1], "fn", "FN")]:
        per_seed_density: list[np.ndarray] = []
        for seed, color in zip(seeds, colors):
            d = np.concatenate([r[f"{side}_distances"] for r in per_seed[seed]
                                if len(r[f"{side}_distances"])]) \
                if any(len(r[f"{side}_distances"]) for r in per_seed[seed]) else np.empty(0)
            if len(d) == 0:
                per_seed_density.append(np.zeros(len(edges) - 1))
                continue
            d_clip = np.clip(d, 0.5, 250.0)
            density, _ = np.histogram(d_clip, bins=edges, density=True)
            per_seed_density.append(density)
            ax.plot(_bin_centers(edges), density, color=color, alpha=0.4, lw=1.2,
                    label=f"seed {seed}")
        # Mean curve across seeds
        if per_seed_density:
            stacked = np.stack(per_seed_density, axis=0)
            mean_density = stacked.mean(axis=0)
            ax.plot(_bin_centers(edges), mean_density, color="black", lw=2.2, label="mean")

        # Median / p90 verticals (mean across seeds)
        all_d_seeds = []
        for seed in seeds:
            d = np.concatenate([r[f"{side}_distances"] for r in per_seed[seed]
                                if len(r[f"{side}_distances"])]) \
                if any(len(r[f"{side}_distances"]) for r in per_seed[seed]) else None
            if d is not None and len(d):
                all_d_seeds.append(d)
        if all_d_seeds:
            medians = [float(np.median(d)) for d in all_d_seeds]
            p90s = [float(np.percentile(d, 90)) for d in all_d_seeds]
            med_mean = float(np.mean(medians))
            p90_mean = float(np.mean(p90s))
            ax.axvline(med_mean, color="grey", linestyle="--", lw=1,
                       label=f"median (mean) = {med_mean:.1f} px")
            ax.axvline(p90_mean, color="grey", linestyle=":", lw=1,
                       label=f"p90 (mean) = {p90_mean:.1f} px")

        ax.set_xscale("log")
        ax.set_xlabel("distance to GT boundary (pixels)")
        ax.set_ylabel("density (normalised)")
        ax.set_title(f"{label} distance distribution across seeds")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


# ---------------------------------------------------------------------------
# Sample plots

def _plot_error_samples(
    grid_path: Path, samples_dir: Path,
    samples: list[str],
    input_cache: dict[str, np.ndarray],
    gt_cache: dict[str, np.ndarray],
    pred_cache: dict[str, np.ndarray],
    dist_cache: dict[str, np.ndarray],
    per_seed: dict[int, list[dict]],
    baseline_seed: int,
    mean_iou_per_stem: dict[str, float],
    band_indices: list[int] | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})

    n_rows = len(samples)
    fig = plt.figure(figsize=(20, 3.8 * n_rows + 0.5))
    gs = fig.add_gridspec(
        nrows=n_rows * 3, ncols=5,
        height_ratios=[14, 0.35, 0.35] * n_rows,
        hspace=0.50, wspace=0.18,
    )

    for r, stem in enumerate(samples):
        base_row = r * 3
        ax_rgb = fig.add_subplot(gs[base_row, 0])
        ax_gt = fig.add_subplot(gs[base_row, 1])
        ax_pred = fig.add_subplot(gs[base_row, 2])
        ax_err = fig.add_subplot(gs[base_row, 3])
        ax_hist = fig.add_subplot(gs[base_row, 4])
        cax_fp = fig.add_subplot(gs[base_row + 1, 3])
        cax_fn = fig.add_subplot(gs[base_row + 2, 3])

        _render_sample_row(
            ax_rgb, ax_gt, ax_pred, ax_err, ax_hist, cax_fp, cax_fn,
            stem, input_cache.get(stem), gt_cache.get(stem),
            pred_cache.get(stem), dist_cache.get(stem),
            per_seed, baseline_seed, mean_iou_per_stem.get(stem, float("nan")),
            band_indices,
        )

    fig.suptitle("Error analysis -- 6 representative samples (2 best / 2 mid / 2 worst by mean IoU)",
                 y=1.005, fontsize=12, fontweight="bold")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Standalone per-sample PNGs
    for stem in samples:
        _save_standalone_sample(
            samples_dir / f"{stem}_panels.png",
            stem, input_cache.get(stem), gt_cache.get(stem),
            pred_cache.get(stem), dist_cache.get(stem),
            per_seed, baseline_seed, mean_iou_per_stem.get(stem, float("nan")),
            band_indices,
        )


def _save_standalone_sample(path, stem, img, gt, pred, dist, per_seed, baseline_seed,
                             mean_iou, band_indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    fig = plt.figure(figsize=(20, 4.4))
    gs = fig.add_gridspec(nrows=3, ncols=5, height_ratios=[14, 0.4, 0.4],
                          hspace=0.40, wspace=0.18)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_err = fig.add_subplot(gs[0, 3])
    ax_hist = fig.add_subplot(gs[0, 4])
    cax_fp = fig.add_subplot(gs[1, 3])
    cax_fn = fig.add_subplot(gs[2, 3])

    _render_sample_row(ax_rgb, ax_gt, ax_pred, ax_err, ax_hist, cax_fp, cax_fn,
                       stem, img, gt, pred, dist, per_seed, baseline_seed,
                       mean_iou, band_indices)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_sample_row(ax_rgb, ax_gt, ax_pred, ax_err, ax_hist, cax_fp, cax_fn,
                        stem, img, gt, pred, dist, per_seed, baseline_seed,
                        mean_iou, band_indices):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.colorbar as mpl_cbar

    fp_cmap = LinearSegmentedColormap.from_list(
        "fp_red_yellow_blue",
        [(0.0, "darkred"), (0.5, "gold"), (1.0, "darkblue")],
    )
    fn_cmap = plt.cm.viridis

    if img is None or gt is None or pred is None or dist is None:
        for a in (ax_rgb, ax_gt, ax_pred, ax_err, ax_hist):
            a.text(0.5, 0.5, f"{stem}\n(data missing)",
                   ha="center", va="center", transform=a.transAxes, fontsize=9)
            a.set_axis_off()
        cax_fp.set_visible(False)
        cax_fn.set_visible(False)
        return

    rgb = _pseudo_rgb(img)
    bruise_frac = float(gt.sum()) / float(gt.size)

    # Per-seed IoU at this stem (for the histogram title)
    per_seed_iou = {}
    for s, records in per_seed.items():
        for r in records:
            if r["stem"] == stem:
                per_seed_iou[s] = r["iou"]
                break

    # Panel 1: pseudo-RGB
    ax_rgb.imshow(rgb)
    bi = band_indices if band_indices else [0, 1, 2]
    ax_rgb.set_title(f"Pseudo-RGB\n(R={bi[0]}, G={bi[1] if len(bi) > 1 else '-'}, "
                     f"B={bi[2] if len(bi) > 2 else '-'})", fontsize=9)
    ax_rgb.set_axis_off()

    # Panel 2: GT mask
    ax_gt.imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax_gt.set_title(f"GT mask\n(bruise area = {bruise_frac:.2%})", fontsize=9)
    ax_gt.set_axis_off()

    # Panel 3: Pred mask (baseline)
    iou_base = per_seed_iou.get(baseline_seed, float("nan"))
    ax_pred.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax_pred.set_title(f"Prediction (seed {baseline_seed})\nIoU = {iou_base:.4f}", fontsize=9)
    ax_pred.set_axis_off()

    # Panel 4: Error overlay
    fp_mask = (pred == 1) & (gt == 0)
    fn_mask = (pred == 0) & (gt == 1)
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    overlay = _error_overlay(rgb, gt, dist, fp_mask, fn_mask,
                              fp_cmap=fp_cmap, fn_cmap=fn_cmap, vmax=100.0)
    ax_err.imshow(overlay)
    ax_err.set_title(f"Error overlay (FP={n_fp}, FN={n_fn})\n"
                     "FP=red->blue (distance), FN=viridis", fontsize=9)
    ax_err.set_axis_off()

    # Colorbars
    if n_fp > 0:
        cb_fp = mpl_cbar.ColorbarBase(cax_fp, cmap=fp_cmap, norm=Normalize(vmin=0, vmax=100),
                                       orientation="horizontal")
        cb_fp.set_label("FP distance to GT boundary (px, clipped at 100)", fontsize=8)
        cb_fp.ax.tick_params(labelsize=7)
    else:
        cax_fp.set_visible(False)
    if n_fn > 0:
        cb_fn = mpl_cbar.ColorbarBase(cax_fn, cmap=fn_cmap, norm=Normalize(vmin=0, vmax=100),
                                       orientation="horizontal")
        cb_fn.set_label("FN distance to GT boundary (px, clipped at 100)", fontsize=8)
        cb_fn.ax.tick_params(labelsize=7)
    else:
        cax_fn.set_visible(False)

    # Panel 5: distance histogram
    fp_d = dist[fp_mask]
    fn_d = dist[fn_mask]
    fp_counts, _ = np.histogram(fp_d, bins=DISTANCE_BIN_EDGES) if len(fp_d) \
        else (np.zeros(len(DISTANCE_BIN_LABELS), dtype=int), None)
    fn_counts, _ = np.histogram(fn_d, bins=DISTANCE_BIN_EDGES) if len(fn_d) \
        else (np.zeros(len(DISTANCE_BIN_LABELS), dtype=int), None)
    x = np.arange(len(DISTANCE_BIN_LABELS))
    w = 0.4
    ax_hist.bar(x - w / 2, fp_counts, width=w, color="tab:red", alpha=0.85,
                edgecolor="black", linewidth=0.3, label="FP")
    ax_hist.bar(x + w / 2, fn_counts, width=w, color="tab:blue", alpha=0.85,
                edgecolor="black", linewidth=0.3, label="FN")
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(DISTANCE_BIN_LABELS, rotation=30, fontsize=8)
    ax_hist.set_ylabel("pixel count")
    ax_hist.set_title(f"Distance histogram\nmean IoU (all seeds) = {mean_iou:.4f}",
                      fontsize=9)
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(axis="y", alpha=0.3)


def _pseudo_rgb(img: np.ndarray) -> np.ndarray:
    """Return (H, W, 3) float [0,1] image from CHW input, per-channel p2-p98 stretch."""
    arr = img
    if arr.ndim == 3 and arr.shape[0] <= arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)  # (H, W, C)
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    H, W, C = arr.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        ci = min(c, C - 1)
        v = arr[:, :, ci].astype(np.float32)
        lo, hi = np.percentile(v, [2, 98])
        if hi - lo < 1e-9:
            out[:, :, c] = 0.5
        else:
            out[:, :, c] = np.clip((v - lo) / (hi - lo), 0, 1)
    return out


def _error_overlay(rgb: np.ndarray, gt: np.ndarray, dist: np.ndarray,
                    fp_mask: np.ndarray, fn_mask: np.ndarray,
                    fp_cmap, fn_cmap, vmax: float = 100.0) -> np.ndarray:
    """Compose: 50%-desaturated pseudo-RGB + white GT boundary + FP/FN coloured overlays."""
    import cv2  # type: ignore

    # 50%-desaturated background
    gray = rgb.mean(axis=-1, keepdims=True)
    bg = 0.5 * rgb + 0.5 * np.broadcast_to(gray, rgb.shape)
    bg = np.clip(bg, 0, 1)

    # GT boundary (white)
    if 0 < gt.sum() < gt.size:
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
        bg = bg.copy()
        bg[boundary] = [1.0, 1.0, 1.0]

    # Coloured overlays
    composite = bg.copy()
    if fp_mask.any():
        d = np.clip(dist, 0.0, vmax) / vmax
        rgba = fp_cmap(d)  # (H, W, 4)
        alpha = 0.7 * fp_mask.astype(np.float32)
        composite = composite * (1 - alpha[..., None]) + rgba[..., :3] * alpha[..., None]
    if fn_mask.any():
        d = np.clip(dist, 0.0, vmax) / vmax
        rgba = fn_cmap(d)
        alpha = 0.7 * fn_mask.astype(np.float32)
        composite = composite * (1 - alpha[..., None]) + rgba[..., :3] * alpha[..., None]
    return np.clip(composite, 0, 1)


# ---------------------------------------------------------------------------
# Report

def _write_report(path: Path, per_seed_summary: dict, aggregate: dict,
                  seeds: list[int], samples: list[str], mean_iou: dict) -> None:
    lines = ["Error distance to GT boundary -- multi-seed report", "=" * 60, ""]
    seeds_sorted = sorted(seeds)
    lines.append(f"Seeds: {seeds_sorted}")
    lines.append(f"Distance bins (pixels): {DISTANCE_BIN_LABELS}")
    lines.append("")

    for side, label in (("fp", "FP"), ("fn", "FN")):
        lines.append(f"[{label}] cumulative fraction within distance (px):")
        header = "  seed   |  n        " + " ".join(f"  <={t:>4d}" for t in CUMULATIVE_THRESHOLDS)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for s in seeds_sorted:
            stats = per_seed_summary[s][side]
            row = (f"  {s:6d} | {stats['n']:8d}  " +
                   " ".join(f"{stats['cumulative'][str(t)] * 100:6.1f}%"
                            for t in CUMULATIVE_THRESHOLDS))
            lines.append(row)
        # Mean +/- std row
        agg = aggregate[side]
        cum_mean = agg["cumulative_mean"]
        cum_std = agg["cumulative_std"]
        row = ("  mean   | " + " " * 10 + " " +
               " ".join(f"{cum_mean[str(t)] * 100:6.1f}%"
                        for t in CUMULATIVE_THRESHOLDS))
        lines.append(row)
        row = ("  +/-std | " + " " * 10 + " " +
               " ".join(f" +/-{cum_std[str(t)] * 100:4.1f}"
                        for t in CUMULATIVE_THRESHOLDS))
        lines.append(row)
        lines.append("")
        lines.append(f"  median: {agg['median_mean']:.2f} +/- {agg['median_std']:.2f} px")
        lines.append(f"  mean:   {agg['mean_mean']:.2f} +/- {agg['mean_std']:.2f} px")
        lines.append(f"  p90:    {agg['p90_mean']:.2f} +/- {agg['p90_std']:.2f} px")
        lines.append("")

    if samples:
        lines.append("Selected samples (mean IoU across seeds):")
        for s in samples:
            lines.append(f"  {s}: mean IoU = {mean_iou.get(s, float('nan')):.4f}")
        lines.append("")
    lines.append("See error_samples_grid.png and samples/<stem>_panels.png for visuals.")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
