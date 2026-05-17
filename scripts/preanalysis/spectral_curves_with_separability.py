"""A1 -- Spectral curves with separability metrics.

Purpose
-------
Replace the older utils/spectral_analysis.py with a rigorous per-band
comparison of healthy vs defect reflectance, using strict pixel
selection rules that avoid mixing border / near-defect pixels into the
"healthy" pool.

Inputs
------
* HSI dataset at ``--data-root`` (default ``/root/autodl-tmp/datasets/full_HSI_dataset``)
  with subdirectories ``images/``, ``masks/`` (defect), ``whole/`` (apple region).

Outputs (under ``outputs/preanalysis/spectral_curves_with_separability/``)
------------------------------------------------------------------------
* ``per_band_stats.json``       : per-band mu/var/n + fisher/cohen_d/auc
* ``sufficient_stats.npz``      : per-image sufficient statistics, reused by A2/A5
* ``sampled_pixels.npz``        : the capped subsample used for AUC (debug aid)
* ``spectral_separability.png`` : 3-panel figure (mean curves, diff, separability)
* ``top_bands_report.txt``      : top-10 ranking by each metric
* ``spectral_curves_with_separability.log``

Typical runtime
---------------
~5-15 minutes for ~200 HSI images on a single workstation (CPU-bound;
two passes -- masks only, then HSI).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

from _common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SEED,
    auc_per_band,
    cohens_d,
    compute_healthy_mask,
    fisher_score,
    list_stems,
    load_hsi,
    load_mask,
    make_output_dir,
    minmax,
    save_json,
    save_sufficient_stats,
    set_seed,
    setup_logging,
    sufficient_stats_path,
    top_k_indices,
)

SCRIPT_NAME = "spectral_curves_with_separability"
DEFAULT_CAP = 500_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    p.add_argument("--band-axis", default="auto", choices=["auto", "0", "-1"])
    p.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help="Per-class cap on sampled pixels for AUC/plotting (default 500k)")
    p.add_argument("--radius-frac", type=float, default=0.7,
                   help="Healthy pixels must be within this fraction of max apple radius")
    p.add_argument("--min-dist-defect", type=int, default=15,
                   help="Minimum distance (pixels) from any defect for a pixel to count as healthy")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit-images", type=int, default=0,
                   help="If >0, only process the first N images (debug)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_output_dir(SCRIPT_NAME, args.output_root)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    data_root = Path(args.data_root)
    stems = list_stems(data_root)
    if args.limit_images:
        stems = stems[: args.limit_images]
    n_imgs = len(stems)
    log.info("found %d images", n_imgs)

    band_axis = args.band_axis if args.band_axis == "auto" else int(args.band_axis)

    # Determine band count from first image
    first_img = load_hsi(data_root / "images" / f"{stems[0]}.npy", band_axis=band_axis)
    H, W, B = first_img.shape
    log.info("HSI shape: H=%d W=%d B=%d", H, W, B)
    del first_img

    # -----------------------------------------------------------------
    # Pass 1: count eligible pixels per class per image (mask-only, fast)
    # -----------------------------------------------------------------
    log.info("Pass 1: counting healthy/defect pixels per image")
    healthy_counts = np.zeros(n_imgs, dtype=np.int64)
    defect_counts = np.zeros(n_imgs, dtype=np.int64)
    healthy_masks: list[np.ndarray] = [None] * n_imgs  # type: ignore
    defect_masks: list[np.ndarray] = [None] * n_imgs   # type: ignore

    for i, stem in enumerate(tqdm(stems, desc="pass1")):
        defect = load_mask(data_root / "masks" / f"{stem}.npy")
        whole = load_mask(data_root / "whole" / f"{stem}.npy")
        h_mask = compute_healthy_mask(
            whole, defect,
            radius_frac=args.radius_frac,
            min_dist_defect=args.min_dist_defect,
        )
        healthy_masks[i] = h_mask
        defect_masks[i] = defect > 0
        healthy_counts[i] = int(h_mask.sum())
        defect_counts[i] = int((defect > 0).sum())

    total_h = int(healthy_counts.sum())
    total_d = int(defect_counts.sum())
    log.info("total eligible healthy pixels: %d", total_h)
    log.info("total defect pixels:           %d", total_d)
    if total_h == 0 or total_d == 0:
        log.error("one of the classes has zero pixels; aborting")
        return 1

    prob_h = min(1.0, args.cap / total_h)
    prob_d = min(1.0, args.cap / total_d)
    log.info("sampling probability healthy=%.4g defect=%.4g", prob_h, prob_d)

    # -----------------------------------------------------------------
    # Pass 2: load HSI, compute per-image sufficient stats AND sample pixels
    # -----------------------------------------------------------------
    log.info("Pass 2: loading HSI cubes, computing sufficient stats + sampling")
    rng = np.random.default_rng(args.seed)

    sum_h = np.zeros((n_imgs, B), dtype=np.float64)
    sumsq_h = np.zeros((n_imgs, B), dtype=np.float64)
    n_h = np.zeros(n_imgs, dtype=np.int64)
    sum_d = np.zeros((n_imgs, B), dtype=np.float64)
    sumsq_d = np.zeros((n_imgs, B), dtype=np.float64)
    n_d = np.zeros(n_imgs, dtype=np.int64)

    healthy_buf: list[np.ndarray] = []
    defect_buf: list[np.ndarray] = []

    for i, stem in enumerate(tqdm(stems, desc="pass2")):
        img = load_hsi(data_root / "images" / f"{stem}.npy", band_axis=band_axis)
        if img.shape[:2] != healthy_masks[i].shape:
            log.warning("shape mismatch for %s (img %s, mask %s); skipping",
                        stem, img.shape, healthy_masks[i].shape)
            continue

        # Healthy: sufficient stats over the FULL eligible pixel set
        h_pixels = img[healthy_masks[i]]  # (n_i_h, B)
        if h_pixels.shape[0] > 0:
            n_h[i] = h_pixels.shape[0]
            sum_h[i] = h_pixels.sum(axis=0)
            sumsq_h[i] = (h_pixels.astype(np.float64) ** 2).sum(axis=0)
            # Subsample for AUC
            k_keep = int(rng.binomial(h_pixels.shape[0], prob_h))
            if k_keep > 0:
                sel = rng.choice(h_pixels.shape[0], size=k_keep, replace=False)
                healthy_buf.append(h_pixels[sel].astype(np.float32))

        d_pixels = img[defect_masks[i]]
        if d_pixels.shape[0] > 0:
            n_d[i] = d_pixels.shape[0]
            sum_d[i] = d_pixels.sum(axis=0)
            sumsq_d[i] = (d_pixels.astype(np.float64) ** 2).sum(axis=0)
            k_keep = int(rng.binomial(d_pixels.shape[0], prob_d))
            if k_keep > 0:
                sel = rng.choice(d_pixels.shape[0], size=k_keep, replace=False)
                defect_buf.append(d_pixels[sel].astype(np.float32))

    healthy = np.concatenate(healthy_buf, axis=0) if healthy_buf else np.empty((0, B), dtype=np.float32)
    defect = np.concatenate(defect_buf, axis=0) if defect_buf else np.empty((0, B), dtype=np.float32)
    log.info("sampled healthy pixels: %d, defect pixels: %d", len(healthy), len(defect))

    # -----------------------------------------------------------------
    # Aggregate sufficient statistics -> mu, var (full data)
    # -----------------------------------------------------------------
    N_h = int(n_h.sum())
    N_d = int(n_d.sum())
    mu_h = sum_h.sum(axis=0) / max(N_h, 1)
    mu_d = sum_d.sum(axis=0) / max(N_d, 1)
    var_h = sumsq_h.sum(axis=0) / max(N_h, 1) - mu_h ** 2
    var_d = sumsq_d.sum(axis=0) / max(N_d, 1) - mu_d ** 2
    var_h = np.maximum(var_h, 0.0)
    var_d = np.maximum(var_d, 0.0)
    sigma_h = np.sqrt(var_h)
    sigma_d = np.sqrt(var_d)

    fisher = fisher_score(mu_d, mu_h, var_d, var_h)
    cohen = cohens_d(mu_d, mu_h, var_d, var_h)
    auc = auc_per_band(healthy, defect) if len(healthy) and len(defect) else np.full(B, np.nan)

    # 95% CI for the mean and for the difference (using full N)
    se_h = sigma_h / np.sqrt(max(N_h, 1))
    se_d = sigma_d / np.sqrt(max(N_d, 1))
    ci95_h = 1.96 * se_h
    ci95_d = 1.96 * se_d
    se_diff = np.sqrt(var_d / max(N_d, 1) + var_h / max(N_h, 1))
    ci95_diff = 1.96 * se_diff

    # -----------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------
    per_band = []
    for b in range(B):
        per_band.append({
            "band": b,
            "mu_h": float(mu_h[b]), "sigma_h": float(sigma_h[b]), "n_h": int(N_h),
            "mu_d": float(mu_d[b]), "sigma_d": float(sigma_d[b]), "n_d": int(N_d),
            "ci95_h": float(ci95_h[b]),
            "ci95_d": float(ci95_d[b]),
            "diff": float(mu_d[b] - mu_h[b]),
            "ci95_diff": float(ci95_diff[b]),
            "fisher": float(fisher[b]),
            "cohen_d": float(cohen[b]),
            "auc": float(auc[b]),
        })

    save_json({
        "config": {
            "data_root": str(data_root),
            "n_images": n_imgs,
            "band_count": B,
            "cap": args.cap,
            "radius_frac": args.radius_frac,
            "min_dist_defect": args.min_dist_defect,
            "seed": args.seed,
        },
        "totals": {
            "healthy_full": N_h, "defect_full": N_d,
            "healthy_sampled": int(len(healthy)),
            "defect_sampled": int(len(defect)),
        },
        "per_band": per_band,
    }, out_dir / "per_band_stats.json")

    save_sufficient_stats(
        sufficient_stats_path(args.output_root),
        stems=np.array(stems),
        sum_h=sum_h, sumsq_h=sumsq_h, n_h=n_h,
        sum_d=sum_d, sumsq_d=sumsq_d, n_d=n_d,
        band_count=np.array([B]),
    )

    np.savez(out_dir / "sampled_pixels.npz", healthy=healthy, defect=defect)

    _write_top_report(out_dir / "top_bands_report.txt", fisher, cohen, auc)
    _plot(out_dir / "spectral_separability.png",
          mu_h, mu_d, ci95_h, ci95_d, ci95_diff, fisher, cohen, auc)

    log.info("done. outputs in %s", out_dir)
    return 0


def _write_top_report(path: Path, fisher: np.ndarray, cohen: np.ndarray, auc: np.ndarray) -> None:
    lines = ["Top-10 bands by each separability metric", "=" * 50, ""]
    lines.append("By Fisher score (higher = more separable)")
    for r, b in enumerate(top_k_indices(fisher, 10), 1):
        lines.append(f"  {r:2d}. band {b:3d}   fisher={fisher[b]:.4f}")
    lines.append("")
    lines.append("By |Cohen's d|")
    for r, b in enumerate(top_k_indices(np.abs(cohen), 10), 1):
        lines.append(f"  {r:2d}. band {b:3d}   d={cohen[b]:+.4f}")
    lines.append("")
    lines.append("By AUC distance from 0.5 (|auc - 0.5|)")
    auc_dist = np.where(np.isnan(auc), 0.0, np.abs(auc - 0.5))
    for r, b in enumerate(top_k_indices(auc_dist, 10), 1):
        lines.append(f"  {r:2d}. band {b:3d}   auc={auc[b]:.4f}  (|auc-0.5|={auc_dist[b]:.4f})")
    path.write_text("\n".join(lines) + "\n")


def _plot(
    path: Path,
    mu_h: np.ndarray, mu_d: np.ndarray,
    ci95_h: np.ndarray, ci95_d: np.ndarray, ci95_diff: np.ndarray,
    fisher: np.ndarray, cohen: np.ndarray, auc: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = np.arange(len(mu_h))
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # (a) mean reflectance with 95% CI
    ax = axes[0]
    ax.plot(bands, mu_h, color="tab:blue", lw=1.5, label="healthy")
    ax.fill_between(bands, mu_h - ci95_h, mu_h + ci95_h, color="tab:blue", alpha=0.2)
    ax.plot(bands, mu_d, color="tab:red", lw=1.5, label="defect")
    ax.fill_between(bands, mu_d - ci95_d, mu_d + ci95_d, color="tab:red", alpha=0.2)
    ax.set_ylabel("mean reflectance")
    ax.set_title("(a) Mean spectral reflectance, 95% CI")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # (b) difference curve
    ax = axes[1]
    diff = mu_d - mu_h
    ax.axhline(0, color="grey", lw=0.5)
    ax.plot(bands, diff, color="tab:purple", lw=1.5, label=r"$\mu_d - \mu_h$")
    ax.fill_between(bands, diff - ci95_diff, diff + ci95_diff, color="tab:purple", alpha=0.2)
    ax.set_ylabel(r"$\Delta$ reflectance")
    ax.set_title("(b) Defect - healthy difference, 95% CI")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # (c) separability metrics, min-max normalised
    ax = axes[2]
    ax.plot(bands, minmax(fisher), color="tab:green", lw=1.5, label="Fisher (norm.)")
    ax.plot(bands, minmax(np.abs(cohen)), color="tab:orange", lw=1.5, label="|Cohen d| (norm.)")
    auc_finite = np.where(np.isnan(auc), 0.5, auc)
    ax.plot(bands, minmax(np.abs(auc_finite - 0.5)), color="tab:brown", lw=1.5,
            label="|AUC - 0.5| (norm.)")
    ax.set_xlabel("HSI band index")
    ax.set_ylabel("normalised separability")
    ax.set_title("(c) Per-band separability metrics (min-max normalised)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
