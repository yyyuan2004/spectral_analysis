"""A2 -- Bootstrap stability of the top-Fisher bands.

Purpose
-------
Show that the bands selected by the Fisher criterion are not a sampling
fluke -- they recur across bootstrap resamples of the dataset.

Limitation (deviation from the original spec)
---------------------------------------------
The original design called for per-apple bootstrapping using a grouping
of files into apple individuals.  The available HSI files
(``REFLECTANCE_<date>_<seq>.npy``) do not encode apple identity and no
mapping was provided, so this script falls back to **image-level**
bootstrapping.  Because the same apple can appear in the resample as
several views, this slightly **over-estimates** stability; the headline
numbers should be read as an upper bound until per-apple grouping is
available.

The combination-level bootstrap (operating on the 84 combo IoUs from
``search_*.json``) is omitted entirely, per the spec's downgrade
clause: there are no per-image IoU records to resample over without
re-training, so we limit ourselves to the single-band Fisher rank.

Inputs
------
* Per-image sufficient statistics from A1 (``sufficient_stats.npz``).
  If absent the script will compute them from scratch using the same
  pixel-selection rules as A1.

Outputs (under ``outputs/preanalysis/bootstrap_band_stability/``)
----------------------------------------------------------------
* ``bootstrap_stability.json``  : per-band frequency, Jaccard stats
* ``selection_frequency.png``   : descending frequency bar chart
* ``jaccard_distribution.png``  : pairwise Jaccard histogram
* ``freq_vs_fisher.png``        : scatter showing freq tracks Fisher
* ``bootstrap_stability_report.txt``
* ``bootstrap_band_stability.log``

Typical runtime
---------------
A few seconds once A1 has produced the sufficient stats; otherwise add
A1's pass-1 + pass-2 cost (~5-15 minutes).
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
    fisher_score,
    load_sufficient_stats,
    make_output_dir,
    save_json,
    set_seed,
    setup_logging,
    sufficient_stats_path,
    top_k_indices,
)

SCRIPT_NAME = "bootstrap_band_stability"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    p.add_argument("--top-k", type=int, default=3, help="Top-K bands selected each iter")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--auto-recompute", action="store_true",
                   help="If sufficient_stats.npz is missing, run A1's logic in-process")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_output_dir(SCRIPT_NAME, args.output_root)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    stats_path = sufficient_stats_path(args.output_root)
    if not stats_path.exists():
        if not args.auto_recompute:
            log.error("missing %s; run A1 first or pass --auto-recompute", stats_path)
            return 1
        log.info("sufficient_stats.npz not found; computing in-process")
        _recompute_sufficient_stats(args.data_root, stats_path, log)

    stats = load_sufficient_stats(stats_path)
    stems = [str(s) for s in stats["stems"].tolist()]
    sum_h = stats["sum_h"]; sumsq_h = stats["sumsq_h"]; n_h = stats["n_h"]
    sum_d = stats["sum_d"]; sumsq_d = stats["sumsq_d"]; n_d = stats["n_d"]
    N_imgs, B = sum_h.shape
    log.info("loaded sufficient stats: N_imgs=%d B=%d", N_imgs, B)

    # Full-data Fisher (reference) ------------------------------------
    fisher_full = _fisher_from_sufficient(sum_h, sumsq_h, n_h, sum_d, sumsq_d, n_d)
    top_full = top_k_indices(fisher_full, args.top_k)
    log.info("full-data top-%d: %s", args.top_k, top_full)

    # Bootstrap -------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    selection_counts = np.zeros(B, dtype=np.int64)
    top_sets: list[set[int]] = []
    per_iter_records: list[dict] = []

    for it in tqdm(range(args.iterations), desc="bootstrap"):
        idx = rng.integers(0, N_imgs, size=N_imgs)  # with replacement
        weights = np.bincount(idx, minlength=N_imgs).astype(np.float64)
        fisher_i = _fisher_from_sufficient(
            weights @ sum_h, weights @ sumsq_h, weights @ n_h,
            weights @ sum_d, weights @ sumsq_d, weights @ n_d,
            _vectorised=True,
        )
        top_i = top_k_indices(fisher_i, args.top_k)
        for b in top_i:
            selection_counts[b] += 1
        top_sets.append(set(top_i))
        per_iter_records.append({"iter": it, "top_k": top_i, "fisher_topk": [float(fisher_i[b]) for b in top_i]})

    frequency = selection_counts / args.iterations

    # Pairwise Jaccard ------------------------------------------------
    jaccards: list[float] = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            a, b = top_sets[i], top_sets[j]
            union = len(a | b)
            jaccards.append(len(a & b) / union if union else 0.0)
    jaccards_arr = np.asarray(jaccards, dtype=np.float64)
    jaccard_mean = float(jaccards_arr.mean()) if len(jaccards_arr) else float("nan")
    jaccard_std = float(jaccards_arr.std()) if len(jaccards_arr) else float("nan")
    log.info("pairwise Jaccard: mean=%.4f std=%.4f", jaccard_mean, jaccard_std)

    # Persist ---------------------------------------------------------
    stable_bands = [int(b) for b in np.argsort(-frequency) if frequency[b] >= 0.5]
    log.info("bands with freq >= 50%%: %s", stable_bands)

    save_json({
        "config": {"iterations": args.iterations, "top_k": args.top_k, "seed": args.seed,
                   "n_images": N_imgs, "band_count": B,
                   "note": "image-level bootstrap (no apple_id mapping available)"},
        "full_data_top_k": top_full,
        "frequency": {int(b): float(frequency[b]) for b in range(B)},
        "fisher_full": {int(b): float(fisher_full[b]) for b in range(B)},
        "jaccard": {"mean": jaccard_mean, "std": jaccard_std,
                    "min": float(jaccards_arr.min()) if len(jaccards_arr) else None,
                    "max": float(jaccards_arr.max()) if len(jaccards_arr) else None,
                    "n_pairs": len(jaccards_arr)},
        "stable_bands_freq_ge_0.5": stable_bands,
        "per_iter": per_iter_records,
    }, out_dir / "bootstrap_stability.json")

    _plot_frequency(out_dir / "selection_frequency.png", frequency, top_full)
    _plot_jaccard(out_dir / "jaccard_distribution.png", jaccards_arr)
    _plot_freq_vs_fisher(out_dir / "freq_vs_fisher.png", fisher_full, frequency, top_full)
    _write_report(out_dir / "bootstrap_stability_report.txt",
                  frequency, fisher_full, top_full, jaccard_mean, jaccard_std, stable_bands)

    log.info("done. outputs in %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Helpers

def _fisher_from_sufficient(
    sum_h, sumsq_h, n_h, sum_d, sumsq_d, n_d, _vectorised: bool = False,
) -> np.ndarray:
    """Compute per-band Fisher from sufficient stats.

    When ``_vectorised`` is False the inputs are per-image arrays
    (shape (N_imgs, B) for sums, (N_imgs,) for counts) and are aggregated.
    When True, the inputs are already aggregated (shape (B,) and scalar).
    """
    if not _vectorised:
        N_h = float(n_h.sum())
        N_d = float(n_d.sum())
        mu_h = sum_h.sum(axis=0) / max(N_h, 1.0)
        mu_d = sum_d.sum(axis=0) / max(N_d, 1.0)
        var_h = np.maximum(sumsq_h.sum(axis=0) / max(N_h, 1.0) - mu_h ** 2, 0.0)
        var_d = np.maximum(sumsq_d.sum(axis=0) / max(N_d, 1.0) - mu_d ** 2, 0.0)
    else:
        N_h = float(n_h)
        N_d = float(n_d)
        mu_h = sum_h / max(N_h, 1.0)
        mu_d = sum_d / max(N_d, 1.0)
        var_h = np.maximum(sumsq_h / max(N_h, 1.0) - mu_h ** 2, 0.0)
        var_d = np.maximum(sumsq_d / max(N_d, 1.0) - mu_d ** 2, 0.0)
    return fisher_score(mu_d, mu_h, var_d, var_h)


def _recompute_sufficient_stats(data_root: str, out_path: Path, log) -> None:
    """Replay A1's pass-1 + pass-2 logic to materialise sufficient_stats.npz."""
    from _common import (
        compute_healthy_mask, list_stems, load_hsi, load_mask, save_sufficient_stats,
    )

    data_root = Path(data_root)
    stems = list_stems(data_root)
    first = load_hsi(data_root / "images" / f"{stems[0]}.npy")
    B = first.shape[-1]
    del first
    N = len(stems)

    sum_h = np.zeros((N, B), dtype=np.float64)
    sumsq_h = np.zeros((N, B), dtype=np.float64)
    n_h = np.zeros(N, dtype=np.int64)
    sum_d = np.zeros((N, B), dtype=np.float64)
    sumsq_d = np.zeros((N, B), dtype=np.float64)
    n_d = np.zeros(N, dtype=np.int64)

    for i, stem in enumerate(tqdm(stems, desc="recompute")):
        defect = load_mask(data_root / "masks" / f"{stem}.npy")
        whole = load_mask(data_root / "whole" / f"{stem}.npy")
        img = load_hsi(data_root / "images" / f"{stem}.npy")
        h_mask = compute_healthy_mask(whole, defect)
        d_mask = defect > 0
        if h_mask.any():
            h_px = img[h_mask]
            n_h[i] = h_px.shape[0]
            sum_h[i] = h_px.sum(axis=0)
            sumsq_h[i] = (h_px.astype(np.float64) ** 2).sum(axis=0)
        if d_mask.any():
            d_px = img[d_mask]
            n_d[i] = d_px.shape[0]
            sum_d[i] = d_px.sum(axis=0)
            sumsq_d[i] = (d_px.astype(np.float64) ** 2).sum(axis=0)

    save_sufficient_stats(
        out_path,
        stems=np.array(stems),
        sum_h=sum_h, sumsq_h=sumsq_h, n_h=n_h,
        sum_d=sum_d, sumsq_d=sumsq_d, n_d=n_d,
        band_count=np.array([B]),
    )
    log.info("wrote %s", out_path)


def _plot_frequency(path: Path, frequency: np.ndarray, top_full: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-frequency)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["tab:red" if b in set(top_full) else "tab:blue" for b in order]
    ax.bar(np.arange(len(frequency)), frequency[order], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(frequency)))
    ax.set_xticklabels(order, fontsize=6, rotation=90)
    ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, label="50% threshold")
    ax.set_ylabel("selection frequency")
    ax.set_xlabel("HSI band (sorted by frequency)")
    ax.set_title("Band selection frequency across bootstrap iterations "
                 "(red = present in full-data top-K)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_jaccard(path: Path, jaccards: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(jaccards):
        ax.hist(jaccards, bins=20, range=(0, 1), color="tab:purple",
                alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axvline(jaccards.mean(), color="black", linestyle="--",
                   label=f"mean={jaccards.mean():.3f}")
    ax.set_xlabel("pairwise Jaccard similarity of top-K sets")
    ax.set_ylabel("pair count")
    ax.set_title("Pairwise Jaccard distribution across bootstrap iterations")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_freq_vs_fisher(path: Path, fisher_full: np.ndarray, frequency: np.ndarray, top_full: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    is_top = np.array([b in set(top_full) for b in range(len(fisher_full))])
    ax.scatter(fisher_full[~is_top], frequency[~is_top], s=18, color="tab:blue",
               alpha=0.7, label="other bands")
    ax.scatter(fisher_full[is_top], frequency[is_top], s=80, color="tab:red",
               marker="*", label="full-data top-K")
    for b in np.argsort(-frequency)[:10]:
        ax.annotate(str(b), (fisher_full[b], frequency[b]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Fisher score (full data)")
    ax.set_ylabel("bootstrap selection frequency")
    ax.set_title("Selection frequency vs full-data Fisher score")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_report(
    path: Path, frequency: np.ndarray, fisher_full: np.ndarray, top_full: list[int],
    jaccard_mean: float, jaccard_std: float, stable_bands: list[int],
) -> None:
    lines = ["Bootstrap band stability report", "=" * 50, ""]
    lines.append(f"Full-data top-K: {top_full}")
    lines.append(f"Mean pairwise Jaccard of top-K sets: {jaccard_mean:.4f} (std {jaccard_std:.4f})")
    lines.append("")
    lines.append("Bands with selection frequency >= 50% (sorted by freq):")
    for b in stable_bands:
        lines.append(f"  band {b:3d}  freq={frequency[b]:.2f}  fisher_full={fisher_full[b]:.4f}")
    lines.append("")
    lines.append("Top-15 by bootstrap selection frequency:")
    for r, b in enumerate(np.argsort(-frequency)[:15], 1):
        marker = "*" if b in set(top_full) else " "
        lines.append(f"  {r:2d}.{marker}band {int(b):3d}  freq={frequency[b]:.2f}  fisher_full={fisher_full[b]:.4f}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
