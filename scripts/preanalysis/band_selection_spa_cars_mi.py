"""Hyperspectral feature band selection: SPA / CARS / MI.

Implements three classical wavelength-selection algorithms commonly used
for hyperspectral imaging analysis:

* **SPA** (Successive Projections Algorithm, Araujo et al. 2001) --
  greedy forward selection that picks the band whose vector has the
  largest projection onto the orthogonal complement of the bands
  already selected; each candidate chain is then scored by PLS-DA
  RMSEP on an independent validation split.

* **CARS** (Competitive Adaptive Reweighted Sampling, Li et al. 2009) --
  iterative Monte-Carlo procedure that fits PLS-DA, ranks bands by
  |regression coefficient|, prunes via an exponentially decreasing
  function (EDF) and adaptively reweighted sampling (ARS), and keeps
  the iteration with minimum 5-fold CV RMSEP.

* **MI** (Mutual Information) -- non-parametric ``mutual_info_classif``
  between each band's pixel reflectance and the binary class label
  (1 = defect, 0 = healthy).

The script is self-contained: it depends only on numpy, scipy,
scikit-learn, matplotlib, tqdm.  It does **not** require the ``msi``
project, and it does **not** require a ``whole/`` subdirectory --
non-defect pixels (optionally eroded away from the defect boundary)
are treated as the healthy class.

Data layout
-----------
Default paths are the user's local Windows paths::

    C:/Users/10730/Desktop/hsi(20-110)/images/<stem>.npy   (H,W,B) or (B,H,W)
    C:/Users/10730/Desktop/hsi(20-110)/masks/<stem>.npy    (H,W), values {0,1}

Override with ``--images-dir`` and ``--masks-dir``.

Outputs (under ``outputs/preanalysis/band_selection_spa_cars_mi/``)
--------------------------------------------------------------------
* ``selected_bands.json``        -- selected indices per method + scores
* ``band_selection_report.txt``  -- human-readable summary
* ``band_selection.png``         -- combined visualisation
* ``cars_trace.png``             -- CARS RMSEP / variable-count traces
* ``spa_rmsep_grid.png``         -- SPA RMSEP heat-map (start x length)
* ``band_selection_spa_cars_mi.log``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

from _common import (
    SEED,
    list_stems,
    load_hsi,
    load_mask,
    make_output_dir,
    save_json,
    set_seed,
    setup_logging,
)

SCRIPT_NAME = "band_selection_spa_cars_mi"

# Default = user's local Windows path (override with --images-dir / --masks-dir)
DEFAULT_IMAGES_DIR = r"C:\Users\10730\Desktop\hsi(20-110)\images"
DEFAULT_MASKS_DIR = r"C:\Users\10730\Desktop\hsi(20-110)\masks"
DEFAULT_OUTPUT_ROOT = "outputs/preanalysis"


# ---------------------------------------------------------------------------
# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, type=str,
                   help="Directory containing HSI .npy cubes")
    p.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR, type=str,
                   help="Directory containing binary defect masks (.npy)")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    p.add_argument("--band-axis", default="auto", choices=["auto", "0", "-1"])

    # Pixel sampling
    p.add_argument("--n-per-class", type=int, default=20000,
                   help="Number of pixels sampled per class (healthy / defect)")
    p.add_argument("--healthy-erosion", type=int, default=5,
                   help="Exclude pixels within this many pixels of any defect "
                        "when picking healthy samples (0 disables)")
    p.add_argument("--limit-images", type=int, default=0,
                   help="If >0, only use the first N HSI images (debug)")
    p.add_argument("--normalize", default="snv",
                   choices=["none", "minmax", "snv"],
                   help="Per-pixel spectrum normalisation before band selection")

    # Algorithm common
    p.add_argument("--n-select", type=int, default=10,
                   help="Number of bands each method should return")
    p.add_argument("--max-components", type=int, default=10,
                   help="Maximum number of PLS latent variables")

    # SPA
    p.add_argument("--spa-max-starts", type=int, default=20,
                   help="SPA evaluates at most this many starting bands")
    p.add_argument("--spa-val-frac", type=float, default=0.3,
                   help="Fraction of pixels held out for SPA chain RMSEP")

    # CARS
    p.add_argument("--cars-iter", type=int, default=50,
                   help="Number of CARS Monte-Carlo iterations")
    p.add_argument("--cars-cv-folds", type=int, default=5,
                   help="K for K-fold RMSECV inside CARS")
    p.add_argument("--cars-mc-ratio", type=float, default=0.8,
                   help="Fraction of samples used for the PLS fit each iter")

    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading

def _erode_mask(m: np.ndarray, k: int) -> np.ndarray:
    """Return mask of pixels at least ``k`` pixels away from any defect pixel."""
    if k <= 0:
        return ~m
    try:
        import cv2

        src = ((~m).astype(np.uint8)) * 255
        dist = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    except ImportError:
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(~m)
    return dist >= float(k)


def _normalize(pixels: np.ndarray, mode: str) -> np.ndarray:
    """Row-wise spectrum normalisation."""
    if mode == "none":
        return pixels
    if mode == "minmax":
        lo = pixels.min(axis=1, keepdims=True)
        hi = pixels.max(axis=1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-8)
        return (pixels - lo) / rng
    if mode == "snv":  # Standard Normal Variate
        mu = pixels.mean(axis=1, keepdims=True)
        sd = pixels.std(axis=1, keepdims=True)
        sd = np.maximum(sd, 1e-8)
        return (pixels - mu) / sd
    raise ValueError(f"unknown normalize mode: {mode}")


def load_pixel_matrix(
    images_dir: Path,
    masks_dir: Path,
    n_per_class: int,
    healthy_erosion: int,
    band_axis,
    normalize: str,
    limit_images: int,
    seed: int,
    log,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Walk the dataset, sample pixels, return (X, y, n_bands, stems_used)."""
    rng = np.random.default_rng(seed)

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    stems = sorted(p.stem for p in images_dir.glob("*.npy"))
    if not stems:
        raise FileNotFoundError(f"no .npy files in {images_dir}")
    if limit_images > 0:
        stems = stems[:limit_images]
    log.info("found %d HSI cubes in %s", len(stems), images_dir)

    # Determine band count from first cube
    first = load_hsi(images_dir / f"{stems[0]}.npy", band_axis=band_axis)
    H, W, B = first.shape
    log.info("HSI shape from first cube: H=%d W=%d B=%d", H, W, B)
    del first

    # -------- pass 1: count eligible pixels per class
    log.info("pass 1/2 -- counting eligible pixels")
    per_img_h = np.zeros(len(stems), dtype=np.int64)
    per_img_d = np.zeros(len(stems), dtype=np.int64)
    h_masks: list[np.ndarray] = [None] * len(stems)  # type: ignore
    d_masks: list[np.ndarray] = [None] * len(stems)  # type: ignore
    skipped: list[str] = []

    for i, stem in enumerate(tqdm(stems, desc="masks")):
        mp = masks_dir / f"{stem}.npy"
        if not mp.exists():
            skipped.append(stem)
            continue
        defect = load_mask(mp) > 0
        healthy = _erode_mask(defect, healthy_erosion)
        d_masks[i] = defect
        h_masks[i] = healthy
        per_img_d[i] = int(defect.sum())
        per_img_h[i] = int(healthy.sum())

    if skipped:
        log.warning("missing mask for %d stems: %s", len(skipped), skipped[:5])

    total_h = int(per_img_h.sum())
    total_d = int(per_img_d.sum())
    log.info("total healthy candidate pixels: %d", total_h)
    log.info("total defect  candidate pixels: %d", total_d)
    if total_h == 0 or total_d == 0:
        raise RuntimeError("one class has zero candidate pixels; check masks")

    target_h = min(n_per_class, total_h)
    target_d = min(n_per_class, total_d)

    # Allocate per-image quotas proportional to image size (with a floor of 0)
    quota_h = np.floor(per_img_h * (target_h / total_h)).astype(np.int64)
    quota_d = np.floor(per_img_d * (target_d / total_d)).astype(np.int64)
    # Top up any short-fall to match target
    while quota_h.sum() < target_h and (per_img_h - quota_h).max() > 0:
        room = per_img_h - quota_h
        quota_h[int(np.argmax(room))] += 1
    while quota_d.sum() < target_d and (per_img_d - quota_d).max() > 0:
        room = per_img_d - quota_d
        quota_d[int(np.argmax(room))] += 1

    # -------- pass 2: load HSI and sample
    log.info("pass 2/2 -- loading HSI cubes and sampling pixels")
    h_buf = []
    d_buf = []
    used_stems = []
    for i, stem in enumerate(tqdm(stems, desc="hsi")):
        if h_masks[i] is None:
            continue
        if quota_h[i] == 0 and quota_d[i] == 0:
            continue
        img = load_hsi(images_dir / f"{stem}.npy", band_axis=band_axis)
        if img.shape[:2] != h_masks[i].shape:
            log.warning("shape mismatch for %s (img %s, mask %s); skipping",
                        stem, img.shape, h_masks[i].shape)
            continue
        used_stems.append(stem)

        if quota_h[i] > 0 and per_img_h[i] > 0:
            ys, xs = np.where(h_masks[i])
            sel = rng.choice(len(ys), size=int(quota_h[i]), replace=False)
            h_buf.append(img[ys[sel], xs[sel]].astype(np.float32))

        if quota_d[i] > 0 and per_img_d[i] > 0:
            ys, xs = np.where(d_masks[i])
            sel = rng.choice(len(ys), size=int(quota_d[i]), replace=False)
            d_buf.append(img[ys[sel], xs[sel]].astype(np.float32))

    healthy = np.concatenate(h_buf, axis=0) if h_buf else np.empty((0, B), dtype=np.float32)
    defect = np.concatenate(d_buf, axis=0) if d_buf else np.empty((0, B), dtype=np.float32)
    log.info("sampled healthy=%d, defect=%d, total=%d", len(healthy), len(defect),
             len(healthy) + len(defect))

    X = np.vstack([healthy, defect])
    y = np.concatenate([np.zeros(len(healthy), dtype=np.float32),
                        np.ones(len(defect), dtype=np.float32)])

    # Optional spectrum-level normalisation
    X = _normalize(X, normalize)

    # Shuffle so PLS CV sees mixed labels
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]

    return X, y, B, used_stems


# ---------------------------------------------------------------------------
# SPA

def _spa_chain(Xm: np.ndarray, start: int, length: int) -> list[int]:
    """Modified-Gram-Schmidt SPA chain of length ``length`` starting at ``start``."""
    n_bands = Xm.shape[1]
    length = min(length, n_bands)
    W = Xm.copy()
    selected = [int(start)]
    remaining = np.ones(n_bands, dtype=bool)
    remaining[start] = False

    for _ in range(length - 1):
        last = W[:, selected[-1]]
        last_norm_sq = float(last @ last)
        if last_norm_sq < 1e-12:
            break
        idx = np.where(remaining)[0]
        if idx.size == 0:
            break
        # Orthogonalise remaining columns against last selected (in-place on W)
        W_r = W[:, idx]
        proj = (W_r.T @ last) / last_norm_sq  # (len(idx),)
        W[:, idx] = W_r - np.outer(last, proj)
        norms = np.linalg.norm(W[:, idx], axis=0)
        best_local = int(np.argmax(norms))
        if norms[best_local] < 1e-12:
            break
        best = int(idx[best_local])
        selected.append(best)
        remaining[best] = False

    return selected


def spa(
    X: np.ndarray,
    y: np.ndarray,
    n_select: int,
    max_starts: int = 20,
    n_components_max: int = 10,
    val_frac: float = 0.3,
    seed: int = SEED,
    log=None,
) -> dict:
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(seed)
    n_samples, n_bands = X.shape
    n_select = min(n_select, n_bands)

    Xm = X - X.mean(axis=0, keepdims=True)

    # Train / val split on samples (used to score chains)
    perm = rng.permutation(n_samples)
    n_val = int(val_frac * n_samples)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xv, yv = X[val_idx], y[val_idx]

    # Starting bands: top ``max_starts`` by column norm of the centered matrix
    col_norm = np.linalg.norm(Xm, axis=0)
    n_starts = min(max_starts, n_bands)
    start_bands = np.argsort(-col_norm)[:n_starts].astype(int)

    rmsep_grid = np.full((n_starts, n_select), np.inf, dtype=np.float64)
    chains: dict[int, list[int]] = {}

    best_rmsep = np.inf
    best_subset: list[int] = []
    best_start = -1
    best_len = 0

    for s_i, start in enumerate(tqdm(start_bands, desc="SPA chains")):
        chain = _spa_chain(Xm, int(start), n_select)
        chains[int(start)] = chain
        for length in range(1, len(chain) + 1):
            cols = chain[:length]
            n_comp = min(n_components_max, length, len(tr_idx) - 1)
            if n_comp < 1:
                continue
            try:
                pls = PLSRegression(n_components=n_comp, scale=False)
                pls.fit(Xtr[:, cols], ytr)
                pred = pls.predict(Xv[:, cols]).ravel()
                rmsep = float(np.sqrt(np.mean((pred - yv) ** 2)))
            except Exception as e:  # noqa: BLE001
                if log is not None:
                    log.debug("PLS failed (start=%d len=%d): %s", start, length, e)
                continue
            rmsep_grid[s_i, length - 1] = rmsep
            if rmsep < best_rmsep:
                best_rmsep = rmsep
                best_subset = list(cols)
                best_start = int(start)
                best_len = length

    return {
        "best_subset": sorted(int(b) for b in best_subset),
        "best_subset_order": [int(b) for b in best_subset],
        "best_start_band": best_start,
        "best_length": int(best_len),
        "best_rmsep": float(best_rmsep),
        "start_bands": [int(s) for s in start_bands],
        "chains": {int(k): [int(x) for x in v] for k, v in chains.items()},
        "rmsep_grid": rmsep_grid.tolist(),
    }


# ---------------------------------------------------------------------------
# CARS

def _pls_rmsecv(X: np.ndarray, y: np.ndarray, n_components: int, n_folds: int,
                seed: int) -> float:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sse = 0.0
    n_total = 0
    for tr, te in kf.split(X):
        n_comp = min(n_components, X.shape[1], len(tr) - 1)
        if n_comp < 1:
            return np.inf
        try:
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X[tr], y[tr])
            pred = pls.predict(X[te]).ravel()
        except Exception:
            return np.inf
        sse += float(np.sum((pred - y[te]) ** 2))
        n_total += len(te)
    return float(np.sqrt(sse / max(n_total, 1)))


def cars(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 50,
    n_components: int = 10,
    cv_folds: int = 5,
    mc_ratio: float = 0.8,
    seed: int = SEED,
    log=None,
) -> dict:
    """Competitive Adaptive Reweighted Sampling.

    EDF schedule: ratio_i = a * exp(-k * i), with
        a = (n_bands / 2) ** (1 / (n_iter - 1))
        k = ln(n_bands / 2)         / (n_iter - 1)
    so that ratio_0 = 1.0 (keep all) and ratio_{n_iter-1} = 2/n_bands.

    Each iteration:
        1. EDF: keep top ``ceil(ratio_i * n_bands)`` bands by current weight.
        2. ARS: weighted bootstrap (with replacement) over kept bands.
        3. Monte-Carlo cal/val split (``mc_ratio`` of samples for calibration).
        4. Fit PLS on calibration with the resampled band subset.
        5. Update weights to |PLS coefficients|.
        6. Score the *unique* band subset by K-fold RMSECV.

    Best subset = subset of the iteration with minimum RMSECV.
    """
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(seed)
    n_samples, n_bands = X.shape
    if n_iter < 2:
        raise ValueError("n_iter must be >= 2")

    a = (n_bands / 2.0) ** (1.0 / (n_iter - 1))
    k = np.log(n_bands / 2.0) / (n_iter - 1)

    weights = np.ones(n_bands, dtype=np.float64)
    kept_history: list[list[int]] = []
    rmsep_history: list[float] = []
    n_kept_history: list[int] = []
    weights_history: list[np.ndarray] = []

    n_cal = max(int(mc_ratio * n_samples), 10)

    for i in tqdm(range(n_iter), desc="CARS iters"):
        # 1. EDF
        ratio_i = a * np.exp(-k * i)
        n_keep_edf = max(2, int(np.ceil(ratio_i * n_bands)))
        order = np.argsort(-weights)
        edf_kept = order[:n_keep_edf]

        # 2. ARS: weighted bootstrap inside the EDF-kept pool
        w_pool = weights[edf_kept]
        if w_pool.sum() <= 0 or i == 0:
            sampled = np.unique(edf_kept)
        else:
            probs = w_pool / w_pool.sum()
            draws = rng.choice(edf_kept, size=n_keep_edf, replace=True, p=probs)
            sampled = np.unique(draws)

        if sampled.size < 1:
            break

        # 3. Monte-Carlo cal split
        cal_idx = rng.choice(n_samples, size=n_cal, replace=False)
        Xc = X[cal_idx][:, sampled]
        yc = y[cal_idx]

        # 4. Fit PLS
        n_comp = min(n_components, sampled.size, n_cal - 1)
        if n_comp < 1:
            continue
        try:
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(Xc, yc)
        except Exception as e:  # noqa: BLE001
            if log is not None:
                log.debug("CARS PLS failed at iter %d: %s", i, e)
            continue

        coef = pls.coef_.ravel()
        if coef.size != sampled.size:
            # sklearn returns (n_features,) for single-output PLS in recent
            # versions; older versions return (1, n_features) -- already
            # handled by .ravel().
            coef = coef[: sampled.size]

        # 5. Update weights
        new_weights = np.zeros(n_bands, dtype=np.float64)
        new_weights[sampled] = np.abs(coef)
        # Avoid all-zero weights collapsing the next iteration
        if new_weights.sum() <= 0:
            new_weights[sampled] = 1.0
        weights = new_weights

        # 6. Score the unique kept subset
        rmsep = _pls_rmsecv(X[:, sampled], y, n_components=n_comp,
                            n_folds=cv_folds, seed=seed + i)
        kept_history.append([int(b) for b in sampled])
        rmsep_history.append(float(rmsep))
        n_kept_history.append(int(sampled.size))
        weights_history.append(weights.copy())

    if not rmsep_history:
        raise RuntimeError("CARS failed to complete any iteration")

    best_i = int(np.argmin(rmsep_history))
    best_subset = sorted(int(b) for b in kept_history[best_i])

    return {
        "best_subset": best_subset,
        "best_iter": best_i,
        "best_rmsep": float(rmsep_history[best_i]),
        "rmsep_history": rmsep_history,
        "n_kept_history": n_kept_history,
        "subset_history": kept_history,
        "final_weights": weights.tolist(),
    }


def cars_topk_by_weight(cars_res: dict, k: int) -> list[int]:
    """Convenience: top-k bands by CARS final-iteration |coefficient|."""
    w = np.asarray(cars_res["final_weights"], dtype=np.float64)
    order = np.argsort(-w)
    return sorted(int(b) for b in order[:k])


# ---------------------------------------------------------------------------
# MI

def mi(X: np.ndarray, y: np.ndarray, n_select: int, seed: int = SEED) -> dict:
    from sklearn.feature_selection import mutual_info_classif

    mi_vals = mutual_info_classif(X, y.astype(int), random_state=seed)
    order = np.argsort(-mi_vals)
    selected = [int(b) for b in order[:n_select]]
    return {
        "best_subset": sorted(selected),
        "best_subset_order": selected,
        "mi_per_band": [float(v) for v in mi_vals],
    }


# ---------------------------------------------------------------------------
# Reporting / plotting

def _write_report(
    path: Path,
    n_bands: int,
    n_samples: int,
    spa_res: dict,
    cars_res: dict,
    mi_res: dict,
    cars_topk: list[int],
    runtimes: dict,
) -> None:
    lines = []
    lines.append("Hyperspectral feature band selection: SPA / CARS / MI")
    lines.append("=" * 60)
    lines.append(f"n_bands  = {n_bands}")
    lines.append(f"n_pixels = {n_samples}")
    lines.append("")

    lines.append(f"SPA   ({runtimes.get('spa', 0):.1f}s)")
    lines.append(f"  best subset (sorted): {spa_res['best_subset']}")
    lines.append(f"  selection order:      {spa_res['best_subset_order']}")
    lines.append(f"  start band:           {spa_res['best_start_band']}")
    lines.append(f"  chain length:         {spa_res['best_length']}")
    lines.append(f"  RMSEP (validation):   {spa_res['best_rmsep']:.4f}")
    lines.append("")

    lines.append(f"CARS  ({runtimes.get('cars', 0):.1f}s)")
    lines.append(f"  best subset:        {cars_res['best_subset']}")
    lines.append(f"  best iteration:     {cars_res['best_iter']}")
    lines.append(f"  RMSECV at min:      {cars_res['best_rmsep']:.4f}")
    lines.append(f"  subset size:        {len(cars_res['best_subset'])}")
    lines.append(f"  top-{len(cars_topk)} by final weights: {cars_topk}")
    lines.append("")

    lines.append(f"MI    ({runtimes.get('mi', 0):.1f}s)")
    lines.append(f"  top {len(mi_res['best_subset'])} by MI: {mi_res['best_subset']}")
    lines.append(f"  selection order:    {mi_res['best_subset_order']}")
    mi_vals = np.asarray(mi_res["mi_per_band"])
    lines.append(f"  MI range:           [{mi_vals.min():.4f}, {mi_vals.max():.4f}]")
    lines.append("")

    # Consensus uses the comparable top-k subsets across the three methods
    spa_set = set(spa_res["best_subset"])
    cars_set = set(cars_topk)
    mi_set = set(mi_res["best_subset"])
    common = spa_set & cars_set & mi_set
    pair_sc = spa_set & cars_set
    pair_sm = spa_set & mi_set
    pair_cm = cars_set & mi_set

    lines.append("Consensus (top-k subsets)")
    lines.append(f"  intersection of all 3:   {sorted(common)}")
    lines.append(f"  SPA ∩ CARS_topk:         {sorted(pair_sc)}")
    lines.append(f"  SPA ∩ MI:                {sorted(pair_sm)}")
    lines.append(f"  CARS_topk ∩ MI:          {sorted(pair_cm)}")
    lines.append(f"  union:                   {sorted(spa_set | cars_set | mi_set)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_combined(
    path: Path,
    n_bands: int,
    mi_vals: np.ndarray,
    spa_res: dict,
    cars_res: dict,
    mi_res: dict,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = np.arange(n_bands)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Top: MI per band + final CARS weights
    ax = axes[0]
    ax.plot(bands, mi_vals, color="tab:brown", lw=1.3, label="MI")
    ax.set_ylabel("Mutual information", color="tab:brown")
    ax.tick_params(axis="y", labelcolor="tab:brown")

    ax2 = ax.twinx()
    cw = np.asarray(cars_res["final_weights"], dtype=np.float64)
    if cw.max() > 0:
        cw = cw / cw.max()
    ax2.plot(bands, cw, color="tab:cyan", lw=1.0, alpha=0.7,
             label="CARS final |coef| (norm.)")
    ax2.set_ylabel("CARS weight (norm.)", color="tab:cyan")
    ax2.tick_params(axis="y", labelcolor="tab:cyan")
    ax.set_title("(a) Per-band importance: MI vs CARS final weights")
    ax.grid(alpha=0.3)

    # Bottom: selected band markers
    ax = axes[1]
    ax.set_xlim(-0.5, n_bands - 0.5)
    ax.set_ylim(0, 4)
    for b in spa_res["best_subset"]:
        ax.axvline(b, color="tab:blue", alpha=0.6, lw=1)
    for b in cars_res["best_subset"]:
        ax.axvline(b, color="tab:green", alpha=0.6, lw=1)
    for b in mi_res["best_subset"]:
        ax.axvline(b, color="tab:red", alpha=0.6, lw=1)
    # Legend handles
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=2, label=f"SPA ({len(spa_res['best_subset'])})"),
            Line2D([0], [0], color="tab:green", lw=2, label=f"CARS ({len(cars_res['best_subset'])})"),
            Line2D([0], [0], color="tab:red", lw=2, label=f"MI ({len(mi_res['best_subset'])})"),
        ],
        loc="upper right",
    )
    ax.set_xlabel("HSI band index")
    ax.set_yticks([])
    ax.set_title("(b) Selected bands per method")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_cars_trace(path: Path, cars_res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rms = np.asarray(cars_res["rmsep_history"], dtype=np.float64)
    nkept = np.asarray(cars_res["n_kept_history"], dtype=np.int64)
    iters = np.arange(len(rms))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(iters, rms, color="tab:red", lw=1.5, label="RMSECV")
    ax1.axvline(cars_res["best_iter"], color="grey", ls="--", lw=1,
                label=f"best iter ({cars_res['best_iter']})")
    ax1.set_xlabel("CARS iteration")
    ax1.set_ylabel("RMSECV", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(iters, nkept, color="tab:blue", lw=1.0, alpha=0.7, label="#bands kept")
    ax2.set_ylabel("#bands kept", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("CARS: RMSECV and kept-band count per iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_spa_grid(path: Path, spa_res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(spa_res["rmsep_grid"], dtype=np.float64)
    grid = np.where(np.isfinite(grid), grid, np.nan)
    starts = spa_res["start_bands"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   origin="lower",
                   extent=[0.5, grid.shape[1] + 0.5, -0.5, grid.shape[0] - 0.5])
    ax.set_xlabel("Chain length")
    ax.set_ylabel("Starting band rank (top-N by column norm)")
    ax.set_yticks(range(len(starts)))
    ax.set_yticklabels([str(s) for s in starts], fontsize=7)
    fig.colorbar(im, ax=ax, label="RMSEP")
    ax.set_title("SPA: RMSEP per (starting band, chain length)")

    # Mark the best cell
    bi = starts.index(spa_res["best_start_band"]) if spa_res["best_start_band"] in starts else -1
    if bi >= 0:
        ax.plot(spa_res["best_length"], bi, marker="*", color="red", ms=14, mec="white")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_output_dir(SCRIPT_NAME, args.output_root)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    band_axis = args.band_axis if args.band_axis == "auto" else int(args.band_axis)

    t0 = time.time()
    X, y, n_bands, used_stems = load_pixel_matrix(
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
        n_per_class=args.n_per_class,
        healthy_erosion=args.healthy_erosion,
        band_axis=band_axis,
        normalize=args.normalize,
        limit_images=args.limit_images,
        seed=args.seed,
        log=log,
    )
    log.info("dataset prepared: X=%s y=%s (%.1fs)", X.shape, y.shape, time.time() - t0)

    if X.shape[0] < 20:
        log.error("not enough samples (%d); aborting", X.shape[0])
        return 1

    # ---- SPA ----
    log.info("running SPA")
    t = time.time()
    spa_res = spa(
        X, y,
        n_select=args.n_select,
        max_starts=args.spa_max_starts,
        n_components_max=args.max_components,
        val_frac=args.spa_val_frac,
        seed=args.seed,
        log=log,
    )
    runtime_spa = time.time() - t
    log.info("SPA  done in %.1fs -- best=%s rmsep=%.4f",
             runtime_spa, spa_res["best_subset"], spa_res["best_rmsep"])

    # ---- CARS ----
    log.info("running CARS")
    t = time.time()
    cars_res = cars(
        X, y,
        n_iter=args.cars_iter,
        n_components=args.max_components,
        cv_folds=args.cars_cv_folds,
        mc_ratio=args.cars_mc_ratio,
        seed=args.seed,
        log=log,
    )
    runtime_cars = time.time() - t
    log.info("CARS done in %.1fs -- best_iter=%d rmsep=%.4f n_kept=%d",
             runtime_cars, cars_res["best_iter"], cars_res["best_rmsep"],
             len(cars_res["best_subset"]))

    # ---- MI ----
    log.info("running MI")
    t = time.time()
    mi_res = mi(X, y, n_select=args.n_select, seed=args.seed)
    runtime_mi = time.time() - t
    log.info("MI   done in %.1fs -- top=%s", runtime_mi, mi_res["best_subset"])

    cars_topk = cars_topk_by_weight(cars_res, k=args.n_select)
    log.info("CARS top-%d by final weights: %s", args.n_select, cars_topk)

    # ---- Persist ----
    runtimes = {"spa": runtime_spa, "cars": runtime_cars, "mi": runtime_mi}
    save_json(
        {
            "config": vars(args),
            "n_bands": n_bands,
            "n_samples": int(X.shape[0]),
            "n_images_used": len(used_stems),
            "runtimes_sec": runtimes,
            "spa": spa_res,
            "cars": cars_res,
            "cars_topk_by_weight": cars_topk,
            "mi": mi_res,
        },
        out_dir / "selected_bands.json",
    )

    _write_report(
        out_dir / "band_selection_report.txt",
        n_bands=n_bands, n_samples=int(X.shape[0]),
        spa_res=spa_res, cars_res=cars_res, mi_res=mi_res,
        cars_topk=cars_topk,
        runtimes=runtimes,
    )

    _plot_combined(
        out_dir / "band_selection.png",
        n_bands=n_bands,
        mi_vals=np.asarray(mi_res["mi_per_band"]),
        spa_res=spa_res, cars_res=cars_res, mi_res=mi_res,
    )
    _plot_cars_trace(out_dir / "cars_trace.png", cars_res)
    _plot_spa_grid(out_dir / "spa_rmsep_grid.png", spa_res)

    log.info("done -- outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
