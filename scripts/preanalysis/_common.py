"""Shared utilities for preanalysis scripts.

Conventions:
  * Data root layout (default ``/root/autodl-tmp/datasets/full_HSI_dataset``):
        images/<stem>.npy   HSI cube, shape (H, W, B) or (B, H, W) auto-detected
        masks/<stem>.npy    defect (bruise) mask, shape (H, W), values {0, 1}
        whole/<stem>.npy    apple region mask, shape (H, W), values {0, 1}
    The same ``<stem>`` appears in all three subdirectories.
  * msi project root (default ``../msi``) is injected onto ``sys.path`` so that
    ``from data.dataset import MSIDataset``, ``from utils.metrics import ...``
    etc. resolve.  See ``_adapters.py`` for the single integration point.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets/full_HSI_dataset"
DEFAULT_MSI_ROOT = "../msi"
DEFAULT_OUTPUT_ROOT = "outputs/preanalysis"
SEED = 42


# ---------------------------------------------------------------------------
# Reproducibility / environment

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_msi_path(msi_root: str | os.PathLike) -> Path:
    p = Path(msi_root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"msi project root not found: {p}\n"
            f"Pass --msi-root /path/to/msi or clone msi as a sibling of this repo."
        )
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p


# ---------------------------------------------------------------------------
# I/O helpers

def make_output_dir(script_name: str, root: str | os.PathLike = DEFAULT_OUTPUT_ROOT) -> Path:
    out = Path(root) / script_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(script_name: str, out_dir: Path) -> logging.Logger:
    log_path = out_dir / f"{script_name}.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"unserializable type: {type(o).__name__}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Dataset traversal

def list_stems(data_root: Path, subdir: str = "images") -> list[str]:
    d = Path(data_root) / subdir
    stems = sorted(p.stem for p in d.glob("*.npy"))
    if not stems:
        raise FileNotFoundError(f"no .npy files found in {d}")
    return stems


def detect_band_axis(arr: np.ndarray) -> int:
    """Heuristic: pick the smaller of axis-0 vs last axis as the band axis."""
    if arr.ndim != 3:
        raise ValueError(f"expected 3D HSI array, got shape {arr.shape}")
    return 0 if arr.shape[0] < arr.shape[-1] else -1


def load_hsi(path: Path, band_axis: int | str = "auto") -> np.ndarray:
    arr = np.load(path)
    if band_axis == "auto":
        ba = detect_band_axis(arr)
    else:
        ba = int(band_axis)
    if ba == 0:
        arr = np.moveaxis(arr, 0, -1)
    return arr.astype(np.float32, copy=False)


def load_mask(path: Path) -> np.ndarray:
    m = np.load(path)
    if m.ndim == 3:
        # Could be (1, H, W) or (H, W, 1)
        m = m.squeeze()
    return (m > 0).astype(np.uint8)


def load_triplet(
    stem: str,
    data_root: Path,
    band_axis: int | str = "auto",
    load_image: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Return (hsi[H,W,B] or None, defect_mask[H,W], whole_mask[H,W])."""
    data_root = Path(data_root)
    defect = load_mask(data_root / "masks" / f"{stem}.npy")
    whole = load_mask(data_root / "whole" / f"{stem}.npy")
    img = load_hsi(data_root / "images" / f"{stem}.npy", band_axis=band_axis) if load_image else None
    return img, defect, whole


# ---------------------------------------------------------------------------
# Pixel selection helpers

def compute_healthy_mask(
    whole: np.ndarray,
    defect: np.ndarray,
    radius_frac: float = 0.7,
    min_dist_defect: int = 15,
) -> np.ndarray:
    """Strict healthy-pixel mask used by A1/A2.

    Conditions (intersection):
      * whole_mask == 1
      * defect_mask == 0
      * distance to apple centroid <= radius_frac * max_radius
        where max_radius = max distance from any whole-mask pixel to centroid
      * distance to nearest defect pixel >= min_dist_defect (in pixels)
    """
    import cv2

    whole = (whole > 0)
    defect = (defect > 0)
    base = whole & (~defect)
    if not whole.any():
        return np.zeros_like(whole, dtype=bool)

    ys, xs = np.where(whole)
    cy = ys.mean()
    cx = xs.mean()
    H, W = whole.shape
    yy, xx = np.indices((H, W))
    dist_to_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = float(dist_to_center[whole].max())
    inside_radius = dist_to_center <= radius_frac * max_radius

    if defect.any():
        src = ((~defect).astype(np.uint8)) * 255  # defect pixels are 0 (sources)
        dist_to_defect = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    else:
        dist_to_defect = np.full(whole.shape, np.inf, dtype=np.float32)

    far_from_defect = dist_to_defect >= float(min_dist_defect)
    return base & inside_radius & far_from_defect


# ---------------------------------------------------------------------------
# Per-band statistics

def fisher_score(mu_d: np.ndarray, mu_h: np.ndarray, var_d: np.ndarray, var_h: np.ndarray) -> np.ndarray:
    denom = var_d + var_h
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide((mu_d - mu_h) ** 2, denom, out=out, where=denom > 0)
    return out


def cohens_d(mu_d: np.ndarray, mu_h: np.ndarray, var_d: np.ndarray, var_h: np.ndarray) -> np.ndarray:
    pooled = np.sqrt((var_d + var_h) / 2.0)
    out = np.zeros_like(pooled, dtype=np.float64)
    np.divide(mu_d - mu_h, pooled, out=out, where=pooled > 0)
    return out


def auc_per_band(healthy: np.ndarray, defect: np.ndarray) -> np.ndarray:
    """Per-band ROC-AUC with label=1 for defect."""
    from sklearn.metrics import roc_auc_score

    n_h, B = healthy.shape
    n_d = defect.shape[0]
    labels = np.concatenate([np.zeros(n_h, dtype=np.int8), np.ones(n_d, dtype=np.int8)])
    aucs = np.zeros(B, dtype=np.float64)
    for b in range(B):
        scores = np.concatenate([healthy[:, b], defect[:, b]])
        aucs[b] = roc_auc_score(labels, scores)
    return aucs


def minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float64)
    return (a - lo) / (hi - lo)


def top_k_indices(a: np.ndarray, k: int = 10) -> list[int]:
    return [int(i) for i in np.argsort(a)[::-1][:k]]


# ---------------------------------------------------------------------------
# Sufficient statistics container (shared by A1/A2)

def sufficient_stats_path(out_root: str | os.PathLike = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(out_root) / "spectral_curves_with_separability" / "sufficient_stats.npz"


def save_sufficient_stats(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_sufficient_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
