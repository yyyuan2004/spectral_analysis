"""Single integration point with the main ``msi`` project.

If the interfaces in your ``msi`` repo differ from the assumptions below,
**edit this file only** -- the analysis scripts depend on these wrappers,
not on the upstream import paths.

Required upstream symbols (defaults):
  * ``scripts.band_range_search.train_and_eval``
        Signature assumed:
            train_and_eval(band_indices: list[int],
                           config_path: str,
                           epochs: int = 80,
                           seed: int = 42) -> float   # defect-class IoU
  * ``data.dataset.MSIDataset``
        Used to build the val DataLoader for A4.
  * A model class (e.g. ``models.unet.UNet`` or whatever baseline uses).
        Used to load best_model.pth for A4.

All scripts call into this module so you only have to fix integration in
one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# A3 / A5 training adapter

def run_training(
    band_indices: Sequence[int],
    config_path: str | Path,
    epochs: int = 80,
    seed: int = 42,
    num_channels: int | None = None,
    extra: dict[str, Any] | None = None,
) -> float:
    """Train a UNet on the given HSI band subset and return defect-class IoU.

    Default implementation delegates to ``msi/scripts/band_range_search.py``'s
    ``train_and_eval``.  If that function's signature differs, override here.
    """
    from scripts.band_range_search import train_and_eval  # type: ignore

    kwargs: dict[str, Any] = dict(
        band_indices=list(band_indices),
        config_path=str(config_path),
        epochs=int(epochs),
        seed=int(seed),
    )
    if num_channels is not None:
        kwargs["num_channels"] = int(num_channels)
    if extra:
        kwargs.update(extra)
    return float(train_and_eval(**kwargs))


# ---------------------------------------------------------------------------
# A4 inference adapter

def load_trained_model(checkpoint_path: str | Path):
    """Load a trained segmentation model from ``best_model.pth``.

    Returns: (model, in_channels, band_indices, num_classes)
    """
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        cfg = ckpt.get("config", {}) or {}
    else:
        state_dict = ckpt
        cfg = {}

    in_channels = int(cfg.get("num_channels") or cfg.get("in_channels") or 3)
    num_classes = int(cfg.get("num_classes", 2))
    band_indices = list(cfg.get("band_indices") or [])

    # Default: assume msi has ``models.unet.UNet``.  Override if needed.
    try:
        from models.unet import UNet  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Could not import models.unet.UNet from the msi project. "
            "Edit scripts/preanalysis/_adapters.py::load_trained_model to "
            "match your model class."
        ) from e

    model = UNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, in_channels, band_indices, num_classes


def get_val_loader(
    data_root: str | Path,
    band_indices: Sequence[int],
    batch_size: int = 1,
    num_workers: int = 0,
):
    """Build a torch DataLoader for the validation split.

    Default: uses ``data.dataset.MSIDataset`` with ``split='val'``.  Adjust
    the kwargs to match your dataset constructor.
    """
    from torch.utils.data import DataLoader  # type: ignore
    try:
        from data.dataset import MSIDataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Could not import MSIDataset from the msi project. "
            "Edit scripts/preanalysis/_adapters.py::get_val_loader."
        ) from e

    ds = MSIDataset(
        root=str(data_root),
        split="val",
        band_indices=list(band_indices),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))


def unpack_batch(batch) -> tuple:
    """Normalise a batch from MSIDataset into (image, defect_mask, whole_mask, stem).

    The msi dataset may return a tuple or a dict.  Adjust if necessary.
    """
    if isinstance(batch, dict):
        img = batch["image"]
        mask = batch.get("mask", batch.get("label"))
        whole = batch.get("whole", batch.get("apple_mask"))
        stem = batch.get("stem", batch.get("filename", ["unknown"]))
        return img, mask, whole, stem
    if isinstance(batch, (list, tuple)):
        # Convention: (image, mask) or (image, mask, whole) or (image, mask, whole, stem)
        if len(batch) == 2:
            return batch[0], batch[1], None, None
        if len(batch) == 3:
            return batch[0], batch[1], batch[2], None
        return batch[0], batch[1], batch[2], batch[3]
    raise TypeError(f"unrecognised batch type: {type(batch).__name__}")
