"""A4 -- Per-pixel error distance to the GT bruise boundary + specular overlap.

Purpose
-------
1. Show that segmentation errors are concentrated within a narrow boundary
   buffer around the ground-truth bruise edge (supports the argument that
   the inter-annotator agreement bound is the practical ceiling).
2. Quantify how much of the FP/FN mass falls inside specular highlight
   regions of the apple, defined per-image as pixels exceeding the
   per-channel p98 (configurable) of the 3 model-input channels, taken
   within the apple region.

Inputs
------
* trained checkpoint (default ``../msi/outputs/baseline_seed42/checkpoints/best_model.pth``)
* val split from the msi project (loaded via ``_adapters.get_val_loader``)
* per-image apple region masks at ``<data-root>/whole/<stem>.npy``
  (falls back to disk if the dataset's batch doesn't carry ``whole``)

Outputs (under ``outputs/preanalysis/error_distance_to_boundary/``)
------------------------------------------------------------------
* ``error_distance.json``         : aggregated stats, specular stats, per-image counts
* ``error_distance_histograms.png``  : FP / FN histograms + CDFs
* ``examples/<stem>.png``         : up to 9 example overlays
* ``error_distance_report.txt``   : the headline numbers + specular overlap section
* ``error_distance_to_boundary.log``

Typical runtime
---------------
Inference is fast (a few minutes for ~55 val images on GPU).  Distance
transform + specular percentile + plotting dominate.  Expect 10-30
minutes total, well under the 8 h budget.
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
    DEFAULT_MSI_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SEED,
    make_output_dir,
    save_json,
    set_seed,
    setup_logging,
    setup_msi_path,
)

SCRIPT_NAME = "error_distance_to_boundary"
DEFAULT_CHECKPOINT = "../msi/outputs/baseline_seed42/checkpoints/best_model.pth"
HIST_BINS = list(range(0, 21)) + [10_000]  # last bucket = "20+"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=str)
    p.add_argument("--msi-root", default=DEFAULT_MSI_ROOT, type=str)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=str)
    p.add_argument("--band-indices", type=int, nargs="*", default=None,
                   help="Override band indices (defaults to those stored in the checkpoint)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-examples", type=int, default=9,
                   help="Number of example overlay images to write")
    p.add_argument("--specular-percentile", type=float, default=98.0,
                   help="Per-channel percentile threshold (within whole_mask) "
                        "for the specular highlight definition (default 98)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    out_dir = make_output_dir(SCRIPT_NAME, args.output_root)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    log = setup_logging(SCRIPT_NAME, out_dir)
    log.info("args: %s", vars(args))

    setup_msi_path(args.msi_root)
    from _adapters import get_val_loader, load_trained_model, unpack_batch  # noqa: E402
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    log.info("loading checkpoint: %s", args.checkpoint)
    model, in_channels, ckpt_band_indices, num_classes = load_trained_model(args.checkpoint)
    model = model.to(device).eval()

    band_indices = args.band_indices if args.band_indices is not None else ckpt_band_indices
    log.info("band_indices=%s, in_channels=%d, num_classes=%d", band_indices, in_channels, num_classes)
    if not band_indices:
        log.warning("no band_indices found in checkpoint and none provided; dataset must default")

    loader = get_val_loader(
        args.data_root,
        band_indices=band_indices or [],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Inference + distance-to-boundary aggregation
    # ------------------------------------------------------------------
    fp_dists_all: list[np.ndarray] = []
    fn_dists_all: list[np.ndarray] = []
    per_image: list[dict] = []
    example_payloads: list[dict] = []
    spec_total_apple = 0
    spec_total_specular = 0
    spec_fp_in_specular = 0
    spec_fn_in_specular = 0
    spec_total_fp = 0
    spec_total_fn = 0
    spec_images_analysed = 0
    spec_images_skipped = 0

    for idx, batch in enumerate(tqdm(loader, desc="inference")):
        img, mask, whole, stem = unpack_batch(batch)
        if img is None or mask is None:
            log.warning("skipping batch %d (missing fields)", idx)
            continue
        if isinstance(stem, (list, tuple)):
            stem_str = str(stem[0])
        else:
            stem_str = str(stem) if stem is not None else f"img_{idx:04d}"

        with torch.no_grad():
            logits = model(img.to(device))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.dim() == 4 and logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).long()
            else:
                pred = logits.argmax(dim=1, keepdim=True)
        # Convention: take first sample in batch; A4 typically runs with batch_size=1
        pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
        if isinstance(mask, torch.Tensor):
            gt_np = mask.cpu().numpy()
        else:
            gt_np = np.asarray(mask)
        gt_np = gt_np.squeeze().astype(np.uint8)
        gt_np = (gt_np > 0).astype(np.uint8)
        if pred_np.shape != gt_np.shape:
            log.warning("shape mismatch %s pred=%s gt=%s", stem_str, pred_np.shape, gt_np.shape)
            continue

        dist_to_boundary = _distance_to_gt_boundary(gt_np)
        fp_mask = (pred_np == 1) & (gt_np == 0)
        fn_mask = (pred_np == 0) & (gt_np == 1)
        fp_d = dist_to_boundary[fp_mask]
        fn_d = dist_to_boundary[fn_mask]
        fp_dists_all.append(fp_d)
        fn_dists_all.append(fn_d)

        # Specular reflection overlap analysis (within whole_mask)
        whole_np = _resolve_whole_mask(whole, stem_str, args.data_root, gt_np.shape, log)
        per_specular = None
        if whole_np is not None:
            specular_mask = _compute_specular_mask(img, whole_np, args.specular_percentile)
            if specular_mask is not None:
                spec_images_analysed += 1
                apple_area = int(whole_np.sum())
                spec_area = int(specular_mask.sum())
                fp_in_spec = int((fp_mask & specular_mask).sum())
                fn_in_spec = int((fn_mask & specular_mask).sum())
                spec_total_apple += apple_area
                spec_total_specular += spec_area
                spec_fp_in_specular += fp_in_spec
                spec_fn_in_specular += fn_in_spec
                spec_total_fp += int(fp_mask.sum())
                spec_total_fn += int(fn_mask.sum())
                per_specular = {
                    "apple_area": apple_area,
                    "specular_area": spec_area,
                    "fp_in_specular": fp_in_spec,
                    "fn_in_specular": fn_in_spec,
                }
            else:
                spec_images_skipped += 1
        else:
            spec_images_skipped += 1

        per_image.append({
            "stem": stem_str,
            "n_fp": int(fp_mask.sum()),
            "n_fn": int(fn_mask.sum()),
            "fp_within_3px": int((fp_d <= 3).sum()),
            "fp_within_5px": int((fp_d <= 5).sum()),
            "fp_within_10px": int((fp_d <= 10).sum()),
            "fn_within_3px": int((fn_d <= 3).sum()),
            "fn_within_5px": int((fn_d <= 5).sum()),
            "fn_within_10px": int((fn_d <= 10).sum()),
            "specular": per_specular,
        })

        # Cache payload for example rendering
        if len(example_payloads) < args.n_examples:
            try:
                img_first_band = _select_display_band(img)
            except Exception:
                img_first_band = None
            example_payloads.append({
                "stem": stem_str,
                "display": img_first_band,
                "gt": gt_np,
                "pred": pred_np,
                "n_err": int(fp_mask.sum() + fn_mask.sum()),
            })

    if not fp_dists_all and not fn_dists_all:
        log.error("no predictions collected; aborting")
        return 1

    fp_all = np.concatenate(fp_dists_all) if fp_dists_all else np.empty(0)
    fn_all = np.concatenate(fn_dists_all) if fn_dists_all else np.empty(0)
    log.info("aggregate FP=%d FN=%d", len(fp_all), len(fn_all))

    summary = _summarise(fp_all, fn_all)
    specular_summary = _specular_summary(
        spec_total_apple, spec_total_specular,
        spec_total_fp, spec_total_fn,
        spec_fp_in_specular, spec_fn_in_specular,
        spec_images_analysed, spec_images_skipped,
        args.specular_percentile,
    )
    log.info("summary: %s", summary)
    log.info("specular: %s", specular_summary)

    save_json({
        "config": {
            "checkpoint": args.checkpoint,
            "data_root": str(args.data_root),
            "band_indices": list(band_indices) if band_indices else [],
            "specular_percentile": float(args.specular_percentile),
        },
        "summary": summary,
        "specular": specular_summary,
        "per_image": per_image,
    }, out_dir / "error_distance.json")

    _plot_histograms(out_dir / "error_distance_histograms.png", fp_all, fn_all)
    _write_report(out_dir / "error_distance_report.txt", summary, specular_summary)

    # Render up to n_examples overlays (ranked by error count)
    example_payloads.sort(key=lambda d: -d["n_err"])
    for payload in example_payloads[: args.n_examples]:
        _plot_example(examples_dir / f"{payload['stem']}.png", payload)

    log.info("done. outputs in %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Helpers

def _distance_to_gt_boundary(gt: np.ndarray) -> np.ndarray:
    """Return per-pixel distance to the nearest GT bruise boundary."""
    import cv2

    if gt.sum() == 0 or gt.sum() == gt.size:
        return np.full(gt.shape, 1e6, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
    src = ((~boundary).astype(np.uint8)) * 255  # boundary pixels are source (value 0)
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def _summarise(fp: np.ndarray, fn: np.ndarray) -> dict:
    def stats(d: np.ndarray) -> dict:
        if len(d) == 0:
            return {"n": 0}
        return {
            "n": int(len(d)),
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)),
            "within_3px": float((d <= 3).mean()),
            "within_5px": float((d <= 5).mean()),
            "within_10px": float((d <= 10).mean()),
        }

    return {"fp": stats(fp), "fn": stats(fn),
            "all": stats(np.concatenate([fp, fn]) if len(fp) + len(fn) else np.empty(0))}


def _plot_histograms(path: Path, fp: np.ndarray, fn: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.arange(0, 22)  # 0,1,...,21 with 20+ in last bucket via clip

    for ax, data, label, color in [
        (axes[0, 0], fp, "FP", "tab:red"),
        (axes[0, 1], fn, "FN", "tab:blue"),
    ]:
        if len(data):
            clipped = np.clip(data, 0, 20)
            ax.hist(clipped, bins=bins, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.set_title(f"{label} distance to GT boundary (n={len(data)})")
        ax.set_xlabel("distance (pixels, last bin = 20+)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

    # CDFs
    for ax_idx, (data, label, color) in enumerate([(fp, "FP", "tab:red"), (fn, "FN", "tab:blue")]):
        ax = axes[1, ax_idx]
        if len(data):
            xs = np.sort(data)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=color, lw=1.5)
        ax.axvline(3, color="grey", linestyle="--", lw=0.8, label="3 px")
        ax.axvline(5, color="grey", linestyle=":", lw=0.8, label="5 px")
        ax.set_xlim(0, max(20, np.max(data) if len(data) else 20))
        ax.set_ylim(0, 1)
        ax.set_xlabel("distance (pixels)")
        ax.set_ylabel("cumulative fraction")
        ax.set_title(f"{label} CDF")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _select_display_band(img):
    """Return a single 2D float array for visual context."""
    import torch

    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    # collapse batch dim if present
    if arr.ndim == 4:
        arr = arr[0]
    # arr shape now (C, H, W) or (H, W, C)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[-1]:
            return arr[0]
        return arr[..., 0]
    return arr


def _plot_example(path: Path, payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = payload["display"]
    gt = payload["gt"].astype(np.uint8)
    pred = payload["pred"].astype(np.uint8)

    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    import cv2
    if gt.sum() and gt.sum() < gt.size:
        kernel = np.ones((3, 3), dtype=np.uint8)
        gt_boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
    else:
        gt_boundary = np.zeros_like(gt, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    if display is not None:
        d = display.astype(np.float32)
        lo, hi = float(np.percentile(d, 2)), float(np.percentile(d, 98))
        if hi - lo > 1e-9:
            d = np.clip((d - lo) / (hi - lo), 0, 1)
        ax.imshow(d, cmap="gray")
    else:
        ax.imshow(np.zeros_like(gt), cmap="gray")

    overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
    overlay[gt_boundary] = [1.0, 1.0, 0.0, 0.9]      # yellow boundary
    overlay[fp] = [1.0, 0.0, 0.0, 0.65]              # red FP
    overlay[fn] = [0.0, 0.7, 1.0, 0.65]              # cyan FN
    ax.imshow(overlay)
    ax.set_title(f"{payload['stem']}  (FP={int(fp.sum())}  FN={int(fn.sum())})\n"
                 "yellow=GT boundary, red=FP, cyan=FN")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, summary: dict, specular_summary: dict | None = None) -> None:
    lines = ["Error distance to GT boundary", "=" * 50, ""]
    for which in ("all", "fp", "fn"):
        s = summary[which]
        lines.append(f"[{which.upper()}] n={s.get('n', 0)}")
        if s.get("n", 0):
            lines.append(f"  mean={s['mean']:.2f} px, median={s['median']:.2f} px, p90={s['p90']:.2f} px")
            lines.append(f"  within 3 px : {s['within_3px']:.1%}")
            lines.append(f"  within 5 px : {s['within_5px']:.1%}")
            lines.append(f"  within 10 px: {s['within_10px']:.1%}")
        lines.append("")
    lines.append("Headline: X% of error pixels are within 3 pixels of the GT boundary "
                 f"-> {summary['all'].get('within_3px', 0):.1%}")

    if specular_summary and specular_summary.get("images_analysed", 0) > 0:
        lines.append("")
        lines.append("-" * 50)
        lines.append("Specular overlap analysis")
        lines.append("-" * 50)
        s = specular_summary
        pct = s["percentile"]
        lines.append(f"Definition: per-channel reflectance > p{pct:g} within whole_mask, "
                     "AND across all 3 input channels.")
        lines.append(f"Images analysed: {s['images_analysed']}  (skipped: {s['images_skipped']})")
        lines.append("")
        lines.append(f"- FP pixels within specular regions: {s['fp_in_specular_frac']:.1%}  "
                     f"({s['fp_in_specular']:,} / {s['total_fp']:,} pixels)")
        lines.append(f"- FN pixels within specular regions: {s['fn_in_specular_frac']:.1%}  "
                     f"({s['fn_in_specular']:,} / {s['total_fn']:,} pixels)")
        lines.append(f"- Specular regions cover {s['specular_frac_of_apple']:.1%} of total apple area  "
                     f"({s['total_specular']:,} / {s['total_apple']:,} pixels)")
        lines.append("")
        lines.append(f"Interpretation: {s['interpretation']}")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Specular reflection overlap helpers

def _resolve_whole_mask(whole, stem_str: str, data_root: str, target_shape, log):
    """Return a (H, W) uint8 mask from the batch field or fall back to disk."""
    try:
        import torch  # type: ignore
        is_tensor = whole is not None and isinstance(whole, torch.Tensor)
    except ImportError:
        is_tensor = False

    if whole is not None:
        if is_tensor:
            arr = whole.detach().cpu().numpy()
        else:
            arr = np.asarray(whole)
        arr = np.squeeze(arr)
        if arr.ndim == 2 and arr.shape == tuple(target_shape):
            return (arr > 0).astype(np.uint8)
        log.warning("unexpected whole shape from batch %s for %s; trying disk fallback",
                    arr.shape, stem_str)

    p = Path(data_root) / "whole" / f"{stem_str}.npy"
    if not p.exists():
        log.warning("whole mask not found at %s; skipping specular analysis for this image", p)
        return None
    try:
        from _common import load_mask
        arr = load_mask(p)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load whole mask %s: %s", p, e)
        return None
    if arr.shape != tuple(target_shape):
        log.warning("whole-mask shape %s != gt shape %s for %s; skipping specular",
                    arr.shape, target_shape, stem_str)
        return None
    return arr


def _compute_specular_mask(img, whole_np: np.ndarray, percentile: float) -> np.ndarray | None:
    """Compute the specular highlight mask within the apple region.

    img: model input tensor / array, expected shape (1, C, H, W), (C, H, W), or (H, W, C).
    whole_np: (H, W) uint8 in {0, 1}.
    Returns (H, W) bool mask: pixels inside whole_mask where ALL C channels
    exceed their per-channel p{percentile}.
    """
    try:
        import torch  # type: ignore
        is_tensor = isinstance(img, torch.Tensor)
    except ImportError:
        is_tensor = False

    if is_tensor:
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        # Heuristic: smaller of (axis 0) vs (axis -1) is channels
        if arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim == 2:
        arr = arr[..., None]
    else:
        return None
    H, W, C = arr.shape
    if (H, W) != whole_np.shape:
        return None

    whole_bool = whole_np > 0
    if not whole_bool.any():
        return None

    specular = np.ones((H, W), dtype=bool)
    for c in range(C):
        vals = arr[:, :, c][whole_bool]
        if vals.size == 0:
            return None
        thresh = float(np.percentile(vals, percentile))
        specular &= (arr[:, :, c] > thresh)
    specular &= whole_bool
    return specular


def _specular_summary(total_apple, total_specular, total_fp, total_fn,
                      fp_in_specular, fn_in_specular,
                      images_analysed, images_skipped, percentile) -> dict:
    def _frac(n, d) -> float:
        return float(n) / float(d) if d else 0.0

    fp_frac = _frac(fp_in_specular, total_fp)
    fn_frac = _frac(fn_in_specular, total_fn)
    spec_frac = _frac(total_specular, total_apple)

    if total_fp == 0 and total_fn == 0:
        interp = "no FP/FN pixels were recorded -- inconclusive."
    elif fp_frac >= 0.30:
        interp = ("specular highlights are a major source of false positives; "
                  "post-hoc masking of specular regions could meaningfully reduce FP.")
    elif fp_frac >= 0.15:
        interp = ("specular highlights contribute non-trivially to false positives; "
                  "consider whether downstream rules can mitigate them.")
    else:
        interp = ("specular highlights are not the main driver of false positives; "
                  "FPs are distributed across non-highlight apple regions.")
    if fn_frac > 0.05:
        interp += (f"  FN overlap with specular ({fn_frac:.1%}) is non-trivial -- "
                   "some bruises may be saturated/occluded by highlights.")
    else:
        interp += f"  FN overlap with specular is small ({fn_frac:.1%}), as expected."

    return {
        "percentile": float(percentile),
        "images_analysed": int(images_analysed),
        "images_skipped": int(images_skipped),
        "total_apple": int(total_apple),
        "total_specular": int(total_specular),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "fp_in_specular": int(fp_in_specular),
        "fn_in_specular": int(fn_in_specular),
        "fp_in_specular_frac": fp_frac,
        "fn_in_specular_frac": fn_frac,
        "specular_frac_of_apple": spec_frac,
        "interpretation": interp,
    }


if __name__ == "__main__":
    raise SystemExit(main())
