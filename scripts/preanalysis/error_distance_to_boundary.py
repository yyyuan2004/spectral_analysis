"""A4 -- Per-pixel error distance to the GT bruise boundary.

Purpose
-------
Show that segmentation errors are concentrated within a narrow boundary
buffer around the ground-truth bruise edge.  This supports the argument
that the inter-annotator agreement bound is the practical ceiling.

Inputs
------
* trained checkpoint (default ``../msi/outputs/baseline_seed42/checkpoints/best_model.pth``)
* val split from the msi project (loaded via ``_adapters.get_val_loader``)

Outputs (under ``outputs/preanalysis/error_distance_to_boundary/``)
------------------------------------------------------------------
* ``error_distance.json``         : aggregated stats + per-image counts
* ``error_distance_histograms.png``  : FP / FN histograms + CDFs
* ``examples/<stem>.png``         : up to 9 example overlays
* ``error_distance_report.txt``   : the headline numbers
* ``error_distance_to_boundary.log``

Typical runtime
---------------
Inference is fast (a few minutes for ~55 val images on GPU).  Distance
transform + plotting dominate.  Expect 10-30 minutes total, well under
the 8 h budget.
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

    for idx, batch in enumerate(tqdm(loader, desc="inference")):
        img, mask, _whole, stem = unpack_batch(batch)
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
    log.info("summary: %s", summary)

    save_json({
        "config": {
            "checkpoint": args.checkpoint,
            "data_root": str(args.data_root),
            "band_indices": list(band_indices) if band_indices else [],
        },
        "summary": summary,
        "per_image": per_image,
    }, out_dir / "error_distance.json")

    _plot_histograms(out_dir / "error_distance_histograms.png", fp_all, fn_all)
    _write_report(out_dir / "error_distance_report.txt", summary)

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


def _write_report(path: Path, summary: dict) -> None:
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
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
