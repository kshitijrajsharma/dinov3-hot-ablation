"""Per-chip pixel IoU and instance F1 on the fAIr-models sample data.

Burns labels.geojson onto each chip via geomltoolkits, runs v5 inference,
aggregates TPs/FPs/FNs across chips, and prints split-level metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from geomltoolkits.raster.burn import burn_labels
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from dinov3_hot.config import load_config, resolve_root
from dinov3_hot.data import load_norm_stats
from dinov3_hot.infer import _normalize, load_model
from dinov3_hot.metrics import _count

REPO = Path(__file__).resolve().parents[1]
FAIR_ROOT = REPO / "data/fair_sample/sample"


def _predict_chip(
    model, chip_path: Path, mean, std, device: str, threshold: float
) -> tuple[np.ndarray, np.ndarray, "rasterio.Affine"]:
    with rasterio.open(chip_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        transform = src.transform
    x = _normalize(img, mean, std).unsqueeze(0).to(device)
    with torch.inference_mode():
        main_logits, _ = model(x)
    logits = main_logits[0]
    mask_prob = torch.sigmoid(logits[0]).cpu().numpy()
    distance = torch.tanh(logits[2]).cpu().numpy()
    fg = mask_prob > threshold
    if not fg.any():
        labels = np.zeros_like(fg, dtype=np.uint32)
        return fg.astype(np.uint8), labels, transform
    coords = peak_local_max(distance, min_distance=4, labels=fg.astype(np.uint8), exclude_border=False)
    if len(coords) == 0:
        labels = ndimage.label(fg)[0].astype(np.uint32)
    else:
        seeds = np.zeros_like(fg, dtype=bool)
        seeds[tuple(coords.T)] = True
        markers, _ = ndimage.label(seeds)
        labels = watershed(-distance, markers=markers, mask=fg).astype(np.uint32)
    return fg.astype(np.uint8), labels, transform


def _gt_instance_labels(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        gt = (src.read(1) > 0).astype(np.uint8)
    return ndimage.label(gt)[0].astype(np.int32)


def evaluate_split(model, mean, std, device: str, oam_dir: Path, masks_dir: Path, threshold: float) -> dict:
    chips = sorted(oam_dir.glob("*.tif"))
    pix_inter = pix_union = 0
    tp = fp = fn = 0
    for chip in chips:
        mask_path = masks_dir / chip.name
        if not mask_path.exists():
            continue
        pred_bin, pred_lbl, _ = _predict_chip(model, chip, mean, std, device, threshold)
        gt_lbl = _gt_instance_labels(mask_path)
        gt_bin = gt_lbl > 0
        pix_inter += int(np.logical_and(pred_bin, gt_bin).sum())
        pix_union += int(np.logical_or(pred_bin, gt_bin).sum())
        c_tp, c_fp, c_fn = _count(pred_lbl.astype(np.int32), gt_lbl)
        tp += c_tp
        fp += c_fp
        fn += c_fn
    pix_iou = pix_inter / pix_union if pix_union > 0 else 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "n_chips": len(chips),
        "pixel_iou": pix_iou,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="conf/train.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = load_norm_stats(cfg.dataset_repo, resolve_root(cfg))
    model = load_model(args.ckpt, cfg, device=device).to(device)

    results = {}
    for split in ("train", "test"):
        oam_dir = FAIR_ROOT / split / "oam"
        masks_dir = FAIR_ROOT / split / "masks"
        labels_geojson = FAIR_ROOT / split / "osm/labels.geojson"
        burn_labels(
            labels_path=str(labels_geojson),
            chips_dir=str(oam_dir),
            output_dir=str(masks_dir),
            burn_value=255,
        )
        results[split] = evaluate_split(model, mean, std, device, oam_dir, masks_dir, args.threshold)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
