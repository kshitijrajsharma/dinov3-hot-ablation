"""Per-tile eval on the hotosm/vhr-building-segmentation HF test split.

Reproduces the HF row in README.md. Each tile carries `bbox_{west,south,east,north}`
in WGS84; we use those for the per-tile transform, no hardcoded pixel scale."""

import argparse
import json

import numpy as np
import rasterio
import torch
from datasets import load_dataset
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from dinov3_hot.config import load_config, resolve_root
from dinov3_hot.data import load_norm_stats
from dinov3_hot.infer import _normalize, load_model, vectorize
from dinov3_hot.metrics import _count, polygon_orthogonality, polygon_vertex_count

_TILE_CRS = rasterio.crs.CRS.from_epsg(4326)  # HF bbox fields are WGS84


def _tile_transform(ex: dict, height: int, width: int) -> rasterio.Affine:
    return rasterio.transform.from_bounds(
        ex["bbox_west"], ex["bbox_south"], ex["bbox_east"], ex["bbox_north"], width, height
    )


def _predict(model, img_uint8: np.ndarray, mean, std, device: str, threshold: float):
    x = _normalize(img_uint8, mean, std).unsqueeze(0).to(device)
    with torch.inference_mode():
        main_logits, _ = model(x)
    logits = main_logits[0]
    mask_prob = torch.sigmoid(logits[0]).cpu().numpy()
    distance = torch.tanh(logits[2]).cpu().numpy()
    fg = mask_prob > threshold
    if not fg.any():
        return fg.astype(np.uint8), np.zeros_like(fg, dtype=np.uint32)
    coords = peak_local_max(distance, min_distance=4, labels=fg.astype(np.uint8), exclude_border=False)
    if len(coords) == 0:
        labels = ndimage.label(fg)[0].astype(np.uint32)
    else:
        seeds = np.zeros_like(fg, dtype=bool)
        seeds[tuple(coords.T)] = True
        markers, _ = ndimage.label(seeds)
        labels = watershed(-distance, markers=markers, mask=fg).astype(np.uint32)
    return fg.astype(np.uint8), labels


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="conf/train.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Eval first N tiles only (for smoke runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = load_norm_stats(cfg.dataset_repo, resolve_root(cfg))
    model = load_model(args.ckpt, cfg, device=device).to(device)

    ds = load_dataset(cfg.dataset_repo, split="test")
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))

    pix_inter = pix_union = 0
    tp = fp = fn = 0
    pred_vertex_sum = 0.0
    pred_orth_sum = 0.0
    pred_poly_count = 0
    gt_vertex_sum = 0.0
    gt_orth_sum = 0.0
    gt_poly_count = 0

    for i in range(n_total):
        ex = ds[i]
        img = np.array(ex["image"].convert("RGB"), copy=True)
        mask = np.array(ex["mask"], copy=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        gt_bin = (mask > 0).astype(np.uint8)
        gt_lbl = ndimage.label(gt_bin)[0].astype(np.int32)
        pred_bin, pred_lbl = _predict(model, img, mean, std, device, args.threshold)
        pix_inter += int(np.logical_and(pred_bin, gt_bin).sum())
        pix_union += int(np.logical_or(pred_bin, gt_bin).sum())
        c_tp, c_fp, c_fn = _count(pred_lbl.astype(np.int32), gt_lbl)
        tp += c_tp
        fp += c_fp
        fn += c_fn

        tile_transform = _tile_transform(ex, *gt_bin.shape)
        pred_gdf = vectorize(
            pred_lbl,
            tile_transform,
            _TILE_CRS,
            min_area_m2=1.0,
            simplify_m=cfg.regularize_simplify_m,
            regularize_area_threshold=cfg.regularize_area_threshold,
            regularize_overlap_tol_m2=cfg.regularize_overlap_tol_m2,
        )
        gt_gdf = vectorize(
            gt_lbl.astype(np.uint32),
            tile_transform,
            _TILE_CRS,
            min_area_m2=1.0,
            simplify_m=cfg.regularize_simplify_m,
            regularize_area_threshold=cfg.regularize_area_threshold,
            regularize_overlap_tol_m2=cfg.regularize_overlap_tol_m2,
        )
        if len(pred_gdf):
            geoms = list(pred_gdf.to_crs(pred_gdf.estimate_utm_crs()).geometry)
            pred_vertex_sum += polygon_vertex_count(geoms) * len(geoms)
            pred_orth_sum += polygon_orthogonality(geoms) * len(geoms)
            pred_poly_count += len(geoms)
        if len(gt_gdf):
            geoms = list(gt_gdf.to_crs(gt_gdf.estimate_utm_crs()).geometry)
            gt_vertex_sum += polygon_vertex_count(geoms) * len(geoms)
            gt_orth_sum += polygon_orthogonality(geoms) * len(geoms)
            gt_poly_count += len(geoms)

        if (i + 1) % 500 == 0:
            print(f"... {i + 1}/{n_total} tiles", flush=True)

    pix_iou = pix_inter / pix_union if pix_union > 0 else 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    result = {
        "n_tiles": n_total,
        "pixel_iou": pix_iou,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_avg_vertices": pred_vertex_sum / pred_poly_count if pred_poly_count else 0.0,
        "pred_orthogonality": pred_orth_sum / pred_poly_count if pred_poly_count else 0.0,
        "pred_polygons": pred_poly_count,
        "gt_avg_vertices": gt_vertex_sum / gt_poly_count if gt_poly_count else 0.0,
        "gt_orthogonality": gt_orth_sum / gt_poly_count if gt_poly_count else 0.0,
        "gt_polygons": gt_poly_count,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
