"""Pixel/instance/shape evaluation against burned-mask ground truth."""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from scipy import ndimage

from dinov3_hot.metrics import instance_prf, polygon_orthogonality, polygon_vertex_count
from dinov3_hot.postprocess import vectorize_binary_mask


def evaluate(
    net: Any,
    chips_dir: Path,
    masks_dir: Path,
    val_chip_names: list[str],
    *,
    mean: list[float],
    std: list[float],
    device: str | None = None,
) -> dict[str, Any]:
    """Pixel IoU + instance P/R/F1@0.5 + polygon shape stats across val chips."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    model = net.to(device).eval()

    pix_inter = pix_union = 0
    tp_total = fp_total = fn_total = 0
    vertex_sum = orth_sum = 0.0
    poly_count = 0

    with torch.no_grad():
        for name in val_chip_names:
            with rasterio.open(chips_dir / name) as src:
                img = src.read([1, 2, 3]).astype(np.float32) / 255.0
                transform = src.transform
                crs = src.crs
            with rasterio.open(masks_dir / name) as src:
                gt = (src.read(1) > 0).astype(np.uint8)
            tensor_in = torch.from_numpy((img - mean_arr) / std_arr).unsqueeze(0).to(device)
            main_logits, _ = model(tensor_in)
            pred = (torch.sigmoid(main_logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)

            pix_inter += int((pred & gt).sum())
            pix_union += int((pred | gt).sum())

            pred_lbl = ndimage.label(pred)[0].astype(np.int32)
            gt_lbl = ndimage.label(gt)[0].astype(np.int32)
            prf = instance_prf(pred_lbl, gt_lbl)
            tp_total += prf["tp"]
            fp_total += prf["fp"]
            fn_total += prf["fn"]

            gdf = vectorize_binary_mask(pred, transform, crs)
            if len(gdf):
                metric_geoms = list(gdf.to_crs(gdf.estimate_utm_crs()).geometry)
                n = len(metric_geoms)
                vertex_sum += polygon_vertex_count(metric_geoms) * n
                orth_sum += polygon_orthogonality(metric_geoms) * n
                poly_count += n

    pixel_iou = pix_inter / pix_union if pix_union else 0.0
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = (
        (2 * tp_total) / (2 * tp_total + fp_total + fn_total) if (2 * tp_total + fp_total + fn_total) else 0.0
    )
    return {
        "pixel_iou": pixel_iou,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "pred_avg_vertices": vertex_sum / poly_count if poly_count else 0.0,
        "pred_orthogonality": orth_sum / poly_count if poly_count else 0.0,
    }
