"""Full-raster sliding-window eval against an OSM ground-truth geojson.

Runs `predict_geotiff` on the given raster, scores the polygons against GT, and
writes a one-row JSON summary. Reproduces the Banepa row in README.md."""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from dinov3_hot.config import load_config
from dinov3_hot.infer import predict_geotiff
from dinov3_hot.metrics import (
    instance_prf,
    polygon_orthogonality,
    polygon_vertex_count,
)


def _rasterise(gdf: gpd.GeoDataFrame, transform, shape) -> np.ndarray:
    if not len(gdf):
        return np.zeros(shape, dtype=np.int32)
    return rasterize(
        ((g, i + 1) for i, g in enumerate(gdf.geometry)),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint path")
    parser.add_argument("--raster", required=True, help="Georeferenced GeoTIFF for the test scene")
    parser.add_argument("--gt", required=True, help="OSM ground-truth polygons GeoJSON")
    parser.add_argument("--out-dir", default="outputs/eval_banepa", help="Where to write prediction geojson")
    parser.add_argument("--config", default="conf/train.yaml", help="OmegaConf training config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.geojson"

    predict_geotiff(cfg, ckpt_path=args.ckpt, raster_path=args.raster, out_geojson=pred_path)

    with rasterio.open(args.raster) as src:
        scene_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width

    pred = gpd.read_file(pred_path).to_crs(scene_crs)
    gt = gpd.read_file(args.gt).to_crs(scene_crs)

    pred_lbl = _rasterise(pred, transform, (height, width))
    gt_lbl = _rasterise(gt, transform, (height, width))
    pred_bin = pred_lbl > 0
    gt_bin = gt_lbl > 0
    inter = int(np.logical_and(pred_bin, gt_bin).sum())
    union = int(np.logical_or(pred_bin, gt_bin).sum())
    pixel_iou = inter / union if union else 0.0
    prf = instance_prf(pred_lbl, gt_lbl)

    shape_crs = gt.estimate_utm_crs()
    pred_shape = pred.to_crs(shape_crs)
    gt_shape = gt.to_crs(shape_crs)

    result = {
        "ckpt": args.ckpt,
        "raster": args.raster,
        "gt": args.gt,
        "scene_crs": str(scene_crs),
        "shape_crs": str(shape_crs),
        "n_pred_polygons": len(pred),
        "n_gt_polygons": len(gt),
        "pixel_iou": pixel_iou,
        "instance_precision": prf["precision"],
        "instance_recall": prf["recall"],
        "instance_f1": prf["f1"],
        "tp": prf["tp"],
        "fp": prf["fp"],
        "fn": prf["fn"],
        "pred_avg_vertices": polygon_vertex_count(list(pred_shape.geometry)),
        "pred_orthogonality": polygon_orthogonality(list(pred_shape.geometry)),
        "gt_avg_vertices": polygon_vertex_count(list(gt_shape.geometry)),
        "gt_orthogonality": polygon_orthogonality(list(gt_shape.geometry)),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
