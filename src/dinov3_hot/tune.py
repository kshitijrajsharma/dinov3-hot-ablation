"""Post-process Optuna HPO, model-agnostic."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import polymetrics
import rasterio
import shapely.geometry as sgeom
import torch
from shapely.ops import unary_union

from dinov3_hot.serve import DEFAULT_INFERENCE_PARAMS

TUNE_MIN_VAL_CHIPS = 8


def cache_val_forwards(
    net: Any,
    chips_dir: Path,
    val_chip_names: list[str],
    *,
    mean: list[float],
    std: list[float],
    device: str,
) -> list[dict[str, Any]]:
    """One forward per val chip; the cache lets each Optuna trial skip the model entirely."""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    model = net.to(device).eval()
    cache: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in val_chip_names:
            with rasterio.open(chips_dir / name) as src:
                img = src.read([1, 2, 3]).astype(np.float32) / 255.0
                transform = src.transform
                crs = src.crs
            tensor_in = torch.from_numpy((img - mean_arr) / std_arr).unsqueeze(0).to(device)
            main_logits, _ = model(tensor_in)
            logits = main_logits[0].cpu().numpy()
            cache.append(
                {
                    "name": name,
                    "mask_prob": 1.0 / (1.0 + np.exp(-logits[0])),
                    "boundary_prob": 1.0 / (1.0 + np.exp(-logits[1])),
                    "distance": np.tanh(logits[2]),
                    "transform": transform,
                    "crs": crs,
                }
            )
    return cache


def gt_clipped_to_val(labels_geojson: Path, cache: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Ground truth restricted to the union of val-chip extents in val CRS."""
    if not cache:
        raise ValueError("cache is empty; call cache_val_forwards with non-empty val_chip_names first")
    boxes = [
        sgeom.box(
            *rasterio.transform.array_bounds(
                entry["mask_prob"].shape[0],
                entry["mask_prob"].shape[1],
                entry["transform"],
            )
        )
        for entry in cache
    ]
    val_union = unary_union(boxes)
    gt = gpd.read_file(labels_geojson)
    target_crs = cache[0]["crs"]
    if gt.crs != target_crs:
        gt = gt.to_crs(target_crs)
    return gt[gt.intersects(val_union)].copy()


def trial_predictions(cache: list[dict[str, Any]], params: dict[str, Any]) -> gpd.GeoDataFrame:
    """Run instance_separate + vectorize per cached chip; concatenate to one gdf."""
    # Lazy: infer pulls torch but trial loops also need numpy outputs; cache already paid the cost.
    from dinov3_hot.infer import instance_separate, vectorize

    per_chip = []
    for entry in cache:
        labels = instance_separate(
            entry["mask_prob"],
            entry["distance"],
            mask_threshold=params["confidence_threshold"],
            seed_min_distance=params["seed_min_distance"],
        )
        gdf = vectorize(
            labels,
            entry["transform"],
            entry["crs"],
            min_area_m2=params["min_area_m2"],
            simplify_m=params["simplify_m"],
            regularize_area_threshold=params["regularize_area_threshold"],
            regularize_overlap_tol_m2=params["regularize_overlap_tol_m2"],
        )
        if len(gdf):
            per_chip.append(gdf)
    if not per_chip:
        return gpd.GeoDataFrame(geometry=[], crs=cache[0]["crs"])
    return gpd.GeoDataFrame(pd.concat(per_chip, ignore_index=True), crs=per_chip[0].crs)


def tune_postprocess_run(
    net: Any,
    chips_dir: Path,
    labels_geojson: Path,
    val_chip_names: list[str],
    *,
    mean: list[float],
    std: list[float],
    n_trials: int = 30,
    seed: int = 42,
    device: str | None = None,
    default_params: dict[str, Any] | None = None,
    min_val_chips: int = TUNE_MIN_VAL_CHIPS,
) -> dict[str, Any]:
    """Optuna TPE over six post-process knobs; objective is `f1 + 0.3 * mean_iou`."""
    defaults = dict(DEFAULT_INFERENCE_PARAMS if default_params is None else default_params)
    if n_trials <= 0 or len(val_chip_names) < min_val_chips:
        return {
            "best_value": None,
            "best_params": defaults,
            "n_trials": 0,
            "skipped": "disabled" if n_trials <= 0 else "val_too_small",
        }

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = cache_val_forwards(net, chips_dir, val_chip_names, mean=mean, std=std, device=device)
    gt = gt_clipped_to_val(labels_geojson, cache)
    if not len(gt):
        raise ValueError(f"No GT polygons intersect the val-chip union from {labels_geojson}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "confidence_threshold": trial.suggest_float("confidence_threshold", 0.3, 0.7),
            "seed_min_distance": trial.suggest_int("seed_min_distance", 2, 16),
            "simplify_m": trial.suggest_float("simplify_m", 0.5, 3.0),
            "regularize_area_threshold": trial.suggest_float("regularize_area_threshold", 0.4, 0.8),
            "regularize_overlap_tol_m2": trial.suggest_float("regularize_overlap_tol_m2", 0.0, 5.0),
            "min_area_m2": trial.suggest_float("min_area_m2", 0.0, 5.0),
        }
        pred = trial_predictions(cache, params)
        if not len(pred):
            # No predictions at these params (e.g. threshold too high) is a valid trial result;
            # surface it as zero score so TPE moves away.
            return 0.0
        result = polymetrics.evaluate(gt, pred, iou_threshold=0.5, compute_map=False)
        return result.f1 + 0.3 * result.mean_iou

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
    }
