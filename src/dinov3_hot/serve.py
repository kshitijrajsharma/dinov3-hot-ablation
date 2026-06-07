"""ONNX inference path for the distroless model image."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely.geometry as sgeom
from rasterio.merge import merge
from rasterio.transform import rowcol
from scipy.signal.windows import gaussian as gauss1d

# Per-channel normalisation for hotosm/vhr-building-segmentation. Mirrors
# `dinov3_hot.data.load_norm_stats` so distroless callers can avoid the torch import.
HOT_MEAN: list[float] = [0.4296737853453577, 0.4001659668453235, 0.34333372802741474]
HOT_STD: list[float] = [0.2056069389373208, 0.16738555558380538, 0.1598986422586595]

MODEL_INPUT_SIZE = 256
SLIDING_STRIDE = 128
INFERENCE_BATCH_SIZE = 8
SEED_MIN_DISTANCE = 4
LARGE_BLOB_AREA_PX = 1500
H_MAXIMA_DEPTH = 0.2

DEFAULT_INFERENCE_PARAMS: dict[str, Any] = {
    "confidence_threshold": 0.5,
    "seed_min_distance": SEED_MIN_DISTANCE,
    "large_blob_area_px": LARGE_BLOB_AREA_PX,
    "h_maxima_depth": H_MAXIMA_DEPTH,
    "simplify_m": 1.0,
    "regularize_area_threshold": 0.55,
    "regularize_overlap_tol_m2": 1.0,
    "min_area_m2": 1.0,
}


def gaussian_kernel(size: int, sigma_frac: float = 0.125) -> np.ndarray:
    """Separable 2D Gaussian, peak-normalised to 1."""
    weights_1d = gauss1d(size, std=sigma_frac * size)
    kernel = np.outer(weights_1d, weights_1d)
    return kernel / kernel.max()


def normalize_chw(image_hwc_uint8: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """HWC uint8 to CHW float32 in [0, 1], per-channel normalised."""
    chw = image_hwc_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return (chw - mean_arr) / std_arr


def sliding_window_onnx(
    session: Any,
    image_hwc: np.ndarray,
    *,
    mean: list[float] = HOT_MEAN,
    std: list[float] = HOT_STD,
    window: int = MODEL_INPUT_SIZE,
    stride: int = SLIDING_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-stitched sliding window; returns (mask_prob, boundary_prob, distance)."""
    height, width, _ = image_hwc.shape
    mask_acc = np.zeros((height, width), dtype=np.float32)
    boundary_acc = np.zeros((height, width), dtype=np.float32)
    distance_acc = np.zeros((height, width), dtype=np.float32)
    weight_acc = np.zeros((height, width), dtype=np.float32)
    kernel = gaussian_kernel(window)

    rows = list(range(0, max(1, height - window + 1), stride))
    cols = list(range(0, max(1, width - window + 1), stride))
    if rows[-1] + window < height:
        rows.append(height - window)
    if cols[-1] + window < width:
        cols.append(width - window)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    # Exported ONNX has a static batch dim; group windows to match it.
    batch = input_meta.shape[0]

    positions = [(row, col) for row in rows for col in cols]
    chips = np.empty((len(positions), 3, window, window), dtype=np.float32)
    for index, (row, col) in enumerate(positions):
        tile = image_hwc[row : row + window, col : col + window, :]
        if tile.shape[0] != window or tile.shape[1] != window:
            pad = np.zeros((window, window, 3), dtype=image_hwc.dtype)
            pad[: tile.shape[0], : tile.shape[1]] = tile
            tile = pad
        chips[index] = normalize_chw(tile, mean, std)

    for start in range(0, len(positions), batch):
        group = chips[start : start + batch]
        filled = group.shape[0]
        if filled < batch:
            padded = np.zeros((batch, 3, window, window), dtype=np.float32)
            padded[:filled] = group
            group = padded
        logits = session.run(None, {input_name: group})[0]
        for offset in range(filled):
            row, col = positions[start + offset]
            mask_acc[row : row + window, col : col + window] += (
                1.0 / (1.0 + np.exp(-logits[offset, 0]))
            ) * kernel
            boundary_acc[row : row + window, col : col + window] += (
                1.0 / (1.0 + np.exp(-logits[offset, 1]))
            ) * kernel
            distance_acc[row : row + window, col : col + window] += np.tanh(logits[offset, 2]) * kernel
            weight_acc[row : row + window, col : col + window] += kernel

    # Every pixel in the image is covered by at least one window;
    assert weight_acc.min() > 0.0, "sliding window left uncovered pixels"
    return mask_acc / weight_acc, boundary_acc / weight_acc, distance_acc / weight_acc


def merge_chips_to_array(input_dir: Path) -> tuple[np.ndarray, Any, Any]:
    """Merge *.tif/*.tiff/*.png chips into one HWC uint8 array via rasterio.merge."""
    patterns = ("*.tif", "*.tiff", "*.png")
    paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No input images found in {input_dir}")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in paths]
        mosaic, transform = merge(sources, indexes=[1, 2, 3])
        crs = sources[0].crs
    return mosaic.transpose(1, 2, 0).astype(np.uint8), transform, crs


def add_scores(
    gdf: Any,
    labels: np.ndarray,
    mask_prob: np.ndarray,
    transform: Any,
    crs: Any,
) -> Any:
    """Attach per-polygon score = mean mask_prob over its watershed instance."""
    if not len(gdf):
        return gdf
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    labels_flat = labels.ravel().astype(np.int64)
    sums = np.bincount(labels_flat, weights=mask_prob.ravel())
    counts = np.bincount(labels_flat)
    label_means = sums / np.maximum(counts, 1)
    height, width = labels.shape
    scores = []
    for geom in gdf.geometry:
        rep_point = geom.representative_point()
        row, col = rowcol(transform, rep_point.x, rep_point.y)
        if 0 <= row < height and 0 <= col < width:
            label_value = int(labels[row, col])
            scores.append(float(label_means[label_value]) if 0 <= label_value < len(label_means) else 0.0)
        else:
            scores.append(0.0)
    gdf = gdf.copy()
    gdf["score"] = scores
    return gdf


def predict_session(
    session: Any,
    input_dir: Path,
    params: dict[str, Any],
    *,
    mean: list[float] = HOT_MEAN,
    std: list[float] = HOT_STD,
) -> dict[str, Any]:
    """Full ONNX predict: merge chips -> sliding window -> watershed -> vectorise -> 4326 GeoJSON."""
    # Lazy import: dinov3_hot.infer
    from dinov3_hot.infer import instance_separate, vectorize

    if "confidence_threshold" not in params:
        raise ValueError("params['confidence_threshold'] is required")
    threshold = float(params["confidence_threshold"])
    seed_min_distance = int(params.get("seed_min_distance", SEED_MIN_DISTANCE))
    large_blob_area_px = int(params.get("large_blob_area_px", LARGE_BLOB_AREA_PX))
    h_maxima_depth = float(params.get("h_maxima_depth", H_MAXIMA_DEPTH))
    stride = int(params.get("sliding_stride", SLIDING_STRIDE))

    image_hwc, transform, crs = merge_chips_to_array(input_dir)
    mask_prob, _boundary, distance = sliding_window_onnx(
        session, image_hwc, mean=mean, std=std, stride=stride
    )
    labels = instance_separate(
        mask_prob,
        distance,
        mask_threshold=threshold,
        seed_min_distance=seed_min_distance,
        large_blob_area_px=large_blob_area_px,
        h_maxima_depth=h_maxima_depth,
    )
    gdf = vectorize(
        labels,
        transform,
        crs,
        min_area_m2=float(params.get("min_area_m2", 1.0)),
        simplify_m=float(params.get("simplify_m", 1.0)),
        regularize_area_threshold=float(params.get("regularize_area_threshold", 0.55)),
        regularize_overlap_tol_m2=float(params.get("regularize_overlap_tol_m2", 1.0)),
    )
    if not len(gdf):
        return {"type": "FeatureCollection", "features": []}
    gdf = add_scores(gdf, labels, mask_prob, transform, crs)
    out_geoms = gdf.to_crs(epsg=4326).geometry
    features = [
        {
            "type": "Feature",
            "properties": {"class": 1, "score": float(score)},
            "geometry": sgeom.mapping(geom),
        }
        for score, geom in zip(gdf["score"], out_geoms, strict=True)
        if not geom.is_empty
    ]
    return {"type": "FeatureCollection", "features": features}
