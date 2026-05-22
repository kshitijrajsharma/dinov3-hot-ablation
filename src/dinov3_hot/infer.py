import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from huggingface_hub import hf_hub_download
from scipy import ndimage
from scipy.signal.windows import gaussian as gauss1d
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from dinov3_hot.config import resolve_root
from dinov3_hot.data import load_norm_stats
from dinov3_hot.model import DinoV3HotLit

log = logging.getLogger(__name__)


def load_model(ckpt_path: str | Path, cfg, device: str = "cuda") -> DinoV3HotLit:
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    model = DinoV3HotLit.load_from_checkpoint(str(ckpt_path), map_location=device, ckpt_path=encoder_ckpt)
    model.eval()
    return model


def _gaussian_kernel(size: int, sigma_frac: float = 0.125) -> np.ndarray:
    w = gauss1d(size, std=sigma_frac * size)
    k = np.outer(w, w)
    return k / k.max()


def _normalize(img_uint8: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    t = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (t - m) / s


def sliding_window_predict(
    model: DinoV3HotLit,
    raster_path: str | Path,
    window: int,
    stride: int,
    mean: list[float],
    std: list[float],
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """Returns (mask_prob, boundary_prob, distance, transform, crs); distance is tanh-normalized
    to [-1, 1], positive inside, negative outside."""
    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs
        img = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)

    mask_acc = np.zeros((h, w), dtype=np.float32)
    boundary_acc = np.zeros((h, w), dtype=np.float32)
    distance_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)
    kernel = _gaussian_kernel(window)

    rows = list(range(0, max(1, h - window + 1), stride))
    cols = list(range(0, max(1, w - window + 1), stride))
    if rows[-1] + window < h:
        rows.append(h - window)
    if cols[-1] + window < w:
        cols.append(w - window)

    with torch.inference_mode():
        for r in rows:
            for c in cols:
                tile = img[r : r + window, c : c + window, :]
                if tile.shape[0] != window or tile.shape[1] != window:
                    pad = np.zeros((window, window, 3), dtype=np.uint8)
                    pad[: tile.shape[0], : tile.shape[1]] = tile
                    tile = pad
                x = _normalize(tile, mean, std).unsqueeze(0).to(device)
                main_logits, _ = model(x)
                logits = main_logits[0]
                mask_prob = torch.sigmoid(logits[0]).cpu().numpy()
                boundary_prob = torch.sigmoid(logits[1]).cpu().numpy()
                distance = torch.tanh(logits[2]).cpu().numpy()
                mask_acc[r : r + window, c : c + window] += mask_prob * kernel
                boundary_acc[r : r + window, c : c + window] += boundary_prob * kernel
                distance_acc[r : r + window, c : c + window] += distance * kernel
                weight_acc[r : r + window, c : c + window] += kernel

    weight_acc = np.maximum(weight_acc, 1e-6)
    return (
        mask_acc / weight_acc,
        boundary_acc / weight_acc,
        distance_acc / weight_acc,
        transform,
        crs,
    )


def instance_separate(
    mask_prob: np.ndarray,
    distance: np.ndarray,
    mask_threshold: float = 0.5,
    seed_min_distance: int = 4,
) -> np.ndarray:
    """Watershed instance labels: seeds are local maxima of the predicted distance map,
    so each building center yields one instance without a hand-tuned threshold."""
    fg = mask_prob > mask_threshold
    if not fg.any():
        return np.zeros_like(fg, dtype=np.uint32)
    coords = peak_local_max(
        distance,
        min_distance=seed_min_distance,
        labels=fg.astype(np.uint8),
        exclude_border=False,
    )
    if len(coords) == 0:
        return ndimage.label(fg)[0].astype(np.uint32)
    seeds = np.zeros_like(fg, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndimage.label(seeds)
    return watershed(-distance, markers=markers, mask=fg).astype(np.uint32)


def vectorize(
    labels: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    min_area_m2: float = 1.0,
    simplify_m: float = 1.0,
    regularize_area_threshold: float = 0.55,
    regularize_overlap_tol_m2: float = 1.0,
) -> gpd.GeoDataFrame:
    """Thin wrapper around `dinov3_hot.postprocess.vectorize_binary_mask` kept for
    backwards compatibility; the implementation now lives in a torch-free module."""
    from dinov3_hot.postprocess import vectorize_binary_mask

    return vectorize_binary_mask(
        labels,
        transform,
        crs,
        min_area_m2=min_area_m2,
        simplify_m=simplify_m,
        regularize_area_threshold=regularize_area_threshold,
        regularize_overlap_tol_m2=regularize_overlap_tol_m2,
    )


def predict_geotiff(
    cfg,
    ckpt_path: str | Path,
    raster_path: str | Path,
    out_geojson: str | Path,
    device: str | None = None,
    min_area_m2: float = 1.0,
    seed_min_distance: int = 4,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm_stats(cfg.dataset_repo, resolve_root(cfg))
    model = load_model(ckpt_path, cfg, device=device)

    mask_prob, _, distance, transform, crs = sliding_window_predict(
        model,
        raster_path,
        window=cfg.tile_window,
        stride=cfg.tile_stride,
        mean=mean,
        std=std,
        device=device,
    )
    labels = instance_separate(
        mask_prob,
        distance,
        mask_threshold=cfg.tile_threshold,
        seed_min_distance=seed_min_distance,
    )
    gdf = vectorize(
        labels,
        transform,
        crs,
        min_area_m2=min_area_m2,
        simplify_m=cfg.regularize_simplify_m,
        regularize_area_threshold=cfg.regularize_area_threshold,
        regularize_overlap_tol_m2=cfg.regularize_overlap_tol_m2,
    )

    out_geojson = Path(out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    log.info("Wrote %d polygons to %s", len(gdf), out_geojson)
    return out_geojson
