import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from dinov3_hot.config import resolve_root
from dinov3_hot.data import load_norm_stats
from dinov3_hot.infer import load_model, sliding_window_predict

log = logging.getLogger(__name__)


def eval_fair_samples(
    cfg,
    ckpt_path: str | Path,
    samples_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*.tif",
    max_tiles: int = 12,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiles = sorted(Path(samples_dir).rglob(pattern))[:max_tiles]
    if not tiles:
        raise FileNotFoundError(f"No tiles matching {pattern} under {samples_dir}")

    mean, std = load_norm_stats(cfg.dataset_repo, resolve_root(cfg))
    model = load_model(ckpt_path, cfg, device=device)
    rows = []

    for tile_path in tiles:
        with rasterio.open(tile_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        mask_prob, boundary_prob, distance, _, _ = sliding_window_predict(
            model,
            tile_path,
            window=cfg.tile_window,
            stride=cfg.tile_stride,
            mean=mean,
            std=std,
            device=device,
        )
        rows.append((tile_path.name, img, mask_prob, boundary_prob, distance))

    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (name, img, mp, bp, dist) in enumerate(rows):
        binary = (mp > cfg.tile_threshold).astype(np.uint8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(name)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(mp, vmin=0, vmax=1, cmap="viridis")
        axes[i, 1].set_title("mask prob")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(bp, vmin=0, vmax=1, cmap="magma")
        axes[i, 2].set_title("boundary prob")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(dist, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[i, 3].set_title("signed distance")
        axes[i, 3].axis("off")
        axes[i, 4].imshow(img)
        axes[i, 4].imshow(binary, alpha=0.45, cmap="Reds")
        axes[i, 4].set_title("prediction overlay")
        axes[i, 4].axis("off")
    fig.tight_layout()
    viz_path = out / "fair_eval_grid.png"
    fig.savefig(viz_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved viz: %s (%d tiles)", viz_path, n)
    return viz_path
