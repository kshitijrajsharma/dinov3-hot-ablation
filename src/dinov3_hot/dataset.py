"""Dataset prep helpers: per-channel stats, spatial split, training-output reader."""

import csv
import re
from pathlib import Path

import numpy as np
import rasterio

# Matches OAM tile names like OAM-12345-67890-19.tif / .tiff.
OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-\d+\.tiff?$")


def compute_dataset_stats(chips_dir: Path) -> tuple[list[float], list[float]]:
    """Per-channel mean and std in [0, 1] over every .tif chip in `chips_dir`."""
    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    px = 0
    for chip in sorted(chips_dir.glob("*.tif")):
        with rasterio.open(chip) as src:
            arr = src.read([1, 2, 3]).astype(np.float64) / 255.0
        sums += arr.sum(axis=(1, 2))
        sums_sq += (arr * arr).sum(axis=(1, 2))
        px += arr.shape[1] * arr.shape[2]
    if px == 0:
        raise FileNotFoundError(f"No .tif chips found in {chips_dir}")
    mean = sums / px
    return mean.tolist(), np.sqrt(sums_sq / px - mean * mean).tolist()


def spatial_split(
    chip_names: list[str],
    val_ratio: float,
    seed: int,
    *,
    block_size: int = 4,
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; entire `(x//K, y//K)` blocks pick a side."""
    matched: dict[tuple[int, int], list[str]] = {}
    unmatched: list[str] = []
    for name in chip_names:
        oam_match = OAM_TILE_RE.match(name)
        if oam_match is None:
            unmatched.append(name)
            continue
        tile_x, tile_y = int(oam_match.group(1)), int(oam_match.group(2))
        matched.setdefault((tile_x // block_size, tile_y // block_size), []).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(matched.keys())
    rng.shuffle(blocks)

    n_total = sum(len(v) for v in matched.values()) + len(unmatched)
    n_val_target = max(1, int(n_total * val_ratio))

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        bucket = val if len(val) < n_val_target else train
        bucket.extend(matched[block])

    if unmatched:
        leftover = sorted(unmatched)
        rng.shuffle(leftover)
        remaining = max(0, n_val_target - len(val))
        val.extend(leftover[:remaining])
        train.extend(leftover[remaining:])

    return sorted(train), sorted(val)


def read_loss_history(out_dir: Path) -> tuple[list[float], list[float]]:
    """Train and val loss per epoch from Lightning CSVLogger; empty lists when not present yet."""
    versions = sorted((out_dir / "lightning").glob("version_*"))
    if not versions:
        return [], []
    metrics_csv = versions[-1] / "metrics.csv"
    if not metrics_csv.exists():
        return [], []
    train_by_epoch: dict[int, float] = {}
    val_by_epoch: dict[int, float] = {}
    with metrics_csv.open() as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            epoch = int(row["epoch"])
            train_loss = row.get("train/loss_epoch") or ""
            val_loss = row.get("val/loss") or ""
            if train_loss:
                train_by_epoch[epoch] = float(train_loss)
            if val_loss:
                val_by_epoch[epoch] = float(val_loss)
    return (
        [train_by_epoch[e] for e in sorted(train_by_epoch)],
        [val_by_epoch[e] for e in sorted(val_by_epoch)],
    )
