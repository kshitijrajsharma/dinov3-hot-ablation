"""Unit tests for dinov3_hot.dataset."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dinov3_hot.dataset import (
    OAM_TILE_RE,
    compute_dataset_stats,
    read_loss_history,
    spatial_split,
)


def _write_chip(path: Path, pixel_value: int) -> None:
    arr = np.full((3, 4, 4), pixel_value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(arr)


def test_oam_tile_re_matches_only_typical_names() -> None:
    assert OAM_TILE_RE.match("OAM-12345-67890-19.tif")
    assert OAM_TILE_RE.match("OAM-1-2-19.tiff")
    assert OAM_TILE_RE.match("foobar.tif") is None
    assert OAM_TILE_RE.match("OAM-abc-def-19.tif") is None


def test_spatial_split_keeps_blocks_together() -> None:
    names = [f"OAM-{x}-{y}-19.tif" for x in range(4) for y in range(4)]
    train, val = spatial_split(names, val_ratio=0.25, seed=0, block_size=2)
    assert sorted(train + val) == sorted(names)
    assert not (set(train) & set(val))
    assert train and val


def test_spatial_split_falls_back_for_non_oam_names() -> None:
    names = ["foo.tif", "bar.tif", "baz.tif", "qux.tif"]
    train, val = spatial_split(names, val_ratio=0.5, seed=42, block_size=2)
    assert sorted(train + val) == sorted(names)
    assert len(val) == 2


def test_dataset_stats_constant_image(tmp_path: Path) -> None:
    _write_chip(tmp_path / "a.tif", 128)
    _write_chip(tmp_path / "b.tif", 128)
    mean, std = compute_dataset_stats(tmp_path)
    expected_mean = 128.0 / 255.0
    assert all(abs(m - expected_mean) < 1e-12 for m in mean)
    assert all(s < 1e-12 for s in std)


def test_dataset_stats_two_intensities(tmp_path: Path) -> None:
    _write_chip(tmp_path / "low.tif", 0)
    _write_chip(tmp_path / "high.tif", 255)
    mean, std = compute_dataset_stats(tmp_path)
    assert all(abs(m - 0.5) < 1e-12 for m in mean)
    assert all(abs(s - 0.5) < 1e-12 for s in std)


def test_dataset_stats_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_dataset_stats(tmp_path)


def test_read_loss_history_returns_empty_when_missing(tmp_path: Path) -> None:
    train, val = read_loss_history(tmp_path)
    assert train == [] and val == []


def test_read_loss_history_parses_csv(tmp_path: Path) -> None:
    version_dir = tmp_path / "lightning" / "version_0"
    version_dir.mkdir(parents=True)
    (version_dir / "metrics.csv").write_text(
        "epoch,train/loss_epoch,val/loss\n0,1.0,0.9\n1,0.5,0.6\n2,0.3,0.45\n",
    )
    train, val = read_loss_history(tmp_path)
    assert train == [1.0, 0.5, 0.3]
    assert val == [0.9, 0.6, 0.45]
