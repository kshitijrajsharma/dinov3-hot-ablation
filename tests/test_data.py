import numpy as np
import pytest
import torch

from dinov3_hot.data import _boundary


def test_boundary_zero_for_empty_mask():
    out = _boundary(np.zeros((32, 32), dtype=np.uint8), width=2)
    assert out.sum() == 0
    assert out.dtype == np.uint8


def test_boundary_traces_perimeter():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    out = _boundary(mask, width=1)
    assert out.sum() > 0
    assert (out & ~mask).sum() == 0
    interior = mask.copy()
    interior[9:23, 9:23] = 0
    assert (out == interior).all()


def test_boundary_width_grows_band():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    narrow = _boundary(mask, width=1).sum()
    wide = _boundary(mask, width=3).sum()
    assert wide > narrow


@pytest.mark.slow
def test_default_keeps_mask_null_tiles_in_train():
    from dinov3_hot.config import load_config
    from dinov3_hot.data import HotBuildingDataModule

    cfg = load_config("conf/train.yaml")
    assert cfg.drop_null_images is True
    dm = HotBuildingDataModule(
        repo_id=cfg.dataset_repo,
        root=cfg.data_root,
        img_size=cfg.img_size,
        boundary_width=cfg.boundary_width,
        distance_clip=cfg.distance_clip,
        drop_null_images=True,
        data_pct=100.0,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=cfg.seed,
    )
    dm.prepare_data()
    dm.setup("fit")
    assert len(dm.train_ds) == 57890, "HOT has no fully-null RGB tiles; filter is a defensive no-op"
    assert len(dm.val_ds) == 7237


@pytest.mark.slow
def test_datamodule_yields_correct_shapes():
    from dinov3_hot.config import load_config
    from dinov3_hot.data import HotBuildingDataModule

    cfg = load_config("conf/train.yaml", overrides=["data_pct=1", "batch_size=2"])
    dm = HotBuildingDataModule(
        repo_id=cfg.dataset_repo,
        root=cfg.data_root,
        img_size=cfg.img_size,
        boundary_width=cfg.boundary_width,
        drop_null_images=True,
        data_pct=cfg.data_pct,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=cfg.seed,
    )
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (2, 3, cfg.img_size, cfg.img_size)
    assert batch["image"].dtype == torch.float32
    assert batch["mask"].shape == (2, cfg.img_size, cfg.img_size)
    assert batch["mask"].dtype == torch.long
    assert set(batch["mask"].unique().tolist()).issubset({0, 1})
    assert batch["boundary"].shape == (2, cfg.img_size, cfg.img_size)
