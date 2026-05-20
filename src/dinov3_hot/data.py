import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningDataModule
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask

log = logging.getLogger(__name__)

_STATS_FILENAME = "norm_stats.json"


def load_norm_stats(repo_id: str, root: str | Path) -> tuple[list[float], list[float]]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stats_path = root / _STATS_FILENAME
    if not stats_path.exists():
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=_STATS_FILENAME, local_dir=str(root))
    stats = json.loads(stats_path.read_text())
    return stats["mean"], stats["std"]


def _boundary(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    eroded = ndimage.binary_erosion(mask.astype(bool), iterations=max(1, width))
    return (mask.astype(bool) & ~eroded).astype(np.uint8)


def _signed_distance(mask: np.ndarray, clip: float) -> np.ndarray:
    """Signed distance to boundary, in pixels, clipped to [-clip, +clip], normalized to [-1, 1]."""
    m = mask.astype(bool)
    if m.sum() == 0 or (~m).sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    inner = ndimage.distance_transform_edt(m)
    outer = ndimage.distance_transform_edt(~m)
    signed = (inner - outer).astype(np.float32)
    return np.clip(signed, -clip, clip) / clip


def _build_transforms(img_size: int, mean: list[float], std: list[float], train: bool) -> v2.Compose:
    spatial = (
        [v2.RandomCrop(img_size, pad_if_needed=True), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
        if train
        else [v2.CenterCrop(img_size)]
    )
    return v2.Compose(
        [
            *spatial,
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


class HotBuildingDataModule(LightningDataModule):
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        img_size: int,
        boundary_width: int,
        distance_clip: float,
        drop_null_images: bool,
        data_pct: float,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root)
        self.img_size = img_size
        self.boundary_width = boundary_width
        self.distance_clip = distance_clip
        self.drop_null_images = drop_null_images
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def prepare_data(self) -> None:
        load_norm_stats(self.repo_id, self.root)

    def _load_split(self, split: str) -> Dataset:
        hf_split = "validation" if split == "val" else split
        ds = load_dataset(self.repo_id, split=hf_split)
        if split == "train" and self.drop_null_images:
            n_before = len(ds)
            ds = ds.filter(lambda ex: int(np.asarray(ex["image"].convert("RGB")).sum()) > 0)
            if len(ds) < n_before:
                log.info("Dropped %d fully-null RGB tiles from train", n_before - len(ds))
        if split == "train" and self.data_pct < 100.0:
            n = max(1, round(len(ds) * self.data_pct / 100.0))
            ds = ds.shuffle(seed=self.seed).select(range(n))
        return ds

    def setup(self, stage: str | None = None) -> None:
        mean, std = load_norm_stats(self.repo_id, self.root)
        self._train_tf = _build_transforms(self.img_size, mean, std, train=True)
        self._eval_tf = _build_transforms(self.img_size, mean, std, train=False)

        if stage in (None, "fit"):
            self.train_ds = self._load_split("train")
        if stage in (None, "fit", "validate"):
            self.val_ds = self._load_split("val")
        if stage in (None, "test"):
            self.test_ds = self._load_split("test")

    def _collate(self, examples: list[dict], train: bool) -> dict[str, torch.Tensor]:
        tf = self._train_tf if train else self._eval_tf
        images, masks, boundaries, distances = [], [], [], []
        for ex in examples:
            img_arr = np.array(ex["image"].convert("RGB"), copy=True)
            img_chw = torch.from_numpy(img_arr).permute(2, 0, 1)
            mask_arr = np.array(ex["mask"], copy=True)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            mask_chw = torch.from_numpy((mask_arr > 0).astype(np.uint8)).unsqueeze(0)
            img_t, mask_t = tf(Image(img_chw), Mask(mask_chw))
            mask_t = mask_t.squeeze(0).long()
            mask_np = mask_t.numpy().astype(np.uint8)
            boundary_t = torch.from_numpy(_boundary(mask_np, self.boundary_width).astype(np.float32))
            distance_t = torch.from_numpy(_signed_distance(mask_np, self.distance_clip))
            images.append(img_t)
            masks.append(mask_t)
            boundaries.append(boundary_t)
            distances.append(distance_t)
        return {
            "image": torch.stack(images),
            "mask": torch.stack(masks),
            "boundary": torch.stack(boundaries),
            "distance": torch.stack(distances),
        }

    def _loader(self, ds: Dataset, batch_size: int, shuffle: bool, train: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=shuffle,
            collate_fn=lambda b: self._collate(b, train=train),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, self.batch_size, shuffle=True, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, self.eval_batch_size, shuffle=False, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, self.eval_batch_size, shuffle=False, train=False)
