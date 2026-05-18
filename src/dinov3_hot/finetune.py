import json
import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import rasterio
import torch
from geomltoolkits.raster.burn import burn_labels
from huggingface_hub import hf_hub_download
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask

from dinov3_hot.config import resolve_root
from dinov3_hot.data import _boundary, _signed_distance
from dinov3_hot.model import DinoV3HotLit

log = logging.getLogger(__name__)


class LocalChipDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, Path]],
        img_size: int,
        boundary_width: int,
        distance_clip: float,
        mean: list[float],
        std: list[float],
        train: bool,
    ):
        self.items = items
        self.boundary_width = boundary_width
        self.distance_clip = distance_clip
        spatial = (
            [v2.RandomCrop(img_size, pad_if_needed=True), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
            if train
            else [v2.CenterCrop(img_size)]
        )
        self.tf = v2.Compose(
            [*spatial, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chip_path, mask_path = self.items[idx]
        with rasterio.open(chip_path) as src:
            img_chw = torch.from_numpy(src.read([1, 2, 3]).astype(np.uint8))
        with rasterio.open(mask_path) as src:
            mask_arr = src.read(1)
        mask_chw = torch.from_numpy((mask_arr > 0).astype(np.uint8)).unsqueeze(0)
        img_t, mask_t = self.tf(Image(img_chw), Mask(mask_chw))
        mask_t = mask_t.squeeze(0).long()
        mask_np = mask_t.numpy().astype(np.uint8)
        boundary_t = torch.from_numpy(_boundary(mask_np, self.boundary_width).astype(np.float32))
        distance_t = torch.from_numpy(_signed_distance(mask_np, self.distance_clip))
        return {"image": img_t, "mask": mask_t, "boundary": boundary_t, "distance": distance_t}


def _pair_chips_with_masks(chips_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for chip in sorted(chips_dir.glob("*.tif")):
        mask = masks_dir / chip.name
        if mask.exists():
            pairs.append((chip, mask))
    if not pairs:
        raise FileNotFoundError(f"No matching (chip, mask) pairs under {chips_dir} / {masks_dir}")
    return pairs


def finetune(
    cfg: DictConfig,
    pretrained_ckpt: str | Path,
    chips_dir: str | Path,
    labels_geojson: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.3,
    ft_lr: float = 5e-5,
    ft_epochs: int = 15,
) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    chips_dir = Path(chips_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks_dir = out / "masks"
    burn_labels(
        labels_path=str(labels_geojson),
        chips_dir=str(chips_dir),
        output_dir=str(masks_dir),
        burn_value=255,
    )

    pairs = _pair_chips_with_masks(chips_dir, masks_dir)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_frac))
    val_pairs = [pairs[i] for i in perm[:n_val]]
    train_pairs = [pairs[i] for i in perm[n_val:]]
    log.info(
        "Banepa FT split: %d train / %d val (from %d total)", len(train_pairs), len(val_pairs), len(pairs)
    )

    stats_path = resolve_root(cfg) / "norm_stats.json"
    if not stats_path.exists():
        hf_hub_download(
            repo_id=cfg.dataset_repo,
            repo_type="dataset",
            filename="norm_stats.json",
            local_dir=str(resolve_root(cfg)),
        )
    stats = json.loads(stats_path.read_text())
    mean, std = stats["mean"], stats["std"]

    train_ds = LocalChipDataset(
        train_pairs,
        cfg.img_size,
        cfg.boundary_width,
        cfg.distance_clip,
        mean,
        std,
        train=True,
    )
    val_ds = LocalChipDataset(
        val_pairs,
        cfg.img_size,
        cfg.boundary_width,
        cfg.distance_clip,
        mean,
        std,
        train=False,
    )
    train_loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(cfg.eval_batch_size, len(val_ds)), shuffle=False)

    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    model = DinoV3HotLit.load_from_checkpoint(
        str(pretrained_ckpt),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_path=encoder_ckpt,
    )
    model.lr = ft_lr

    trainer_zero = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    pre_iou = float(trainer_zero.validate(model, dataloaders=val_loader, verbose=False)[0]["val/iou"])

    ckpt_cb = ModelCheckpoint(
        dirpath=out / "ckpts",
        filename="ft-{epoch:02d}-{val/iou:.4f}",
        monitor="val/iou",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(monitor="val/iou", mode="max", patience=5)
    trainer = pl.Trainer(
        max_epochs=ft_epochs,
        precision=cfg.precision,
        accelerator="auto",
        devices=1,
        gradient_clip_val=cfg.grad_clip,
        callbacks=[ckpt_cb, early],
        logger=CSVLogger(save_dir=str(out), name="lightning"),
        log_every_n_steps=1,
        default_root_dir=str(out),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    post_iou = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")

    summary = {
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "val_iou_pretrained": pre_iou,
        "val_iou_finetuned": post_iou,
        "delta": post_iou - pre_iou,
        "best_ckpt": ckpt_cb.best_model_path,
        "output_dir": str(out),
    }
    (out / "summary.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(summary)))
    log.info("Banepa FT summary: %s", summary)
    return summary
