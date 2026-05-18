import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from kornia.losses import total_variation
from lightning.pytorch import LightningModule
from segmentation_models_pytorch.losses import DiceLoss
from terratorch.models.decoders.upernet_decoder import UperNetDecoder
from terratorch.models.necks import LearnedInterpolateToPyramidal, ReshapeTokensToImage
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn
from torchmetrics.classification import BinaryJaccardIndex

log = logging.getLogger(__name__)

N_HEAD_CHANNELS = 3  # mask, boundary, distance


def _download_ckpt(repo: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, filename=filename))


class _SelectIndices(nn.Module):
    def __init__(self, backbone: nn.Module, indices: Sequence[int]):
        super().__init__()
        self.backbone = backbone
        self._indices = list(indices)
        total = len(backbone.out_channels)  # type: ignore[attr-defined]
        resolved = [i if i >= 0 else total + i for i in self._indices]
        self.out_channels = [backbone.out_channels[i] for i in resolved]  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(x)
        n = len(feats)
        return [feats[i if i >= 0 else n + i] for i in self._indices]


class DinoV3UperNet(nn.Module):
    def __init__(
        self,
        ckpt_path: str | Path | None,
        img_size: int,
        seg_out_indices: Sequence[int],
        decoder_channels: int,
        head_dropout: float,
    ):
        super().__init__()
        build_kwargs = {"pretrained": ckpt_path is not None, "img_size": img_size}
        if ckpt_path is not None:
            build_kwargs["ckpt_path"] = str(ckpt_path)
        raw = BACKBONE_REGISTRY.build("terratorch_dinov3_vitl16", **build_kwargs)
        self.backbone = _SelectIndices(raw, seg_out_indices)
        ch = list(self.backbone.out_channels)
        self.reshape = ReshapeTokensToImage(channel_list=ch, remove_cls_token=True)
        self.pyramid = LearnedInterpolateToPyramidal(channel_list=ch)
        self.decoder = UperNetDecoder(
            embed_dim=list(self.pyramid.embedding_dim),
            channels=decoder_channels,
            pool_scales=(1, 2, 3, 6),
        )
        self.dropout = nn.Dropout2d(head_dropout)
        self.head = nn.Conv2d(decoder_channels, N_HEAD_CHANNELS, kernel_size=1)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.reshape(feats)
        feats = self.pyramid(feats)
        logits = self.head(self.dropout(self.decoder(feats)))
        return torch.nn.functional.interpolate(
            logits, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
        )


class DinoV3HotLit(LightningModule):
    def __init__(
        self,
        img_size: int,
        seg_out_indices: Sequence[int],
        decoder_channels: int,
        head_dropout: float,
        lr: float,
        weight_decay: float,
        boundary_loss_weight: float,
        distance_loss_weight: float,
        tv_loss_weight: float = 0.0,
        onecycle_pct_start: float = 0.05,
        onecycle_div_factor: float = 25.0,
        onecycle_final_div_factor: float = 1e4,
        ckpt_path: str | Path | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ckpt_path"])
        self.net = DinoV3UperNet(
            ckpt_path=ckpt_path,
            img_size=img_size,
            seg_out_indices=seg_out_indices,
            decoder_channels=decoder_channels,
            head_dropout=head_dropout,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="binary", from_logits=True)
        self.huber = nn.HuberLoss(delta=1.0)
        self.lr = lr
        self.weight_decay = weight_decay
        self.boundary_loss_weight = boundary_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.tv_loss_weight = tv_loss_weight
        self.onecycle_pct_start = onecycle_pct_start
        self.onecycle_div_factor = onecycle_div_factor
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.val_iou = BinaryJaccardIndex()
        self.test_iou = BinaryJaccardIndex()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(batch["image"])
        mask_logit = logits[:, 0]
        boundary_logit = logits[:, 1]
        distance_pred = torch.tanh(logits[:, 2])
        mask_target = batch["mask"].float()
        boundary_target = batch["boundary"].float()
        distance_target = batch["distance"].float()
        loss_mask = self.bce(mask_logit, mask_target) + self.dice(mask_logit, mask_target)
        loss_boundary = self.bce(boundary_logit, boundary_target)
        loss_distance = self.huber(distance_pred, distance_target)
        loss = (
            loss_mask + self.boundary_loss_weight * loss_boundary + self.distance_loss_weight * loss_distance
        )
        if self.tv_loss_weight > 0.0:
            mask_prob = torch.sigmoid(mask_logit)
            loss = loss + self.tv_loss_weight * total_variation(mask_prob, reduction="mean").mean()
        return loss, mask_logit, mask_target

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mask_logit, mask_target = self._step(batch)
        self.val_iou.update((torch.sigmoid(mask_logit) > 0.5).int(), mask_target.int())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val/iou", self.val_iou.compute(), prog_bar=True)
        self.val_iou.reset()

    def test_step(self, batch, batch_idx):
        _, mask_logit, mask_target = self._step(batch)
        self.test_iou.update((torch.sigmoid(mask_logit) > 0.5).int(), mask_target.int())

    def on_test_epoch_end(self):
        self.log("test/iou", self.test_iou.compute())
        self.test_iou.reset()

    def configure_optimizers(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=self.onecycle_pct_start,
            anneal_strategy="cos",
            div_factor=self.onecycle_div_factor,
            final_div_factor=self.onecycle_final_div_factor,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_model(cfg) -> DinoV3HotLit:
    ckpt_path = _download_ckpt(cfg.hf_ckpt_repo, cfg.hf_ckpt_file)
    return DinoV3HotLit(
        ckpt_path=ckpt_path,
        img_size=cfg.img_size,
        seg_out_indices=tuple(cfg.seg_out_indices),
        decoder_channels=cfg.decoder_channels,
        head_dropout=cfg.head_dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        boundary_loss_weight=cfg.boundary_loss_weight,
        distance_loss_weight=cfg.distance_loss_weight,
        tv_loss_weight=cfg.tv_loss_weight,
        onecycle_pct_start=cfg.onecycle_pct_start,
        onecycle_div_factor=cfg.onecycle_div_factor,
        onecycle_final_div_factor=cfg.onecycle_final_div_factor,
    )
