import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from kornia.losses import total_variation
from lightning.pytorch import LightningModule
from segmentation_models_pytorch.losses import DiceLoss
from terratorch.models.decoders.upernet_decoder import UperNetDecoder
from terratorch.models.necks import LearnedInterpolateToPyramidal
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn
from torchmetrics.classification import BinaryJaccardIndex

log = logging.getLogger(__name__)

N_HEAD_CHANNELS = 3  # mask, boundary, distance


def _download_ckpt(repo: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, filename=filename))


class DinoV3UperNet(nn.Module):
    def __init__(
        self,
        ckpt_path: str | Path | None,
        seg_out_indices: Sequence[int],
        decoder_channels: int,
        aux_in_index: int,
    ):
        super().__init__()
        build_kwargs = {"ckpt_path": str(ckpt_path)} if ckpt_path else {}
        wrapper = BACKBONE_REGISTRY.build("terratorch_dinov3_vitl16", **build_kwargs)
        # Bypass terratorch's wrapper to pass norm=True (it omits the arg, so intermediate features
        # would reach the decoder without the backbone's final LayerNorm applied).
        self.backbone = wrapper.dinov3
        self.indices = list(seg_out_indices)
        self.aux_in_index = aux_in_index
        embed_dim = self.backbone.embed_dim
        channel_list = [embed_dim] * len(self.indices)
        self.pyramid = LearnedInterpolateToPyramidal(channel_list=channel_list)
        self.decoder = UperNetDecoder(
            embed_dim=list(self.pyramid.embedding_dim),
            channels=decoder_channels,
            pool_scales=(1, 2, 3, 6),  # ty: ignore[invalid-argument-type]
        )
        self.dropout = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(decoder_channels, N_HEAD_CHANNELS, kernel_size=1)
        self.aux_head = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, N_HEAD_CHANNELS, kernel_size=1),
        )

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        device_type = "cuda" if x.is_cuda else "cpu"
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=x.is_cuda),
        ):
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.indices,
                reshape=True,
                norm=True,
                return_class_token=False,
            )
        feats = [f.float() for f in feats]
        pyramid_feats = self.pyramid(tuple(feats))
        decoded = self.decoder(pyramid_feats)
        upsample = torch.nn.functional.interpolate
        main_logits = self.head(self.dropout(decoded))
        main_logits = upsample(main_logits, size=(h, w), mode="bilinear", align_corners=False)
        aux_logits = self.aux_head(feats[self.aux_in_index])
        aux_logits = upsample(aux_logits, size=(h, w), mode="bilinear", align_corners=False)
        return main_logits, aux_logits


class DinoV3HotLit(LightningModule):
    def __init__(
        self,
        seg_out_indices: Sequence[int],
        decoder_channels: int,
        aux_in_index: int,
        aux_loss_weight: float,
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
            seg_out_indices=seg_out_indices,
            decoder_channels=decoder_channels,
            aux_in_index=aux_in_index,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="binary", from_logits=True)
        self.huber = nn.HuberLoss(delta=1.0)
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_loss_weight = aux_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.tv_loss_weight = tv_loss_weight
        self.onecycle_pct_start = onecycle_pct_start
        self.onecycle_div_factor = onecycle_div_factor
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.val_iou = BinaryJaccardIndex()
        self.test_iou = BinaryJaccardIndex()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def _heads_loss(
        self, logits: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return loss, mask_logit, mask_target

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_logits, aux_logits = self.net(batch["image"])
        main_loss, mask_logit, mask_target = self._heads_loss(main_logits, batch)
        aux_loss, _, _ = self._heads_loss(aux_logits, batch)
        loss = main_loss + self.aux_loss_weight * aux_loss
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
        preds = torch.from_numpy((torch.sigmoid(mask_logit) > 0.5).int().cpu().numpy())
        target = torch.from_numpy(mask_target.int().cpu().numpy())
        self.val_iou.update(preds, target)  # ty: ignore[invalid-argument-type]
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()  # ty: ignore[missing-argument]
        self.log("val/iou", iou, prog_bar=True)
        val_loss = self.trainer.callback_metrics.get("val/loss")
        log.info(
            "epoch %d val/loss=%.4f val/iou=%.4f",
            self.current_epoch,
            float(val_loss) if val_loss is not None else float("nan"),
            float(iou),
        )
        self.val_iou.reset()

    def test_step(self, batch, batch_idx):
        _, mask_logit, mask_target = self._step(batch)
        preds = torch.from_numpy((torch.sigmoid(mask_logit) > 0.5).int().cpu().numpy())
        target = torch.from_numpy(mask_target.int().cpu().numpy())
        self.test_iou.update(preds, target)  # ty: ignore[invalid-argument-type]

    def on_test_epoch_end(self):
        self.log("test/iou", self.test_iou.compute())  # ty: ignore[missing-argument]
        self.test_iou.reset()

    def configure_optimizers(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
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
        seg_out_indices=tuple(cfg.seg_out_indices),
        decoder_channels=cfg.decoder_channels,
        aux_in_index=cfg.aux_in_index,
        aux_loss_weight=cfg.aux_loss_weight,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        boundary_loss_weight=cfg.boundary_loss_weight,
        distance_loss_weight=cfg.distance_loss_weight,
        tv_loss_weight=cfg.tv_loss_weight,
        onecycle_pct_start=cfg.onecycle_pct_start,
        onecycle_div_factor=cfg.onecycle_div_factor,
        onecycle_final_div_factor=cfg.onecycle_final_div_factor,
    )
