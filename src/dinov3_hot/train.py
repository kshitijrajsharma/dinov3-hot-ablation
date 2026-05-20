import logging

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from dinov3_hot.config import resolve_output, resolve_root
from dinov3_hot.data import HotBuildingDataModule
from dinov3_hot.model import build_model

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> dict:
    if cfg.hpo.enabled:
        from dinov3_hot.hpo import run_hpo

        hpo_summary = run_hpo(cfg)
        for k, v in hpo_summary["best_params"].items():
            cfg[k] = v
        log.info("Applying best HPO params then training at full schedule: %s", hpo_summary["best_params"])

    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    out_dir = resolve_output(cfg)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    dm = HotBuildingDataModule(
        repo_id=cfg.dataset_repo,
        root=resolve_root(cfg),
        img_size=cfg.img_size,
        boundary_width=cfg.boundary_width,
        distance_clip=cfg.distance_clip,
        drop_null_images=cfg.drop_null_images,
        data_pct=cfg.data_pct,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        seed=cfg.seed,
    )

    model = build_model(cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "ckpts",
        filename="best-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(monitor="val/loss", mode="min", patience=cfg.early_stop_patience)
    # refresh_rate=0 makes TQDM print one line per epoch instead of live-updating,
    # so redirected logs (nohup ... > log) show clean per-epoch metrics.
    progress = TQDMProgressBar(refresh_rate=0)
    csv_logger = CSVLogger(save_dir=str(out_dir), name="lightning")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=cfg.grad_clip,
        callbacks=[ckpt_cb, early, progress],
        logger=csv_logger,
        log_every_n_steps=10,
        default_root_dir=str(out_dir),
        deterministic="warn",
    )

    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path="best")

    summary = {
        "best_ckpt": ckpt_cb.best_model_path,
        "best_val_loss": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "test_iou": float(test_metrics[0]["test/iou"]) if test_metrics else None,
        "output_dir": str(out_dir),
    }
    log.info("Run summary: %s", summary)
    return summary
