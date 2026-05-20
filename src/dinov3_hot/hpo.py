import logging
from copy import deepcopy

import lightning.pytorch as pl
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from dinov3_hot.config import resolve_output, resolve_root
from dinov3_hot.data import HotBuildingDataModule
from dinov3_hot.model import build_model

log = logging.getLogger(__name__)


def _objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    cfg = deepcopy(base_cfg)
    cfg.lr = trial.suggest_float("lr", cfg.hpo.lr_range[0], cfg.hpo.lr_range[1], log=True)
    cfg.weight_decay = trial.suggest_float(
        "weight_decay", cfg.hpo.weight_decay_range[0], cfg.hpo.weight_decay_range[1], log=True
    )
    cfg.boundary_loss_weight = trial.suggest_float(
        "boundary_loss_weight",
        cfg.hpo.boundary_loss_weight_range[0],
        cfg.hpo.boundary_loss_weight_range[1],
    )
    cfg.distance_loss_weight = trial.suggest_float(
        "distance_loss_weight",
        cfg.hpo.distance_loss_weight_range[0],
        cfg.hpo.distance_loss_weight_range[1],
    )
    cfg.aux_loss_weight = trial.suggest_float(
        "aux_loss_weight",
        cfg.hpo.aux_loss_weight_range[0],
        cfg.hpo.aux_loss_weight_range[1],
    )
    cfg.tv_loss_weight = trial.suggest_float(
        "tv_loss_weight",
        cfg.hpo.tv_loss_weight_range[0],
        cfg.hpo.tv_loss_weight_range[1],
    )
    cfg.data_pct = cfg.hpo.data_pct
    cfg.max_epochs = cfg.hpo.max_epochs
    cfg.run_name = f"{base_cfg.run_name}_hpo_t{trial.number:03d}"

    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

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
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=cfg.grad_clip,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic="warn",
    )
    trainer.fit(model, datamodule=dm)
    val_loss = float(trainer.callback_metrics.get("val/loss", torch.tensor(float("inf"))))
    val_iou = float(trainer.callback_metrics.get("val/iou", torch.tensor(0.0)))
    log.info(
        "Trial %d: val/loss=%.4f val/iou=%.4f lr=%g wd=%g bdy=%g dist=%g aux=%g tv=%g",
        trial.number,
        val_loss,
        val_iou,
        cfg.lr,
        cfg.weight_decay,
        cfg.boundary_loss_weight,
        cfg.distance_loss_weight,
        cfg.aux_loss_weight,
        cfg.tv_loss_weight,
    )
    return val_loss


def run_hpo(cfg: DictConfig) -> dict:
    out_dir = resolve_output(cfg)
    db_path = out_dir / "hpo.db"
    if db_path.exists() and cfg.hpo.storage is None:
        db_path.unlink()
    storage = cfg.hpo.storage or f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="minimize",
        study_name=cfg.hpo.study_name,
        storage=storage,
        load_if_exists=cfg.hpo.storage is not None,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: _objective(t, cfg), n_trials=cfg.hpo.n_trials, gc_after_trial=True)

    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": dict(best.params),
        "n_trials": len(study.trials),
        "storage": storage,
    }
    (out_dir / "hpo_best.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(summary)))
    log.info("HPO done: %s", summary)
    return summary
