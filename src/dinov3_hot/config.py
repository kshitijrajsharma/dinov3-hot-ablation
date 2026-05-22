from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf


@dataclass
class HpoConfig:
    enabled: bool = False
    n_trials: int = 8
    data_pct: float = 10.0
    max_epochs: int = 5
    # Wide ranges so HPO can find configurations that improve instance F1, not just pixel IoU.
    lr_range: tuple[float, float] = (3e-4, 3e-3)
    weight_decay_range: tuple[float, float] = (1e-4, 1e-2)
    boundary_loss_weight_range: tuple[float, float] = (0.1, 2.0)
    distance_loss_weight_range: tuple[float, float] = (0.1, 2.0)
    aux_loss_weight_range: tuple[float, float] = (0.2, 0.6)
    tv_loss_weight_range: tuple[float, float] = (0.0, 0.15)
    storage: str | None = None
    study_name: str = "dinov3_hot_hpo"


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    backbone: str = "terratorch_dinov3_vitl16_lvd"
    hf_ckpt_repo: str = "kshitijrajsharma/dinov3"
    hf_ckpt_file: str = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    img_size: int = 256
    patch_size: int = 16
    seg_out_indices: tuple[int, ...] = (5, 11, 17, 23)
    aux_in_index: int = 2
    decoder_channels: int = 512

    dataset_repo: str = "hotosm/vhr-building-segmentation"
    data_root: str = "data/hot_building"
    data_pct: float = 100.0
    drop_null_images: bool = True
    boundary_width: int = 2
    distance_clip: float = 15.0

    batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 50
    early_stop_patience: int = 5
    aux_loss_weight: float = 0.4
    boundary_loss_weight: float = 0.43
    distance_loss_weight: float = 0.52
    tv_loss_weight: float = 0.0
    onecycle_pct_start: float = 0.05
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    precision: str = "bf16-mixed"
    seed: int = 42
    grad_clip: float = 1.0

    output_dir: str = "outputs"
    run_name: str = "dinov3l_v5"

    tile_window: int = 256
    tile_stride: int = 170
    tile_threshold: float = 0.5

    regularize_simplify_m: float = 0.5
    regularize_area_threshold: float = 0.70
    regularize_overlap_tol_m2: float = 1.0

    hpo: HpoConfig = field(default_factory=HpoConfig)


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> DictConfig:
    base = OmegaConf.structured(TrainConfig)
    if path is not None:
        file_cfg = OmegaConf.load(Path(path))
        base = OmegaConf.merge(base, file_cfg)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    return cast(DictConfig, base)


def resolve_root(cfg: DictConfig) -> Path:
    p = Path(cfg.data_root)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_output(cfg: DictConfig) -> Path:
    p = Path(cfg.output_dir)
    out = (p if p.is_absolute() else REPO_ROOT / p) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    return out
