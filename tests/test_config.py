from dinov3_hot.config import load_config


def test_default_config_loads():
    cfg = load_config(None)
    assert cfg.backbone == "terratorch_dinov3_vitl16"
    assert cfg.img_size == 256
    assert tuple(cfg.seg_out_indices) == (5, 11, 17, 23)
    assert cfg.aux_in_index == 2
    assert cfg.aux_loss_weight == 0.4
    assert cfg.decoder_channels == 512


def test_yaml_overrides_apply():
    cfg = load_config("conf/train.yaml", overrides=["data_pct=1", "max_epochs=3", "run_name=t"])
    assert cfg.data_pct == 1.0
    assert cfg.max_epochs == 3
    assert cfg.run_name == "t"


def test_typed_override_coerces():
    cfg = load_config(None, overrides=["lr=5e-5", "batch_size=4"])
    assert cfg.lr == 5e-5
    assert cfg.batch_size == 4
