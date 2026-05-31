"""Smoke tests for the cheap guard paths in dinov3_hot.tune. The HPO body
needs a real model + chips and is exercised by the fair-models integration suite."""

from dinov3_hot.serve import DEFAULT_INFERENCE_PARAMS
from dinov3_hot.tune import TUNE_MIN_VAL_CHIPS, tune_postprocess_run


def test_skips_when_n_trials_zero() -> None:
    result = tune_postprocess_run(
        net=None,
        chips_dir=None,  # ty: ignore[invalid-argument-type]
        labels_geojson=None,  # ty: ignore[invalid-argument-type]
        val_chip_names=[],
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        n_trials=0,
    )
    assert result["best_value"] is None
    assert result["best_params"] == DEFAULT_INFERENCE_PARAMS
    assert result["skipped"] == "disabled"


def test_skips_when_val_smaller_than_minimum() -> None:
    result = tune_postprocess_run(
        net=None,
        chips_dir=None,  # ty: ignore[invalid-argument-type]
        labels_geojson=None,  # ty: ignore[invalid-argument-type]
        val_chip_names=["a.tif", "b.tif"],
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        n_trials=5,
        min_val_chips=TUNE_MIN_VAL_CHIPS,
    )
    assert result["skipped"] == "val_too_small"
    assert result["best_params"] == DEFAULT_INFERENCE_PARAMS
