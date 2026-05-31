"""Unit tests for the torch-free helpers in dinov3_hot.serve."""

import numpy as np

from dinov3_hot.serve import (
    HOT_MEAN,
    HOT_STD,
    MODEL_INPUT_SIZE,
    SLIDING_STRIDE,
    gaussian_kernel,
    normalize_chw,
)


def test_gaussian_kernel_shape_peak_and_separability() -> None:
    kernel = gaussian_kernel(8)
    assert kernel.shape == (8, 8)
    assert kernel.max() == 1.0
    assert kernel[3, 3] == kernel.max()
    assert kernel[0, 0] < kernel[3, 3]
    assert np.allclose(kernel, kernel.T, atol=1e-12)


def test_normalize_chw_shape_dtype_and_values() -> None:
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = normalize_chw(img, HOT_MEAN, HOT_STD)
    assert out.shape == (3, 4, 4)
    assert out.dtype == np.float32
    mean_arr = np.asarray(HOT_MEAN).reshape(3, 1, 1)
    std_arr = np.asarray(HOT_STD).reshape(3, 1, 1)
    expected = (np.full((3, 1, 1), 128 / 255.0) - mean_arr) / std_arr
    assert np.allclose(out, expected.astype(np.float32))


def test_normalize_chw_orders_brightness_correctly() -> None:
    white = normalize_chw(np.full((2, 2, 3), 255, dtype=np.uint8), HOT_MEAN, HOT_STD)
    black = normalize_chw(np.zeros((2, 2, 3), dtype=np.uint8), HOT_MEAN, HOT_STD)
    for channel in range(3):
        assert white[channel].mean() > black[channel].mean()


def test_module_constants_are_plausible() -> None:
    assert MODEL_INPUT_SIZE == 256
    assert SLIDING_STRIDE == 128
    assert all(0 < m < 1 for m in HOT_MEAN)
    assert all(0 < s < 1 for s in HOT_STD)
