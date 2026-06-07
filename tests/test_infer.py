"""Unit tests for the adaptive watershed seeding in dinov3_hot.infer.instance_separate.

The shipped behavior: small foreground blobs seed on distance local maxima (unchanged),
while blobs larger than `large_blob_area_px` seed on the distance h-maxima so a single
large roof with several spurious distance bumps collapses to one instance, yet a large
blob with a genuinely deep valley still splits into two.
"""

import numpy as np

from dinov3_hot.infer import instance_separate


def _bump(shape: tuple[int, int], cy: int, cx: int, sigma: float, amp: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))


def _n_instances(labels: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(labels)))


def test_two_separate_small_blobs_stay_two() -> None:
    """Small blobs (below the large-blob threshold) keep distance-peak seeding."""
    shape = (80, 80)
    mask = np.zeros(shape, dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    mask[50:70, 50:70] = 1.0
    distance = _bump(shape, 20, 20, 5.0, 0.8) + _bump(shape, 60, 60, 5.0, 0.8)
    distance *= mask

    labels = instance_separate(mask, distance, mask_threshold=0.5, large_blob_area_px=1500)
    assert _n_instances(labels) == 2


def test_large_blob_shallow_bumps_collapse_to_one() -> None:
    """A big roof whose distance has two shallow bumps (valley < h) is one instance."""
    shape = (90, 90)
    mask = np.zeros(shape, dtype=np.float32)
    mask[15:75, 15:75] = 1.0  # 3600 px > 1500
    distance = 0.6 + _bump(shape, 45, 30, 12.0, 0.1) + _bump(shape, 45, 60, 12.0, 0.1)
    distance *= mask

    labels = instance_separate(
        mask, distance, mask_threshold=0.5, large_blob_area_px=1500, h_maxima_depth=0.2
    )
    assert _n_instances(labels) == 1


def test_large_blob_deep_valley_splits_to_two() -> None:
    """A big blob with a genuinely deep valley (>> h) still separates into two instances."""
    shape = (90, 90)
    mask = np.zeros(shape, dtype=np.float32)
    mask[15:75, 15:75] = 1.0
    distance = _bump(shape, 45, 30, 8.0, 0.9) + _bump(shape, 45, 60, 8.0, 0.9)
    distance *= mask

    labels = instance_separate(
        mask, distance, mask_threshold=0.5, large_blob_area_px=1500, h_maxima_depth=0.2
    )
    assert _n_instances(labels) == 2


def test_empty_foreground_returns_zeros() -> None:
    shape = (40, 40)
    mask = np.zeros(shape, dtype=np.float32)
    distance = np.zeros(shape, dtype=np.float32)
    labels = instance_separate(mask, distance, mask_threshold=0.5)
    assert _n_instances(labels) == 0
