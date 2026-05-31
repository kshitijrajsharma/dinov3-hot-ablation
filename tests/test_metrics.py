import numpy as np
import pytest
import torch
from scipy.ndimage import label
from shapely.geometry import Polygon

from dinov3_hot.metrics import (
    BinaryInstanceF1,
    instance_prf,
    polygon_orthogonality,
    polygon_vertex_count,
)


def _two_instance_gt() -> np.ndarray:
    gt = np.zeros((32, 32), dtype=np.int32)
    gt[2:10, 2:10] = 1
    gt[15:25, 15:25] = 1
    return gt


def _one_match_one_spurious() -> np.ndarray:
    pr = np.zeros((32, 32), dtype=np.int32)
    pr[2:10, 2:10] = 1
    pr[1:5, 25:30] = 1
    return pr


def test_instance_prf_one_match_one_spurious():
    gt = _two_instance_gt()
    pr = _one_match_one_spurious()
    gt_lbl, _ = label(gt)
    pr_lbl, _ = label(pr)
    out = instance_prf(pr_lbl, gt_lbl)
    assert out["tp"] == 1
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert out["f1"] == 0.5


def test_binary_instance_f1_matches_numpy_api():
    gt = _two_instance_gt()
    pr = _one_match_one_spurious()
    pr_t = torch.from_numpy(pr).unsqueeze(0)
    gt_t = torch.from_numpy(gt).unsqueeze(0)
    m = BinaryInstanceF1()
    m.update(pr_t, gt_t)  # ty: ignore[invalid-argument-type]
    assert m.compute().item() == 0.5  # ty: ignore[missing-argument]
    prf = m.precision_recall_f1()
    assert prf["tp"] == 1 and prf["fp"] == 1 and prf["fn"] == 1


def test_binary_instance_f1_accumulates_across_batches():
    gt = _two_instance_gt()
    pr = _one_match_one_spurious()
    pr_t = torch.from_numpy(pr).unsqueeze(0)
    gt_t = torch.from_numpy(gt).unsqueeze(0)
    m = BinaryInstanceF1()
    m.update(pr_t, gt_t)  # ty: ignore[invalid-argument-type]
    m.update(gt_t, gt_t)  # ty: ignore[invalid-argument-type]
    prf = m.precision_recall_f1()
    assert prf["tp"] == 3
    assert prf["fp"] == 1
    assert prf["fn"] == 1


def test_binary_instance_f1_perfect_and_empty():
    gt = _two_instance_gt()
    gt_t = torch.from_numpy(gt).unsqueeze(0)
    m_perfect = BinaryInstanceF1()
    m_perfect.update(gt_t, gt_t)  # ty: ignore[invalid-argument-type]
    assert m_perfect.compute().item() == 1.0  # ty: ignore[missing-argument]

    m_empty = BinaryInstanceF1()
    empty = torch.zeros((1, 32, 32), dtype=torch.int32)
    m_empty.update(empty, empty)  # ty: ignore[invalid-argument-type]
    assert m_empty.compute().item() == 0.0  # ty: ignore[missing-argument]


def test_polygon_vertex_count_single_square_is_four():
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert polygon_vertex_count([sq]) == 4.0


def test_polygon_vertex_count_mean_across_polygons():
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    tri = Polygon([(0, 0), (2, 0), (1, 2)])
    assert polygon_vertex_count([sq, tri]) == 3.5


def test_polygon_vertex_count_empty_input_is_zero():
    assert polygon_vertex_count([]) == 0.0


def test_polygon_orthogonality_axis_aligned_square_is_one():
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert polygon_orthogonality([sq]) == 1.0


def test_polygon_orthogonality_rotated_square_is_one():
    diamond = Polygon([(0, 1), (1, 0), (0, -1), (-1, 0)])
    assert polygon_orthogonality([diamond]) == 1.0


def test_polygon_orthogonality_equilateral_triangle_is_below_one():
    tri = Polygon([(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)])
    assert 0.0 <= polygon_orthogonality([tri]) < 1.0


def test_polygon_orthogonality_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError):
        polygon_orthogonality([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], tol_deg=0)


def test_polygon_orthogonality_empty_input_is_zero():
    assert polygon_orthogonality([]) == 0.0
