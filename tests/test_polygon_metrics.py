"""Unit tests for the polygon eval helpers added in metrics.py:
polygon_c_iou, polygon_mta, polygon_n_ratio, match_polygons_by_iou."""

from shapely.geometry import Polygon

from dinov3_hot.metrics import (
    match_polygons_by_iou,
    polygon_c_iou,
    polygon_mta,
    polygon_n_ratio,
)


def test_c_iou_perfect_match_is_one() -> None:
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert abs(polygon_c_iou(p, p) - 1.0) < 1e-9


def test_c_iou_vertex_mismatch_lowers_score() -> None:
    rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    octagon = Polygon([(0, 0), (5, -1), (10, 0), (11, 5), (10, 10), (5, 11), (0, 10), (-1, 5)])
    full = polygon_c_iou(rect, rect)
    penalised = polygon_c_iou(rect, octagon)
    # Same coverage region roughly, but vertex-count mismatch penalises c_iou.
    assert penalised < full


def test_c_iou_empty_inputs_return_zero() -> None:
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    empty = Polygon()
    assert polygon_c_iou(p, empty) == 0.0
    assert polygon_c_iou(empty, p) == 0.0


def test_mta_aligned_polygons_is_zero() -> None:
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert polygon_mta(p, p) == 0.0


def test_mta_45_degree_rotation_is_45() -> None:
    rect = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    rotated = Polygon([(0, 0), (1, 1), (0, 2), (-1, 1)])
    err = polygon_mta(rect, rotated)
    assert abs(err - 45.0) < 1e-6


def test_n_ratio_equal_vertex_counts_is_one() -> None:
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    q = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    assert abs(polygon_n_ratio([p], [q]) - 1.0) < 1e-9


def test_n_ratio_excess_vertices_above_one() -> None:
    pred = Polygon([(0, 0), (5, -1), (10, 0), (11, 5), (10, 10), (5, 11), (0, 10), (-1, 5)])
    gt = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    r = polygon_n_ratio([pred], [gt])
    assert r > 1.0


def test_match_polygons_pairs_overlapping_geoms() -> None:
    rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    overlap = Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])
    other = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    pairs = match_polygons_by_iou([overlap], [rect, other], iou_threshold=0.5)
    assert len(pairs) == 1
    pi, gi, iou = pairs[0]
    assert (pi, gi) == (0, 0)
    assert iou >= 0.5


def test_match_polygons_no_pairs_below_threshold() -> None:
    a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    pairs = match_polygons_by_iou([a], [b], iou_threshold=0.5)
    assert pairs == []


def test_match_polygons_each_gt_matched_at_most_once() -> None:
    rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    a = Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])
    b = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    pairs = match_polygons_by_iou([a, b], [rect], iou_threshold=0.3)
    # Only one rect to match; first pred takes it greedily.
    assert len(pairs) == 1
    assert pairs[0][0] == 0
