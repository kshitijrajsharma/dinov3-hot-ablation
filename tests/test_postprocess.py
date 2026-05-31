"""Unit tests for dinov3_hot.postprocess, including the _ensure_valid_polygon
repair for self-intersecting DP output."""

from shapely.geometry import LinearRing, Polygon

from dinov3_hot.postprocess import _dp_then_mbr_safe, _ensure_valid_polygon


def test_ensure_valid_passes_through_valid_polygon() -> None:
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert _ensure_valid_polygon(p) is p


def test_ensure_valid_repairs_self_intersecting_bowtie() -> None:
    # Classic bowtie: two triangles meeting at the centre, exterior self-crosses.
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert not bowtie.is_valid
    repaired = _ensure_valid_polygon(bowtie)
    assert repaired.is_valid
    assert repaired.geom_type == "Polygon"
    # The largest part of a bowtie repair is one of the two triangles, each area 1.
    assert abs(repaired.area - 1.0) < 1e-9


def test_dp_then_mbr_safe_handles_simplification_invalidation() -> None:
    # A reasonable rectangle plus a worthwhile spike that DP at tolerance 5.0 would prune.
    coords = [(0, 0), (10, 0), (10, 10), (5, 10), (5.1, 11), (5.2, 10), (0, 10), (0, 0)]
    p = Polygon(coords)
    out = _dp_then_mbr_safe([p], simplify_m=2.0, area_threshold=0.95, overlap_tol_m2=0.5)
    assert len(out) == 1
    assert out[0].is_valid
    assert out[0].geom_type == "Polygon"


def test_dp_then_mbr_safe_substitutes_mbr_for_near_rectangle() -> None:
    # A near-rectangle should be replaced with its minimum rotated rectangle.
    almost_rect = Polygon([(0, 0), (10, 0), (10.05, 5), (10, 10), (0, 10), (0, 0)])
    out = _dp_then_mbr_safe([almost_rect], simplify_m=0.0, area_threshold=0.95, overlap_tol_m2=1.0)
    assert len(out) == 1
    # MBR of an almost-axis-aligned rectangle is itself axis-aligned with 4 corners
    coords = list(LinearRing(out[0].exterior.coords).coords)
    assert len(coords) == 5  # 4 corners + repeated close
