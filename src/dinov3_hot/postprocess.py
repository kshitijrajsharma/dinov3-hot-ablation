"""Polygon postprocessing helpers with no torch dependency.

Lives in its own module so distroless inference images can pull only
shapely + geopandas + rasterio (not the full training stack) and still
get the production DP+MBR-safe regularisation.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import shapely
import shapely.geometry as sgeom
from rasterio.features import shapes
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree


def _ensure_valid_polygon(g: BaseGeometry) -> BaseGeometry:
    """DP simplify can emit self-intersecting polygons even with preserve_topology=True,
    which then throws TopologyException in downstream intersection ops. Repair via
    make_valid; if it splits into multiple parts, keep the largest by area."""
    if g.is_valid:
        return g
    fixed = shapely.make_valid(g)
    if fixed.geom_type == "Polygon":
        return fixed
    polys = [p for p in getattr(fixed, "geoms", []) if p.geom_type == "Polygon"]
    return max(polys, key=lambda p: p.area) if polys else g


def _dp_then_mbr_safe(
    geoms: list[BaseGeometry],
    simplify_m: float,
    area_threshold: float,
    overlap_tol_m2: float,
) -> list[BaseGeometry]:
    """DP-simplify each polygon, replace with its minimum rotated rectangle when the polygon
    is already near-rectangular and the swap doesn't add more than overlap_tol_m2 of new
    overlap with any neighbour. Operates in the CRS of the input geometries (metres)."""
    simplified = [_ensure_valid_polygon(g.simplify(simplify_m, preserve_topology=True)) for g in geoms]
    tree = STRtree(simplified)
    out: list[BaseGeometry] = []
    for i, s in enumerate(simplified):
        m = s.minimum_rotated_rectangle
        if not m.area or s.area / m.area <= area_threshold:
            out.append(s)
            continue
        bad = False
        for j in tree.query(m):
            if j == i:
                continue
            neighbour = simplified[j]
            if (
                m.intersects(neighbour)
                and not m.touches(neighbour)
                and m.intersection(neighbour).area - s.intersection(neighbour).area > overlap_tol_m2
            ):
                bad = True
                break
        out.append(s if bad else m)
    return out


def vectorize_binary_mask(
    mask: np.ndarray,
    transform: "rasterio.Affine",
    crs: Any,
    min_area_m2: float = 1.0,
    simplify_m: float = 1.0,
    regularize_area_threshold: float = 0.55,
    regularize_overlap_tol_m2: float = 1.0,
) -> gpd.GeoDataFrame:
    """Binary (or label) raster -> regularised polygons. Projects to EPSG:3857 for the metric-
    space DP+MBR step, drops by min_area_m2, returns polygons in the input CRS."""
    polys = [sgeom.shape(g) for g, v in shapes(mask, mask=mask > 0, transform=transform) if v > 0]
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs).to_crs(epsg=3857)
    gdf["geometry"] = _dp_then_mbr_safe(
        list(gdf.geometry),
        simplify_m=simplify_m,
        area_threshold=regularize_area_threshold,
        overlap_tol_m2=regularize_overlap_tol_m2,
    )
    if min_area_m2 > 0:
        gdf = gdf[gdf.geometry.area > min_area_m2].reset_index(drop=True)
    return gdf.to_crs(crs)
