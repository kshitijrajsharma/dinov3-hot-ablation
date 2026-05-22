"""Instance F1 (torchmetrics PanopticQuality wrapper) and polygon shape stats."""

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import torch
from scipy import ndimage
from shapely.geometry import Polygon
from torchmetrics import Metric
from torchmetrics.detection import PanopticQuality

# PanopticQuality stores per-class state things-first then stuffs (each sorted),
# so with things={1}, stuffs={0} the building class lives at index 0.
_THING_IDX = 0


def _to_panoptic(labels: np.ndarray) -> torch.Tensor:
    cat = (labels > 0).astype(np.int32)
    return torch.from_numpy(np.stack([cat, labels.astype(np.int32)], axis=-1))


def _count(pred_labels: np.ndarray, gt_labels: np.ndarray) -> tuple[int, int, int]:
    pq = PanopticQuality(things={1}, stuffs={0})
    preds = _to_panoptic(pred_labels).unsqueeze(0)
    target = _to_panoptic(gt_labels).unsqueeze(0)
    pq.update(preds, target)  # ty: ignore[invalid-argument-type]
    return (
        int(pq.true_positives[_THING_IDX].item()),
        int(pq.false_positives[_THING_IDX].item()),
        int(pq.false_negatives[_THING_IDX].item()),
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


class BinaryInstanceF1(Metric):
    full_state_update: bool = False
    higher_is_better: bool = True

    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for p, t in zip(preds.cpu().numpy(), target.cpu().numpy(), strict=True):
            p_lbl, _ = ndimage.label(p > 0)
            t_lbl, _ = ndimage.label(t > 0)
            tp, fp, fn = _count(p_lbl, t_lbl)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def compute(self) -> torch.Tensor:
        tp = self.tp.float()
        denom = 2 * tp + self.fp.float() + self.fn.float()
        return torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))

    def precision_recall_f1(self) -> dict[str, float]:
        return _prf(int(self.tp.item()), int(self.fp.item()), int(self.fn.item()))


def instance_prf(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict[str, float]:
    return _prf(*_count(pred_labels, gt_labels))


def polygon_vertex_count(geoms: Iterable[Polygon]) -> float:
    """Mean exterior vertex count (drops shapely's closing duplicate)."""
    counts = [max(0, len(p.exterior.coords) - 1) for p in geoms if not p.is_empty]
    return float(np.mean(counts)) if counts else 0.0


def polygon_orthogonality(geoms: Iterable[Polygon], tol_deg: float = 5.0) -> float:
    """Mean per-polygon fraction of edges within tol_deg of the minimum-rotated-rectangle axis."""
    if tol_deg <= 0:
        raise ValueError("tol_deg must be positive")
    per_poly: list[float] = []
    for poly in geoms:
        if poly.is_empty:
            continue
        coords = list(poly.exterior.coords)
        n_edges = len(coords) - 1
        if n_edges < 3:
            per_poly.append(1.0)
            continue
        rect_coords = list(poly.minimum_rotated_rectangle.exterior.coords)
        ref_deg = np.degrees(
            np.arctan2(rect_coords[1][1] - rect_coords[0][1], rect_coords[1][0] - rect_coords[0][0])
        )
        hits = 0
        for (x0, y0), (x1, y1) in pairwise(coords):
            ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
            mod = (ang - ref_deg) % 90.0
            if min(mod, 90.0 - mod) < tol_deg:
                hits += 1
        per_poly.append(hits / n_edges)
    return float(np.mean(per_poly)) if per_poly else 0.0
