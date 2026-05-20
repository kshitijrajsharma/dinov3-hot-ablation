"""Instance precision/recall/F1 at IoU > 0.5 via torchmetrics PanopticQuality."""

import numpy as np
import torch
from scipy import ndimage
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
    pq.update(_to_panoptic(pred_labels).unsqueeze(0), _to_panoptic(gt_labels).unsqueeze(0))
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
