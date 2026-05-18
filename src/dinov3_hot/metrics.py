"""Instance-level Precision/Recall/F1 at IoU thresholds.

Composed of established library primitives:
- sklearn.metrics.cluster.contingency_matrix for per-(pred,gt) pixel counts
- scipy.optimize.linear_sum_assignment for Hungarian matching
- scipy.ndimage.label for binary-mask -> instance-ids conversion
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix


def _label_iou_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    p_max = int(pred_labels.max())
    g_max = int(gt_labels.max())
    if p_max == 0 or g_max == 0:
        return np.zeros((p_max, g_max), dtype=np.float32)
    cm = contingency_matrix(pred_labels.ravel(), gt_labels.ravel(), sparse=False)
    intersection = cm[1:, 1:].astype(np.float32)
    pred_areas = cm[1:, :].sum(axis=1).astype(np.float32)
    gt_areas = cm[:, 1:].sum(axis=0).astype(np.float32)
    union = pred_areas[:, None] + gt_areas[None, :] - intersection
    return intersection / np.maximum(union, 1.0)


def instance_prf(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision/Recall/F1 by Hungarian matching at given IoU threshold."""
    n_pred = int(pred_labels.max())
    n_gt = int(gt_labels.max())
    if n_pred == 0 and n_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": n_gt}
    if n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": n_pred, "fn": 0}

    iou = _label_iou_matrix(pred_labels, gt_labels)
    row_ind, col_ind = linear_sum_assignment(-iou)
    matched_iou = iou[row_ind, col_ind]
    tp = int((matched_iou >= iou_threshold).sum())
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def binarize_gt_to_instances(gt_mask: np.ndarray) -> np.ndarray:
    """Connected components of a binary GT mask -> uint32 instance labels."""
    return ndimage.label(gt_mask > 0)[0].astype(np.uint32)
