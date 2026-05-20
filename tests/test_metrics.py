import numpy as np
import torch
from scipy.ndimage import label

from dinov3_hot.metrics import BinaryInstanceF1, instance_prf


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
    m = BinaryInstanceF1()
    m.update(torch.from_numpy(pr).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0))
    assert m.compute().item() == 0.5
    prf = m.precision_recall_f1()
    assert prf["tp"] == 1 and prf["fp"] == 1 and prf["fn"] == 1


def test_binary_instance_f1_accumulates_across_batches():
    gt = _two_instance_gt()
    pr = _one_match_one_spurious()
    m = BinaryInstanceF1()
    m.update(torch.from_numpy(pr).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0))
    m.update(torch.from_numpy(gt).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0))
    prf = m.precision_recall_f1()
    assert prf["tp"] == 3
    assert prf["fp"] == 1
    assert prf["fn"] == 1


def test_binary_instance_f1_perfect_and_empty():
    gt = _two_instance_gt()
    m_perfect = BinaryInstanceF1()
    m_perfect.update(torch.from_numpy(gt).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0))
    assert m_perfect.compute().item() == 1.0

    m_empty = BinaryInstanceF1()
    empty = torch.zeros((1, 32, 32), dtype=torch.int32)
    m_empty.update(empty, empty)
    assert m_empty.compute().item() == 0.0
