import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from dinov3_hot.model import DinoV3HotLit
from dinov3_hot.serve import INFERENCE_BATCH_SIZE

log = logging.getLogger(__name__)


class _MainHeadOnly(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main, _ = self.net(x)
        return main


def export_onnx(
    cfg,
    ckpt_path: str | Path,
    out_path: str | Path,
    opset: int = 17,
    parity_atol: float = 5e-2,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    model = DinoV3HotLit.load_from_checkpoint(str(ckpt_path), map_location=device, ckpt_path=encoder_ckpt)
    model.eval()
    # Export only the main head; the FCN aux head is supervision-only and unused at inference.
    inner = _MainHeadOnly(model.net).to(device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(INFERENCE_BATCH_SIZE, 3, cfg.img_size, cfg.img_size, device=device)
    onnx_program = torch.onnx.export(
        inner,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["logits"],
        dynamo=True,
    )
    if onnx_program is not None:
        onnx_program.optimize()
        onnx_program.save(str(out_path))

    with torch.inference_mode():
        torch_out = inner(dummy).cpu().numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = np.asarray(session.run(None, {"image": dummy.cpu().numpy()})[0])
    diff = float(np.abs(torch_out - ort_out).max())
    # agrees pixel-for-pixel; small logit drift below saturation is invisible downstream.
    mask_torch = (1.0 / (1.0 + np.exp(-torch_out[:, 0]))) > 0.5
    mask_onnx = (1.0 / (1.0 + np.exp(-ort_out[:, 0]))) > 0.5
    mask_disagree = int((mask_torch != mask_onnx).sum())
    if mask_disagree > 0:
        raise RuntimeError(
            f"ONNX mask parity failed: {mask_disagree}/{mask_torch.size} pixels disagree at threshold 0.5"
        )
    if diff > parity_atol:
        raise RuntimeError(f"ONNX parity failed: max abs diff {diff:.4e} > {parity_atol:.4e}")
    log.info(
        "ONNX exported to %s | logit max-abs-diff=%.2e | mask pixels disagree=0/%d",
        out_path,
        diff,
        mask_torch.size,
    )
    return out_path
