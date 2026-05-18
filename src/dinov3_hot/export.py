import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download

from dinov3_hot.model import DinoV3HotLit

log = logging.getLogger(__name__)


def export_onnx(
    cfg,
    ckpt_path: str | Path,
    out_path: str | Path,
    opset: int = 17,
    parity_atol: float = 1e-2,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    model = DinoV3HotLit.load_from_checkpoint(str(ckpt_path), map_location=device, ckpt_path=encoder_ckpt)
    model.eval()
    inner = model.net.to(device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
    onnx_program = torch.onnx.export(
        inner,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        dynamo=True,
    )
    if onnx_program is not None:
        onnx_program.optimize()
        onnx_program.save(str(out_path))

    with torch.inference_mode():
        torch_out = inner(dummy).cpu().numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"image": dummy.cpu().numpy()})[0]
    diff = float(np.abs(torch_out - ort_out).max())
    if diff > parity_atol:
        raise RuntimeError(f"ONNX parity failed: max abs diff {diff:.4e} > {parity_atol:.4e}")
    log.info("ONNX exported to %s | parity max-abs-diff = %.2e", out_path, diff)
    return out_path
