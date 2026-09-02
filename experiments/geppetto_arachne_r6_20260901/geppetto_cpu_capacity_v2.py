from __future__ import annotations

import time
import torch

from .geppetto_candidate_v2 import GeppettoCandidateConfigV2, GeppettoCandidateV2


def geppetto_cpu_capacity_probe_v2(
    *,
    surface_token_count: int = 328,
    decode_steps: int = 328,
) -> dict[str, float | int | bool | str]:
    """Synthetic CPU architecture-capacity probe; performs no optimization/training."""
    if surface_token_count < 1 or decode_steps < 1 or decode_steps > surface_token_count:
        raise ValueError("invalid capacity probe cardinality")
    cfg = GeppettoCandidateConfigV2(
        model_dim=24,
        knn_k=4,
        local_layers=1,
        global_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=48,
        support_topk=4,
    )
    model = GeppettoCandidateV2(cfg).cpu().eval()
    features = torch.zeros((1, surface_token_count, 24), dtype=torch.float32)
    positions = torch.linspace(-1.0, 1.0, surface_token_count, dtype=torch.float32)[:, None].repeat(1, 3)[None]
    mask = torch.ones((1, surface_token_count), dtype=torch.bool)
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(features, positions, mask, decode_steps=decode_steps)
    elapsed = time.perf_counter() - t0
    finite = all(torch.isfinite(x).all().item() for x in (
        output.positions_normalized,
        output.position_log_sigma,
        output.existence_logits,
        output.stop_logits,
        output.root_logits,
        output.parent_logits,
        output.support_logits,
    ))
    return {
        "schema": "RealSaS.GeppettoCPUCapacityProbe.v2",
        "surface_token_count": int(surface_token_count),
        "decode_steps": int(decode_steps),
        "finite": bool(finite),
        "parent_matrix_rows": int(output.parent_logits.shape[1]),
        "parent_matrix_cols": int(output.parent_logits.shape[2]),
        "support_width": int(output.support_logits.shape[2]),
        "elapsed_seconds": float(elapsed),
        "optimizer_steps": 0,
        "product_max_joint_count": None,
    }
