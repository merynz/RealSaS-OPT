from __future__ import annotations

import torch

from models.arachne.v4.arachne_candidate_v4 import EdgeMessageBlock


def test_edge_message_block_bf16_autocast_forward_backward_cpu():
    """Regression: AMP Linear emits BF16 while graph accumulator is FP32."""
    torch.manual_seed(11)
    block = EdgeMessageBlock(dim=64, edge_dim=4, ratio=2, dropout=0.0).train()
    x = torch.randn(1, 18, 64, dtype=torch.float32, requires_grad=True)
    edge_index = torch.randint(0, 18, (1, 24, 2), dtype=torch.long)
    edge_features = torch.randn(1, 24, 4, dtype=torch.float32)
    edge_mask = torch.ones(1, 24, dtype=torch.bool)
    node_mask = torch.ones(1, 18, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        y = block(x, edge_index, edge_features, edge_mask, node_mask)
        loss = y.float().square().mean()

    assert y.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
