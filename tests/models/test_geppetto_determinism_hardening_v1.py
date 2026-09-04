from __future__ import annotations

import torch

from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateV2, _LocalGeometryAttention


def test_equal_distance_knn_uses_stable_canonical_surface_row_order() -> None:
    layer = _LocalGeometryAttention(dim=4, k=2, dropout=0.0)
    h = torch.zeros((1, 3, 4))
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    captured: list[torch.Tensor] = []
    original = layer._gather

    def recording_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        captured.append(idx.detach().cpu())
        return original(x, idx)

    layer._gather = recording_gather  # type: ignore[method-assign]
    layer(h, xyz, mask)
    # First gather is the hidden-neighbor gather. For the center row, self is
    # distance 0 and rows 1/2 are exactly tied at distance 1; stable canonical
    # row order must choose row 1 as the second neighbor.
    assert captured
    assert captured[0][0, 0].tolist() == [0, 1]


def test_equal_map_logits_use_explicit_stable_mode_slot_policy() -> None:
    modes = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    log_sigma = torch.zeros_like(modes)
    logits = torch.tensor([[4.0, 4.0, 3.0]])
    pos, sigma, idx = GeppettoCandidateV2._map_representative(modes, log_sigma, logits)
    assert idx.tolist() == [0]
    torch.testing.assert_close(pos, modes[:, 0])
    torch.testing.assert_close(sigma, log_sigma[:, 0])


def test_candidate_source_contains_no_backend_undefined_topk_or_bare_map_argmax() -> None:
    import inspect

    source = inspect.getsource(_LocalGeometryAttention.forward)
    assert "torch.topk" not in source
    assert "stable=True" in source
    map_source = inspect.getsource(GeppettoCandidateV2._map_representative)
    assert "torch.argmax" not in map_source
    assert "stable=True" in map_source
