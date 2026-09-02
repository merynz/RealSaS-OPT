from __future__ import annotations

from dataclasses import replace
import torch

from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v2 import GeppettoCandidateConfigV2, GeppettoCandidateV2
from experiments.geppetto_arachne_r6_20260901.geppetto_loss_v2 import GeppettoLossV2


def _model() -> GeppettoCandidateV2:
    return GeppettoCandidateV2(GeppettoCandidateConfigV2(
        model_dim=24,
        knn_k=4,
        local_layers=1,
        global_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=48,
        support_topk=4,
        position_modes=3,
        parent_pair_chunk=2,
    ))


def _forward(model: GeppettoCandidateV2, n: int = 7, k: int = 5):
    torch.manual_seed(12)
    f = torch.randn(1, n, 24)
    p = torch.randn(1, n, 3).clamp(-1, 1)
    m = torch.ones((1, n), dtype=torch.bool)
    return model(f, p, m, decode_steps=k)


def test_control_locus_is_multimodal_and_primary_is_real_map_hypothesis():
    model = _model()
    out = _forward(model)
    assert model.config.position_modes == 3
    assert out.position_modes_normalized.shape == (1, 5, 3, 3)
    assert out.position_mode_log_sigma.shape == (1, 5, 3, 3)
    assert out.position_mode_logits.shape == (1, 5, 3)
    idx = torch.argmax(out.position_mode_logits, dim=-1)
    gather3 = idx[..., None, None].expand(1, 5, 1, 3)
    representative = torch.gather(out.position_modes_normalized, 2, gather3).squeeze(2)
    representative_sigma = torch.gather(out.position_mode_log_sigma, 2, gather3).squeeze(2)
    torch.testing.assert_close(out.positions_normalized, representative)
    torch.testing.assert_close(out.position_log_sigma, representative_sigma)


def test_map_representative_does_not_create_nonexistent_midpoint():
    modes = torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.4, 0.0]]])
    sigma = torch.zeros_like(modes)
    logits = torch.tensor([[5.0, 4.0, -3.0]])
    pos, _, idx = GeppettoCandidateV2._map_representative(modes, sigma, logits)
    assert int(idx.item()) == 0
    torch.testing.assert_close(pos, torch.tensor([[-1.0, 0.0, 0.0]]))
    mixture_mean = (torch.softmax(logits, dim=-1)[..., None] * modes).sum(dim=1)
    assert not torch.allclose(pos, mixture_mean)


def test_mixture_position_nll_can_explain_alternative_valid_locus():
    model = _model()
    out = _forward(model, n=6, k=3)
    target = torch.tensor([[-0.4, 0.1, 0.0], [0.0, 0.2, 0.1], [0.45, -0.1, 0.0]])
    q = torch.arange(3)
    exact_modes = out.position_modes_normalized.clone()
    exact_modes[0, q, 1] = target
    logits = torch.full_like(out.position_mode_logits, -4.0)
    logits[0, q, 1] = 4.0
    sigma = torch.full_like(out.position_mode_log_sigma, -3.0)
    exact = replace(out, position_modes_normalized=exact_modes, position_mode_logits=logits, position_mode_log_sigma=sigma)
    shifted = replace(exact, position_modes_normalized=exact_modes + 0.7)
    loss = GeppettoLossV2(support_topk=2)
    a = loss._mixture_position_nll(exact, 0, q, target)
    b = loss._mixture_position_nll(shifted, 0, q, target)
    assert a < b


def test_chunked_parent_relation_matches_materialized_reference():
    model = _model().eval()
    out = _forward(model, n=7, k=5)
    h = out.control_states
    pos = out.positions_normalized
    B, K, D = h.shape
    hp = h[:, :, None, :].expand(B, K, K, D)
    hc = h[:, None, :, :].expand(B, K, K, D)
    delta = pos[:, None, :, :] - pos[:, :, None, :]
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    reference = model.parent_pair(torch.cat([hp, hc, delta, dist], dim=-1)).squeeze(-1)
    reference = reference.masked_fill(torch.eye(K, dtype=torch.bool)[None], -1e4)
    torch.testing.assert_close(out.parent_logits, reference, atol=1e-6, rtol=1e-6)
