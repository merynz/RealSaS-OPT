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


def test_control_locus_is_multimodal_not_single_gaussian_contract():
    model = _model()
    out = _forward(model)
    assert model.config.position_modes == 3
    assert out.position_modes_normalized.shape == (1, 5, 3, 3)
    assert out.position_mode_log_sigma.shape == (1, 5, 3, 3)
    assert out.position_mode_logits.shape == (1, 5, 3)
    probability = torch.softmax(out.position_mode_logits, dim=-1)
    representative = (probability[..., None] * out.position_modes_normalized).sum(dim=2)
    torch.testing.assert_close(out.positions_normalized, representative)


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
