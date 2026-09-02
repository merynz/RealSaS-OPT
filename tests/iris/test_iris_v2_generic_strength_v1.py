from __future__ import annotations

import math
import pytest
import torch
from torch import nn

from experiments.iris_reprojection_v2_20260831.depth_output_head_v2 import DepthOutputV2
from experiments.iris_reprojection_v2_20260831.dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    DINO_SOURCE_REVISION,
    DINO_S_WEIGHT_SHA256,
)
from experiments.iris_reprojection_v2_20260831.evidence_field_v2 import EvidenceFieldV2, EvidenceFieldOutputV2
from experiments.iris_reprojection_v2_20260831.eval_v2 import iris_v2_scientific_metrics
from experiments.iris_reprojection_v2_20260831.iris_apparatus_v2 import IrisDINOv2SApparatusV2
from experiments.iris_reprojection_v2_20260831.model_v2 import IrisReprojectionV2, IrisReprojectionOutputV2
from experiments.iris_reprojection_v2_20260831.q_descriptor_sampler_v2 import NativeResolutionPyramidV2
from experiments.iris_reprojection_v2_20260831.q_domain_v2 import RayHypothesisDomainV2, pixel_center_grid_v2
from experiments.iris_reprojection_v2_20260831.q_spatial_graph_v2 import build_anchor_neighbor_graph_v2
from experiments.iris_reprojection_v2_20260831.ray_modes_v2 import RayModesV2
from experiments.iris_reprojection_v2_20260831.train_v2 import _multitarget_mode_nll, train_step_production_v2
from experiments.iris_reprojection_v2_20260831.world_regularizer_v2 import isotropic_world_regularizer_v2


def test_exact_dino_foundation_authority_is_frozen_and_fail_closed_contract():
    authority = DINOv2FoundationAuthorityV2()
    authority.validate()
    contract = authority.feature_contract()
    assert authority.source_revision == DINO_SOURCE_REVISION == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    assert authority.constructor == "dinov2_vits14"
    assert authority.embed_dim == 384
    assert authority.patch_size == 14
    assert authority.input_hw == (518, 518)
    assert authority.patch_grid_hw == (37, 37)
    assert authority.weight_sha256 == DINO_S_WEIGHT_SHA256 == "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    assert contract.level_ids == ("x_norm_patchtokens",)
    assert contract.level_dims == (384,)
    assert "WHOLE1024_TO518" in contract.preprocessing_id
    assert "TORCHVISION_TF_RESIZE_BICUBIC_AA" in contract.preprocessing_id


def test_production_apparatus_binds_frozen_foundation_and_bounded_view_chunks():
    class FrozenFakeDINO(nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)

    learner = IrisReprojectionV2((384,), hidden_dim=24, max_modes=3)
    apparatus = IrisDINOv2SApparatusV2(learner, FrozenFakeDINO(), foundation_view_chunk=2)
    assert apparatus.runtime_seal.output_level == "x_norm_patchtokens"
    assert apparatus.runtime_seal.output_dim == 384
    assert apparatus.runtime_seal.patch_grid_hw == (37, 37)
    assert apparatus.runtime_seal.foundation_view_chunk == 2
    assert all(not p.requires_grad for p in apparatus.foundation.parameters())
    apparatus.train(True)
    assert apparatus.foundation.training is False
    assert tuple(apparatus.learner.sampler.foundation_dims) == (384,)


def test_production_train_entrypoint_rejects_arbitrary_foundation_injection_before_step():
    class DummyApparatus:
        runtime_seal = object()
        source_contract_hash = "sealed"

    with pytest.raises(ValueError, match="forbids caller-supplied foundation_maps"):
        train_step_production_v2(
            DummyApparatus(),
            optimizer=None,
            batch={"foundation_maps": (torch.zeros(1),)},
        )


def test_native_pyramid_preserves_preregistered_five_scale_capacity():
    torch.manual_seed(0)
    pyramid = NativeResolutionPyramidV2()
    x = torch.randn(1, 8, 4, 32, 32, requires_grad=True)
    levels = pyramid(x)
    assert pyramid.widths == (32, 48, 64, 96, 128)
    assert len(levels) == 5
    assert tuple(int(y.shape[2]) for y in levels) == pyramid.widths
    assert tuple(int(y.shape[-1]) for y in levels) == (32, 16, 8, 4, 2)
    sum(y.mean() for y in levels).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_evidence_field_has_ray_reasoning_and_sparse_spatial_message_passing():
    field = EvidenceFieldV2(hidden_dim=24, heads=4, ray_layers=2, spatial_neighbors=7)
    assert len(field.ray_reasoner.layers) == 2
    assert field.spatial_neighbors == 7
    assert hasattr(field, "spatial_message")
    assert hasattr(field, "spatial_score")
    assert not hasattr(field, "pointwise_only_decoder")


def _grid(xs, ys):
    return pixel_center_grid_v2(torch.tensor(xs), torch.tensor(ys), 1024).unsqueeze(0)


def test_sparse_anchor_graph_is_enumeration_equivariant_and_not_q_index_adjacency():
    g = _grid([100, 108, 100, 108], [100, 108, 108, 100])
    valid = torch.ones((1, 4), dtype=torch.bool)
    idx, mask = build_anchor_neighbor_graph_v2(g, valid, max_neighbors=8)
    assert mask.any()
    coordinates = g[0]
    neighbors = []
    for q in range(4):
        neighbors.append({tuple(coordinates[j].tolist()) for j in idx[0, q][mask[0, q]].tolist()})
    perm = torch.tensor([2, 0, 3, 1])
    gp = g[:, perm]
    ip, mp = build_anchor_neighbor_graph_v2(gp, valid, max_neighbors=8)
    perm_neighbors = []
    for q in range(4):
        perm_neighbors.append({tuple(gp[0, j].tolist()) for j in ip[0, q][mp[0, q]].tolist()})
    inverse = torch.empty_like(perm); inverse[perm] = torch.arange(4)
    for original_q in range(4):
        assert neighbors[original_q] == perm_neighbors[int(inverse[original_q])]


def test_multitarget_mode_objective_rewards_coverage_of_both_ray_intersections():
    depth_values = torch.linspace(0.0, 1.0, 9).view(1, 1, 9)
    teacher = torch.tensor([[[0.25, 0.75]]])
    valid = torch.ones_like(teacher, dtype=torch.bool)
    both = torch.full((1, 1, 9), -8.0); both[..., 2] = 5.0; both[..., 6] = 5.0
    one = torch.full((1, 1, 9), -8.0); one[..., 2] = 5.0
    assert _multitarget_mode_nll(both, depth_values, teacher, valid) < _multitarget_mode_nll(one, depth_values, teacher, valid)


def test_scientific_eval_reports_multimodal_tail_support_uncertainty_and_world_metrics():
    B, Q, D, K, H = 1, 2, 5, 2, 6
    depth_values = torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]]])
    q_points = torch.zeros(B, Q, D, 3)
    q_points[..., 0] = torch.tensor([0.0, 1.0]).view(1, Q, 1)
    q_points[..., 2] = depth_values
    candidate_valid = torch.ones(B, Q, D, dtype=torch.bool)
    domain = RayHypothesisDomainV2(
        anchor_view=torch.zeros(B, Q, dtype=torch.long),
        anchor_grid=torch.zeros(B, Q, 2),
        depth_values=depth_values,
        q_points=q_points,
        projected_grid=torch.zeros(B, Q, D, 8, 2),
        projected_depth=torch.zeros(B, Q, D, 8),
        in_frame=torch.ones(B, Q, D, 8, dtype=torch.bool),
        foreground_support=torch.ones(B, Q, D, 8, dtype=torch.bool),
        candidate_valid=candidate_valid,
        contract_hashes=("synthetic",),
    )
    field = EvidenceFieldOutputV2(
        score_logits=torch.tensor([[[0.0, 5.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 5.0, 0.0]]]),
        hidden=torch.zeros(B, Q, D, H),
        support_fraction=torch.ones(B, Q, D),
        neighbor_indices=torch.tensor([[[1], [0]]]),
        neighbor_mask=torch.ones(B, Q, 1, dtype=torch.bool),
    )
    modes = RayModesV2(
        mode_indices=torch.tensor([[[1, 3], [3, 1]]]),
        mode_scores=torch.ones(B, Q, K),
        mode_probabilities=torch.full((B, Q, K), 0.5),
        mode_valid=torch.ones(B, Q, K, dtype=torch.bool),
        ambiguous=torch.ones(B, Q, dtype=torch.bool),
    )
    refined = torch.tensor([[[0.25, 0.75], [0.75, 0.25]]])
    depth_output = DepthOutputV2(
        support_logits=torch.full((B, Q, K), 5.0),
        log_sigma=torch.full((B, Q, K), -2.0),
        support_probability=torch.full((B, Q, K), 0.95),
    )
    output = IrisReprojectionOutputV2(field, modes, refined, depth_output)
    teacher_depth = torch.tensor([[[0.25, 0.75], [0.25, 0.75]]])
    teacher_support = torch.ones_like(teacher_depth, dtype=torch.bool)
    metrics = iris_v2_scientific_metrics(output, domain, teacher_depth, teacher_support)
    for key in (
        "coverage_p95_abs",
        "coverage_tail_mean",
        "teacher_mode_recall_at_1p5_spacing",
        "support_f1",
        "uncertainty_mean_abs_z",
        "uncertainty_coverage_1sigma",
        "world_consistency_regularizer",
    ):
        assert key in metrics and math.isfinite(metrics[key])
    assert metrics["coverage_mae"] == 0.0
    assert metrics["teacher_mode_recall_at_1p5_spacing"] == 1.0


def test_world_regularizer_is_q_permutation_rotation_and_scale_invariant():
    torch.manual_seed(4)
    q = torch.randn(1, 12, 7, 3)
    logits = torch.randn(1, 12, 7)
    valid = torch.ones_like(logits, dtype=torch.bool)
    base = isotropic_world_regularizer_v2(q, logits, valid_mask=valid)
    perm = torch.randperm(q.shape[1])
    permuted = isotropic_world_regularizer_v2(q[:, perm], logits[:, perm], valid_mask=valid[:, perm])
    theta = 0.63
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    transformed = isotropic_world_regularizer_v2((q @ R.T) * 2.7, logits, valid_mask=valid)
    torch.testing.assert_close(base, permuted, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(base, transformed, atol=1e-5, rtol=1e-5)
