from __future__ import annotations

import math
import torch

from experiments.iris_reprojection_v2_20260831.dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    DINO_SOURCE_REVISION,
    DINO_S_WEIGHT_SHA256,
)
from experiments.iris_reprojection_v2_20260831.evidence_field_v2 import EvidenceFieldV2
from experiments.iris_reprojection_v2_20260831.q_descriptor_sampler_v2 import NativeResolutionPyramidV2
from experiments.iris_reprojection_v2_20260831.q_domain_v2 import pixel_center_grid_v2
from experiments.iris_reprojection_v2_20260831.q_spatial_graph_v2 import build_anchor_neighbor_graph_v2
from experiments.iris_reprojection_v2_20260831.train_v2 import _multitarget_mode_nll
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
    # Four rays on an 8-pixel lattice, deliberately enumerated in non-raster order.
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
