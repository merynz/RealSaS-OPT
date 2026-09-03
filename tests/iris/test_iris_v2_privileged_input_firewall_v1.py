from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
from torch import nn

from experiments.iris_reprojection_v2_20260831.iris_apparatus_v2 import IrisDINOv2SApparatusV2
from experiments.iris_reprojection_v2_20260831.model_v2 import IrisReprojectionV2
from experiments.iris_reprojection_v2_20260831.observation_contract_v2 import OrthographicCameraV2, ObservationContractV2
from experiments.iris_reprojection_v2_20260831.q_descriptor_sampler_v2 import (
    NativeResolutionPyramidV2,
    SampledQDescriptorsV2,
)
from experiments.iris_reprojection_v2_20260831.q_domain_v2 import (
    MASK_DIAGNOSTIC_Q_DOMAIN_AUTHORITY_V2,
    PRODUCTION_Q_DOMAIN_AUTHORITY_V2,
    QCandidatePolicyV2,
    RayHypothesisDomainV2,
    build_production_observation_ray_lattice_v2,
    build_q_domain_v2,
    validate_production_observation_domain_v2,
)
from experiments.iris_reprojection_v2_20260831.q_evidence_encoder_v2 import QEvidenceEncoderV2


def _camera(v: int, half: float = 1.5) -> OrthographicCameraV2:
    a = 2.0 * math.pi * v / 8.0
    return OrthographicCameraV2(
        v,
        (0.0, 0.0, 0.0),
        (math.cos(a), 0.0, -math.sin(a)),
        (0.0, 1.0, 0.0),
        (-math.sin(a), 0.0, -math.cos(a)),
        half,
        1024,
    )


def _contract() -> ObservationContractV2:
    return ObservationContractV2(
        "privileged-firewall-synthetic",
        tuple(_camera(v) for v in range(8)),
        tuple(f"r{v}" for v in range(8)),
    )


def _unsealed_domain() -> RayHypothesisDomainV2:
    c = _contract()
    av = torch.zeros((1, 2), dtype=torch.long)
    ag = torch.tensor([[[-0.2, 0.0], [0.2, 0.0]]], dtype=torch.float32)
    dv = torch.linspace(0.2, 0.8, 4)
    return build_q_domain_v2((c,), av, ag, dv)


def test_alpha_is_not_causally_reachable_by_learned_native_path():
    torch.manual_seed(20260903)
    pyramid = NativeResolutionPyramidV2().eval()
    assert pyramid.INPUT_CHANNELS == 3

    rgb = torch.randn(1, 8, 3, 32, 32)
    rgba0 = torch.cat([rgb, torch.zeros(1, 8, 1, 32, 32)], dim=2)
    rgba1 = torch.cat([rgb, torch.ones(1, 8, 1, 32, 32)], dim=2)

    with pytest.raises(ValueError, match="RGB"):
        pyramid(rgba0)

    learner0 = IrisDINOv2SApparatusV2.learner_rgb_from_rgba(rgba0)
    learner1 = IrisDINOv2SApparatusV2.learner_rgb_from_rgba(rgba1)
    assert torch.equal(learner0, learner1)

    with torch.no_grad():
        f0 = pyramid(learner0)
        f1 = pyramid(learner1)
    assert all(torch.equal(a, b) for a, b in zip(f0, f1))


def test_view_reenumeration_is_equivariant_without_absolute_slot_identity():
    torch.manual_seed(20260904)
    B = Q = D = 1
    V, C = 8, 7
    descriptors = torch.randn(B, Q, D, V, C)
    valid = torch.ones(B, Q, D, V, dtype=torch.bool)
    projected_grid = torch.randn(B, Q, D, V, 2).clamp(-0.8, 0.8)
    projected_depth = torch.linspace(0.2, 0.9, V).view(B, Q, D, V)
    domain = RayHypothesisDomainV2(
        anchor_view=torch.zeros(B, Q, dtype=torch.long),
        anchor_grid=torch.zeros(B, Q, 2),
        depth_values=torch.ones(B, Q, D),
        q_points=torch.zeros(B, Q, D, 3),
        projected_grid=projected_grid,
        projected_depth=projected_depth,
        in_frame=valid,
        foreground_support=valid,
        candidate_valid=torch.ones(B, Q, D, dtype=torch.bool),
        contract_hashes=("synthetic",),
    )
    enc = QEvidenceEncoderV2(C, hidden_dim=24, heads=4, layers=1).eval()
    perm = torch.tensor([3, 4, 5, 6, 7, 0, 1, 2])
    moved_domain = replace(
        domain,
        projected_grid=domain.projected_grid[..., perm, :],
        projected_depth=domain.projected_depth[..., perm],
        in_frame=domain.in_frame[..., perm],
        foreground_support=domain.foreground_support[..., perm],
    )
    with torch.no_grad():
        base = enc(SampledQDescriptorsV2(descriptors, valid, C), domain)
        moved = enc(
            SampledQDescriptorsV2(descriptors[..., perm, :], valid[..., perm], C),
            moved_domain,
        )

    torch.testing.assert_close(base.pooled, moved.pooled, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base.view_weights[..., perm], moved.view_weights, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base.view_tokens[..., perm, :], moved.view_tokens, atol=1e-6, rtol=1e-6)


def test_production_q_domain_is_camera_only_full_frame_and_forgery_fails_closed():
    c = _contract()
    depth_values = torch.tensor([0.25, 0.75], dtype=torch.float32)
    domain = build_production_observation_ray_lattice_v2(
        (c,),
        anchor_view_index=0,
        anchor_stride_px=512,
        depth_values=depth_values,
    )
    assert domain.construction_authority == PRODUCTION_Q_DOMAIN_AUTHORITY_V2
    assert domain.candidate_policy is None
    assert torch.equal(domain.foreground_support, domain.in_frame)
    assert torch.equal(domain.candidate_valid, domain.in_frame.any(dim=-1))
    validate_production_observation_domain_v2(domain)

    pruned = replace(
        domain,
        anchor_view=domain.anchor_view[:, :-1],
        anchor_grid=domain.anchor_grid[:, :-1],
    )
    with pytest.raises(ValueError, match="full-frame"):
        validate_production_observation_domain_v2(pruned)

    masks = torch.ones((1, 8, 32, 32), dtype=torch.bool)
    masks[:, 7] = False
    policy = QCandidatePolicyV2(minimum_foreground_support_views=2)
    masked = build_q_domain_v2(
        (c,),
        torch.zeros((1, 2), dtype=torch.long),
        torch.tensor([[[-0.2, 0.0], [0.2, 0.0]]], dtype=torch.float32),
        depth_values,
        foreground_masks=masks,
        candidate_policy=policy,
    )
    assert masked.construction_authority == MASK_DIAGNOSTIC_Q_DOMAIN_AUTHORITY_V2
    with pytest.raises(ValueError, match="forbids unsealed/mask Q domain"):
        validate_production_observation_domain_v2(masked)

    forged = replace(
        masked,
        construction_authority=PRODUCTION_Q_DOMAIN_AUTHORITY_V2,
        candidate_policy=None,
        foreground_support=masked.in_frame,
        candidate_valid=masked.in_frame.any(dim=-1),
        anchor_stride_px=512,
    )
    with pytest.raises(ValueError):
        validate_production_observation_domain_v2(forged)


def test_apparatus_rejects_unsealed_domain_before_observation_or_foundation_execution():
    class FrozenFakeDINO(nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)

    apparatus = IrisDINOv2SApparatusV2(
        IrisReprojectionV2((384,), hidden_dim=24, max_modes=1),
        FrozenFakeDINO(),
    )
    # Deliberately malformed image tensor: the expected error must still be the
    # Q-domain firewall, proving domain rejection precedes image/foundation work.
    with pytest.raises(ValueError, match="forbids unsealed/mask Q domain"):
        apparatus(torch.zeros(1), _unsealed_domain())
