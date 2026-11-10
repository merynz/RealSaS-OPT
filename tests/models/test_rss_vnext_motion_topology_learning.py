from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.court_v2 import evaluate_decoded_topology_v2
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.motion_topology_candidates import propose_component_safe_edge_flips
from models.rss_vnext.motion_aware_topology_learning import (
    RSSvNextMotionCourt,
    RSSvNextTopologyPreferenceConfig,
    topology_candidate_log_probs,
    source_relative_topology_preference_loss,
)
from models.rss_vnext.animation_ready_mesh import (
    posed_lbs_from_joint_matrices, deformation_bridge_loss,
    source_pose_surface_consequence_loss, _area_weighted_surface_witnesses,
    _two_way_surface_chamfer,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(2026)
    torch.set_num_threads(1)


def examples(cfg):
    # Two triangulations of the same square, source-neutral toy provenance.
    # Synthetic preference values here test gradient wiring ONLY.
    verts = np.array([
        [-.25, -.25, 0], [.25, -.25, 0],
        [.25, .25, 0], [-.25, .25, 0],
    ], dtype=np.float64)
    seq_cfg = TESSAConfigV1(coordinate_bins=cfg.tessa.coordinate_bins)
    mesh_a = encode_adjacent_teacher_mesh_v1(
        verts, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        component_id="body", chart_id="body", cfg=seq_cfg,
    )
    mesh_b = encode_adjacent_teacher_mesh_v1(
        verts, np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64),
        component_id="body", chart_id="body", cfg=seq_cfg,
    )
    full = [mesh_a.token_ids, mesh_b.token_ids]
    max_len = max(map(len, full))
    tensor = torch.full((1, 2, max_len), cfg.tessa.PAD, dtype=torch.long)
    for i, seq in enumerate(full):
        tensor[0, i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        court = evaluate_decoded_topology_v2(seq, config=cfg.tessa)
        assert court.grammar_valid, court.decode_failure
        assert court.nonmanifold_edge_count == 0
        decoded = decode_tessa_asset_sequence_v1(seq, cfg=seq_cfg)
        assert len(decoded.faces) == 2
    return tensor


def evidence(tokens, device="cpu", *, static=(0.02, 0.025), dynamic=(0.2, 0.8),
             valid=(True, True), bad_hash=False, bad_motion_suite=False):
    hashes = []
    for i in range(tokens.shape[1]):
        hashes.append(hashlib.sha256(tokens[0, i].cpu().numpy().tobytes()).hexdigest())
    if bad_hash:
        hashes[0] = "tampered!"
    return RSSvNextMotionCourt(
        source_sha256="fake-synthetic-source-sha-for-unit-tests",
        motion_suite_hash="synthetic-same-pose-suite",
        canonical_binding_hash="synthetic-canonical-binding",
        source_skin_protocol_hash="synthetic-same-teacher-kinematics-and-weight-transfer",
        evaluation_stage="PRE_DYNAMIC_REFINEMENT_STRUCTURAL_MESH",
        candidate_hashes=tuple(hashes),
        reference_authority="QUALIFIED_SOURCE_RELATIVE_MOTION_COURT",
        static_cost=torch.tensor(static, dtype=torch.float32, device=device),
        dynamic_cost=torch.tensor(dynamic, dtype=torch.float32, device=device),
        valid_mask=torch.tensor(valid, dtype=torch.bool, device=device),
        candidate_source_sha256=("fake-synthetic-source-sha-for-unit-tests",) * 2,
        candidate_motion_suite_hash=(
            "mismatched" if bad_motion_suite else "synthetic-same-pose-suite",
            "synthetic-same-pose-suite",
        ),
        candidate_canonical_binding_hash=("synthetic-canonical-binding",) * 2,
        candidate_source_skin_protocol_hash=(
            "synthetic-same-teacher-kinematics-and-weight-transfer",
        ) * 2,
    )


def test_motion_preference_directly_teaches_topology_probability():
    cfg = RSSvNextConfig.tiny_for_test()
    tokens = examples(cfg)
    logp = torch.tensor([-0.5, -0.4], requires_grad=True)
    out = source_relative_topology_preference_loss(logp, tokens, evidence(tokens))
    assert out.preferred_pairs == ((0, 1),)
    out.loss.backward()
    assert logp.grad[0] < 0  # minimizing increases preference of motion winner
    assert logp.grad[1] > 0
    assert out.authority == "TRAINING_SOURCE_RELATIVE_PREFERENCE_ONLY"


def test_source_binding_and_comparable_geometry_are_mandatory():
    cfg = RSSvNextConfig.tiny_for_test()
    tokens = examples(cfg)
    scores = torch.tensor([-1.5, -1.0], requires_grad=True)
    with pytest.raises(ValueError, match="TOKEN_HASH_DRIFT"):
        source_relative_topology_preference_loss(scores, tokens, evidence(tokens, bad_hash=True))
    with pytest.raises(ValueError, match="MOTION_CANDIDATE_PROVENANCE_DRIFT"):
        source_relative_topology_preference_loss(scores, tokens, evidence(tokens, bad_motion_suite=True))
    with pytest.raises(ValueError, match="POSTHOC_REPAIR_REWARD_FORBIDDEN"):
        source_relative_topology_preference_loss(
            scores, tokens,
            replace(evidence(tokens), evaluation_stage="POST_DYNAMIC_REFINEMENT"),
        )
    with pytest.raises(ValueError, match="NO_QUALIFIED"):
        source_relative_topology_preference_loss(
            scores, tokens, evidence(tokens, static=(0.02, 0.9)),
        )
    with pytest.raises(ValueError, match="NO_QUALIFIED"):
        source_relative_topology_preference_loss(
            scores, tokens, evidence(tokens, valid=(True, False)),
        )


def test_same_two_view_carrier_receives_motion_preference_gradient():
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    tokens = examples(cfg)
    rgba = torch.rand(1, 2, 4, 8, 8)
    cameras = torch.zeros(1, 2, 13)
    cameras[:, :, 3] = 1
    cameras[:, :, 7] = 1
    cameras[:, :, 11] = 1
    cameras[:, :, 12] = 1
    out = net.motion_aware_structural_objective(
        rgba, cameras, candidate_tokens=tokens,
        qualified_motion_court=evidence(tokens),
    )
    assert out.candidate_log_prob.shape == (2,)
    assert out.pair_count == 1
    out.loss.backward()
    assert net.tessa.lm_head.weight.grad is not None
    assert torch.isfinite(net.tessa.lm_head.weight.grad).all()
    assert net.articulation_prior[-1].weight.grad is not None
    assert net.articulation_prior[-1].weight.grad.abs().sum() > 0
    assert net.affordance_to_memory.weight.grad is not None
    assert net.affordance_to_memory.weight.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.vision.patch.weight.grad is not None


def test_grammar_does_not_confuse_decoded_topology_with_dynamic_truth():
    cfg = RSSvNextConfig.tiny_for_test()
    tokens = examples(cfg)
    decoded = decode_tessa_asset_sequence_v1(
        tokens[0, 0][tokens[0, 0] != cfg.tessa.PAD].tolist(),
        cfg=TESSAConfigV1(coordinate_bins=cfg.tessa.coordinate_bins),
    )
    # Valid static topology by itself does NOT imply this passes dynamic G3.
    assert decoded.orientation_qualified is False
    assert len(decoded.faces) == 2



def test_deformation_source_pose_loss_reaches_shared_mesh_and_skin():
    """A moving teacher surface must penalize bad student skin, not merely static XYZ."""
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = 1
    cam[:, :, 7] = 1
    cam[:, :, 11] = 1
    cam[:, :, 12] = 1
    state, _, _ = net.encode_observations(rgba, cam)
    seq = examples(cfg)[0, 0]
    seq = seq[seq != cfg.tessa.PAD]
    mesh = net.build_mesh_candidate(
        state, seq, source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    mesh.validate()
    assert mesh.faces.shape == (2, 3)
    n = mesh.rest_vertices_iris.shape[1]
    logits = torch.zeros((1, n, 2), requires_grad=True)
    weights = torch.softmax(logits, dim=-1)
    mats = torch.eye(4)[None, None, None].repeat(1, 2, 2, 1, 1)
    mats[0, 1, 0, 0, 3] = 0.25
    mats[0, 1, 1, 1, 3] = -0.25

    # A *different* native rig's influence distribution produces a different
    # posed surface. This stays teacher-only and outside image encoder input.
    gt_weights = weights.detach().clone()
    gt_weights[0, :, 0] = 0.88
    gt_weights[0, :, 1] = 0.12
    gt_posed = posed_lbs_from_joint_matrices(
        mesh.rest_vertices_iris.detach(), gt_weights, mats,
    )
    source_rest = _area_weighted_surface_witnesses(
        mesh.rest_vertices_iris.detach(), mesh.faces,
    ).detach()
    source_poses = torch.stack([
        _area_weighted_surface_witnesses(gt_posed[:, p].detach(), mesh.faces)
        for p in range(2)
    ])
    result = source_pose_surface_consequence_loss(
        mesh, weights, mats, source_rest, source_poses,
        valid_pose_mask=torch.tensor([True, True]),
    )
    assert float(result["posed_surface"]) > 0.0
    result["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None
    assert torch.isfinite(net.iris.scene.scene_queries.grad).all()
    assert net.surface_refine[-1].weight.grad is not None


def test_pose_loss_refuses_missing_motion_supervision():
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = cam[:, :, 7] = cam[:, :, 11] = cam[:, :, 12] = 1
    state, _, _ = net.encode_observations(rgba, cam)
    seq = examples(cfg)[0, 0]
    mesh = net.build_mesh_candidate(
        state, seq[seq != cfg.tessa.PAD],
        source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    weights = torch.full((1, mesh.rest_vertices_iris.shape[1], 2), 0.5)
    mats = torch.eye(4)[None, None, None].repeat(1, 2, 2, 1, 1)
    rest = _area_weighted_surface_witnesses(mesh.rest_vertices_iris.detach(), mesh.faces)
    source_poses = torch.stack([rest, rest])
    with pytest.raises(ValueError, match="MASK_INVALID"):
        source_pose_surface_consequence_loss(
            mesh, weights, mats, rest, source_poses,
            valid_pose_mask=torch.tensor([False, False]),
        )



def test_two_static_equivalent_topologies_have_distinct_posed_surfaces():
    """Actual geometric consequence, not a fabricated reward score.

    At rest, the two meshes tessellate the same flat square. Under a joint
    movement that lifts exactly one corner, their interpolated interiors differ.
    That difference is observable by deterministic surface quadrature and can
    be fed to the discrete topology preference objective.
    """
    rest = torch.tensor([
        [-.5, -.5, 0.0],
        [.5, -.5, 0.0],
        [.5, .5, 0.0],
        [-.5, .5, 0.0],
    ], dtype=torch.float32)[None]
    face_a = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    face_b = torch.tensor([[0, 1, 3], [1, 2, 3]], dtype=torch.long)
    weights = torch.tensor([[
        [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 0.0],
    ]])
    source_joint_matrices = torch.eye(4)[None, None, None].repeat(1, 1, 2, 1, 1)
    source_joint_matrices[0, 0, 1, 2, 3] = 0.8
    posed = posed_lbs_from_joint_matrices(rest, weights, source_joint_matrices)[:, 0]
    assert torch.allclose(posed[0, 2, 2], torch.tensor(.8))
    source_surface = _area_weighted_surface_witnesses(posed, face_a)
    candidate_a = _area_weighted_surface_witnesses(posed, face_a)
    candidate_b = _area_weighted_surface_witnesses(posed, face_b)
    cost_a = _two_way_surface_chamfer(candidate_a, source_surface)
    cost_b = _two_way_surface_chamfer(candidate_b, source_surface)
    assert float(cost_a.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(cost_b.detach()) > 0.05
    cfg = RSSvNextConfig.tiny_for_test()
    tokens = examples(cfg)
    court = evidence(tokens, dynamic=(float(cost_a.detach()), float(cost_b.detach())))
    logp = torch.tensor([-1.0, -1.0], requires_grad=True)
    out = source_relative_topology_preference_loss(logp, tokens, court)
    out.loss.backward()
    assert out.preferred_pairs == ((0, 1),)
    assert logp.grad[0] < 0 and logp.grad[1] > 0



def test_safe_edge_flip_generates_real_alternative_face_connectivity():
    rest = torch.tensor([
        [-.5, -.5, 0.0], [.5, -.5, 0.0],
        [.5, .5, 0.0], [-.5, .5, 0.0],
    ])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    comp = torch.zeros(4, dtype=torch.long)
    candidates = propose_component_safe_edge_flips(
        rest, faces, comp, protected_edges=frozenset(),
        protection_court_identity="toy-source-no-protected-edge",
    )
    assert len(candidates) == 1
    alt = candidates[0]
    assert alt.removed_edge == (0, 2)
    assert alt.new_edge == (1, 3)
    assert alt.faces.shape == faces.shape
    assert alt.authority == "UNQUALIFIED_COUNTERFACTUAL_TRAINING_CANDIDATE"
    # Face indices remain in exactly the same vertex array and component.
    assert alt.faces.max() == 3 and alt.faces.min() == 0
    # Under a single-joint corner lift, the counterfactual has a different
    # posed triangle surface. This can be ranked by source motion consequence.
    lifted = rest.clone()
    lifted[2, 2] = .8
    source = _area_weighted_surface_witnesses(lifted[None], faces)
    alternative = _area_weighted_surface_witnesses(lifted[None], alt.faces)
    assert float(_two_way_surface_chamfer(alternative, source)) > .05


def test_edge_flip_abstains_across_protected_seam_or_nonplanar_hinge():
    rest = torch.tensor([
        [-.5, -.5, 0.0], [.5, -.5, 0.0],
        [.5, .5, 0.0], [-.5, .5, 0.0],
    ])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    comp = torch.zeros(4, dtype=torch.long)
    assert not propose_component_safe_edge_flips(
        rest, faces, comp, protected_edges=frozenset({(0, 2)}),
        protection_court_identity="toy-protected-mechanical-seam",
    )
    altered = rest.clone()
    altered[3, 2] = .9
    assert not propose_component_safe_edge_flips(
        altered, faces, comp, protected_edges=frozenset(),
        protection_court_identity="toy-nonplanar-hinge-no-safe-flip",
    )
    with pytest.raises(ValueError, match="PROTECTION_COURT_REQUIRED"):
        propose_component_safe_edge_flips(
            rest, faces, comp, protected_edges=frozenset(),
            protection_court_identity="",
        )
