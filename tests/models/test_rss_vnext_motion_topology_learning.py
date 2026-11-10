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
from models.rss_vnext.motion_aware_topology_learning import (
    RSSvNextMotionCourt,
    RSSvNextTopologyPreferenceConfig,
    topology_candidate_log_probs,
    source_relative_topology_preference_loss,
)
from models.rss_vnext.animation_ready_mesh import (
    posed_lbs_from_joint_matrices, deformation_bridge_loss,
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
        assert court.grammar_valid and court.nonmanifold_edge_count == 0
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
