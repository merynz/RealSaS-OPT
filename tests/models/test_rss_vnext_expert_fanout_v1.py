from __future__ import annotations

"""Single unified RSS checkpoint shared carrier -> typed expert consumers.

Uses the real RSSvNextUnifiedCharacterModel, not the deprecated independent
IRIS V3 inference worker. The model is tiny and untrained: these are E2E
wiring / no-hallucinated-authority tests, NOT unseen model qualification.
"""

from dataclasses import replace

import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_expert_fanout_v1 import (
    GRAPHICS_CHANNELS, REGION_CHANNELS,
    open_rss_vnext_expert_fanout_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.inference_evidence_export_v1 import export_inference_evidence_v1
from models.rss_vnext.neural_scene_archive_v1 import (
    encode_neural_scene_carrier_v1,
)
from models.rss_vnext.training_engine import architecture_digest


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(151)
    torch.set_num_threads(1)


def _unified_abstaining_evidence(*, corrupt_symmetric_relation: bool = False):
    cfg = RSSvNextConfig.tiny_for_test()
    model = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).eval()
    source_rgba = torch.rand(1, 2, 4, 8, 8)
    cameras = torch.zeros(1, 2, 13)
    cameras[:, :, 3] = cameras[:, :, 7] = cameras[:, :, 11] = cameras[:, :, 12] = 1.
    with torch.no_grad():
        state, _, _ = model.encode_observations(source_rgba, cameras)
    if corrupt_symmetric_relation:
        relation = state.region_relation_logits.clone()
        relation[0, 0, 1, 0] += 0.125
        state = replace(state, region_relation_logits=relation)
    result = {
        "state": state,
        "mesh_status": "ABSTAIN",
        "mesh_candidate": None,
        "mesh_failure": "SYNTHETIC_FREE_RUNNING_TOPOLOGY_INCOMPLETE",
        "observed_view_count": 2,
        "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
    }
    evidence = export_inference_evidence_v1(
        result,
        original_rgba=source_rgba,
        original_camera_features=cameras,
        checkpoint_sha256="a" * 64,
    )
    carrier = encode_neural_scene_carrier_v1(state, evidence=evidence)
    return cfg, state, evidence, carrier


def _open(cfg, evidence, carrier, *, expected_config_digest=None):
    return open_rss_vnext_expert_fanout_v1(
        evidence, carrier, model_config=cfg,
        expected_checkpoint_sha256="a" * 64,
        expected_model_config_sha256=(
            expected_config_digest or architecture_digest(cfg)
        ),
    )


def test_four_rss_specialists_share_exact_h_even_when_tessa_abstains():
    cfg, state, evidence, carrier = _unified_abstaining_evidence()
    fanout = _open(cfg, evidence, carrier)

    assert fanout.availability.decoded_mesh_candidate == "ABSTAIN_NO_TOPOLOGY"
    assert fanout.availability.causal_rig_program_candidate == "NO_TYPED_EVENTS"
    assert fanout.availability.semantic_and_presentation == "TYPED_NEURAL_CANDIDATE"
    assert fanout.availability.relational_mechanics == (
        "TYPED_AFFORDANCE_AND_DIRECTED_SLOT_RELATIONS"
    )
    assert fanout.availability.canonical_mesh == "NOT_QUALIFIED"
    assert fanout.availability.canonical_rig_skin == "NOT_QUALIFIED"
    assert fanout.availability.pose_conditioned_source_visibility == "NOT_QUALIFIED"
    assert fanout.witness_count == cfg.query_grid_side ** 3
    assert fanout.region_slot_count == cfg.region_slots

    for index in range(fanout.witness_count):
        row = fanout.query(index)
        # The same scene H(x) simultaneously conditions geometry, region
        # semantics, presentation, graphics and mechanical affordances.
        assert row.local_h == tuple(
            state.local_features[0, index].float().tolist()
        )
        assert row.xyz == tuple(state.query_xyz[0, index].float().tolist())
        assert row.region_membership_logits == tuple(
            state.semantic_logits[0, index].float().tolist()
        )
        assert row.presentation_role_logits == tuple(
            state.presentation_role_logits[0, index].float().tolist()
        )
        assert row.graphics_field_logits == tuple(
            state.graphics_logits[0, index].float().tolist()
        )
        assert row.deformation_affordance_logits == tuple(
            state.affordance_logits[0, index].float().tolist()
        )
        geometry = dict(row.geometry_field_values)
        assert geometry["H"] == row.local_h
        assert "unsigned_distance" in geometry
        assert "sign_logit" in geometry
        assert len(row.graphics_field_logits) == len(GRAPHICS_CHANNELS) == 5
        assert len(row.region_membership_probabilities) == cfg.region_slots + 1
        assert sum(row.region_membership_probabilities) == pytest.approx(1., abs=1e-6)
        assert row.region_membership_probabilities[-1] > 0  # UNKNOWN retained
        assert len(row.presentation_role_probabilities) == len(cfg.semantic_role_names)
        assert row.canonical_vertex_id is None
        assert not row.source_rgb_qualification
        assert not row.source_visibility_qualification
        assert not row.executable_mechanics_qualification

    for view_index in (0, 1):
        patch = fanout.original_source_patch(view_index, 0)
        assert patch.observed_view_index == view_index
        assert patch.semantic_evidence_logits == tuple(
            state.source_semantic_patch_logits[0, view_index, 0].float().tolist()
        )
        assert not patch.is_original_source_label
    with pytest.raises(QualificationError, match="SOURCE_VIEW_NOT_OBSERVED"):
        fanout.original_source_patch(2, 0)
    with pytest.raises(QualificationError, match="QUERY_ADDRESS_INVALID"):
        fanout.query(fanout.witness_count)


def test_attachment_head_is_directed_and_never_fabricates_anchor_or_joint():
    cfg, state, evidence, carrier = _unified_abstaining_evidence()
    fanout = _open(cfg, evidence, carrier)
    forward = fanout.region_relation(0, 1)
    reverse = fanout.region_relation(1, 0)

    assert forward.relation_channel_names == REGION_CHANNELS
    assert forward.raw_logits == tuple(
        state.region_relation_logits[0, 0, 1].float().tolist()
    )
    assert reverse.raw_logits == tuple(
        state.region_relation_logits[0, 1, 0].float().tolist()
    )
    assert forward.raw_logits[:2] == reverse.raw_logits[:2]
    assert forward.directed_attachment_affinity == pytest.approx(
        torch.sigmoid(state.region_relation_logits[0, 0, 1, 2]).item()
    )
    assert reverse.directed_attachment_affinity == pytest.approx(
        torch.sigmoid(state.region_relation_logits[0, 1, 0, 2]).item()
    )
    assert forward.attachment_parent_joint_id is None
    assert forward.attachment_anchor_xyz is None
    assert forward.source_contact_proven is False
    assert forward.independent_drawable_proven is False
    assert fanout.causal_control_events() == ()
    with pytest.raises(QualificationError, match="RELATION_SELF_EDGE_UNOBSERVED"):
        fanout.region_relation(0, 0)
    # Latent attachment proposals can survive topology ABSTAIN, but cannot
    # become a RigProgram or explicit mechanical parent without AXIS events.
    with pytest.raises(QualificationError, match="RIG_REQUIRES_VALID_NEURAL_MESH"):
        fanout.qualify_causal_rig_proposal(mira_codec_binding=None)


def test_shared_carrier_cannot_hide_asymmetric_same_drawable_or_cut_logits():
    cfg, _, evidence, carrier = _unified_abstaining_evidence(
        corrupt_symmetric_relation=True,
    )
    # JSON evidence can be re-sealed with bad-but-finite relations. This
    # stronger *cross-expert* court checks actual RSS architecture invariants.
    with pytest.raises(QualificationError, match="SYMMETRIC_REGION_CHANNEL_DRIFT"):
        _open(cfg, evidence, carrier)


def test_fanout_refuses_wrong_checkpoint_and_wrong_expert_config():
    cfg, state, evidence, carrier = _unified_abstaining_evidence()
    with pytest.raises(QualificationError, match="CHECKPOINT_BINDING_DRIFT"):
        open_rss_vnext_expert_fanout_v1(
            evidence, carrier, model_config=cfg,
            expected_checkpoint_sha256="b" * 64,
            expected_model_config_sha256=architecture_digest(cfg),
        )
    with pytest.raises(QualificationError, match="MODEL_EXPERT_SHAPE_DRIFT"):
        _open(replace(cfg, affordance_classes=cfg.affordance_classes + 1), evidence, carrier)
    # Same tensor widths are NOT enough: checkpoint architecture config is
    # pinned in the source-bound receipt independently of values/shapes.
    role_names = tuple(reversed(cfg.semantic_role_names))
    with pytest.raises(QualificationError, match="MODEL_CONFIG_BINDING_DRIFT"):
        _open(replace(cfg, semantic_role_names=role_names), evidence, carrier,
              expected_config_digest=architecture_digest(cfg))
    with pytest.raises(QualificationError, match="FROM_SLOT_INVALID"):
        _open(cfg, evidence, carrier).region_relation(-1, 1)



def test_compiler_batched_query_relations_match_training_marginal_not_argmax_slots():
    cfg, state, evidence, carrier = _unified_abstaining_evidence()
    fanout = _open(cfg, evidence, carrier)
    assert evidence.mesh_status == "ABSTAIN"  # not a prerequisite for shared H
    pairs = [(0, 1), (1, 0), (2, 3), (6, 7)]
    rows = fanout.query_pair_relations(pairs)
    assert len(rows) == len(pairs)
    assert fanout.query_pair_relation(0, 1) == rows[0]
    assert fanout.query_pair_relations([]) == ()

    # The model's qualified_source_typed_relation_loss must predict exactly
    # the same query-addressed marginal that Compiler now consumes.
    # Using argmax slot assignments would silently discard relation evidence.
    known = torch.softmax(state.semantic_logits[0, :, :-1].float(), dim=-1)
    full = torch.softmax(state.semantic_logits[0].float(), dim=-1)
    scored = torch.sigmoid(state.region_relation_logits[0].float())
    valid = state.region_relation_valid_mask[0].to(scored.dtype)
    for ((i, j), row) in zip(pairs, rows):
        cross = known[i, :, None] * known[j, None, :] * valid
        expected = (cross[..., None] * scored).sum(dim=(0, 1)) / cross.sum()
        assert row.same_drawable_affinity == pytest.approx(expected[0].item(), abs=2.e-6)
        assert row.cut_affinity == pytest.approx(expected[1].item(), abs=2.e-6)
        assert row.directed_attachment_affinity == pytest.approx(expected[2].item(), abs=2.e-6)
        assert row.nonself_slot_pair_mass == pytest.approx(cross.sum().item(), abs=2.e-6)
        assert row.parent_predicted_unknown_mass == pytest.approx(full[i, -1].item(), abs=2.e-6)
        assert row.child_predicted_unknown_mass == pytest.approx(full[j, -1].item(), abs=2.e-6)
        assert row.parent_predicted_known_region_mass + row.parent_predicted_unknown_mass == pytest.approx(1., abs=2.e-6)
        assert row.inference_evidence_hash == evidence.evidence_hash
        assert row.neural_carrier_sha256 == fanout.neural_carrier_sha256
        assert not any((
            row.source_contact_proven, row.attachment_anchor_proven,
            row.canonical_face_rebinding_proven, row.source_drawable_proven,
            row.source_visibility_proven,
        ))
        assert row.authority == "NEURAL_EVIDENCE_ONLY_NO_PRODUCT_AUTHORITY"

    # Graph channels 0/1 are symmetric by model architecture; attachment
    # channel 2 is a separate ORDERED calculation, never symmetrized.
    assert rows[0].same_drawable_affinity == pytest.approx(rows[1].same_drawable_affinity, abs=1.e-6)
    assert rows[0].cut_affinity == pytest.approx(rows[1].cut_affinity, abs=1.e-6)
    rev = fanout.query_pair_relations(list(reversed(pairs)))
    assert rev == tuple(reversed(rows))


def test_compiler_relation_batch_enforces_addresses_and_resource_budget_without_carrier_reinfer():
    cfg, _, evidence, carrier = _unified_abstaining_evidence()
    fanout = _open(cfg, evidence, carrier)
    with pytest.raises(QualificationError, match="QUERY_PAIR_ADDRESS_INVALID"):
        fanout.query_pair_relations([(0, fanout.witness_count)])
    with pytest.raises(QualificationError, match="QUERY_PAIR_ADDRESS_INVALID"):
        fanout.query_pair_relations([(True, 1)])
    with pytest.raises(QualificationError, match="QUERY_PAIR_SELF_EDGE_FORBIDDEN"):
        fanout.query_pair_relations([(0, 0)])
    with pytest.raises(QualificationError, match="PAIR_BATCH_TYPE_INVALID"):
        fanout.query_pair_relations("0:1")
    with pytest.raises(QualificationError, match="PAIR_BATCH_RESOURCE_LIMIT"):
        fanout.query_pair_relations(
            [(0, 1)] * (cfg.max_pair_queries + 1)
        )
    assert len(fanout.query_pair_relations(
        [(0, 1)] * cfg.max_pair_queries,
    )) == cfg.max_pair_queries
