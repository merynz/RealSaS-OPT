from __future__ import annotations

"""Synthetic contract court only: no checkpoint quality or product proof claim."""

from dataclasses import asdict, replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR, CanonicalMeshVertexCandidateIR,
    canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.rss_vnext_canonical_rig_lowering_v1 import (
    lower_rss_vnext_rig_program_to_canonical_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_decode_v6 import AXISV6CodecBinding
from models.axis.rig_program_model_v6 import RigProgramEventV6, RigTokenKindV6
from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
from models.rss_vnext.contracts import RSSvNextConfig
from tests.compiler.test_canonical_mesh_ir_v1 import _mesh as canonical_test_mesh
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import (
    _candidate as rss_test_inference,
    _mesh as rss_test_mesh,
)


def _fixture():
    model_cfg = RSSvNextConfig.tiny_for_test()
    kinds = (
        RigTokenKindV6.BOS,
        RigTokenKindV6.JOINT,
        RigTokenKindV6.FRAMES_START,
        RigTokenKindV6.FRAME,
        RigTokenKindV6.CONTROLS_START,
        RigTokenKindV6.CONTROL,
        RigTokenKindV6.CONTROL_TARGET,
        RigTokenKindV6.CHAINS_START,
        RigTokenKindV6.CONSTRAINTS_START,
        RigTokenKindV6.ANCHORS_START,
        RigTokenKindV6.SKIN_START,
        *([RigTokenKindV6.SKIN_CODE] * model_cfg.skin_tokens_per_joint),
        RigTokenKindV6.EOS,
    )
    events = []
    for kind in kinds:
        options = {}
        if kind == RigTokenKindV6.JOINT:
            options = {"coord_indices": (128, 128, 128), "pointer_a": -1}
        elif kind == RigTokenKindV6.FRAME:
            options = {"pointer_a": 0, "frame_6d": (1., 0., 0., 0., 1., 0.)}
        elif kind == RigTokenKindV6.CONTROL:
            options = {
                "coord_indices": (128, 128, 128),
                "pointer_a": -1,
                "aux_type_id": 1,
            }
        elif kind == RigTokenKindV6.CONTROL_TARGET:
            options = {"pointer_a": 0}
        elif kind == RigTokenKindV6.SKIN_CODE:
            options = {"skin_code_id": 7}
        row = asdict(RigProgramEventV6(kind, **options))
        row["kind"] = int(kind)
        events.append(row)
    evidence = seal_inference_evidence_v1(
        rss_test_inference(control_events=tuple(events))
    )
    source = rss_test_mesh()
    old = canonical_test_mesh()
    new_vertices = []
    candidates = []
    for i, template in enumerate(old.vertices):
        name = f"RSSV:{i:08d}"
        point = tuple(source["rest_xyz"][i])
        candidates.append(CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=name,
            support_binding=template.support_binding,
            component_id=template.component_id,
            P=point,
        ))
        new_vertices.append(replace(
            template, P=point, source_candidate_vertex_id=name,
        ))
    candidate = CanonicalMeshCandidateIR(
        vertices=tuple(candidates),
        faces=(tuple(v.candidate_vertex_id for v in candidates),),
        edges=tuple(sorted({
            tuple(sorted((candidates[a].candidate_vertex_id,
                          candidates[b].candidate_vertex_id)))
            for a, b in ((0, 1), (1, 2), (2, 0))
        })),
        surface_binding_hash=old.source_evidence_binding_hash,
        partition_binding_hash=old.partition_binding_hash,
        carrier_policy_binding_hash=old.carrier_policy_binding_hash,
        producer_id="TEST_NEURAL_CANDIDATE_NO_SOURCE_TRUTH",
        producer_policy_hash="c" * 64,
        candidate_lineage_hash="",
        metadata={
            "neural_proposal_basis": True,
            "source_geometry_was_not_independently_observed": True,
            "inference_evidence_hash": evidence.evidence_hash,
            "original_mesh_sha256": content_sha256(source),
        },
    )
    candidate = replace(candidate, candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate))
    canonical = replace(
        old, vertices=tuple(new_vertices),
        source_candidate_lineage_hash=candidate.candidate_lineage_hash,
        mesh_lineage_hash="",
    )
    canonical = replace(canonical, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(canonical))
    codec = AXISV6CodecBinding(
        MIRA_V6_CODEC_ID, "e" * 64,
        model_cfg.axis_config().skin_vocab_size,
        model_cfg.skin_tokens_per_joint,
    )
    return canonical, candidate, evidence, model_cfg, codec


def test_rss_vnext_causal_program_canonical_lowering_uses_one_checkpoint_only():
    mesh, candidate, evidence, config, codec = _fixture()
    lowered = lower_rss_vnext_rig_program_to_canonical_v1(
        canonical_mesh=mesh, neural_mesh_candidate=candidate,
        evidence=evidence, model_config=config, codec_binding=codec,
    )
    assert lowered.canonical_rig.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert len(lowered.canonical_rig.joints) == 1
    assert len(lowered.canonical_rig.controls) == 1
    assert lowered.skin_code_proposal.codec_checkpoint_hash == "e" * 64
    assert lowered.skin_code_proposal.rig_program_proposal_hash == lowered.qualified_program.proposal.proposal_lineage_hash
    assert lowered.model_release_count_loaded == 0
    assert lowered.compiler_parent_reselection_performed is False
    assert lowered.product_revision_minted is False


def test_rss_vnext_canonical_lowering_rejects_other_mesh_identity_and_reordered_vertices():
    mesh, candidate, evidence, config, codec = _fixture()
    swapped = replace(mesh, source_candidate_lineage_hash="f" * 64, mesh_lineage_hash="")
    swapped = replace(swapped, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(swapped))
    with pytest.raises(QualificationError, match="NEURAL_CANDIDATE_CANONICAL_PARENT_DRIFT"):
        lower_rss_vnext_rig_program_to_canonical_v1(
            canonical_mesh=swapped, neural_mesh_candidate=candidate,
            evidence=evidence, model_config=config, codec_binding=codec,
        )
    ordered = replace(mesh, vertices=tuple(reversed(mesh.vertices)), mesh_lineage_hash="")
    ordered = replace(ordered, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(ordered))
    with pytest.raises(QualificationError, match="TESSA_VERTEX_ORDER_OR_REFINEMENT_DRIFT"):
        lower_rss_vnext_rig_program_to_canonical_v1(
            canonical_mesh=ordered, neural_mesh_candidate=candidate,
            evidence=evidence, model_config=config, codec_binding=codec,
        )
