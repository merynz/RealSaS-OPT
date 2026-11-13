from __future__ import annotations

"""Model-Compiler boundary: one inference source, one canonical mesh.

These fixtures exercise typed ABI and identity, not learned checkpoint quality,
source 3D ground truth or ProductRevision admission.
"""
from dataclasses import asdict, replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    mint_canonical_mesh_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
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
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    build_surface_addressing, static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_decode_v6 import AXISV6CodecBinding
from models.axis.rig_program_model_v6 import RigProgramEventV6, RigTokenKindV6
from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
from models.rss_vnext.contracts import RSSvNextConfig
from tests.compiler.test_canonical_mesh_ir_v1 import _fixture as mesh_fixture
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import (
    _candidate as neural_fixture,
)


def _fixture():
    config = RSSvNextConfig.tiny_for_test()
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
        *([RigTokenKindV6.SKIN_CODE] * config.skin_tokens_per_joint),
        RigTokenKindV6.EOS,
    )
    events = []
    for kind in kinds:
        fields = {}
        if kind == RigTokenKindV6.JOINT:
            fields = {"coord_indices": (128, 128, 128), "pointer_a": -1}
        elif kind == RigTokenKindV6.FRAME:
            fields = {"pointer_a": 0, "frame_6d": (1., 0., 0., 0., 1., 0.)}
        elif kind == RigTokenKindV6.CONTROL:
            fields = {"coord_indices": (128, 128, 128), "pointer_a": -1, "aux_type_id": 1}
        elif kind == RigTokenKindV6.CONTROL_TARGET:
            fields = {"pointer_a": 0}
        elif kind == RigTokenKindV6.SKIN_CODE:
            fields = {"skin_code_id": 7}
        item = asdict(RigProgramEventV6(kind, **fields))
        item["kind"] = int(kind)
        events.append(item)
    evidence = seal_inference_evidence_v1(
        neural_fixture(control_events=tuple(events))
    )
    old_candidate, old_static, _ = mesh_fixture()
    source_ids = tuple(f"RSSV:{i:08d}" for i in range(3))
    vertices = tuple(
        CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=source_ids[i],
            support_binding=old_candidate.vertices[i].support_binding,
            component_id=old_candidate.vertices[i].component_id,
            P=tuple(evidence.mesh["rest_xyz"][i]),
        )
        for i in range(3)
    )
    candidate = CanonicalMeshCandidateIR(
        vertices=vertices, faces=(source_ids,),
        edges=tuple(sorted({
            tuple(sorted(pair)) for pair in (
                (source_ids[0], source_ids[1]),
                (source_ids[1], source_ids[2]),
                (source_ids[0], source_ids[2]),
            )
        })),
        surface_binding_hash=old_candidate.surface_binding_hash,
        partition_binding_hash=old_candidate.partition_binding_hash,
        carrier_policy_binding_hash=old_candidate.carrier_policy_binding_hash,
        producer_id="TEST_ONLY_UNQUALIFIED_NEURAL_BASIS",
        producer_policy_hash=content_sha256({"fixture": True}),
        candidate_lineage_hash="",
        metadata={
            "neural_proposal_basis": True,
            "source_geometry_was_not_independently_observed": True,
            "inference_evidence_hash": evidence.evidence_hash,
            "original_mesh_sha256": content_sha256(evidence.mesh),
        },
    )
    candidate = replace(
        candidate, candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate),
    )
    addressing = build_surface_addressing(candidate)
    static = replace(
        old_static,
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        qualification_hash="",
    )
    static = replace(static, qualification_hash=static_mesh_qualification_hash(static))
    carrier = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addressing,
    )
    mesh = mint_canonical_mesh_v1(
        candidate, static_qualification=static, carrier_evidence=carrier,
        proposal_evidence_refs=(evidence.evidence_hash,),
    )
    codec = AXISV6CodecBinding(
        MIRA_V6_CODEC_ID, "e" * 64,
        config.axis_config().skin_vocab_size, config.skin_tokens_per_joint,
    )
    return mesh, candidate, evidence, config, codec


def _lower(mesh, candidate, evidence, config, codec):
    return lower_rss_vnext_rig_program_to_canonical_v1(
        canonical_mesh=mesh, neural_mesh_candidate=candidate,
        evidence=evidence, model_config=config, codec_binding=codec,
    )


def test_exact_rig_program_and_skin_codes_rebind_without_second_inference():
    mesh, candidate, evidence, config, codec = _fixture()
    result = _lower(mesh, candidate, evidence, config, codec)
    assert result.canonical_rig.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert len(result.canonical_rig.joints) == 1
    assert len(result.canonical_rig.controls) == 1
    assert result.skin_code_proposal.codec_checkpoint_hash == "e" * 64
    assert result.skin_code_proposal.rig_program_proposal_hash == (
        result.qualified_program.proposal.proposal_lineage_hash
    )
    assert result.model_release_count_loaded == 0
    assert result.compiler_parent_reselection_performed is False
    assert result.compiler_control_synthesis_performed is False
    assert result.product_revision_minted is False


def test_rig_rejects_foreign_mesh_or_reordered_tessa_vertex_parent():
    mesh, candidate, evidence, config, codec = _fixture()
    swapped = replace(mesh, source_candidate_lineage_hash="f" * 64)
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
        canonical_mesh_lineage_hash_v1,
    )
    swapped = replace(swapped, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(swapped))
    with pytest.raises(QualificationError, match="NEURAL_CANDIDATE_CANONICAL_PARENT_DRIFT"):
        _lower(swapped, candidate, evidence, config, codec)
    wrong_order = replace(mesh, vertices=tuple(reversed(mesh.vertices)))
    wrong_order = replace(
        wrong_order, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(wrong_order),
    )
    with pytest.raises(QualificationError, match="TESSA_VERTEX_ORDER_OR_REFINEMENT_DRIFT"):
        _lower(wrong_order, candidate, evidence, config, codec)
