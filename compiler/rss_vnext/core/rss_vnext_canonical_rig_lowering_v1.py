from __future__ import annotations

"""One-checkpoint RSS.vNext AXIS program -> canonical Rig.v2 + SkinCode.

No second AXIS or MIRA model runs here. TESSA's accepted CanonicalMesh is the
sole address and identity owner. The learned token sequence is decoded twice
only to verify that canonical rebinding changes *no* parent, control, joint,
constraint, anchor or FSQ code choice. This cannot convert an unqualified
neural mesh into a source-observed 3D surface or a product PASS.
"""

from dataclasses import dataclass
from typing import Any

from .canonical_mesh_ir_v1 import (
    CanonicalMeshIR, canonical_mesh_lineage_hash_v1,
    validate_canonical_mesh_domain_v1,
)
from .canonical_rig_ir_v2 import CanonicalRigIRV2, mint_canonical_rig_v2
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR, canonical_mesh_candidate_lineage_hash,
)
from .rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1, validate_rig_constraint_operator_v1,
)
from .rig_program_ir_v1 import (
    QualifiedRigProgramIR, qualify_rig_program_v1,
)
from .skin_code_ir_v1 import (
    SkinCodeProposalIR, validate_skin_code_proposal_v1,
)
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
)
from .types import QualificationError
from models.axis.rig_program_decode_v6 import (
    AXISV6CodecBinding, decode_axis_v6_events,
)
from models.rss_vnext.candidate_rig_court_v1 import (
    _events, qualify_candidate_rig_program_v1,
)


@dataclass(frozen=True)
class RSSvNextCanonicalRigLoweringV1:
    canonical_rig: CanonicalRigIRV2
    qualified_program: QualifiedRigProgramIR
    skin_code_proposal: SkinCodeProposalIR
    original_inference_evidence_hash: str
    original_candidate_mesh_hash: str
    accepted_canonical_mesh_hash: str
    codec_submodule_sha256: str
    model_release_count_loaded: int = 0
    compiler_control_synthesis_performed: bool = False
    compiler_parent_reselection_performed: bool = False
    source_3d_observation_claimed: bool = False
    source_rig_teacher_required_at_inference: bool = False
    product_revision_minted: bool = False
    schema: str = "RealSaS.RSSvNext.CanonicalRigLowering.v1"


def lower_rss_vnext_rig_program_to_canonical_v1(
    *,
    canonical_mesh: CanonicalMeshIR,
    neural_mesh_candidate: CanonicalMeshCandidateIR,
    evidence: RSSvNextInferenceEvidenceIRV1,
    model_config: Any,
    codec_binding: AXISV6CodecBinding,
) -> RSSvNextCanonicalRigLoweringV1:
    if type(canonical_mesh) is not CanonicalMeshIR:
        raise QualificationError("RSS_VNEXT_RIG_CANONICAL_MESH_REQUIRED")
    validate_canonical_mesh_domain_v1(canonical_mesh)
    if (
        canonical_mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(canonical_mesh)
        or canonical_mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH"
        or canonical_mesh.metadata.get("dynamic_qualification_claimed") is not False
    ):
        raise QualificationError("RSS_VNEXT_RIG_MESH_AUTHORITY_OR_HASH_DRIFT")
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=evidence.checkpoint_sha256,
    )
    if (
        evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh is None
        or not evidence.control_events
        or type(neural_mesh_candidate) is not CanonicalMeshCandidateIR
    ):
        raise QualificationError("RSS_VNEXT_RIG_TESSA_AND_CONTROL_EVIDENCE_REQUIRED")
    candidate = neural_mesh_candidate
    info = dict(candidate.metadata or {})
    if (
        candidate.candidate_lineage_hash != canonical_mesh_candidate_lineage_hash(candidate)
        or candidate.candidate_lineage_hash != canonical_mesh.source_candidate_lineage_hash
        or candidate.surface_binding_hash != canonical_mesh.source_evidence_binding_hash
        or candidate.partition_binding_hash != canonical_mesh.partition_binding_hash
        or candidate.carrier_policy_binding_hash != canonical_mesh.carrier_policy_binding_hash
        or info.get("neural_proposal_basis") is not True
        or info.get("source_geometry_was_not_independently_observed") is not True
        or info.get("inference_evidence_hash") != evidence.evidence_hash
        or info.get("original_mesh_sha256") != content_sha256(evidence.mesh)
    ):
        raise QualificationError("RSS_VNEXT_RIG_NEURAL_CANDIDATE_CANONICAL_PARENT_DRIFT")

    # The positional/indexed domain must be EXACTLY the admitted TESSA mesh.
    # It is not safe to transfer weights and joints by nearest vertex.
    vertices = tuple(candidate.vertices)
    bound = tuple(canonical_mesh.vertices)
    if (
        len(vertices) != len(bound)
        or len(vertices) != len(evidence.mesh["rest_xyz"])
        or len(candidate.faces) != len(canonical_mesh.faces)
    ):
        raise QualificationError("RSS_VNEXT_RIG_VERTEX_FACE_CARDINALITY_DRIFT")
    for i, (candidate_vertex, canon_vertex) in enumerate(zip(vertices, bound)):
        rest = tuple(float(x) for x in evidence.mesh["rest_xyz"][i])
        if (
            candidate_vertex.candidate_vertex_id != f"RSSV:{i:08d}"
            or canon_vertex.source_candidate_vertex_id != candidate_vertex.candidate_vertex_id
            or canon_vertex.P != candidate_vertex.P
            or candidate_vertex.P != rest
            or canon_vertex.component_id != candidate_vertex.component_id
        ):
            raise QualificationError("RSS_VNEXT_RIG_TESSA_VERTEX_ORDER_OR_REFINEMENT_DRIFT")

    candidate_court = qualify_candidate_rig_program_v1(
        evidence, model_config=model_config,
        expected_checkpoint_sha256=evidence.checkpoint_sha256,
        mira_codec_binding=codec_binding,
    )
    if candidate_court.candidate_mesh_hash != content_sha256(evidence.mesh):
        raise QualificationError("RSS_VNEXT_RIG_CANDIDATE_COURT_PARENT_DRIFT")

    proposal, codes = decode_axis_v6_events(
        _events(evidence.control_events),
        config=model_config.axis_config(),
        normalization_center=(0., 0., 0.),
        normalization_scale=1.,
        mesh_lineage_hash=canonical_mesh.mesh_lineage_hash,
        feature_evidence_binding_hash=canonical_mesh.source_evidence_binding_hash,
        carrier_evidence_hash=canonical_mesh.carrier_evidence_hash,
        carrier_topology_hash=canonical_mesh.carrier_topology_hash,
        carrier_geometry_hash=canonical_mesh.carrier_geometry_hash,
        model_provenance=(
            "RSS.vNext.UnifiedInference:" + evidence.checkpoint_sha256
        ),
        codec_binding=codec_binding,
    )
    original = candidate_court.rig_program.proposal
    if (
        proposal.joints != original.joints
        or proposal.local_frames != original.local_frames
        or proposal.controls != original.controls
        or proposal.chains != original.chains
        or proposal.constraints != original.constraints
        or proposal.attachment_anchors != original.attachment_anchors
        or tuple((r.proposal_joint_id, r.code_ids) for r in codes.rows)
        != tuple((r.proposal_joint_id, r.code_ids)
                 for r in candidate_court.skin_codes.rows)
    ):
        raise QualificationError("RSS_VNEXT_RIG_REBIND_CHANGED_LEARNED_PROGRAM")
    for constraint in proposal.constraints:
        if constraint.operator_id not in RIG_CONSTRAINT_OPERATOR_IDS_V1:
            raise QualificationError("RSS_VNEXT_RIG_UNEXECUTABLE_CONSTRAINT")
        validate_rig_constraint_operator_v1(
            operator_id=constraint.operator_id,
            input_control_count=len(constraint.input_proposal_control_ids),
            output_joint_count=len(constraint.output_proposal_joint_ids),
            has_chain=constraint.proposal_chain_id is not None,
            parameters=constraint.parameters,
        )
    qualified = qualify_rig_program_v1(proposal)
    validate_skin_code_proposal_v1(
        codes,
        expected_proposal_joint_ids=tuple(j.proposal_joint_id for j in proposal.joints),
        expected_rig_program_proposal_hash=proposal.proposal_lineage_hash,
    )
    rig = mint_canonical_rig_v2(
        canonical_mesh, qualified_program=qualified,
    )
    if rig.mesh_lineage_hash != canonical_mesh.mesh_lineage_hash:
        raise QualificationError("RSS_VNEXT_RIG_CANONICAL_BINDING_DRIFT")
    return RSSvNextCanonicalRigLoweringV1(
        canonical_rig=rig, qualified_program=qualified,
        skin_code_proposal=codes,
        original_inference_evidence_hash=evidence.evidence_hash,
        original_candidate_mesh_hash=candidate.candidate_lineage_hash,
        accepted_canonical_mesh_hash=canonical_mesh.mesh_lineage_hash,
        codec_submodule_sha256=codec_binding.checkpoint_hash,
    )


__all__ = [
    "RSSvNextCanonicalRigLoweringV1",
    "lower_rss_vnext_rig_program_to_canonical_v1",
]
