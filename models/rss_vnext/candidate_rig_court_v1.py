from __future__ import annotations

"""Compiler's AXIS V6 program court for RSS.vNext single-checkpoint proposals.

Decode the existing causal RigProgram language rather than inventing another
controller schema. This court admits a *structurally valid proposal* only. Its
mesh identity is the neural candidate evidence hash, NOT CanonicalMeshIR.
Product rig minting, mechanics qualification and ProductRevision remain blocked
until real CanonicalMesh and accepted source qualification are supplied.
"""

from dataclasses import asdict, dataclass, fields
from typing import Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1, validate_rig_constraint_operator_v1,
)
from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    QualifiedRigProgramIR, qualify_rig_program_v1,
)
from compiler.realsas_compiler_core.skin_code_ir_v1 import (
    SkinCodeProposalIR, validate_skin_code_proposal_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_decode_v6 import AXISV6CodecBinding, decode_axis_v6_events
from models.axis.rig_program_model_v6 import RigProgramEventV6, RigTokenKindV6
from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
from .contracts import RSSvNextConfig


@dataclass(frozen=True)
class RSSvNextCandidateRigCourtV1:
    rig_program: QualifiedRigProgramIR
    skin_codes: SkinCodeProposalIR
    candidate_mesh_hash: str
    source_evidence_hash: str
    product_authority: bool = False
    status: str = "QUALIFIED_RIG_PROGRAM_ONLY_CODEC_PARITY_NOT_PRODUCT_PROVEN"


def _events(raw: tuple[Mapping, ...]) -> tuple[RigProgramEventV6, ...]:
    allowed = {f.name for f in fields(RigProgramEventV6)}
    output = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != allowed:
            raise QualificationError("RSS_VNEXT_RIG_EVENT_FIELDS_DRIFT")
        data = dict(row)
        for key in ("coord_indices", "dof_mask", "frame_6d", "scalar_params"):
            value = data[key]
            if not isinstance(value, (tuple, list)):
                raise QualificationError("RSS_VNEXT_RIG_EVENT_VECTOR_TYPE")
            data[key] = tuple(value)
        try:
            data["kind"] = RigTokenKindV6(data["kind"])
            output.append(RigProgramEventV6(**data))
        except (TypeError, ValueError) as exc:
            raise QualificationError("RSS_VNEXT_RIG_EVENT_DECODE_INVALID") from exc
    return tuple(output)


def qualify_candidate_rig_program_v1(
    evidence: RSSvNextInferenceEvidenceIRV1,
    *,
    model_config: RSSvNextConfig,
    expected_checkpoint_sha256: str,
    mira_codec_binding: AXISV6CodecBinding,
) -> RSSvNextCandidateRigCourtV1:
    """Execute the existing decoder, structural court and solver vocabulary gate.

    No canonical mesh, source 3D or teacher joints are silently introduced.
    A structurally qualified RigProgram remains unusable for runtime until
    CanonicalMeshIR, skin weights and mechanics/graphics are independently
    qualified and bound to its exact candidate geometry.
    """
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    model_config.validate()
    if evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE" or evidence.mesh is None:
        raise QualificationError("RSS_VNEXT_RIG_REQUIRES_VALID_NEURAL_MESH")
    if not evidence.control_events:
        raise QualificationError("RSS_VNEXT_RIG_CONTROL_EVENTS_REQUIRED")
    mesh = evidence.mesh
    # These hashes name proposals only. No CanonicalMesh, carrier or geometry
    # authority can be minted by choosing a convenient normalization.
    candidate_hash = content_sha256(mesh)
    source_hash = content_sha256({
        "front": evidence.observed_front_sha256,
        "back": evidence.observed_back_sha256,
        "cameras": evidence.camera_binding_sha256,
    })
    topology_hash = content_sha256({
        "tokens": mesh["topology_tokens"], "faces": mesh["faces"],
        "component_ids": mesh["vertex_components"],
    })
    geometry_hash = content_sha256({
        "base_xyz": mesh["base_xyz"], "rest_xyz": mesh["rest_xyz"],
    })
    config = model_config.axis_config()
    # The RSS unified inference checkpoint and a MIRA V6 runtime codec are
    # different artifacts. A model checkpoint digest is NOT automatically a
    # decoder digest; false parity here would break exact FSQ token meaning.
    # This binding is still only launcher-provided structural provenance;
    # proving embedded-soft vs deployed-discrete codec parity is a separate
    # product qualification gate.
    if (
        type(mira_codec_binding) is not AXISV6CodecBinding
        or mira_codec_binding.architecture_id != MIRA_V6_CODEC_ID
        or type(mira_codec_binding.checkpoint_hash) is not str
        or len(mira_codec_binding.checkpoint_hash) != 64
        or any(c not in "0123456789abcdef" for c in mira_codec_binding.checkpoint_hash)
        or mira_codec_binding.checkpoint_hash == evidence.checkpoint_sha256
        or type(mira_codec_binding.codebook_size) is not int
        or mira_codec_binding.codebook_size != config.skin_vocab_size
        or type(mira_codec_binding.tokens_per_joint) is not int
        or mira_codec_binding.tokens_per_joint != model_config.skin_tokens_per_joint
    ):
        raise QualificationError("RSS_VNEXT_RIG_INDEPENDENT_MIRA_CODEC_BINDING_REQUIRED")
    proposal, skin = decode_axis_v6_events(
        _events(evidence.control_events),
        config=config,
        normalization_center=(0.0, 0.0, 0.0),
        normalization_scale=1.0,
        mesh_lineage_hash=candidate_hash,
        feature_evidence_binding_hash=source_hash,
        carrier_evidence_hash=evidence.evidence_hash,
        carrier_topology_hash=topology_hash,
        carrier_geometry_hash=geometry_hash,
        model_provenance="RSS.vNext.UnifiedInference:" + evidence.checkpoint_sha256,
        codec_binding=mira_codec_binding,
    )
    for constraint in proposal.constraints:
        if constraint.operator_id not in RIG_CONSTRAINT_OPERATOR_IDS_V1:
            raise QualificationError("RSS_VNEXT_RIG_OPERATOR_NOT_EXECUTABLE")
        validate_rig_constraint_operator_v1(
            operator_id=constraint.operator_id,
            input_control_count=len(constraint.input_proposal_control_ids),
            output_joint_count=len(constraint.output_proposal_joint_ids),
            has_chain=constraint.proposal_chain_id is not None,
            parameters=constraint.parameters,
        )
    qualified = qualify_rig_program_v1(proposal)
    validate_skin_code_proposal_v1(
        skin,
        expected_proposal_joint_ids=tuple(j.proposal_joint_id for j in proposal.joints),
        expected_rig_program_proposal_hash=proposal.proposal_lineage_hash,
    )
    if evidence.dynamic is not None:
        weights = evidence.dynamic["weights"]
        if len(weights[0]) != len(proposal.joints):
            raise QualificationError("RSS_VNEXT_RIG_SKIN_WEIGHT_JOINT_ORDER_DRIFT")
    return RSSvNextCandidateRigCourtV1(
        rig_program=qualified, skin_codes=skin,
        candidate_mesh_hash=candidate_hash, source_evidence_hash=evidence.evidence_hash,
    )


__all__ = [
    "RSSvNextCandidateRigCourtV1",
    "qualify_candidate_rig_program_v1",
]
