from __future__ import annotations

"""RSS.vNext unified candidate → existing canonical Rig v2/Skin v2/Mechanics.

Intentional authority split:
  * A learned model can propose mesh, RigProgram, FSQ codes and W[V,J].
  * An independent source-/render-qualified Compiler stage must FIRST mint a
    CanonicalMeshIR with exactly the same vertex/face/component identities.
  * This bridge can then translate causal rig tokens, qualify RigProgram,
    mint canonical Rig and qualify the exact-mesh learned skin field using
    the EXISTING code, without transferring weights to a new mesh.
  * It executes real deterministic rest-pose mechanics as a consumer test.
  * It CANNOT mint ProductRevision, Graphics, atlas or runtime qualification.

No canonical fake mesh or arbitrary source proof is created in this adapter.
"""

from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Callable
import math

import numpy as np

from models.axis.rig_program_decode_v6 import AXISV6CodecBinding, decode_axis_v6_events
from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    RigProgramEventV6,
    RigTokenKindV6,
)
from models.rss_vnext.inference_candidate_v1 import RSSvNextInferenceEvidenceBundleV1

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v2 import CanonicalRigIRV2, mint_canonical_rig_v2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2, mint_canonical_skin_v2
from .carrier_skin_v1 import (
    CarrierSkinInfluenceProposal, CarrierSkinProposalIR,
    qualify_carrier_skin_v1,
)
from .mechanics_control_solver_v1 import (
    MechanicsPoseIRV1, DeformedMeshIRV1,
    solve_control_pose_v1, deform_mesh_lbs_v1,
)
from .rig_constraint_schema_v1 import validate_rig_constraint_operator_v1
from .rig_program_ir_v1 import QualifiedRigProgramIR, qualify_rig_program_v1
from .skin_code_ir_v1 import SkinCodeProposalIR, validate_skin_code_proposal_v1
from .types import QualificationError


@dataclass(frozen=True)
class RSSvNextCanonicalMeshCandidateBindingV1:
    """External independent surface-court mapping; no implicit remeshing.

    The learned field uses IRIS xyz (-1..1), so the binder declares an
    explicit uniform normalization map to source-approved 3D world coords.
    The mapping is bijective with CanonicalMesh.source_candidate_vertex_id.
    """
    candidate_index_to_carrier_vertex_id: tuple[str, ...]
    component_names_by_model_index: tuple[str, ...]
    normalization_center: tuple[float, float, float]
    normalization_scale: float
    canonical_mesh_source_qualification_receipt_sha256: str
    max_position_residual: float = 1e-5


@dataclass(frozen=True)
class RSSvNextMechanicsBridgeResultV1:
    rig: CanonicalRigIRV2
    skin: CanonicalSkinIRV2
    qualified_program: QualifiedRigProgramIR
    skin_code: SkinCodeProposalIR
    rest_pose: MechanicsPoseIRV1
    rest_deformation: DeformedMeshIRV1
    source_candidate_hash: str
    authority: str = "COMPILER_MECHANICS_QUALIFIED_NOT_A_PRODUCT_REVISION"


def _parse_events(raw: list[dict]) -> tuple[RigProgramEventV6, ...]:
    fields = RigProgramEventV6.__dataclass_fields__
    rows: list[RigProgramEventV6] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != set(fields):
            raise QualificationError("RSS_VNEXT_BRIDGE_RIG_EVENT_FIELDS_INVALID")
        try:
            payload = dict(item)
            payload["kind"] = RigTokenKindV6(int(payload["kind"]))
            for key in ("coord_indices", "dof_mask", "frame_6d", "scalar_params"):
                payload[key] = tuple(payload[key])
            rows.append(RigProgramEventV6(**payload))
        except (ValueError, TypeError, KeyError) as exc:
            raise QualificationError("RSS_VNEXT_BRIDGE_RIG_EVENT_INVALID") from exc
    return tuple(rows)


def _oriented(face: tuple[str, str, str]) -> tuple[str, str, str]:
    return min(face, face[1:] + face[:1], face[2:] + face[:2])


def compile_rss_vnext_mechanics_candidate_v1(
    evidence: RSSvNextInferenceEvidenceBundleV1,
    canonical_mesh: CanonicalMeshIR,
    binding: RSSvNextCanonicalMeshCandidateBindingV1,
    *,
    axis_config: AXISRigProgramConfigV6,
    codec_architecture_id: str,
    verify_independent_mesh_qualification: Callable[
        [CanonicalMeshIR, RSSvNextCanonicalMeshCandidateBindingV1], bool
    ],
    max_influences_per_vertex: int = 32,
) -> RSSvNextMechanicsBridgeResultV1:
    """Bridge learned 3D mechanism into real Compiler and run its rest state.

    External verification is required and cannot be substituted by the
    candidate's own hash. No attempt at finding or repairing missing joints.
    """
    evidence.validate()
    m = evidence.manifest
    if m["status"] != "CONTROL_SKIN_CANDIDATE":
        raise QualificationError("RSS_VNEXT_BRIDGE_REQUIRES_CONTROL_SKIN_EVIDENCE")
    if canonical_mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(canonical_mesh):
        raise QualificationError("RSS_VNEXT_BRIDGE_CANONICAL_MESH_HASH_DRIFT")
    if canonical_mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("RSS_VNEXT_BRIDGE_CANONICAL_MESH_AUTHORITY_REQUIRED")
    if not callable(verify_independent_mesh_qualification):
        raise QualificationError("RSS_VNEXT_BRIDGE_EXTERNAL_MESH_COURT_REQUIRED")
    if not verify_independent_mesh_qualification(canonical_mesh, binding):
        raise QualificationError("RSS_VNEXT_BRIDGE_EXTERNAL_MESH_COURT_REJECTED")
    proof = binding.canonical_mesh_source_qualification_receipt_sha256
    if len(proof) != 64 or any(c not in "0123456789abcdef" for c in proof):
        raise QualificationError("RSS_VNEXT_BRIDGE_SOURCE_MESH_RECEIPT_REQUIRED")
    if (not 0 < float(binding.normalization_scale) <= 1e6
            or not math.isfinite(binding.normalization_scale)
            or len(binding.normalization_center) != 3
            or not all(math.isfinite(float(x)) for x in binding.normalization_center)):
        raise QualificationError("RSS_VNEXT_BRIDGE_NORMALIZATION_INVALID")
    if not 0 < float(binding.max_position_residual) <= 0.0001:
        raise QualificationError("RSS_VNEXT_BRIDGE_POSITION_TOLERANCE_INVALID")
    if not 1 <= max_influences_per_vertex <= 256:
        raise QualificationError("RSS_VNEXT_BRIDGE_SKIN_INFLUENCE_CAP_INVALID")

    with np.load(BytesIO(evidence.array_archive_npz), allow_pickle=False) as archive:
        xyz = archive["mesh_rest_xyz"][0].astype(np.float64)
        faces = archive["mesh_faces"].astype(np.int64)
        components = archive["mesh_vertex_component"]
        weights = archive["skin_weights"][0].astype(np.float64)
        joint_mask = archive["skin_joint_mask"][0].astype(bool)

    n = len(xyz)
    mapping = binding.candidate_index_to_carrier_vertex_id
    if (len(mapping) != n or len(set(mapping)) != n or
            set(mapping) != {v.source_candidate_vertex_id for v in canonical_mesh.vertices}):
        raise QualificationError("RSS_VNEXT_BRIDGE_VERTEX_BIJECTION_REQUIRED")
    source_by = {v.source_candidate_vertex_id: v for v in canonical_mesh.vertices}
    if not binding.component_names_by_model_index:
        raise QualificationError("RSS_VNEXT_BRIDGE_COMPONENT_MAPPING_REQUIRED")
    for index, source_id in enumerate(mapping):
        vertex = source_by[source_id]
        component_index = int(components[index])
        if (component_index < 0 or component_index >= len(binding.component_names_by_model_index)
                or vertex.component_id != binding.component_names_by_model_index[component_index]):
            raise QualificationError("RSS_VNEXT_BRIDGE_COMPONENT_OWNERSHIP_DRIFT")
        expected = (
            np.asarray(binding.normalization_center, dtype=np.float64)
            + float(binding.normalization_scale) * xyz[index]
        )
        if float(np.max(np.abs(np.asarray(vertex.P) - expected))) > binding.max_position_residual:
            raise QualificationError("RSS_VNEXT_BRIDGE_REST_GEOMETRY_DRIFT")
    proposed_faces = {
        _oriented(tuple(source_by[mapping[int(i)]].canonical_mesh_vertex_id for i in face))
        for face in faces
    }
    if len(proposed_faces) != len(faces):
        raise QualificationError("RSS_VNEXT_BRIDGE_DUPLICATE_CANDIDATE_FACE")
    if proposed_faces != {_oriented(face) for face in canonical_mesh.faces}:
        raise QualificationError("RSS_VNEXT_BRIDGE_FACE_TOPOLOGY_OR_WINDING_DRIFT")

    if not codec_architecture_id.strip():
        raise QualificationError("RSS_VNEXT_BRIDGE_CODEC_ARCHITECTURE_REQUIRED")
    events = _parse_events(m["rig_events"])
    if not joint_mask.all():
        raise QualificationError("RSS_VNEXT_BRIDGE_SKIN_JOINT_COVERAGE_UNQUALIFIED")
    codebook_size = int(axis_config.skin_vocab_size)
    if codebook_size < 2:
        raise QualificationError("RSS_VNEXT_BRIDGE_SKIN_CODEC_VOCAB_INVALID")
    rig_proposal, skin_codes = decode_axis_v6_events(
        events, config=axis_config,
        normalization_center=binding.normalization_center,
        normalization_scale=binding.normalization_scale,
        mesh_lineage_hash=canonical_mesh.mesh_lineage_hash,
        feature_evidence_binding_hash=canonical_mesh.source_evidence_binding_hash,
        carrier_evidence_hash=canonical_mesh.carrier_evidence_hash,
        carrier_topology_hash=canonical_mesh.carrier_topology_hash,
        carrier_geometry_hash=canonical_mesh.carrier_geometry_hash,
        model_provenance=m["checkpoint_sha256"],
        codec_binding=AXISV6CodecBinding(
            architecture_id=codec_architecture_id,
            checkpoint_hash=m["checkpoint_sha256"],
            codebook_size=codebook_size,
            tokens_per_joint=int(m["skin_tokens_per_joint"]),
        ),
    )
    program = qualify_rig_program_v1(rig_proposal)
    for row in rig_proposal.constraints:
        validate_rig_constraint_operator_v1(
            operator_id=row.operator_id,
            input_control_count=len(row.input_proposal_control_ids),
            output_joint_count=len(row.output_proposal_joint_ids),
            has_chain=row.proposal_chain_id is not None,
            parameters=row.parameters,
        )
    expected_joint_ids = tuple(j.proposal_joint_id for j in rig_proposal.joints)
    validate_skin_code_proposal_v1(
        skin_codes, expected_proposal_joint_ids=expected_joint_ids,
        expected_rig_program_proposal_hash=rig_proposal.proposal_lineage_hash,
    )
    rig = mint_canonical_rig_v2(canonical_mesh, qualified_program=program)
    if len(rig.joints) != weights.shape[1]:
        raise QualificationError("RSS_VNEXT_BRIDGE_SKIN_JOINT_DIMENSION_DRIFT")
    influences: list[CarrierSkinInfluenceProposal] = []
    for i, source_id in enumerate(mapping):
        for j, weight in enumerate(weights[i]):
            if weight > 0.0:
                influences.append(CarrierSkinInfluenceProposal(
                    carrier_vertex_id=source_id,
                    canonical_joint_id=rig.joints[j].canonical_joint_id,
                    weight=float(weight),
                ))
    carrier = SimpleNamespace(
        ordered_vertex_ids=mapping,
        carrier_evidence_hash=canonical_mesh.carrier_evidence_hash,
        topology_hash=canonical_mesh.carrier_topology_hash,
        geometry_hash=canonical_mesh.carrier_geometry_hash,
    )
    skeletal = SimpleNamespace(
        joints=rig.joints, skeleton_lineage_hash=rig.deform_structure_hash,
    )
    proposal = CarrierSkinProposalIR(
        influences=tuple(influences),
        carrier_evidence_hash=carrier.carrier_evidence_hash,
        carrier_topology_hash=carrier.topology_hash,
        carrier_geometry_hash=carrier.geometry_hash,
        skeleton_binding_hash=rig.deform_structure_hash,
        model_provenance=m["checkpoint_sha256"],
        metadata={
            "output_domain":"MECHANICAL_CARRIER_M",
            "semantic_skin_transfer_performed":False,
            "gsa_skin_field_authority_minted":False,
            "direct_simplex_readout":True,
            "source_skin_runtime_authority":False,
            "mesh_mutation_invalidates_prediction":True,
            "rss_vnext_source_candidate_hash":m["array_archive_sha256"],
        },
    )
    qualified_skin = qualify_carrier_skin_v1(
        carrier, skeletal, proposal,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=max(1e-4, 1e-6 * n),
        negative_tolerance=1e-8,
        max_influences=max_influences_per_vertex,
    )
    skin = mint_canonical_skin_v2(
        canonical_mesh, rig, qualified_skin=qualified_skin,
        skin_code_proposal_hash=skin_codes.proposal_lineage_hash,
        skin_codec_architecture_id=codec_architecture_id,
        skin_codec_checkpoint_hash=m["checkpoint_sha256"],
    )
    # Mandatory runtime consumer smoke. The current solver is not a proof of
    # complete full-body control, merely an exact current capability court.
    rest = solve_control_pose_v1(rig, ())
    deformed = deform_mesh_lbs_v1(canonical_mesh, rig, skin, rest)
    if tuple(deformed.canonical_mesh_vertex_ids) != tuple(
        v.canonical_mesh_vertex_id for v in canonical_mesh.vertices
    ):
        raise QualificationError("RSS_VNEXT_BRIDGE_REST_SKIN_ADDRESS_DRIFT")
    return RSSvNextMechanicsBridgeResultV1(
        rig=rig, skin=skin, qualified_program=program,
        skin_code=skin_codes, rest_pose=rest,
        rest_deformation=deformed,
        source_candidate_hash=m["array_archive_sha256"],
    )


__all__ = [
    "RSSvNextCanonicalMeshCandidateBindingV1",
    "RSSvNextMechanicsBridgeResultV1",
    "compile_rss_vnext_mechanics_candidate_v1",
]
