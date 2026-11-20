from __future__ import annotations

"""Editor-authored OPEN pin approval on one canonical TESSA M* surface.

The compiler consumes a typed neural REVIEW candidate and a distinct editor
authoring command. Neither is a source-native witness, a user authentication
token, a material teacher, a ModelRelease qualification, or an automatic
physics activation. The editor/controller is responsible for capturing a real
user action. This module revalidates the complete canonical topology, actual
pin-host skin consequences and the existing XPBD domain before returning a
declared (disabled) domain for optional runtime activation.
"""

from dataclasses import asdict, dataclass, replace

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .hashing import content_sha256
from .mechanics_source_qualified_surface_pin_v1 import _admit_open_patch
from .mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1,
    XPBDSurfaceDomainV1,
    prepare_declared_xpbd_surface_domain_v1,
)
from .types import QualificationError


CANDIDATE_SCHEMA = "RealSaS.Mechanics.EditorOpenPinNeuralReview.v1"
APPROVAL_SCHEMA = "RealSaS.Mechanics.EditorOpenPinAuthoringCommand.v1"
PREPARED_SCHEMA = "RealSaS.Mechanics.EditorOpenPinPreparedDomain.v1"
REVIEW_STATES = frozenset({
    "REVIEW_READY_NEURAL_PIN_AND_HOST_HYPOTHESIS",
    "REVIEW_NEEDS_EXPLICIT_HOST_PIN_BINDING_REPAIR",
})
ACTION = "USER_EDITOR_APPROVE_EXPLICIT_OPEN_PIN_SELECTION"


def _hash_record(record: object, hash_field: str) -> str:
    fields = asdict(record)
    fields.pop(hash_field)
    return content_sha256(fields)


@dataclass(frozen=True)
class EditorOpenPinNeuralReviewV1:
    anchor_event_index: int
    source_mstar_topology_digest: str
    canonical_mesh_lineage_hash: str
    canonical_rig_lineage_hash: str
    canonical_skin_lineage_hash: str
    held_canonical_face: tuple[str, str, str]
    holder_canonical_face: tuple[str, str, str]
    candidate_pin_vertex_ids: tuple[str, ...]
    candidate_component_id: str
    suggested_host_chain_id: str
    state: str
    review_hash: str = ""
    schema_version: str = CANDIDATE_SCHEMA
    authority: str = "NEURAL_REVIEW_ONLY_NOT_SOURCE_OR_COMPILER_AUTHORITY"
    source_pin_qualified: bool = False
    compiler_xpbd_activation_qualified: bool = False
    inferred_physical_parameters: bool = False

    def sealed(self) -> "EditorOpenPinNeuralReviewV1":
        return replace(self, review_hash=_hash_record(self, "review_hash"))


@dataclass(frozen=True)
class EditorOpenPinAuthoringCommandV1:
    review_hash: str
    canonical_mesh_lineage_hash: str
    canonical_rig_lineage_hash: str
    canonical_skin_lineage_hash: str
    authoring_revision_id: str
    editor_action_id: str
    approved_pin_vertex_ids: tuple[str, ...]
    approved_host_chain_id: str
    action: str = ACTION
    command_hash: str = ""
    schema_version: str = APPROVAL_SCHEMA
    authority: str = "EXPLICIT_EDITOR_USER_AUTHORED_NOT_SOURCE_WITNESS"
    source_pin_qualified: bool = False
    automatic_xpbd_activation: bool = False
    source_material_truth_claimed: bool = False

    def sealed(self) -> "EditorOpenPinAuthoringCommandV1":
        return replace(self, command_hash=_hash_record(self, "command_hash"))


@dataclass(frozen=True)
class EditorOpenPinPreparedDomainV1:
    domain: XPBDSurfaceDomainV1
    review_hash: str
    command_hash: str
    authoring_revision_id: str
    canonical_open_hem_sha256: str
    approval_receipt_sha256: str
    schema_version: str = PREPARED_SCHEMA
    authority: str = "EXPLICIT_USER_AUTHORED_DECLARED_XPBD_NOT_SOURCE_WITNESS"
    source_pin_qualified: bool = False
    automatic_xpbd_activation: bool = False
    source_material_truth_claimed: bool = False
    product_release_qualified: bool = False


def prepare_editor_approved_open_pin_domain_v1(
    *,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    review: EditorOpenPinNeuralReviewV1,
    command: EditorOpenPinAuthoringCommandV1,
    max_pinned_fraction: float = 0.5,
) -> EditorOpenPinPreparedDomainV1:
    """Validate an editor-authored pin command, never source-native authority.

    This does NOT invoke enable_declared_xpbd_surface. A caller must separately
    select a runtime physics policy and explicitly activate the returned
    domain. An editor command hash is only integrity of the command bytes;
    it is NOT authentication of a person or independent source observation.
    """
    if type(review) is not EditorOpenPinNeuralReviewV1 or (
        review.schema_version != CANDIDATE_SCHEMA
        or review.authority != "NEURAL_REVIEW_ONLY_NOT_SOURCE_OR_COMPILER_AUTHORITY"
        or review.source_pin_qualified is not False
        or review.compiler_xpbd_activation_qualified is not False
        or review.inferred_physical_parameters is not False
        or review.state not in REVIEW_STATES
        or type(review.anchor_event_index) is not int
        or review.anchor_event_index < 0
        or type(review.source_mstar_topology_digest) is not str
        or len(review.source_mstar_topology_digest) != 64
        or not review.review_hash
        or review.review_hash != _hash_record(review, "review_hash")
        or not review.candidate_component_id
        or type(review.candidate_pin_vertex_ids) is not tuple
        or not review.candidate_pin_vertex_ids
        or len(set(review.candidate_pin_vertex_ids)) != len(review.candidate_pin_vertex_ids)
        or not review.suggested_host_chain_id
    ):
        raise QualificationError("EDITOR_PIN_REVIEW_NOT_NEURAL_CANDIDATE")
    if type(command) is not EditorOpenPinAuthoringCommandV1 or (
        command.schema_version != APPROVAL_SCHEMA
        or command.authority != "EXPLICIT_EDITOR_USER_AUTHORED_NOT_SOURCE_WITNESS"
        or command.source_pin_qualified is not False
        or command.automatic_xpbd_activation is not False
        or command.source_material_truth_claimed is not False
        or command.action != ACTION
        or not command.authoring_revision_id or not command.editor_action_id
        or not command.command_hash
        or command.command_hash != _hash_record(command, "command_hash")
        or command.review_hash != review.review_hash
        or type(command.approved_pin_vertex_ids) is not tuple
        or not command.approved_pin_vertex_ids
        or len(set(command.approved_pin_vertex_ids)) != len(command.approved_pin_vertex_ids)
        or not command.approved_host_chain_id
    ):
        raise QualificationError("EDITOR_PIN_EXPLICIT_AUTHORING_COMMAND_REQUIRED")
    if (
        review.canonical_mesh_lineage_hash != mesh.mesh_lineage_hash
        or review.canonical_rig_lineage_hash != rig.rig_lineage_hash
        or review.canonical_skin_lineage_hash != skin.skin_lineage_hash
        or command.canonical_mesh_lineage_hash != mesh.mesh_lineage_hash
        or command.canonical_rig_lineage_hash != rig.rig_lineage_hash
        or command.canonical_skin_lineage_hash != skin.skin_lineage_hash
    ):
        raise QualificationError("EDITOR_PIN_CANONICAL_PARENT_LINEAGE_DRIFT")
    if (
        type(review.held_canonical_face) is not tuple
        or type(review.holder_canonical_face) is not tuple
        or len(review.held_canonical_face) != 3
        or len(review.holder_canonical_face) != 3
        or review.held_canonical_face not in mesh.faces
        or review.holder_canonical_face not in mesh.faces
        or not isinstance(max_pinned_fraction, float)
        or not 0.0 < max_pinned_fraction <= 0.5
    ):
        raise QualificationError("EDITOR_PIN_DIRECTED_FACE_OR_POLICY_INVALID")
    owners = {v.canonical_mesh_vertex_id: v.component_id for v in mesh.vertices}
    held = {owners[vid] for vid in review.held_canonical_face}
    holder = {owners[vid] for vid in review.holder_canonical_face}
    if (
        held != {review.candidate_component_id}
        or len(holder) != 1
        or holder == held
    ):
        raise QualificationError("EDITOR_PIN_HELD_HOLDER_COMPONENT_DRIFT")
    face_indices = tuple(i for i,face in enumerate(mesh.faces)
                         if all(owners[v] == review.candidate_component_id
                                for v in face))
    selected, hem_hash = _admit_open_patch(
        mesh, canonical_component_id=review.candidate_component_id,
        selected_face_indices=face_indices,
    )
    boundary_counts: dict[tuple[str,str], int] = {}
    for a,b,c in selected:
        for x,y in ((a,b),(b,c),(c,a)):
            edge = tuple(sorted((x,y)))
            boundary_counts[edge] = boundary_counts.get(edge,0) + 1
    boundary = {vid for edge,count in boundary_counts.items()
                if count == 1 for vid in edge}
    selected_vertices = {vid for face in selected for vid in face}
    approved = command.approved_pin_vertex_ids
    candidate = review.candidate_pin_vertex_ids
    if (
        not set(candidate) <= boundary
        or not set(approved) <= boundary
        or len(approved) >= len(selected_vertices)
        or len(approved) / len(selected_vertices) > max_pinned_fraction
        or len(approved) > 8
    ):
        raise QualificationError("EDITOR_PIN_NOT_OPEN_HEM_OR_OVERPINNED")
    chains = {c.canonical_chain_id:set(c.canonical_joint_ids) for c in rig.chains}
    joints = chains.get(command.approved_host_chain_id)
    if not joints:
        raise QualificationError("EDITOR_PIN_HOST_CHAIN_NOT_DECLARED")
    weight_rows = {row.canonical_mesh_vertex_id:dict(row.influences)
                   for row in skin.rows}
    if (
        any(not any(weight_rows[vid].get(j,0.) > 0.0 for j in joints)
            for vid in approved)
        or not any(
            sum(weight_rows[vid].get(j,0.) for j in joints) > 0.0
            for vid in review.holder_canonical_face
        )
    ):
        # An independently skinned cape can be reviewed, but the existing
        # native XPBD implementation needs actual kinematic host support
        # on each PIN. The editor must first repair and requalify Skin.
        raise QualificationError("EDITOR_PIN_HOST_BINDING_REPAIR_REQUIRED")
    domain = prepare_declared_xpbd_surface_domain_v1(
        mesh,rig,skin,mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=selected,pinned_vertex_ids=approved,
        ),
    )
    receipt_hash = content_sha256({
        "schema": PREPARED_SCHEMA,
        "review_hash": review.review_hash,
        "command_hash": command.command_hash,
        "authoring_revision_id": command.authoring_revision_id,
        "canonical_open_hem_sha256": hem_hash,
        "domain_hash": domain.domain_hash,
        "source_pin_qualified": False,
        "automatic_xpbd_activation": False,
        "product_release_qualified": False,
    })
    return EditorOpenPinPreparedDomainV1(
        domain=domain,
        review_hash=review.review_hash,
        command_hash=command.command_hash,
        authoring_revision_id=command.authoring_revision_id,
        canonical_open_hem_sha256=hem_hash,
        approval_receipt_sha256=receipt_hash,
    )


__all__ = [
    "CANDIDATE_SCHEMA", "APPROVAL_SCHEMA", "PREPARED_SCHEMA", "ACTION",
    "EditorOpenPinNeuralReviewV1", "EditorOpenPinAuthoringCommandV1",
    "EditorOpenPinPreparedDomainV1",
    "prepare_editor_approved_open_pin_domain_v1",
]
