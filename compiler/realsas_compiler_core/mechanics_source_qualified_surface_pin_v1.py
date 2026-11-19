from __future__ import annotations

"""Source-qualified OPEN surface pin program -> existing canonical XPBD court.

This is a Compiler admission boundary, NOT an alternative solver, learned
material policy or automatic semantics-to-physics converter. The final M*
address, independently verified native open hem/pin witness, source-to-
canonical face/vertex transport, host-only intervention and all canonical
lineages must agree before a caller can explicitly activate a surface patch.

Source qualification and executable XPBD remain distinct. Missing evidence
ABSTAINS by raising QualificationError. Neither an OPEN neural prediction
nor a source garment name activates this program by itself.
"""

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR, validate_canonical_mesh_domain_v1
from .canonical_rig_ir_v2 import CanonicalRigIRV2, validate_canonical_rig_domain_v2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2, validate_canonical_skin_domain_v2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .hashing import content_sha256
from .mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1, XPBDSurfaceDomainV1,
    prepare_declared_xpbd_surface_domain_v1,
)
from .types import QualificationError

SCHEMA = "RealSaS.Mechanics.SourceQualifiedOpenSurfacePinProgram.v1"
TRANSPORT_SCHEMA = "RealSaS.Mechanics.SourceNativeFacePinToCanonicalMstar.v2"
Barycentric3 = tuple[float, float, float]
NativeFaceCornersV1 = tuple[Barycentric3, Barycentric3, Barycentric3]
RECEIPT_SCHEMA = "RealSaS.Mechanics.SourceQualifiedXPBDSurfaceActivation.v1"
PIN_ADMISSION = "QUALIFIED_SOURCE_NATIVE_OPEN_PIN_WITH_HOST_INTERVENTION"
TRANSPORT_ADMISSION = "COMPILER_FINAL_MSTAR_TO_CANONICAL_REINDEX_VERIFIED"


def _sha(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class SourceQualifiedOpenSurfacePinProgramV1:
    """One native open component; all pairs are *independently* reindexed.

    Native face/vertex IDs are not student slots, part labels, or a nearest
    point guess. They must correspond in order to final canonical addresses.
    Multi-pin regions are allowed, including kinematic pins moving on a
    common host chain while remaining distinct from free XPBD particles.
    """
    source_asset_sha256: str
    dataset_manifest_sha256: str
    source_qualification_key: str
    candidate_mstar_topology_address_sha256: str
    canonical_mesh_lineage_hash: str
    canonical_component_id: str
    host_canonical_chain_id: str
    source_native_face_indices: tuple[int, ...]
    canonical_face_indices: tuple[int, ...]
    # For each final canonical face, THREE source-parent barycentric corner
    # addresses in matching winding. One native face may have multiple TESSA
    # children; no one-to-one native-face assumption is allowed here.
    source_native_face_corner_barycentric: tuple[NativeFaceCornersV1, ...]
    source_native_pin_vertex_ids: tuple[str, ...]
    canonical_pin_vertex_ids: tuple[str, ...]
    native_open_boundary_sha256: str
    native_pin_witness_sha256: str
    host_only_intervention_sha256: str
    source_face_pin_transport_sha256: str
    schema_version: str = SCHEMA

    def transport_hash(self) -> str:
        return content_sha256({
            "schema": TRANSPORT_SCHEMA,
            "source_asset_sha256": self.source_asset_sha256,
            "candidate_mstar_topology_address_sha256":
                self.candidate_mstar_topology_address_sha256,
            "canonical_mesh_lineage_hash": self.canonical_mesh_lineage_hash,
            "source_native_face_indices": self.source_native_face_indices,
            "canonical_face_indices": self.canonical_face_indices,
            "source_native_face_corner_barycentric":
                self.source_native_face_corner_barycentric,
            "source_native_pin_vertex_ids": self.source_native_pin_vertex_ids,
            "canonical_pin_vertex_ids": self.canonical_pin_vertex_ids,
        })

    def witness_hash(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class SourceQualifiedXPBDSurfaceActivationV1:
    domain: XPBDSurfaceDomainV1
    program_witness_sha256: str
    candidate_mstar_topology_address_sha256: str
    source_face_pin_transport_sha256: str
    host_canonical_chain_id: str
    canonical_component_id: str
    qualification_receipt_sha256: str
    activation_authority: str = "COMPILER_SOURCE_QUALIFIED_PIN_EXPLICIT_RUNTIME_ACTIVATION"
    source_material_truth_claimed: bool = False
    product_release_admission: bool = False
    schema_version: str = RECEIPT_SCHEMA


def _validate_native_face_corners_v1(
    program: SourceQualifiedOpenSurfacePinProgramV1,
) -> None:
    """Exact native-parent barycentric corners for every final M* child face.

    An original native triangle can parent many TESSA refinement children.
    The source must independently attest this transport; this court verifies
    shape, valid barycentrics and orientation, never guesses a nearest face.
    """
    if len(program.source_native_face_corner_barycentric) != len(
        program.canonical_face_indices
    ):
        raise QualificationError("SOURCE_PIN_FACE_BARYCENTRIC_CARDINALITY")
    for corners in program.source_native_face_corner_barycentric:
        try:
            values = np.asarray(corners, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise QualificationError("SOURCE_PIN_FACE_BARYCENTRIC_INVALID") from exc
        if (
            values.shape != (3,3) or not bool(np.isfinite(values).all())
            or bool(np.any(values < -1e-7))
            or bool(np.any(values > 1. + 1e-7))
            or not bool(np.allclose(
                values.sum(axis=1), 1., atol=1e-7, rtol=0.,
            ))
            or float(np.linalg.det(values)) <= 1e-12
        ):
            raise QualificationError("SOURCE_PIN_FACE_BARYCENTRIC_INVALID")


def _admit_open_patch(
    mesh: CanonicalMeshIR,
    *,
    canonical_component_id: str,
    selected_face_indices: tuple[int, ...],
) -> tuple[tuple[str, str, str], ...]:
    """A complete, disjoint OPEN manifold component; never a body crop."""
    if (
        not selected_face_indices
        or len(set(selected_face_indices)) != len(selected_face_indices)
        or any(type(i) is not int or not 0 <= i < len(mesh.faces)
               for i in selected_face_indices)
    ):
        raise QualificationError("SOURCE_PIN_CANONICAL_FACE_INDEX_INVALID")
    selected = tuple(mesh.faces[i] for i in selected_face_indices)
    owners = {v.canonical_mesh_vertex_id: v.component_id for v in mesh.vertices}
    component_faces = tuple(
        face for face in mesh.faces
        if all(owners[vid] == canonical_component_id for vid in face)
    )
    if (
        not component_faces or len(component_faces) != len(selected)
        or set(component_faces) != set(selected)
        or any(len({owners[vid] for vid in face}) != 1 for face in mesh.faces)
    ):
        raise QualificationError("SOURCE_PIN_CROSS_COMPONENT_OR_PARTIAL_PATCH")
    vertices = set(v for face in selected for v in face)
    owned = {v.canonical_mesh_vertex_id for v in mesh.vertices
             if v.component_id == canonical_component_id}
    if vertices != owned:
        raise QualificationError("SOURCE_PIN_OPEN_SURFACE_UNUSED_VERTEX")
    edge_count: dict[tuple[str, str], int] = {}
    neighbors: dict[str, set[str]] = {vid: set() for vid in owned}
    for a,b,c in selected:
        for x,y in ((a,b),(b,c),(c,a)):
            key = tuple(sorted((x,y)))
            edge_count[key] = edge_count.get(key,0) + 1
            neighbors[x].add(y)
            neighbors[y].add(x)
    if any(count > 2 for count in edge_count.values()):
        raise QualificationError("SOURCE_PIN_OPEN_SURFACE_NONMANIFOLD")
    boundary = [e for e,n in edge_count.items() if n == 1]
    if len(boundary) < 3:
        raise QualificationError("SOURCE_PIN_OPEN_HEM_MISSING")
    degree: dict[str, int] = {}
    for a,b in boundary:
        degree[a] = degree.get(a,0) + 1
        degree[b] = degree.get(b,0) + 1
    if any(n != 2 for n in degree.values()):
        raise QualificationError("SOURCE_PIN_OPEN_HEM_NOT_MANIFOLD")
    visited = {next(iter(owned))}
    todo = list(visited)
    while todo:
        v = todo.pop()
        for u in neighbors[v] - visited:
            visited.add(u)
            todo.append(u)
    if visited != owned:
        raise QualificationError("SOURCE_PIN_OPEN_COMPONENT_DISCONNECTED")
    # Geometry quality and self-contact remain separate runtime/product courts.
    return selected


def prepare_source_qualified_open_surface_xpbd_v1(
    *,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    program: SourceQualifiedOpenSurfacePinProgramV1,
    trusted_native_pin_registry: Mapping[str, Mapping[str, object]],
    trusted_compiler_transport_registry: Mapping[str, Mapping[str, object]],
) -> SourceQualifiedXPBDSurfaceActivationV1:
    """Compile native, intervened pin evidence to *declared* XPBD domain.

    Both registries are independently admitted trusted inputs, not model
    predictions. This method never runs a timestep or selects stiffness/
    mass/gravity. The owning runtime still must activate an explicit
    XPBDSurfacePolicyV1 and pass its disjoint physics ownership court.
    """
    if type(program) is not SourceQualifiedOpenSurfacePinProgramV1:
        raise QualificationError("SOURCE_PIN_PROGRAM_TYPE_INVALID")
    if (
        program.schema_version != SCHEMA
        or any(not _sha(s) for s in (
            program.source_asset_sha256,
            program.dataset_manifest_sha256,
            program.candidate_mstar_topology_address_sha256,
            program.canonical_mesh_lineage_hash,
            program.native_open_boundary_sha256,
            program.native_pin_witness_sha256,
            program.host_only_intervention_sha256,
            program.source_face_pin_transport_sha256,
        ))
        or not program.source_qualification_key
        or not program.canonical_component_id
        or not program.host_canonical_chain_id
        or not program.source_native_face_indices
        or not program.source_native_pin_vertex_ids
        or len(program.source_native_face_indices)
            != len(program.canonical_face_indices)
        or len(program.source_native_pin_vertex_ids)
            != len(program.canonical_pin_vertex_ids)
        # Source faces may repeat when TESSA refines one native parent into
        # several canonical children; canonical face indices remain unique.
        or len(set(program.source_native_pin_vertex_ids))
            != len(program.source_native_pin_vertex_ids)
        or len(set(program.canonical_pin_vertex_ids))
            != len(program.canonical_pin_vertex_ids)
        or any(type(i) is not int or i < 0
               for i in program.source_native_face_indices)
        or any(type(v) is not str or not v
               for v in program.source_native_pin_vertex_ids)
        or program.canonical_mesh_lineage_hash != mesh.mesh_lineage_hash
        or program.transport_hash() != program.source_face_pin_transport_sha256
    ):
        raise QualificationError("SOURCE_PIN_PROGRAM_ADDRESS_OR_HASH_DRIFT")
    _validate_native_face_corners_v1(program)
    source_row = trusted_native_pin_registry.get(program.source_qualification_key)
    transport_row = trusted_compiler_transport_registry.get(
        program.source_face_pin_transport_sha256
    )
    if not isinstance(source_row, Mapping) or not isinstance(transport_row, Mapping):
        raise QualificationError("SOURCE_PIN_SOURCE_AND_COMPILER_RECEIPTS_REQUIRED")
    expected_source = {
        "admission_status": PIN_ADMISSION,
        "source_qualification_key": program.source_qualification_key,
        "source_asset_sha256": program.source_asset_sha256,
        "dataset_manifest_sha256": program.dataset_manifest_sha256,
        "native_open_boundary_sha256": program.native_open_boundary_sha256,
        "native_pin_witness_sha256": program.native_pin_witness_sha256,
        "host_only_intervention_sha256": program.host_only_intervention_sha256,
        "candidate_mstar_topology_address_sha256":
            program.candidate_mstar_topology_address_sha256,
        "source_face_pin_transport_sha256":
            program.source_face_pin_transport_sha256,
        "program_witness_sha256": program.witness_hash(),
        "canonical_component_id": program.canonical_component_id,
        "host_canonical_chain_id": program.host_canonical_chain_id,
        "source_rights_independently_verified": True,
        "native_open_hem_and_pins_independently_verified": True,
        "heldout_host_only_intervention_confirmed": True,
        "generated_views_used_as_source_truth": False,
        "student_slot_ids_used_as_source_truth": False,
        "material_coefficients_claimed_as_source_truth": False,
    }
    expected_transport = {
        "admission_status": TRANSPORT_ADMISSION,
        "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
        "source_candidate_lineage_hash": mesh.source_candidate_lineage_hash,
        "candidate_mstar_topology_address_sha256":
            program.candidate_mstar_topology_address_sha256,
        "source_asset_sha256": program.source_asset_sha256,
        "native_pin_witness_sha256": program.native_pin_witness_sha256,
        "source_face_pin_transport_sha256":
            program.source_face_pin_transport_sha256,
        "independently_qualified_final_mstar_address": True,
    }
    if (
        any(source_row.get(k) != v for k,v in expected_source.items())
        or any(transport_row.get(k) != v for k,v in expected_transport.items())
    ):
        raise QualificationError("SOURCE_PIN_NATIVE_OR_MSTAR_RECEIPT_DRIFT")
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    selected = _admit_open_patch(
        mesh,
        canonical_component_id=program.canonical_component_id,
        selected_face_indices=program.canonical_face_indices,
    )
    selected_ids = set(v for face in selected for v in face)
    pins = program.canonical_pin_vertex_ids
    if not set(pins).issubset(selected_ids) or len(pins) == len(selected_ids):
        raise QualificationError("SOURCE_PIN_NOT_ON_OPEN_COMPONENT_OR_NO_FREE_SURFACE")
    chains = {c.canonical_chain_id: c for c in rig.chains}
    chain = chains.get(program.host_canonical_chain_id)
    if chain is None or not chain.canonical_joint_ids:
        raise QualificationError("SOURCE_PIN_HOST_CHAIN_NOT_IN_CANONICAL_RIG")
    rows = {row.canonical_mesh_vertex_id: row for row in skin.rows}
    host_joints = set(chain.canonical_joint_ids)
    if any(
        p not in rows or not any(
            jid in host_joints and weight > 0.
            for jid,weight in rows[p].influences
        )
        for p in pins
    ):
        raise QualificationError("SOURCE_PIN_HOST_SKIN_CONSEQUENCE_UNSUPPORTED")
    # No topology/rig/skin/Mechanics redesign or invented physical parameters.
    domain = prepare_declared_xpbd_surface_domain_v1(
        mesh,rig,skin,mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=selected,
            pinned_vertex_ids=pins,
            activation_authority="EXPLICIT_RUNTIME_POLICY",
        ),
    )
    if domain.mesh_lineage_hash != program.canonical_mesh_lineage_hash:
        raise QualificationError("SOURCE_PIN_XPBD_DOMAIN_PARENT_DRIFT")
    digest = content_sha256({
        "schema": RECEIPT_SCHEMA,
        "domain_hash": domain.domain_hash,
        "source_witness_sha256": program.witness_hash(),
        "source_face_pin_transport_sha256":
            program.source_face_pin_transport_sha256,
        "source_registry_admission": PIN_ADMISSION,
        "compiler_registry_admission": TRANSPORT_ADMISSION,
        "host_canonical_chain_id": program.host_canonical_chain_id,
    })
    return SourceQualifiedXPBDSurfaceActivationV1(
        domain=domain,
        program_witness_sha256=program.witness_hash(),
        candidate_mstar_topology_address_sha256=
            program.candidate_mstar_topology_address_sha256,
        source_face_pin_transport_sha256=
            program.source_face_pin_transport_sha256,
        host_canonical_chain_id=program.host_canonical_chain_id,
        canonical_component_id=program.canonical_component_id,
        qualification_receipt_sha256=digest,
    )


__all__ = [
    "SCHEMA", "PIN_ADMISSION", "TRANSPORT_ADMISSION",
    "SourceQualifiedOpenSurfacePinProgramV1",
    "SourceQualifiedXPBDSurfaceActivationV1",
    "prepare_source_qualified_open_surface_xpbd_v1",
]
