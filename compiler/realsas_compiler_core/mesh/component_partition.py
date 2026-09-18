from __future__ import annotations

"""Render projection for Compiler-owned mechanical components.

The continuity substrate is intentionally the complete qualified source mesh. Rigid
mechanical-role components are exact pure-face overlays extracted from that same mesh.
This avoids creating render holes at component boundaries: every source face is always
present in BODY_UNDERLAY, while rigid overlays add only faces whose complete qualified
surface support belongs to one rigid component.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..mechanical_component_partition import validate_mechanical_component_partition
from ..types import QualificationError
from ..v4 import build_appearance_binding
from .mesh_binding import (
    mesh_lineage_hash,
    mesh_skin_lineage_hash,
    validate_qualified_mesh,
    validate_qualified_mesh_skin,
)

BODY_COMPONENT_ID = "BODY_UNDERLAY"


@dataclass(frozen=True)
class ProjectedMechanicalComponentIR:
    component_id: str
    source_face_indices: tuple[int, ...]
    mesh: Any
    mesh_skin: Any
    appearance: Any
    schema_version: str = "RealSaS.ProjectedMechanicalComponentIR.v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "source_face_indices": list(self.source_face_indices),
            "mesh": self.mesh.to_dict(),
            "mesh_skin": self.mesh_skin.to_dict(),
            "appearance": self.appearance.to_dict(),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class MechanicalComponentMeshProjectionIR:
    source_mesh_lineage_hash: str
    source_mesh_skin_lineage_hash: str
    source_appearance_lineage_hash: str
    component_partition_hash: str
    components: tuple[ProjectedMechanicalComponentIR, ...]
    ambiguous_vertex_ids: tuple[str, ...]
    cross_component_face_indices: tuple[int, ...]
    qualification_report: dict[str, Any]
    schema_version: str = "RealSaS.MechanicalComponentMeshProjectionIR.v2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vertex_component(mesh_vertex, assignment_by_surface: dict[str, str]):
    components = set()
    for sid, coeff in mesh_vertex.support_binding.coefficients:
        if float(coeff) <= 1.0e-12:
            continue
        component_id = assignment_by_surface.get(str(sid))
        if component_id is None:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_MESH_UNKNOWN_SURFACE_SUPPORT:{sid}"
            )
        components.add(component_id)
    if not components:
        raise QualificationError(
            f"MECHANICAL_COMPONENT_MESH_EMPTY_SURFACE_SUPPORT:{mesh_vertex.canonical_mesh_vertex_id}"
        )
    if len(components) != 1:
        return None
    return next(iter(components))


def _subset_edges(faces):
    edges = set()
    for face in faces:
        if len(face) != 3:
            raise QualificationError("MECHANICAL_COMPONENT_MESH_REQUIRES_TRIANGLES")
        for i in range(3):
            a, b = str(face[i]), str(face[(i + 1) % 3])
            if a == b:
                raise QualificationError("MECHANICAL_COMPONENT_MESH_DEGENERATE_FACE")
            edges.add(tuple(sorted((a, b))))
    return tuple(sorted(edges))


def _subset_component(
    *,
    component_id: str,
    rows: list[tuple[int, tuple[str, ...]]],
    source_mesh,
    source_mesh_skin,
    source_appearance,
    mechanical,
    partition,
) -> ProjectedMechanicalComponentIR:
    source_face_indices = tuple(old_index for old_index, _face in rows)
    faces = tuple(face for _old_index, face in rows)
    used = {vid for face in faces for vid in face}
    vertices = tuple(
        vertex for vertex in source_mesh.vertices
        if str(vertex.canonical_mesh_vertex_id) in used
    )
    edges = _subset_edges(faces)

    mesh = replace(
        source_mesh,
        vertices=vertices,
        faces=faces,
        edges=edges,
        qualification_report={
            "status": "PASS_EXACT_RIGID_MECHANICAL_OVERLAY_FACE_SUBSET",
            "passed": True,
            "source_mesh_lineage_hash": str(source_mesh.mesh_lineage_hash),
            "component_partition_hash": str(partition.partition_hash),
            "component_id": component_id,
            "source_face_count": len(source_mesh.faces),
            "retained_face_count": len(faces),
            "vertices_moved": False,
            "support_bindings_mutated": False,
            "retriangulated": False,
            "new_geometry_generated": False,
        },
        mesh_lineage_hash="",
        support_coverage_classification="MECHANICAL_RIGID_PURE_FACE_OVERLAY",
        metadata={
            **dict(source_mesh.metadata or {}),
            "source_mesh_lineage_hash": str(source_mesh.mesh_lineage_hash),
            "component_partition_hash": str(partition.partition_hash),
            "mechanical_component_id": component_id,
            "exact_face_subset_projection": True,
            "continuity_underlay_owns_boundary_faces": True,
            "new_geometry_generated": False,
        },
    )
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))
    validate_qualified_mesh(mesh, mechanical.surface)

    mesh_skin = replace(
        source_mesh_skin,
        rows=tuple(
            row for row in source_mesh_skin.rows
            if str(row.canonical_mesh_vertex_id) in used
        ),
        mesh_binding_hash=str(mesh.mesh_lineage_hash),
        qualification_report={
            **dict(source_mesh_skin.qualification_report or {}),
            "status": "PASS_EXACT_RIGID_MECHANICAL_OVERLAY_SKIN_SUBSET",
            "source_mesh_skin_lineage_hash": str(source_mesh_skin.mesh_skin_lineage_hash),
            "component_partition_hash": str(partition.partition_hash),
            "component_id": component_id,
            "weight_rows_mutated": False,
            "only_row_subset_and_mesh_binding_changed": True,
        },
        mesh_skin_lineage_hash="",
        metadata={
            **dict(source_mesh_skin.metadata or {}),
            "source_mesh_skin_lineage_hash": str(source_mesh_skin.mesh_skin_lineage_hash),
            "component_partition_hash": str(partition.partition_hash),
            "mechanical_component_id": component_id,
            "weight_values_preserved_exactly": True,
        },
    )
    mesh_skin = replace(
        mesh_skin,
        mesh_skin_lineage_hash=mesh_skin_lineage_hash(mesh_skin),
    )
    validate_qualified_mesh_skin(
        mesh_skin,
        surface=mechanical.surface,
        skeleton=mechanical.skeleton,
        skin=mechanical.skin,
        mesh=mesh,
    )

    corner_by_key = {
        (int(corner.face_index), int(corner.corner_index)): corner
        for corner in source_appearance.corner_bindings
    }
    corners = []
    for new_face_index, old_face_index in enumerate(source_face_indices):
        for corner_index in range(len(source_mesh.faces[old_face_index])):
            source_corner = corner_by_key.get((old_face_index, corner_index))
            if source_corner is None:
                raise QualificationError(
                    f"MECHANICAL_COMPONENT_MESH_APPEARANCE_CORNER_MISSING:{old_face_index}:{corner_index}"
                )
            corners.append(replace(source_corner, face_index=int(new_face_index)))
    appearance = build_appearance_binding(
        target_view_index=int(source_appearance.target_view_index),
        mesh_binding_hash=str(mesh.mesh_lineage_hash),
        camera_binding_hash=str(source_appearance.camera_binding_hash),
        corner_bindings=tuple(corners),
        atlas_payload_hash=str(source_appearance.atlas_payload_hash),
        metadata={
            **dict(source_appearance.metadata or {}),
            "source_appearance_lineage_hash": str(source_appearance.appearance_lineage_hash),
            "component_partition_hash": str(partition.partition_hash),
            "mechanical_component_id": component_id,
            "artist_corner_payload_preserved": True,
            "new_pixels_generated": False,
        },
    )
    return ProjectedMechanicalComponentIR(
        component_id=component_id,
        source_face_indices=source_face_indices,
        mesh=mesh,
        mesh_skin=mesh_skin,
        appearance=appearance,
    )


def project_mesh_to_mechanical_components(
    *,
    source_mesh,
    source_mesh_skin,
    source_appearance,
    partition,
    mechanical,
    require_nonempty_components: bool = True,
) -> MechanicalComponentMeshProjectionIR:
    """Keep full qualified geometry as continuity underlay + exact rigid overlays.

    This is deliberately not a disjoint face partition. BODY_UNDERLAY contains every
    source face and therefore owns coverage/continuity. Rigid mechanical-role overlays
    contain only faces whose every vertex has support exclusively inside that rigid
    component. Mixed/cross-component faces are never guessed, clipped, or deleted;
    they remain source-backed in the underlay.
    """
    validate_mechanical_component_partition(
        partition,
        surface=mechanical.surface,
        skeleton=mechanical.skeleton,
        skin=mechanical.skin,
    )
    validate_qualified_mesh(source_mesh, mechanical.surface)
    validate_qualified_mesh_skin(
        source_mesh_skin,
        surface=mechanical.surface,
        skeleton=mechanical.skeleton,
        skin=mechanical.skin,
        mesh=source_mesh,
    )
    if source_appearance.mesh_binding_hash != source_mesh.mesh_lineage_hash:
        raise QualificationError("MECHANICAL_COMPONENT_MESH_APPEARANCE_MESH_DRIFT")
    if source_appearance.target_view_index != source_mesh.view_index:
        raise QualificationError("MECHANICAL_COMPONENT_MESH_APPEARANCE_VIEW_DRIFT")
    if BODY_COMPONENT_ID not in partition.component_surface_ids:
        raise QualificationError("MECHANICAL_COMPONENT_MESH_BODY_PARTITION_MISSING")

    assignment_by_surface = {
        str(row.surface_id): str(row.component_id)
        for row in partition.assignments
    }
    vertex_by_id = {
        str(vertex.canonical_mesh_vertex_id): vertex
        for vertex in source_mesh.vertices
    }
    if len(vertex_by_id) != len(source_mesh.vertices):
        raise QualificationError("MECHANICAL_COMPONENT_MESH_DUPLICATE_VERTEX_ID")
    skin_by_id = {
        str(row.canonical_mesh_vertex_id): row for row in source_mesh_skin.rows
    }
    if set(skin_by_id) != set(vertex_by_id):
        raise QualificationError("MECHANICAL_COMPONENT_MESH_SKIN_VERTEX_ACCOUNTING_DRIFT")

    vertex_component = {}
    ambiguous = []
    for vid, vertex in vertex_by_id.items():
        cid = _vertex_component(vertex, assignment_by_surface)
        if cid is None:
            ambiguous.append(vid)
        else:
            vertex_component[vid] = cid

    rigid_ids = tuple(
        sorted(
            cid
            for cid, mechanical_class in partition.component_mechanical_classes.items()
            if str(mechanical_class) == "RIGID_SKINNED_COMPONENT"
        )
    )
    rigid_face_rows: dict[str, list[tuple[int, tuple[str, ...]]]] = {
        cid: [] for cid in rigid_ids
    }
    cross_faces = []
    pure_body_face_count = 0
    for face_index, face in enumerate(source_mesh.faces):
        ids = tuple(map(str, face))
        labels = [vertex_component.get(vid) for vid in ids]
        if None in labels or len(set(labels)) != 1:
            cross_faces.append(int(face_index))
            continue
        cid = str(labels[0])
        if cid == BODY_COMPONENT_ID:
            pure_body_face_count += 1
        elif cid in rigid_face_rows:
            rigid_face_rows[cid].append((int(face_index), ids))
        else:
            raise QualificationError("MECHANICAL_COMPONENT_MESH_FACE_COMPONENT_UNKNOWN")

    # Full source mesh is the continuity owner. No face is removed from BODY_UNDERLAY.
    projected = [
        ProjectedMechanicalComponentIR(
            component_id=BODY_COMPONENT_ID,
            source_face_indices=tuple(range(len(source_mesh.faces))),
            mesh=source_mesh,
            mesh_skin=source_mesh_skin,
            appearance=source_appearance,
        )
    ]
    for cid in rigid_ids:
        rows = rigid_face_rows[cid]
        if not rows:
            if require_nonempty_components:
                raise QualificationError(
                    f"MECHANICAL_COMPONENT_MESH_RIGID_COMPONENT_HAS_NO_PURE_FACE:{cid}"
                )
            continue
        projected.append(
            _subset_component(
                component_id=cid,
                rows=rows,
                source_mesh=source_mesh,
                source_mesh_skin=source_mesh_skin,
                source_appearance=source_appearance,
                mechanical=mechanical,
                partition=partition,
            )
        )

    source_face_count = len(source_mesh.faces)
    rigid_overlay_face_count = sum(
        len(row.source_face_indices)
        for row in projected
        if row.component_id != BODY_COMPONENT_ID
    )
    if len(projected[0].source_face_indices) != source_face_count:
        raise QualificationError("MECHANICAL_COMPONENT_MESH_UNDERLAY_FACE_ACCOUNTING_DRIFT")
    return MechanicalComponentMeshProjectionIR(
        source_mesh_lineage_hash=str(source_mesh.mesh_lineage_hash),
        source_mesh_skin_lineage_hash=str(source_mesh_skin.mesh_skin_lineage_hash),
        source_appearance_lineage_hash=str(source_appearance.appearance_lineage_hash),
        component_partition_hash=str(partition.partition_hash),
        components=tuple(projected),
        ambiguous_vertex_ids=tuple(sorted(ambiguous)),
        cross_component_face_indices=tuple(cross_faces),
        qualification_report={
            "passed": True,
            "source_face_count": source_face_count,
            "underlay_face_count": source_face_count,
            "source_face_coverage_fraction": 1.0,
            "pure_body_face_count": int(pure_body_face_count),
            "rigid_overlay_face_count": int(rigid_overlay_face_count),
            "cross_component_face_count": len(cross_faces),
            "ambiguous_vertex_count": len(ambiguous),
            "boundary_completion_performed": False,
            "boundary_faces_deleted": False,
            "teacher_truth_used": False,
            "new_geometry_generated": False,
        },
        metadata={
            "authority": "QUALIFIED_MECHANICAL_COMPONENT_PARTITION",
            "continuity_policy": "FULL_SOURCE_MESH_UNDERLAY_PLUS_EXACT_RIGID_PURE_FACE_OVERLAYS",
            "cross_component_faces_remain_in_underlay": True,
            "disjoint_render_face_partition_claimed": False,
        },
    )


__all__ = [
    "BODY_COMPONENT_ID",
    "ProjectedMechanicalComponentIR",
    "MechanicalComponentMeshProjectionIR",
    "project_mesh_to_mechanical_components",
]
