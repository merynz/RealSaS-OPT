from __future__ import annotations

"""Compiler-owned per-component source-backed mesh materialization.

This module materializes each qualified mechanical component directly from current
Surface/Skeleton/Skin authority plus exact source observations.  It deliberately does
not consume a historical full-subject render mesh, source-component truth, owner
rasters, semantic filenames, or teacher topology.

The full observation alpha is a hard containment authority for every component mesh.
Coverage is judged on the UNION of all qualified component meshes, while each
individual component is qualified against its exact mechanical surface-id domain.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..appearance import build_observed_appearance_binding
from ..hashing import content_sha256
from ..mechanical_component_partition import validate_mechanical_component_partition
from ..types import QualifiedMeshVertex, QualificationError
from .mwb2_cdt import (
    build_mwb2_observation_cdt_candidate,
    build_mwb2_supported_kernel_id_triangles,
    qualify_mwb2_component_observation_cdt_mesh,
)
from .mesh_binding import mesh_lineage_hash, validate_qualified_mesh
from .mwb2_skin import bind_mwb2_mesh_skin
from .quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    _triangle_metrics,
    coverage_gate_failures,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
)


COMPONENT_MATERIALIZATION_SCHEMA = "RealSaS.MechanicalComponentObservationMaterialization.v1"


@dataclass(frozen=True)
class MaterializedMechanicalComponentIR:
    component_id: str
    mesh: Any
    mesh_skin: Any
    appearance: Any
    qualification_report: Mapping[str, Any]
    schema_version: str = COMPONENT_MATERIALIZATION_SCHEMA + ".Component"

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": str(self.component_id),
            "mesh": self.mesh.to_dict(),
            "mesh_skin": self.mesh_skin.to_dict(),
            "appearance": self.appearance.to_dict(),
            "qualification_report": dict(self.qualification_report),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class MechanicalComponentViewMaterializationIR:
    view_index: int
    component_partition_hash: str
    components: tuple[MaterializedMechanicalComponentIR, ...]
    union_coverage_report: Mapping[str, Any]
    qualification_report: Mapping[str, Any]
    materialization_hash: str
    schema_version: str = COMPONENT_MATERIALIZATION_SCHEMA + ".View"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_index": int(self.view_index),
            "component_partition_hash": str(self.component_partition_hash),
            "components": [row.to_dict() for row in self.components],
            "union_coverage_report": dict(self.union_coverage_report),
            "qualification_report": dict(self.qualification_report),
            "materialization_hash": str(self.materialization_hash),
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }


def mechanical_component_view_materialization_hash(
    value: MechanicalComponentViewMaterializationIR,
) -> str:
    payload = value.to_dict()
    payload.pop("materialization_hash", None)
    return content_sha256(payload)


def _raster_triangles(mesh) -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
]:
    by_id: dict[str, tuple[float, float]] = {}
    for vertex in mesh.vertices:
        metadata = dict(getattr(vertex, "metadata", {}) or {})
        xy = metadata.get("raster_xy")
        if xy is None or len(tuple(xy)) != 2:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_MATERIALIZATION_RASTER_BINDING_MISSING:"
                f"{vertex.canonical_mesh_vertex_id}"
            )
        by_id[str(vertex.canonical_mesh_vertex_id)] = tuple(map(float, xy))
    out = []
    for face in mesh.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(vertex_id not in by_id for vertex_id in ids):
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_FACE_BINDING_INVALID"
            )
        out.append((by_id[ids[0]], by_id[ids[1]], by_id[ids[2]]))
    return tuple(out)


def _repair_component_policy_slivers(
    mesh,
    *,
    surface,
    view_index: int,
    component_id: str,
):
    """Remove only frozen-policy sliver faces, without losing any admitted vertex.

    This is deliberately narrower than remeshing: no vertex moves, no new vertex is
    generated, no support binding changes, and no component identity changes.  The
    repair is admitted only when every original component-local vertex remains incident
    to at least one retained face. Full-subject coverage is re-qualified later on the
    union of all component meshes.
    """

    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    before = mesh_raster_quality_report(
        mesh,
        surface=surface,
        view_index=int(view_index),
    )
    failures = tuple(
        raster_quality_gate_failures(before, policy=policy)
    )
    if not failures:
        return mesh, {
            "performed": False,
            "removed_face_indices": (),
            "before": before,
            "after": before,
        }

    allowed = {
        "min_raster_triangle_angle_deg",
        "max_raster_triangle_aspect_ratio",
    }
    if not set(failures).issubset(allowed):
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_UNREPAIRABLE_RASTER_QUALITY:"
            f"V{view_index}:{component_id}:{','.join(failures)}"
        )

    raster_xy = {}
    surface_raster = {}
    for node in surface.surface_nodes:
        rows = [
            tuple(map(float, xy))
            for v, xy in node.raster_bindings
            if int(v) == int(view_index)
        ]
        if len(rows) > 1:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_DUPLICATE_SURFACE_RASTER:"
                f"V{view_index}:{node.surface_id}"
            )
        if rows:
            surface_raster[str(node.surface_id)] = rows[0]

    for vertex in mesh.vertices:
        cached = dict(getattr(vertex, "metadata", {}) or {}).get("raster_xy")
        if cached is not None:
            raster_xy[str(vertex.canonical_mesh_vertex_id)] = tuple(
                map(float, cached)
            )
            continue
        x = y = total = 0.0
        for sid, coeff in vertex.support_binding.coefficients:
            xy = surface_raster.get(str(sid))
            if xy is None:
                raise QualificationError(
                    "MECHANICAL_COMPONENT_MATERIALIZATION_SUPPORT_RASTER_MISSING:"
                    f"V{view_index}:{sid}"
                )
            w = float(coeff)
            x += w * xy[0]
            y += w * xy[1]
            total += w
        if abs(total - 1.0) > 1.0e-8:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_SUPPORT_SIMPLEX_DRIFT"
            )
        raster_xy[str(vertex.canonical_mesh_vertex_id)] = (x, y)

    bad = []
    for face_index, face in enumerate(mesh.faces):
        ids = tuple(map(str, face))
        pa, pb, pc = (raster_xy[vid] for vid in ids)
        area, angle, aspect = _triangle_metrics(pa, pb, pc)
        if float(area) <= 1.0e-12:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_DEGENERATE_FACE:"
                f"V{view_index}:{component_id}:F{face_index}"
            )
        if (
            float(angle) < float(policy.min_raster_triangle_angle_deg)
            or float(aspect) > float(policy.max_raster_triangle_aspect_ratio)
        ):
            bad.append(int(face_index))

    if not bad:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_QUALITY_FAILURE_NOT_LOCALIZED:"
            f"V{view_index}:{component_id}:{','.join(failures)}"
        )
    if len(bad) >= len(mesh.faces):
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_SLIVER_REPAIR_WOULD_EMPTY_COMPONENT:"
            f"V{view_index}:{component_id}"
        )

    removed = set(bad)
    retained_faces = tuple(
        face for index, face in enumerate(mesh.faces) if index not in removed
    )
    before_vertices = {
        str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices
    }
    retained_vertices = {
        str(vertex_id) for face in retained_faces for vertex_id in face
    }
    if retained_vertices != before_vertices:
        lost = tuple(sorted(before_vertices - retained_vertices))
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_SLIVER_REPAIR_ORPHANS_VERTEX:"
            f"V{view_index}:{component_id}:{lost[:8]}"
        )

    retained_edges = tuple(
        sorted(
            {
                tuple(sorted((str(face[i]), str(face[(i + 1) % 3]))))
                for face in retained_faces
                for i in range(3)
            }
        )
    )
    repaired = replace(
        mesh,
        faces=retained_faces,
        edges=retained_edges,
        qualification_report={
            **dict(mesh.qualification_report or {}),
            "status": "PASS_COMPONENT_LOCAL_EXACT_SLIVER_FACE_SUBSET_REPAIR",
            "source_mesh_lineage_hash": str(mesh.mesh_lineage_hash),
            "removed_face_indices": tuple(bad),
            "vertices_preserved_exactly": True,
            "support_bindings_preserved_exactly": True,
            "new_geometry_generated": False,
            "retriangulated": False,
            "policy_thresholds_changed": False,
        },
        metadata={
            **dict(mesh.metadata or {}),
            "component_local_exact_sliver_face_subset_repair": True,
            "source_mesh_lineage_hash": str(mesh.mesh_lineage_hash),
            "removed_face_indices": tuple(bad),
        },
        mesh_lineage_hash="",
    )
    repaired = replace(
        repaired,
        mesh_lineage_hash=mesh_lineage_hash(repaired),
    )
    validate_qualified_mesh(repaired, surface)
    after = mesh_raster_quality_report(
        repaired,
        surface=surface,
        view_index=int(view_index),
    )
    after_failures = tuple(
        raster_quality_gate_failures(after, policy=policy)
    )
    if after_failures:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_SLIVER_REPAIR_INCOMPLETE:"
            f"V{view_index}:{component_id}:{','.join(after_failures)}"
        )
    return repaired, {
        "performed": True,
        "removed_face_indices": tuple(bad),
        "before": before,
        "after": after,
        "vertices_preserved_exactly": True,
        "new_geometry_generated": False,
        "retriangulated": False,
    }


def _identity_surface_id(vertex) -> str:
    binding = vertex.support_binding
    coeffs = tuple(binding.coefficients)
    if (
        binding.mode != "IDENTITY_SURFACE_NODE"
        or len(coeffs) != 1
        or abs(float(coeffs[0][1]) - 1.0) > 1.0e-12
    ):
        raise QualificationError(
            "MECHANICAL_COMPONENT_BOUNDARY_SEAM_REQUIRES_IDENTITY_SUPPORT"
        )
    return str(coeffs[0][0])


def _augment_body_with_boundary_seam(
    body_mesh,
    *,
    surface,
    partition,
    view_index: int,
    camera_binding_hash: str,
    observation_domain,
):
    """Add only compiler-owned mixed-component seam faces from current S.

    The five mechanical interiors remain component-local CDT meshes.  This operator
    uses a fresh current-S observation-domain CDT only as a seam proposal source and
    admits exclusively faces whose vertices span multiple qualified mechanical
    components.  It never consumes historical P1/P1Q topology, teacher component art,
    owner rasters, or semantic filenames.

    Mixed seam vertices keep their exact current SurfaceSupportBinding and therefore
    receive exact current skin weights downstream.  The seam is carried by the
    deformable BODY render component solely as continuity geometry; it does not mutate
    the MechanicalComponentPartitionIR ownership table.
    """

    view_index = int(view_index)
    candidate = build_mwb2_observation_cdt_candidate(
        surface,
        view_index=view_index,
        camera_binding_hash=str(camera_binding_hash),
        observation_domain=observation_domain,
        candidate_namespace="BOUNDARY_SEAM",
    )
    assignment = {
        str(row.surface_id): str(row.component_id)
        for row in partition.assignments
    }
    candidate_vertex = {
        str(vertex.candidate_vertex_id): vertex
        for vertex in candidate.vertices
    }
    if len(candidate_vertex) != len(candidate.vertices):
        raise QualificationError(
            "MECHANICAL_COMPONENT_BOUNDARY_SEAM_DUPLICATE_CANDIDATE_VERTEX"
        )

    body_sid_to_vid = {}
    vertices = list(body_mesh.vertices)
    for vertex in body_mesh.vertices:
        sid = _identity_surface_id(vertex)
        if sid in body_sid_to_vid:
            raise QualificationError(
                "MECHANICAL_COMPONENT_BOUNDARY_SEAM_DUPLICATE_BODY_SURFACE"
            )
        body_sid_to_vid[sid] = str(vertex.canonical_mesh_vertex_id)

    faces = [tuple(map(str, face)) for face in body_mesh.faces]
    face_keys = {tuple(sorted(face)) for face in faces}
    edge_incidence: dict[tuple[str, str], int] = {}
    for face in faces:
        for index in range(3):
            edge = tuple(sorted((face[index], face[(index + 1) % 3])))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1

    seam_vertex_by_sid: dict[str, str] = {}
    accepted = []
    rejected_quality = 0
    rejected_topology = 0
    mixed_proposal_count = 0
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1

    for source_face_index, face in enumerate(candidate.faces):
        raw_vertices = tuple(candidate_vertex[str(vertex_id)] for vertex_id in face)
        sids = tuple(_identity_surface_id(vertex) for vertex in raw_vertices)
        try:
            component_ids = tuple(assignment[sid] for sid in sids)
        except KeyError as exc:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_BOUNDARY_SEAM_UNKNOWN_SURFACE:{exc.args[0]}"
            ) from exc
        if len(set(component_ids)) <= 1:
            continue
        mixed_proposal_count += 1

        tri_xy = tuple(
            tuple(map(float, dict(vertex.metadata or {})["raster_xy"]))
            for vertex in raw_vertices
        )
        area, angle, aspect = _triangle_metrics(*tri_xy)
        if (
            float(area) <= 1.0e-12
            or float(angle) < float(policy.min_raster_triangle_angle_deg)
            or float(aspect) > float(policy.max_raster_triangle_aspect_ratio)
        ):
            rejected_quality += 1
            continue
        if not observation_domain.triangle_inside(tri_xy):
            raise QualificationError(
                "MECHANICAL_COMPONENT_BOUNDARY_SEAM_ESCAPES_OBSERVATION_ALPHA"
            )

        canonical_ids = []
        staged_new = []
        for raw_vertex, sid in zip(raw_vertices, sids):
            canonical_id = body_sid_to_vid.get(sid)
            if canonical_id is None:
                canonical_id = seam_vertex_by_sid.get(sid)
            if canonical_id is None:
                canonical_id = f"SEAM:V{view_index}:{sid}"
                staged_new.append((sid, canonical_id, raw_vertex))
            canonical_ids.append(canonical_id)
        canonical_face = tuple(canonical_ids)
        if len(set(canonical_face)) != 3:
            rejected_topology += 1
            continue
        key = tuple(sorted(canonical_face))
        if key in face_keys:
            continue

        face_edges = tuple(
            tuple(
                sorted(
                    (
                        canonical_face[index],
                        canonical_face[(index + 1) % 3],
                    )
                )
            )
            for index in range(3)
        )
        if any(edge_incidence.get(edge, 0) >= 2 for edge in face_edges):
            rejected_topology += 1
            continue

        for sid, canonical_id, raw_vertex in staged_new:
            if sid in seam_vertex_by_sid or sid in body_sid_to_vid:
                continue
            seam_vertex_by_sid[sid] = canonical_id
            vertices.append(
                QualifiedMeshVertex(
                    canonical_mesh_vertex_id=canonical_id,
                    P=tuple(map(float, raw_vertex.P)),
                    support_binding=raw_vertex.support_binding,
                    source_candidate_vertex_id=str(raw_vertex.candidate_vertex_id),
                    metadata={
                        **dict(raw_vertex.metadata or {}),
                        "compiler_boundary_seam_vertex": True,
                        "mechanical_component_id": assignment[sid],
                    },
                )
            )
        faces.append(canonical_face)
        face_keys.add(key)
        for edge in face_edges:
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
        accepted.append(
            {
                "source_face_index": int(source_face_index),
                "surface_ids": sids,
                "component_ids": component_ids,
            }
        )

    if not accepted:
        return body_mesh, {
            "performed": False,
            "mixed_proposal_count": int(mixed_proposal_count),
            "accepted_face_count": 0,
            "added_vertex_count": 0,
            "rejected_quality_face_count": int(rejected_quality),
            "rejected_topology_face_count": int(rejected_topology),
            "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
        }

    edges = tuple(sorted(edge_incidence))
    merged = replace(
        body_mesh,
        vertices=tuple(vertices),
        faces=tuple(faces),
        edges=edges,
        qualification_report={
            **dict(body_mesh.qualification_report or {}),
            "status": "PASS_COMPONENT_LOCAL_PLUS_CURRENT_S_BOUNDARY_SEAM",
            "source_component_local_mesh_lineage_hash": str(
                body_mesh.mesh_lineage_hash
            ),
            "boundary_seam_source_candidate_lineage_hash": str(
                candidate.candidate_lineage_hash
            ),
            "boundary_seam_face_count": len(accepted),
            "boundary_seam_added_vertex_count": len(seam_vertex_by_sid),
            "mechanical_partition_mutated": False,
            "historical_full_subject_mesh_used": False,
            "teacher_topology_used": False,
        },
        support_coverage_classification=(
            "MECHANICAL_COMPONENT_LOCAL_PLUS_CURRENT_S_BOUNDARY_SEAM"
        ),
        metadata={
            **dict(body_mesh.metadata or {}),
            "compiler_boundary_seam": True,
            "boundary_seam_source_candidate_lineage_hash": str(
                candidate.candidate_lineage_hash
            ),
            "boundary_seam_face_count": len(accepted),
            "boundary_seam_added_vertex_count": len(seam_vertex_by_sid),
            "boundary_seam_authority": (
                "CURRENT_QUALIFIED_S_PLUS_MECHANICAL_PARTITION_ONLY"
            ),
            "mechanical_partition_mutated": False,
            "historical_full_subject_mesh_used": False,
            "teacher_topology_used": False,
        },
        mesh_lineage_hash="",
    )
    merged = replace(merged, mesh_lineage_hash=mesh_lineage_hash(merged))
    validate_qualified_mesh(merged, surface)
    raster_report = mesh_raster_quality_report(
        merged,
        surface=surface,
        view_index=view_index,
    )
    failures = tuple(
        raster_quality_gate_failures(
            raster_report,
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
    )
    if failures:
        raise QualificationError(
            "MECHANICAL_COMPONENT_BOUNDARY_SEAM_RASTER_QUALITY_FAIL:"
            f"V{view_index}:{','.join(failures)}"
        )
    return merged, {
        "performed": True,
        "mixed_proposal_count": int(mixed_proposal_count),
        "accepted_face_count": len(accepted),
        "added_vertex_count": len(seam_vertex_by_sid),
        "rejected_quality_face_count": int(rejected_quality),
        "rejected_topology_face_count": int(rejected_topology),
        "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
        "accepted_faces": tuple(accepted),
        "raster_quality": raster_report,
        "mechanical_partition_mutated": False,
        "historical_full_subject_mesh_used": False,
        "teacher_topology_used": False,
    }


def _support_binding_key(binding) -> tuple[tuple[str, float], ...]:
    coeffs = tuple(
        sorted(
            (
                str(sid),
                round(float(weight), 14),
            )
            for sid, weight in binding.coefficients
            if float(weight) > 1.0e-12
        )
    )
    total = float(sum(weight for _sid, weight in coeffs))
    if not coeffs or abs(total - 1.0) > 1.0e-9:
        raise QualificationError(
            "MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_SUPPORT_SIMPLEX_DRIFT"
        )
    return coeffs


def _rational_support_key(
    parent: tuple[str, str, str],
    nums: tuple[int, int, int],
    n: int,
) -> tuple[tuple[str, int], ...]:
    acc: dict[str, int] = {}
    for sid, num in zip(parent, nums):
        if int(num) <= 0:
            continue
        acc[str(sid)] = acc.get(str(sid), 0) + int(num)
    if sum(acc.values()) != int(n):
        raise QualificationError(
            "MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_RATIONAL_SIMPLEX_DRIFT"
        )
    return tuple(sorted((sid, int(num)) for sid, num in acc.items()))


def _binding_from_rational_support(
    key: tuple[tuple[str, int], ...],
    n: int,
    *,
    view_index: int,
):
    from ..types import SurfaceSupportBinding

    coeffs = tuple(
        (str(sid), float(num) / float(n))
        for sid, num in key
        if int(num) > 0
    )
    if len(coeffs) == 1 and int(key[0][1]) == int(n):
        return SurfaceSupportBinding(
            "IDENTITY_SURFACE_NODE",
            ((str(coeffs[0][0]), 1.0),),
            metadata={
                "observed_view": int(view_index),
                "compiler_residual_recovery": True,
                "generated_geometry": False,
            },
        )
    return SurfaceSupportBinding(
        "LOCAL_CONVEX_INTERPOLATION",
        coeffs,
        metadata={
            "observed_view": int(view_index),
            "compiler_residual_recovery": True,
            "generated_geometry": True,
            "support_denominator": int(n),
        },
    )


def _uniform_subtriangle_support_keys(
    parent: tuple[str, str, str],
    n: int,
):
    def q(i: int, j: int):
        return _rational_support_key(
            parent,
            (int(n) - int(i) - int(j), int(i), int(j)),
            int(n),
        )

    out = []
    for i in range(int(n)):
        for j in range(int(n) - i):
            out.append((q(i, j), q(i + 1, j), q(i, j + 1)))
            if i + j <= int(n) - 2:
                out.append(
                    (
                        q(i + 1, j),
                        q(i + 1, j + 1),
                        q(i, j + 1),
                    )
                )
    if len(out) != int(n) * int(n):
        raise QualificationError(
            "MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_SUBDIVISION_CARDINALITY_DRIFT"
        )
    return tuple(out)


def _derive_support_point(
    *,
    key: tuple[tuple[str, int], ...],
    n: int,
    nodes,
    raster_by_surface,
    view_index: int,
):
    binding = _binding_from_rational_support(
        key,
        int(n),
        view_index=int(view_index),
    )
    px = py = pz = rx = ry = 0.0
    for sid, weight in binding.coefficients:
        node = nodes.get(str(sid))
        raster = raster_by_surface.get(str(sid))
        if node is None or raster is None:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_SUPPORT_NOT_VISIBLE:{sid}"
            )
        coeff = float(weight)
        px += coeff * float(node.P[0])
        py += coeff * float(node.P[1])
        pz += coeff * float(node.P[2])
        rx += coeff * float(raster[0])
        ry += coeff * float(raster[1])
    return binding, (px, py, pz), (rx, ry)


def _mesh_union_mask(meshes, observation_domain):
    mask = bytearray(
        int(observation_domain.height) * int(observation_domain.width)
    )
    width = int(observation_domain.width)
    for mesh in meshes:
        by_id = {
            str(vertex.canonical_mesh_vertex_id): tuple(
                map(float, dict(vertex.metadata or {})["raster_xy"])
            )
            for vertex in mesh.vertices
        }
        for face in mesh.faces:
            tri = tuple(by_id[str(vertex_id)] for vertex_id in face)
            for x, y in observation_domain.triangle_pixels(tri):
                mask[int(y) * width + int(x)] = 1
    return mask


def _augment_body_with_uniform_residual_recovery(
    body_mesh,
    *,
    component_meshes,
    surface,
    view_index: int,
    camera_binding_hash: str,
    observation_domain,
    subdivision_factor: int,
):
    """Recover only source-alpha pixels missing after component-first materialization.

    Candidate geometry comes from the pre-alpha CURRENT-S safe CDT kernel. Every new
    vertex is either an exact Surface carrier or a bounded local convex interpolation
    inside one current-S parent triangle. Only quality-admissible triangles fully
    inside exact source alpha and gaining at least one previously uncovered pixel are
    added. The operator never uses historical P1/P1Q topology or teacher semantics.
    """

    n = int(subdivision_factor)
    if n < 2:
        raise ValueError("subdivision_factor must be >=2")
    view_index = int(view_index)
    kernel = build_mwb2_supported_kernel_id_triangles(
        surface,
        view_index=view_index,
    )
    nodes = {str(node.surface_id): node for node in surface.surface_nodes}
    raster_by_surface = {}
    for sid, node in nodes.items():
        rows = [
            tuple(map(float, xy))
            for raw_view, xy in node.raster_bindings
            if int(raw_view) == view_index
        ]
        if len(rows) > 1:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_DUPLICATE_RASTER:{sid}"
            )
        if rows:
            raster_by_surface[sid] = rows[0]

    authority = observation_domain.mask_bytes
    predicted = _mesh_union_mask(component_meshes, observation_domain)
    if any(
        bool(predicted[index]) and not bool(authority[index])
        for index in range(len(predicted))
    ):
        raise QualificationError(
            "MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_EXISTING_ALPHA_SPILL"
        )
    missing = bytearray(
        1 if bool(authority[index]) and not bool(predicted[index]) else 0
        for index in range(len(predicted))
    )
    missing_before = int(sum(missing))
    if missing_before == 0:
        return body_mesh, {
            "performed": False,
            "subdivision_factor": n,
            "missing_pixels_before": 0,
            "missing_pixels_after": 0,
            "accepted_face_count": 0,
            "added_vertex_count": 0,
        }

    vertices = list(body_mesh.vertices)
    faces = [tuple(map(str, face)) for face in body_mesh.faces]
    face_keys = {tuple(sorted(face)) for face in faces}
    edge_incidence: dict[tuple[str, str], int] = {}
    for face in faces:
        for index in range(3):
            edge = tuple(sorted((face[index], face[(index + 1) % 3])))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1

    existing_by_binding = {}
    for vertex in body_mesh.vertices:
        key = _support_binding_key(vertex.support_binding)
        existing_by_binding.setdefault(
            key,
            str(vertex.canonical_mesh_vertex_id),
        )
    generated_by_binding: dict[tuple[tuple[str, float], ...], str] = {}
    point_cache = {}
    accepted_face_count = 0
    recovered_pixel_count = 0
    rejected_quality = 0
    rejected_alpha = 0
    rejected_no_gain = 0
    rejected_topology = 0
    candidate_subtriangle_count = 0
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1

    for _component_index, parent in kernel["triangles"]:
        parent = tuple(map(str, parent))
        for tri_keys in _uniform_subtriangle_support_keys(parent, n):
            candidate_subtriangle_count += 1
            payload = []
            for key in tri_keys:
                cache_key = (key, n)
                row = point_cache.get(cache_key)
                if row is None:
                    row = _derive_support_point(
                        key=key,
                        n=n,
                        nodes=nodes,
                        raster_by_surface=raster_by_surface,
                        view_index=view_index,
                    )
                    point_cache[cache_key] = row
                payload.append(row)
            tri_xy = tuple(row[2] for row in payload)
            area, angle, aspect = _triangle_metrics(*tri_xy)
            if (
                float(area) <= 1.0e-12
                or float(angle) < float(policy.min_raster_triangle_angle_deg)
                or float(aspect) > float(policy.max_raster_triangle_aspect_ratio)
            ):
                rejected_quality += 1
                continue
            if not observation_domain.triangle_inside(tri_xy):
                rejected_alpha += 1
                continue
            pixels = observation_domain.triangle_pixels(tri_xy)
            if not pixels:
                rejected_no_gain += 1
                continue
            width = int(observation_domain.width)
            pixel_indices = tuple(
                int(y) * width + int(x)
                for x, y in pixels
            )
            gain = int(
                sum(1 for index in pixel_indices if missing[index])
            )
            if gain <= 0:
                rejected_no_gain += 1
                continue

            canonical_ids = []
            staged = []
            for binding, P, raster_xy in payload:
                binding_key = _support_binding_key(binding)
                vertex_id = existing_by_binding.get(binding_key)
                if vertex_id is None:
                    vertex_id = generated_by_binding.get(binding_key)
                if vertex_id is None:
                    vertex_id = (
                        f"RECOVERY:V{view_index}:N{n}:"
                        + content_sha256(
                            {
                                "surface_lineage_hash": surface.geometry_lineage_hash,
                                "binding": binding_key,
                            }
                        )[:24]
                    )
                    staged.append(
                        (
                            binding_key,
                            vertex_id,
                            binding,
                            P,
                            raster_xy,
                        )
                    )
                canonical_ids.append(vertex_id)

            ax, ay = tri_xy[0]
            bx, by = tri_xy[1]
            cx, cy = tri_xy[2]
            area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if area2 < 0.0:
                canonical_ids = [
                    canonical_ids[0],
                    canonical_ids[2],
                    canonical_ids[1],
                ]
            face = tuple(canonical_ids)
            if len(set(face)) != 3:
                rejected_topology += 1
                continue
            face_key = tuple(sorted(face))
            if face_key in face_keys:
                rejected_topology += 1
                continue
            face_edges = tuple(
                tuple(sorted((face[index], face[(index + 1) % 3])))
                for index in range(3)
            )
            if any(edge_incidence.get(edge, 0) >= 2 for edge in face_edges):
                rejected_topology += 1
                continue

            for (
                binding_key,
                vertex_id,
                binding,
                P,
                raster_xy,
            ) in staged:
                if (
                    binding_key in existing_by_binding
                    or binding_key in generated_by_binding
                ):
                    continue
                generated_by_binding[binding_key] = vertex_id
                vertices.append(
                    QualifiedMeshVertex(
                        canonical_mesh_vertex_id=vertex_id,
                        P=tuple(map(float, P)),
                        support_binding=binding,
                        source_candidate_vertex_id="",
                        metadata={
                            "raster_xy": tuple(map(float, raster_xy)),
                            "compiler_residual_recovery_vertex": True,
                            "subdivision_factor": n,
                            "generated_geometry": (
                                binding.mode == "LOCAL_CONVEX_INTERPOLATION"
                            ),
                            "source_mesh_used": False,
                            "teacher_topology_used": False,
                        },
                    )
                )

            faces.append(face)
            face_keys.add(face_key)
            for edge in face_edges:
                edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
            accepted_face_count += 1
            recovered_pixel_count += gain
            for index in pixel_indices:
                predicted[index] = 1
                missing[index] = 0

    if accepted_face_count == 0:
        return body_mesh, {
            "performed": False,
            "subdivision_factor": n,
            "missing_pixels_before": missing_before,
            "missing_pixels_after": int(sum(missing)),
            "accepted_face_count": 0,
            "added_vertex_count": 0,
            "candidate_subtriangle_count": candidate_subtriangle_count,
            "rejected_quality_face_count": rejected_quality,
            "rejected_alpha_face_count": rejected_alpha,
            "rejected_no_gain_face_count": rejected_no_gain,
            "rejected_topology_face_count": rejected_topology,
        }

    merged = replace(
        body_mesh,
        vertices=tuple(vertices),
        faces=tuple(faces),
        edges=tuple(sorted(edge_incidence)),
        qualification_report={
            **dict(body_mesh.qualification_report or {}),
            "status": "PASS_COMPONENT_FIRST_PLUS_CURRENT_S_RESIDUAL_RECOVERY",
            "source_body_mesh_lineage_hash": str(body_mesh.mesh_lineage_hash),
            "residual_recovery_subdivision_factor": n,
            "residual_recovery_face_count": accepted_face_count,
            "residual_recovery_added_vertex_count": len(generated_by_binding),
            "historical_full_subject_mesh_used": False,
            "teacher_topology_used": False,
            "mechanical_partition_mutated": False,
        },
        support_coverage_classification=(
            "MECHANICAL_COMPONENT_FIRST_PLUS_CURRENT_S_RESIDUAL_RECOVERY"
        ),
        metadata={
            **dict(body_mesh.metadata or {}),
            "compiler_residual_recovery": True,
            "residual_recovery_subdivision_factor": n,
            "residual_recovery_face_count": accepted_face_count,
            "residual_recovery_added_vertex_count": len(generated_by_binding),
            "residual_recovery_authority": (
                "CURRENT_QUALIFIED_S_LOCAL_CONVEX_SUPPORT_PLUS_EXACT_ALPHA"
            ),
            "source_mesh_used": False,
            "historical_full_subject_mesh_used": False,
            "teacher_topology_used": False,
            "mechanical_partition_mutated": False,
        },
        mesh_lineage_hash="",
    )
    merged = replace(merged, mesh_lineage_hash=mesh_lineage_hash(merged))
    validate_qualified_mesh(merged, surface)
    raster_report = mesh_raster_quality_report(
        merged,
        surface=surface,
        view_index=view_index,
    )
    failures = tuple(
        raster_quality_gate_failures(
            raster_report,
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
    )
    if failures:
        raise QualificationError(
            "MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_RASTER_QUALITY_FAIL:"
            f"V{view_index}:N{n}:{','.join(failures)}"
        )
    return merged, {
        "performed": True,
        "subdivision_factor": n,
        "missing_pixels_before": missing_before,
        "missing_pixels_after": int(sum(missing)),
        "recovered_pixel_count": int(recovered_pixel_count),
        "accepted_face_count": int(accepted_face_count),
        "added_vertex_count": len(generated_by_binding),
        "candidate_subtriangle_count": int(candidate_subtriangle_count),
        "rejected_quality_face_count": int(rejected_quality),
        "rejected_alpha_face_count": int(rejected_alpha),
        "rejected_no_gain_face_count": int(rejected_no_gain),
        "rejected_topology_face_count": int(rejected_topology),
        "kernel_parent_face_count": len(kernel["triangles"]),
        "raster_quality": raster_report,
        "historical_full_subject_mesh_used": False,
        "teacher_topology_used": False,
        "mechanical_partition_mutated": False,
    }


def _verify_rigid_skin(mesh_skin, *, component_id: str, parent_joint_id: str) -> None:
    parent = str(parent_joint_id)
    if not parent:
        raise QualificationError(
            f"MECHANICAL_COMPONENT_MATERIALIZATION_RIGID_PARENT_MISSING:{component_id}"
        )
    for row in mesh_skin.rows:
        influences = {str(jid): float(weight) for jid, weight in row.influences}
        owner = float(influences.get(parent, 0.0))
        other = float(sum(weight for jid, weight in influences.items() if jid != parent))
        if owner < 0.999 - 1.0e-12 or other > 0.001 + 1.0e-12:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_RIGID_SKIN_DRIFT:"
                f"{component_id}:{row.canonical_mesh_vertex_id}:"
                f"owner={owner}:other={other}"
            )


def materialize_mechanical_component_view(
    *,
    surface,
    skeleton,
    skin,
    mechanical,
    partition,
    view_index: int,
    camera_binding_hash: str,
    observation_domain,
    observation_hash_by_view: Mapping[int, str],
    atlas_payload_hash: str,
    require_product_mesh_quality: bool = True,
) -> MechanicalComponentViewMaterializationIR:
    """Build one view as disjoint mechanical domains directly from current S/G/W.

    Mechanical interiors are emitted from component-local current-S domains. If their
    union cannot satisfy the frozen source-alpha coverage gate, an explicit compiler
    boundary-seam pass may add only mixed-component faces from a fresh current-S CDT
    into BODY continuity geometry. No historical full-subject mesh is subset, clipped,
    or reinterpreted.
    """

    view_index = int(view_index)
    validate_mechanical_component_partition(
        partition,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    observation_domain.validate()
    if int(observation_domain.view_index) != view_index:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_OBSERVATION_VIEW_DRIFT"
        )
    if not camera_binding_hash:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_CAMERA_HASH_REQUIRED"
        )
    if view_index not in observation_hash_by_view:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_OBSERVATION_HASH_MISSING"
        )

    rows: list[MaterializedMechanicalComponentIR] = []
    union_triangles = []
    component_reports: dict[str, Any] = {}

    for component_id in sorted(partition.component_surface_ids):
        allowed_surface_ids = tuple(partition.component_surface_ids[component_id])
        if not allowed_surface_ids:
            raise QualificationError(
                f"MECHANICAL_COMPONENT_MATERIALIZATION_EMPTY_DOMAIN:{component_id}"
            )

        candidate = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view_index,
            camera_binding_hash=str(camera_binding_hash),
            observation_domain=observation_domain,
            allowed_surface_ids=allowed_surface_ids,
            candidate_namespace=str(component_id),
        )
        mesh = qualify_mwb2_component_observation_cdt_mesh(
            surface,
            candidate,
            allowed_surface_ids=allowed_surface_ids,
            min_precision_inside_alpha=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.min_precision_inside_alpha,
        )
        if require_product_mesh_quality:
            mesh, sliver_repair = _repair_component_policy_slivers(
                mesh,
                surface=surface,
                view_index=view_index,
                component_id=str(component_id),
            )
        else:
            raster_report = mesh_raster_quality_report(
                mesh,
                surface=surface,
                view_index=view_index,
            )
            sliver_repair = {
                "performed": False,
                "removed_face_indices": (),
                "before": raster_report,
                "after": raster_report,
            }
        raster_report = mesh_raster_quality_report(
            mesh,
            surface=surface,
            view_index=view_index,
        )
        raster_failures = raster_quality_gate_failures(
            raster_report,
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        if require_product_mesh_quality and raster_failures:
            raise QualificationError(
                "MECHANICAL_COMPONENT_MATERIALIZATION_RASTER_QUALITY_FAIL:"
                f"V{view_index}:{component_id}:{','.join(raster_failures)}:"
                f"min_angle={raster_report['min_raster_triangle_angle_deg']}:"
                f"max_aspect={raster_report['max_raster_triangle_aspect_ratio']}"
            )

        mesh_skin = bind_mwb2_mesh_skin(
            surface,
            skeleton,
            skin,
            mesh,
        )
        mechanical_class = str(
            partition.component_mechanical_classes[component_id]
        )
        parent_joint_id = str(
            partition.component_parent_joint_ids[component_id]
        )
        if mechanical_class == "RIGID_SKINNED_COMPONENT":
            _verify_rigid_skin(
                mesh_skin,
                component_id=str(component_id),
                parent_joint_id=parent_joint_id,
            )

        appearance = build_observed_appearance_binding(
            surface=surface,
            mesh=mesh,
            target_view_index=view_index,
            camera_binding_hash=str(camera_binding_hash),
            observation_hash_by_view={
                int(k): str(v) for k, v in observation_hash_by_view.items()
            },
            atlas_payload_hash=str(atlas_payload_hash),
        )
        triangles = _raster_triangles(mesh)
        union_triangles.extend(triangles)
        report = {
            "passed": True,
            "component_id": str(component_id),
            "mechanical_class": mechanical_class,
            "parent_joint_id": parent_joint_id,
            "allowed_surface_count": len(allowed_surface_ids),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "precision_inside_alpha": float(
                candidate.residual_report.get("precision_inside_alpha", -1.0)
            ),
            "source_alpha_recall_diagnostic_only": float(
                candidate.residual_report.get("source_alpha_recall", -1.0)
            ),
            "raster_quality": raster_report,
            "sliver_face_subset_repair": sliver_repair,
            "teacher_truth_used": False,
            "source_mesh_used": False,
            "cross_component_faces_possible": False,
        }
        component_reports[str(component_id)] = report
        rows.append(
            MaterializedMechanicalComponentIR(
                component_id=str(component_id),
                mesh=mesh,
                mesh_skin=mesh_skin,
                appearance=appearance,
                qualification_report=report,
            )
        )

    expected_components = set(map(str, partition.component_surface_ids))
    actual_components = {row.component_id for row in rows}
    if actual_components != expected_components:
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_COMPONENT_ACCOUNTING_DRIFT"
        )

    union_coverage = observation_domain.coverage(union_triangles)
    coverage_failures = coverage_gate_failures(
        union_coverage,
        policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    )
    seam_report = {
        "performed": False,
        "accepted_face_count": 0,
        "added_vertex_count": 0,
    }
    residual_recovery_attempts = []
    adopted_residual_recovery = {}
    if require_product_mesh_quality and coverage_failures:
        body_index = next(
            (
                index
                for index, row in enumerate(rows)
                if str(row.component_id) == "BODY_UNDERLAY"
            ),
            None,
        )
        if body_index is None:
            raise QualificationError(
                "MECHANICAL_COMPONENT_BOUNDARY_SEAM_BODY_MISSING"
            )
        body_row = rows[body_index]
        seam_mesh, seam_report = _augment_body_with_boundary_seam(
            body_row.mesh,
            surface=surface,
            partition=partition,
            view_index=view_index,
            camera_binding_hash=str(camera_binding_hash),
            observation_domain=observation_domain,
        )
        if seam_report.get("performed") is True:
            seam_skin = bind_mwb2_mesh_skin(
                surface,
                skeleton,
                skin,
                seam_mesh,
            )
            seam_appearance = build_observed_appearance_binding(
                surface=surface,
                mesh=seam_mesh,
                target_view_index=view_index,
                camera_binding_hash=str(camera_binding_hash),
                observation_hash_by_view={
                    int(k): str(v) for k, v in observation_hash_by_view.items()
                },
                atlas_payload_hash=str(atlas_payload_hash),
            )
            seam_raster = mesh_raster_quality_report(
                seam_mesh,
                surface=surface,
                view_index=view_index,
            )
            body_report = {
                **dict(body_row.qualification_report),
                "vertex_count": len(seam_mesh.vertices),
                "face_count": len(seam_mesh.faces),
                "raster_quality": seam_raster,
                "compiler_boundary_seam": seam_report,
                "cross_component_faces_possible": True,
            }
            rows[body_index] = MaterializedMechanicalComponentIR(
                component_id=body_row.component_id,
                mesh=seam_mesh,
                mesh_skin=seam_skin,
                appearance=seam_appearance,
                qualification_report=body_report,
            )
            component_reports["BODY_UNDERLAY"] = body_report
            union_triangles = [
                triangle
                for row in rows
                for triangle in _raster_triangles(row.mesh)
            ]
            union_coverage = observation_domain.coverage(union_triangles)
            coverage_failures = coverage_gate_failures(
                union_coverage,
                policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
            )

    if require_product_mesh_quality and coverage_failures:
        base_rows = list(rows)
        body_index = next(
            (
                index
                for index, row in enumerate(base_rows)
                if str(row.component_id) == "BODY_UNDERLAY"
            ),
            None,
        )
        if body_index is None:
            raise QualificationError(
                "MECHANICAL_COMPONENT_RESIDUAL_RECOVERY_BODY_MISSING"
            )
        base_body = base_rows[body_index]
        best = None
        for subdivision_factor in (2, 3, 4):
            recovered_mesh, recovery_report = (
                _augment_body_with_uniform_residual_recovery(
                    base_body.mesh,
                    component_meshes=tuple(
                        row.mesh for row in base_rows
                    ),
                    surface=surface,
                    view_index=view_index,
                    camera_binding_hash=str(camera_binding_hash),
                    observation_domain=observation_domain,
                    subdivision_factor=subdivision_factor,
                )
            )
            if recovery_report.get("performed") is not True:
                residual_recovery_attempts.append(recovery_report)
                continue

            recovered_skin = bind_mwb2_mesh_skin(
                surface,
                skeleton,
                skin,
                recovered_mesh,
            )
            recovered_appearance = build_observed_appearance_binding(
                surface=surface,
                mesh=recovered_mesh,
                target_view_index=view_index,
                camera_binding_hash=str(camera_binding_hash),
                observation_hash_by_view={
                    int(k): str(v)
                    for k, v in observation_hash_by_view.items()
                },
                atlas_payload_hash=str(atlas_payload_hash),
            )
            recovered_raster = mesh_raster_quality_report(
                recovered_mesh,
                surface=surface,
                view_index=view_index,
            )
            recovered_body_report = {
                **dict(base_body.qualification_report),
                "vertex_count": len(recovered_mesh.vertices),
                "face_count": len(recovered_mesh.faces),
                "raster_quality": recovered_raster,
                "compiler_residual_recovery": recovery_report,
                "cross_component_faces_possible": True,
            }
            candidate_rows = list(base_rows)
            candidate_rows[body_index] = MaterializedMechanicalComponentIR(
                component_id=base_body.component_id,
                mesh=recovered_mesh,
                mesh_skin=recovered_skin,
                appearance=recovered_appearance,
                qualification_report=recovered_body_report,
            )
            candidate_union_triangles = [
                triangle
                for row in candidate_rows
                for triangle in _raster_triangles(row.mesh)
            ]
            candidate_coverage = observation_domain.coverage(
                candidate_union_triangles
            )
            candidate_failures = coverage_gate_failures(
                candidate_coverage,
                policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
            )
            attempt = {
                **dict(recovery_report),
                "union_source_alpha_recall": float(
                    candidate_coverage["source_alpha_recall"]
                ),
                "union_precision_inside_alpha": float(
                    candidate_coverage["precision_inside_alpha"]
                ),
                "union_alpha_iou": float(
                    candidate_coverage["alpha_iou"]
                ),
                "union_largest_uncovered_component_fraction": float(
                    candidate_coverage[
                        "largest_uncovered_component_fraction"
                    ]
                ),
                "union_coverage_failures": tuple(candidate_failures),
            }
            residual_recovery_attempts.append(attempt)
            score = (
                float(candidate_coverage["source_alpha_recall"]),
                -float(
                    candidate_coverage[
                        "largest_uncovered_component_fraction"
                    ]
                ),
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    candidate_rows,
                    candidate_coverage,
                    candidate_failures,
                    recovered_body_report,
                    attempt,
                )
            if not candidate_failures:
                rows = candidate_rows
                union_triangles = candidate_union_triangles
                union_coverage = candidate_coverage
                coverage_failures = candidate_failures
                component_reports["BODY_UNDERLAY"] = recovered_body_report
                adopted_residual_recovery = dict(attempt)
                break

        if coverage_failures and best is not None:
            (
                _score,
                _best_rows,
                best_coverage,
                best_failures,
                _best_body_report,
                _best_attempt,
            ) = best
            # Keep the component-first authoritative state unchanged on failure.
            # Best-attempt metrics are diagnostics only and cannot silently weaken
            # the frozen admission policy.
            residual_recovery_attempts.append(
                {
                    "diagnostic_best_only": True,
                    "union_source_alpha_recall": float(
                        best_coverage["source_alpha_recall"]
                    ),
                    "union_precision_inside_alpha": float(
                        best_coverage["precision_inside_alpha"]
                    ),
                    "union_alpha_iou": float(best_coverage["alpha_iou"]),
                    "union_largest_uncovered_component_fraction": float(
                        best_coverage[
                            "largest_uncovered_component_fraction"
                        ]
                    ),
                    "union_coverage_failures": tuple(best_failures),
                }
            )

    if require_product_mesh_quality and coverage_failures:
        component_diag = ";".join(
            (
                f"{cid}:faces={component_reports[cid]['face_count']}:"
                f"diag_recall={component_reports[cid]['source_alpha_recall_diagnostic_only']}:"
                f"sliver_removed={len(component_reports[cid]['sliver_face_subset_repair']['removed_face_indices'])}"
            )
            for cid in sorted(component_reports)
        )
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_UNION_COVERAGE_FAIL:"
            f"V{view_index}:{','.join(coverage_failures)}:"
            f"recall={union_coverage['source_alpha_recall']}:"
            f"precision={union_coverage['precision_inside_alpha']}:"
            f"iou={union_coverage['alpha_iou']}:"
            f"largest_hole={union_coverage['largest_uncovered_component_fraction']}:"
            f"predicted_pixels={union_coverage['predicted_pixel_count']}:"
            f"foreground_pixels={union_coverage['foreground_pixel_count']}:"
            f"seam_faces={seam_report.get('accepted_face_count', 0)}:"
            f"seam_vertices={seam_report.get('added_vertex_count', 0)}:"
            f"recovery_attempts={residual_recovery_attempts}:"
            f"components={component_diag}"
        )

    provisional = MechanicalComponentViewMaterializationIR(
        view_index=view_index,
        component_partition_hash=str(partition.partition_hash),
        components=tuple(rows),
        union_coverage_report=union_coverage,
        qualification_report={
            "passed": True,
            "component_count": len(rows),
            "component_reports": component_reports,
            "coverage_failures": tuple(coverage_failures),
            "product_mesh_quality_required": bool(require_product_mesh_quality),
            "teacher_truth_used": False,
            "source_component_truth_used": False,
            "historical_full_subject_mesh_used": False,
            "source_owner_raster_used": False,
            "cross_component_faces_generated": bool(
                seam_report.get("accepted_face_count", 0)
                or adopted_residual_recovery.get("accepted_face_count", 0)
            ),
            "compiler_boundary_seam": seam_report,
            "compiler_residual_recovery_attempts": tuple(
                residual_recovery_attempts
            ),
            "adopted_compiler_residual_recovery": dict(
                adopted_residual_recovery
            ),
        },
        materialization_hash="",
        metadata={
            "authority": "CURRENT_QUALIFIED_S_G_W_PLUS_EXACT_SOURCE_OBSERVATION",
            "mesh_order": (
                "MECHANICAL_PARTITION_FIRST_THEN_COMPONENT_LOCAL_CDT"
                "_THEN_CURRENT_S_BOUNDARY_SEAM_IF_COVERAGE_REQUIRED"
                "_THEN_BOUNDED_LOCAL_CONVEX_RESIDUAL_RECOVERY"
            ),
            "observation_alpha_role": "HARD_CONTAINMENT_AND_UNION_COVERAGE_AUTHORITY",
            "component_identity_authority": "COMPILER_MECHANICAL_PARTITION",
        },
    )
    value = replace(
        provisional,
        materialization_hash=mechanical_component_view_materialization_hash(provisional),
    )
    if value.materialization_hash != mechanical_component_view_materialization_hash(value):
        raise QualificationError(
            "MECHANICAL_COMPONENT_MATERIALIZATION_HASH_MISMATCH"
        )
    return value


__all__ = [
    "COMPONENT_MATERIALIZATION_SCHEMA",
    "MaterializedMechanicalComponentIR",
    "MechanicalComponentViewMaterializationIR",
    "mechanical_component_view_materialization_hash",
    "materialize_mechanical_component_view",
]
