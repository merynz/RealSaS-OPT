from __future__ import annotations

"""Canonical local-chart adapter for the sealed historical CDT kernel.

The historical v0.5 CDT is numerical machinery only. This adapter owns all current
IR/provenance semantics:
- no view/camera input,
- charts originate from current structural relation faces,
- every emitted point is rebound to admitted RiggingSurfaceIR support,
- no boundary split is admitted in V1, so adjacent chart seams remain exact,
- candidate output remains subordinate to QualifiedMesh G1-G5 admission.
"""

import math

from .canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from .hashing import content_sha256
from .mesh._historical_v05 import (
    HISTORICAL_CDT_SOURCE_SHA256,
    triangulate_production_cdt,
)
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
    validate_component_carrier_policy,
    validate_mechanical_partition,
    validate_mesh_qualification_policy,
)
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding

_BARY_TOL = 1e-8
_POINT_TOL = 1e-10


def _identity_surface_id(vertex: CanonicalMeshVertexCandidateIR) -> str:
    binding = vertex.support_binding
    if binding.mode != "IDENTITY_SURFACE_NODE" or len(binding.coefficients) != 1:
        raise QualificationError("CANONICAL_CDT_BASELINE_REQUIRES_IDENTITY_SUPPORT")
    sid, coeff = binding.coefficients[0]
    if abs(float(coeff) - 1.0) > 1e-9:
        raise QualificationError("CANONICAL_CDT_BASELINE_IDENTITY_WEIGHT_INVALID")
    return sid


def _triangle_chart(pa, pb, pc):
    a = tuple(map(float, pa)); b = tuple(map(float, pb)); c = tuple(map(float, pc))
    ab = tuple(b[i] - a[i] for i in range(3))
    ac = tuple(c[i] - a[i] for i in range(3))
    lab = math.sqrt(sum(x*x for x in ab))
    if not math.isfinite(lab) or lab <= 0.0:
        raise QualificationError("CANONICAL_CDT_CHART_DEGENERATE")
    ex = tuple(x / lab for x in ab)
    cx = sum(ac[i] * ex[i] for i in range(3))
    residual = tuple(ac[i] - cx * ex[i] for i in range(3))
    cy = math.sqrt(sum(x*x for x in residual))
    if not math.isfinite(cy) or cy <= lab * 1e-12:
        raise QualificationError("CANONICAL_CDT_CHART_DEGENERATE")
    return ((0.0, 0.0), (lab, 0.0), (cx, cy))


def _barycentric_2d(point, tri):
    x, y = map(float, point)
    (x0,y0),(x1,y1),(x2,y2) = tri
    denom = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
    if not math.isfinite(denom) or abs(denom) <= 1e-18:
        raise QualificationError("CANONICAL_CDT_BARYCENTRIC_DEGENERATE")
    a = ((y1-y2)*(x-x2) + (x2-x1)*(y-y2)) / denom
    b = ((y2-y0)*(x-x2) + (x0-x2)*(y-y2)) / denom
    c = 1.0 - a - b
    coeffs = [float(a), float(b), float(c)]
    coeffs = [0.0 if abs(v) <= _BARY_TOL else (1.0 if abs(v-1.0) <= _BARY_TOL else v) for v in coeffs]
    if min(coeffs) < -_BARY_TOL or max(coeffs) > 1.0 + _BARY_TOL:
        raise QualificationError("CANONICAL_CDT_POINT_OUTSIDE_SOURCE_SIMPLEX")
    coeffs = [max(0.0, min(1.0, v)) for v in coeffs]
    total = sum(coeffs)
    if total <= 0.0:
        raise QualificationError("CANONICAL_CDT_BARYCENTRIC_ZERO_SUM")
    coeffs = [v / total for v in coeffs]
    return tuple(coeffs)


def _support_key(surface_ids, coeffs):
    return tuple((sid, round(float(w), 14)) for sid, w in zip(surface_ids, coeffs) if float(w) > 1e-14)


def _lift(surface_nodes, surface_ids, coeffs):
    out = [0.0, 0.0, 0.0]
    for sid, weight in zip(surface_ids, coeffs):
        p = surface_nodes[sid].P
        for axis in range(3):
            out[axis] += float(weight) * float(p[axis])
    return tuple(out)


def build_canonical_cdt_candidate(
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    mesh_policy: MeshQualificationPolicyIR,
    *,
    relation_baseline_policy_hash: str,
    max_constraint_recovery_iterations: int = 96,
    max_quality_iterations: int = 96,
) -> CanonicalMeshCandidateIR:
    """Refine current relation triangles with hash-sealed CDT local charts.

    V1 forbids boundary splits. This intentionally leaves some skinny parent faces
    unrepairable rather than creating an unstitched chart seam. A later adapter may
    admit synchronized edge splits only with an explicit stitch proof.
    """
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    validate_mesh_qualification_policy(mesh_policy)
    if not relation_baseline_policy_hash:
        raise QualificationError("CANONICAL_CDT_BASELINE_POLICY_MISSING")

    baseline = build_canonical_relation_candidate(
        surface,
        partition,
        carrier_policy,
        producer_policy_hash=relation_baseline_policy_hash,
    )
    baseline_by_id = {vertex.candidate_vertex_id: vertex for vertex in baseline.vertices}
    surface_nodes = {node.surface_id: node for node in surface.surface_nodes}

    vertex_by_support = {}
    output_faces = set()
    patch_reports = []
    generated_count = 0

    def admit_vertex(surface_ids, coeffs, component_id, *, patch_id, point2d):
        nonlocal generated_count
        key = (component_id, _support_key(surface_ids, coeffs))
        existing = vertex_by_support.get(key)
        if existing is not None:
            return existing.candidate_vertex_id
        support = key[1]
        if not support:
            raise QualificationError("CANONICAL_CDT_EMPTY_SUPPORT")
        mode = "IDENTITY_SURFACE_NODE" if len(support) == 1 and abs(support[0][1]-1.0) <= 1e-12 else "LOCAL_CONVEX_INTERPOLATION"
        position = _lift(surface_nodes, tuple(sid for sid,_ in support), tuple(w for _,w in support))
        candidate_id = "CDTV:" + content_sha256({
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "partition_lineage_hash": partition.partition_lineage_hash,
            "component_id": component_id,
            "support": support,
        })[:24]
        vertex = CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=candidate_id,
            support_binding=SurfaceSupportBinding(
                mode,
                support,
                metadata={
                    "source": "HISTORICAL_CDT_V05_LOCAL_CHART",
                    "historical_cdt_source_sha256": HISTORICAL_CDT_SOURCE_SHA256,
                },
            ),
            component_id=component_id,
            P=position,
            metadata={
                "patch_id": patch_id,
                "chart_xy": tuple(map(float, point2d)),
                "source_mesh_used": False,
            },
        )
        vertex_by_support[key] = vertex
        if mode != "IDENTITY_SURFACE_NODE":
            generated_count += 1
        return candidate_id

    for patch_index, face in enumerate(baseline.faces):
        base_vertices = tuple(baseline_by_id[vid] for vid in face)
        component_ids = {vertex.component_id for vertex in base_vertices}
        if len(component_ids) != 1:
            raise QualificationError("CANONICAL_CDT_BASELINE_CROSS_COMPONENT_FACE")
        component_id = next(iter(component_ids))
        surface_ids = tuple(_identity_surface_id(vertex) for vertex in base_vertices)
        points3 = tuple(surface_nodes[sid].P for sid in surface_ids)
        chart = _triangle_chart(*points3)
        patch_id = "CDTP:" + content_sha256({
            "baseline_candidate": baseline.candidate_lineage_hash,
            "face": tuple(face),
            "surface_ids": surface_ids,
        })[:24]

        result = triangulate_production_cdt(
            chart,
            target_min_angle_deg=float(mesh_policy.g3_min_angle_deg),
            max_constraint_recovery_iterations=int(max_constraint_recovery_iterations),
            max_quality_iterations=int(max_quality_iterations),
        )
        if not bool(result.success):
            raise QualificationError(f"CANONICAL_CDT_KERNEL_FAIL:{result.reason}")
        if int(result.missing_constraint_count) != 0:
            raise QualificationError("CANONICAL_CDT_MISSING_CONSTRAINT")
        if int(result.constraint_split_count) != 0 or int(result.boundary_vertex_count) != 3:
            raise QualificationError("CANONICAL_CDT_BOUNDARY_SPLIT_REQUIRES_STITCH_PROOF")
        if len(result.vertices) < 3 or not result.triangles:
            raise QualificationError("CANONICAL_CDT_EMPTY_PATCH_RESULT")

        local_ids = []
        for point in result.vertices:
            coeffs = _barycentric_2d(point, chart)
            local_ids.append(admit_vertex(
                surface_ids,
                coeffs,
                component_id,
                patch_id=patch_id,
                point2d=point,
            ))

        for tri in result.triangles:
            if len(tri) != 3:
                raise QualificationError("CANONICAL_CDT_TRIANGLE_INDEX_INVALID")
            try:
                ids = tuple(local_ids[int(i)] for i in tri)
            except (IndexError, TypeError, ValueError) as exc:
                raise QualificationError("CANONICAL_CDT_TRIANGLE_INDEX_INVALID") from exc
            if len(set(ids)) != 3:
                raise QualificationError("CANONICAL_CDT_COLLAPSED_OUTPUT_FACE")
            output_faces.add(tuple(ids))

        patch_reports.append({
            "patch_id": patch_id,
            "component_id": component_id,
            "source_surface_ids": surface_ids,
            "vertex_count": len(result.vertices),
            "triangle_count": len(result.triangles),
            "inserted_steiner_count": int(result.inserted_steiner_count),
            "quality_insert_count": int(result.quality_insert_count),
            "constraint_split_count": int(result.constraint_split_count),
            "min_angle_deg": float(result.min_angle_deg),
            "worst_radius_edge_ratio": float(result.worst_radius_edge_ratio),
            "backend_name": str(result.backend_name),
        })

    if not output_faces:
        raise QualificationError("CANONICAL_CDT_NO_OUTPUT_FACE")

    faces = tuple(sorted(output_faces))
    used_ids = {vid for face in faces for vid in face}
    vertices = tuple(sorted(
        (vertex for vertex in vertex_by_support.values() if vertex.candidate_vertex_id in used_ids),
        key=lambda vertex: vertex.candidate_vertex_id,
    ))
    edges = tuple(sorted({
        tuple(sorted((face[i], face[j])))
        for face in faces
        for i,j in ((0,1),(1,2),(2,0))
    }))

    producer_policy_hash = content_sha256({
        "schema": "RealSaS.CanonicalCDTLocalChartProducerPolicy.v1",
        "historical_cdt_source_sha256": HISTORICAL_CDT_SOURCE_SHA256,
        "relation_baseline_policy_hash": relation_baseline_policy_hash,
        "mesh_qualification_policy_hash": mesh_policy.qualification_policy_lineage_hash,
        "target_min_angle_deg": float(mesh_policy.g3_min_angle_deg),
        "max_constraint_recovery_iterations": int(max_constraint_recovery_iterations),
        "max_quality_iterations": int(max_quality_iterations),
        "boundary_split_policy": "FORBIDDEN_UNTIL_EXACT_STITCH_PROOF",
        "chart_family": "SOURCE_RELATION_TRIANGLE_ISOMETRIC_CHARTS",
    })
    provisional = CanonicalMeshCandidateIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        producer_id="RealSaS.HistoricalCDTV05.LocalTriangleChartAdapter.v1",
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata={
            "view_independent": True,
            "camera_authority_used": False,
            "source_mesh_used": False,
            "historical_cdt_source_sha256": HISTORICAL_CDT_SOURCE_SHA256,
            "baseline_candidate_lineage_hash": baseline.candidate_lineage_hash,
            "patch_count": len(patch_reports),
            "generated_local_convex_vertex_count": generated_count,
            "boundary_split_policy": "FAIL_CLOSED",
            "patch_reports": patch_reports,
        },
    )
    return CanonicalMeshCandidateIR(
        **{
            **provisional.__dict__,
            "candidate_lineage_hash": canonical_mesh_candidate_lineage_hash(provisional),
        }
    )
