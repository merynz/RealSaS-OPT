from __future__ import annotations

"""Compiler-owned static Mesh capability.

Consumes immutable source truth plus ShapeProposal.v2.  Learned topology is
proposal evidence only.  Compiler independently qualifies topology, resolves
winding from role-free IRIS surface normals, reprojects the exact candidate into
all source views under the frozen Stage13 P999 geometry policy, and only then
mints CanonicalMeshIR.v1.
"""

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import mint_canonical_mesh_v1
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import silhouette_distance_metrics
from compiler.realsas_compiler_services.platform_worker.input_capability_v1 import SourceObservationV1
from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import (
    DecodedShapeProposalV2,
    MESH_OBJECT_FRAME_TRANSFORM_V2,
)


MESH_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductMeshCompile.v1"
MESH_OUTPUT_ARTIFACT_TYPE_V1 = "RealSaS.CanonicalMesh"
MESH_OUTPUT_AUTHORITY_V1 = "COMPILER_STATIC_CANONICAL_MESH"

FROZEN_SOURCE_FIDELITY_POLICY_V1 = {
    "schema": "RealSaS.ProductMeshSourceFidelityPolicy.v1",
    "authority_source": "canonical/STAGE13_REST_REPROJECTION_POLICY_V2_20260919.json",
    "profile": "P999",
    "metric_contract": "G5_COVERAGE_PLUS_SYMMETRIC_NEAREST_BOUNDARY_DISTANCE_PIXELS",
    "min_recall": 0.999,
    "min_precision": 0.999,
    "max_largest_coherent_hole_fraction": 0.00025,
    "max_interior_uncovered_fraction": 0.0005,
    "min_component_recall": 0.999,
    "component_min_foreground_fraction": 0.0,
    "max_silhouette_edge_p95_px": 0.5,
    "every_source_view_must_pass": True,
    "aggregate_compensation_forbidden": True,
}
FROZEN_SOURCE_FIDELITY_POLICY_SHA256_V1 = content_sha256(FROZEN_SOURCE_FIDELITY_POLICY_V1)


@dataclass(frozen=True)
class MeshCompileResultV1:
    payload: bytes
    diagnostics: dict[str, Any]
    schema: str = "RealSaS.ProductMeshCompileResult.v1"


def _topology_court(shape: DecodedShapeProposalV2) -> dict[str, Any]:
    court = dict(shape.metadata.get("topology_court") or {})
    required = {
        "grammar_valid",
        "vertex_count",
        "face_count",
        "declared_component_count",
        "geometric_component_count",
        "nonmanifold_edge_count",
        "zero_area_face_count",
        "open_boundary_chain_count",
        "finite_geometry",
        "within_scale_policy",
    }
    if not required.issubset(court):
        raise RuntimeError("MESH_TOPOLOGY_COURT_FIELDS_MISSING")
    passed = bool(
        court["grammar_valid"] is True
        and court.get("decode_failure") is None
        and court["finite_geometry"] is True
        and court["within_scale_policy"] is True
        and int(court["vertex_count"]) == len(shape.decoded_vertices_tessa)
        and int(court["face_count"]) == len(shape.decoded_faces)
        and int(court["declared_component_count"]) == int(court["geometric_component_count"])
        and int(court["nonmanifold_edge_count"]) == 0
        and int(court["zero_area_face_count"]) == 0
        and int(court["open_boundary_chain_count"]) == 0
    )
    return {
        "schema": "RealSaS.ProductMeshTopologyCourt.v1",
        "passed": passed,
        "grammar_valid": bool(court["grammar_valid"]),
        "vertex_count": int(court["vertex_count"]),
        "face_count": int(court["face_count"]),
        "declared_component_count": int(court["declared_component_count"]),
        "geometric_component_count": int(court["geometric_component_count"]),
        "boundary_loop_count": int(court.get("boundary_loop_count", 0)),
        "open_boundary_chain_count": int(court["open_boundary_chain_count"]),
        "nonmanifold_edge_count": int(court["nonmanifold_edge_count"]),
        "zero_area_face_count": int(court["zero_area_face_count"]),
        "source_court_schema": str(court.get("schema") or ""),
        "learned_court_is_evidence_not_authority": True,
    }


def _edge_direction(face: np.ndarray, u: int, v: int) -> int:
    a, b, c = map(int, face.tolist())
    for x, y in ((a, b), (b, c), (c, a)):
        if x == u and y == v:
            return 1
        if x == v and y == u:
            return -1
    raise RuntimeError("MESH_ORIENTATION_EDGE_NOT_IN_FACE")


def _orient_faces(
    vertices_object: np.ndarray,
    faces: np.ndarray,
    face_components: np.ndarray,
    witness_xyz_object: np.ndarray,
    witness_normals: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    tri = np.asarray(faces, dtype=np.int64).copy()
    components = np.asarray(face_components, dtype=np.int64).reshape(-1)
    if len(components) != len(tri):
        raise RuntimeError("MESH_ORIENTATION_FACE_COMPONENT_DRIFT")
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, row in enumerate(tri.tolist()):
        for u, v in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
            edge_to_faces.setdefault(tuple(sorted((int(u), int(v)))), []).append(fi)
    if any(len(rows) > 2 for rows in edge_to_faces.values()):
        raise RuntimeError("MESH_ORIENTATION_NONMANIFOLD_EDGE")

    signs: dict[int, int] = {}
    conflicts = 0
    for seed in range(len(tri)):
        if seed in signs:
            continue
        signs[seed] = 1
        stack = [seed]
        while stack:
            fi = stack.pop()
            face = tri[fi]
            for raw_u, raw_v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted((int(raw_u), int(raw_v))))
                incident = edge_to_faces[key]
                if len(incident) != 2:
                    continue
                gi = incident[0] if incident[1] == fi else incident[1]
                if components[gi] != components[fi]:
                    raise RuntimeError("MESH_ORIENTATION_EDGE_CROSSES_COMPONENT")
                u, v = key
                df = _edge_direction(tri[fi], u, v)
                dg = _edge_direction(tri[gi], u, v)
                expected = -df * signs[fi] * dg
                if gi in signs:
                    if signs[gi] != expected:
                        conflicts += 1
                    continue
                signs[gi] = expected
                stack.append(gi)
    if conflicts:
        raise RuntimeError("MESH_ORIENTATION_NONORIENTABLE_COMPONENT")
    for fi, sign in signs.items():
        if sign < 0:
            tri[fi, 1], tri[fi, 2] = tri[fi, 2], tri[fi, 1]

    witness_xyz = np.asarray(witness_xyz_object, dtype=np.float64).reshape(-1, 3)
    witness_n = np.asarray(witness_normals, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(witness_xyz).all(axis=1) & np.isfinite(witness_n).all(axis=1)
    norm = np.linalg.norm(witness_n, axis=1)
    valid &= norm > 1.0e-8
    if not np.any(valid):
        raise RuntimeError("MESH_ORIENTATION_NORMAL_EVIDENCE_EMPTY")
    witness_xyz = witness_xyz[valid]
    witness_n = witness_n[valid] / norm[valid, None]
    tree = cKDTree(witness_xyz)

    rows: list[dict[str, Any]] = []
    global_flips = 0
    for component in sorted(set(map(int, components.tolist()))):
        indices = np.where(components == component)[0]
        if len(indices) == 0:
            continue
        xyz = vertices_object[tri[indices]]
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        areas2 = np.linalg.norm(cross, axis=1)
        if np.any(areas2 <= 1.0e-12) or not np.isfinite(cross).all():
            raise RuntimeError("MESH_ORIENTATION_DEGENERATE_FACE")
        normals = cross / areas2[:, None]
        centroids = np.mean(xyz, axis=1)
        _, nearest = tree.query(centroids, k=1)
        evidence = witness_n[np.asarray(nearest, dtype=np.int64)]
        dots = np.sum(normals * evidence, axis=1)
        weighted_score = float(np.sum(dots * areas2) / np.sum(areas2))
        if not math.isfinite(weighted_score) or abs(weighted_score) < 1.0e-3:
            raise RuntimeError("MESH_ORIENTATION_NORMAL_EVIDENCE_AMBIGUOUS")
        flipped = weighted_score < 0.0
        if flipped:
            tmp = tri[indices, 1].copy()
            tri[indices, 1] = tri[indices, 2]
            tri[indices, 2] = tmp
            global_flips += 1
        rows.append(
            {
                "component_index": int(component),
                "face_count": int(len(indices)),
                "pre_global_flip_weighted_normal_agreement": weighted_score,
                "global_flip_applied": bool(flipped),
                "absolute_orientation_support": float(abs(weighted_score)),
            }
        )
    return tri, {
        "schema": "RealSaS.ProductMeshOrientationCourt.v1",
        "passed": True,
        "adjacency_conflict_count": int(conflicts),
        "component_count": int(len(rows)),
        "global_flip_count": int(global_flips),
        "components": rows,
        "authority": "COMPILER_ORIENTATION_FROM_IRIS_SURFACE_NORMAL_EVIDENCE",
        "semantic_segmentation_used": False,
    }


def _normalized_cameras(source: SourceObservationV1):
    camera = dict(source.camera_bundle)
    center = np.asarray(camera["common_center_xyz"], dtype=np.float64)
    common_half = float(camera["common_half_extent"])
    order = list(camera["view_order"])
    result = []
    for index, row in enumerate(camera["cameras"]):
        result.append(
            qualify_camera_v3(
                {
                    "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
                    "view_index": index,
                    "origin": (
                        (np.asarray(row["origin"], dtype=np.float64) - center) / common_half
                    ).tolist(),
                    "right": list(row["right"]),
                    "screen_up": list(row["screen_up"]),
                    "forward": list(row["forward"]),
                    "half_extent": float(row["half_extent"]) / common_half,
                    "resolution": 1024,
                },
                view_id=str(order[index]),
                view_index=index,
            )
        )
    return tuple(result)


def _source_fidelity_court(
    *,
    source: SourceObservationV1,
    vertices_object: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    source.validate()
    cameras = _normalized_cameras(source)
    rgba = source.native_rgba.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    policy = FROZEN_SOURCE_FIDELITY_POLICY_V1
    for view_index, camera in enumerate(cameras):
        source_mask_np = np.asarray(rgba[0, view_index, 3] > 0, dtype=np.bool_)
        if not bool(np.any(source_mask_np)):
            raise RuntimeError(f"MESH_SOURCE_FOREGROUND_EMPTY:{view_index}")
        source_mask = bytes(source_mask_np.astype(np.uint8).reshape(-1))
        projected = project_points_xyz_v3(vertices_object, camera)
        triangles = tuple(
            tuple((float(projected[int(i), 0]), float(projected[int(i), 1])) for i in face)
            for face in faces.tolist()
        )
        predicted = rasterize_triangles_half_integer_top_left(
            triangles,
            width=1024,
            height=1024,
        )
        coverage = coverage_metrics(source_mask, predicted, width=1024, height=1024)
        component = source_connected_component_recall_metrics(
            source_mask,
            predicted,
            width=1024,
            height=1024,
            minimum_foreground_fraction=float(policy["component_min_foreground_fraction"]),
        )
        predicted_np = np.frombuffer(predicted, dtype=np.uint8).reshape(1024, 1024).astype(bool)
        edge_mean, edge_p95, edge_max = silhouette_distance_metrics(
            source_mask_np,
            predicted_np,
        )
        component_recall = float(component["minimum_eligible_component_recall"])
        passed = bool(
            float(coverage["recall"]) >= float(policy["min_recall"])
            and float(coverage["precision"]) >= float(policy["min_precision"])
            and float(coverage["largest_coherent_hole_fraction"])
            <= float(policy["max_largest_coherent_hole_fraction"])
            and float(coverage["interior_uncovered_fraction"])
            <= float(policy["max_interior_uncovered_fraction"])
            and component_recall >= float(policy["min_component_recall"])
            and float(edge_p95) <= float(policy["max_silhouette_edge_p95_px"])
        )
        rows.append(
            {
                "view_index": int(view_index),
                "view_id": str(camera.view_id),
                "passed": passed,
                "silhouette_recall": float(coverage["recall"]),
                "silhouette_precision": float(coverage["precision"]),
                "largest_coherent_hole_fraction": float(
                    coverage["largest_coherent_hole_fraction"]
                ),
                "interior_uncovered_fraction": float(
                    coverage["interior_uncovered_fraction"]
                ),
                "component_recall": component_recall,
                "silhouette_edge_mean_px": float(edge_mean),
                "silhouette_edge_p95_px": float(edge_p95),
                "silhouette_edge_max_px": float(edge_max),
                "source_foreground_pixel_count": int(coverage["foreground_pixel_count"]),
                "predicted_pixel_count": int(coverage["predicted_pixel_count"]),
            }
        )
    return {
        "schema": "RealSaS.ProductMeshSourceFidelityCourt.v1",
        "passed": bool(len(rows) == 8 and all(row["passed"] for row in rows)),
        "policy_sha256": FROZEN_SOURCE_FIDELITY_POLICY_SHA256_V1,
        "policy": dict(FROZEN_SOURCE_FIDELITY_POLICY_V1),
        "views": rows,
        "source_alpha_is_geometry_fidelity_authority": True,
        "appearance_color_used_as_geometry_authority": False,
        "aggregate_compensation_used": False,
    }


def compile_mesh_v1(
    shape: DecodedShapeProposalV2,
    source: SourceObservationV1,
    *,
    source_shape_content_sha256: str,
    source_observation_content_sha256: str,
) -> MeshCompileResultV1:
    if shape.metadata.get("authority") != "LEARNED_SHAPE_PROPOSAL_ONLY":
        raise RuntimeError("MESH_SHAPE_AUTHORITY_DRIFT")
    if shape.metadata.get("canonical_mesh_authority") is not False:
        raise RuntimeError("MESH_SHAPE_PREMATURE_AUTHORITY")
    if shape.metadata.get("mesh_object_frame_transform") != MESH_OBJECT_FRAME_TRANSFORM_V2:
        raise RuntimeError("MESH_SHAPE_COORDINATE_TRANSFORM_DRIFT")
    topology = _topology_court(shape)
    if topology["passed"] is not True:
        raise RuntimeError("MESH_TOPOLOGY_COURT_FAILED")

    vertices_object = np.asarray(shape.decoded_vertices_tessa, dtype=np.float64) * 2.0
    faces = np.asarray(shape.decoded_faces, dtype=np.int64)
    oriented_faces, orientation = _orient_faces(
        vertices_object,
        faces,
        shape.decoded_face_component_indices,
        np.asarray(shape.surface_xyz_tessa, dtype=np.float64) * 2.0,
        np.asarray(shape.surface_normal, dtype=np.float64),
    )
    source_fidelity = _source_fidelity_court(
        source=source,
        vertices_object=vertices_object,
        faces=oriented_faces,
    )
    if source_fidelity["passed"] is not True:
        raise RuntimeError("MESH_SOURCE_FIDELITY_COURT_FAILED")

    mesh = mint_canonical_mesh_v1(
        vertices_object=vertices_object,
        faces=oriented_faces,
        vertex_component_indices=shape.decoded_vertex_component_indices,
        source_shape_content_sha256=source_shape_content_sha256,
        source_observation_content_sha256=source_observation_content_sha256,
        decoded_geometry_sha256=str(shape.metadata["decoded_geometry_sha256"]),
        topology_report=topology,
        orientation_report=orientation,
        source_fidelity_policy_sha256=FROZEN_SOURCE_FIDELITY_POLICY_SHA256_V1,
        source_fidelity_report=source_fidelity,
    )
    payload = json.dumps(
        mesh.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return MeshCompileResultV1(
        payload=payload,
        diagnostics={
            "schema": "RealSaS.ProductMeshCompileDiagnostics.v1",
            "authority": MESH_OUTPUT_AUTHORITY_V1,
            "mesh_lineage_sha256": mesh.mesh_lineage_sha256,
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "component_count": len(set(v.component_id for v in mesh.vertices)),
            "topology_report_sha256": mesh.topology_report_sha256,
            "orientation_report_sha256": mesh.orientation_report_sha256,
            "source_fidelity_report_sha256": mesh.source_fidelity_report_sha256,
            "dynamic_qualification_claimed": False,
        },
    )
