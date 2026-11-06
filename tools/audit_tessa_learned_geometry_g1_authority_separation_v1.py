from __future__ import annotations

"""TESSA learned-geometry vs GSA field-support authority separation court.

This court intentionally does *not* qualify TESSA geometry.  It measures the exact
current TESSAMaterialSupportFieldV1 lift for a decoded TESSA mesh and demonstrates
whether those coefficients reconstruct the learned vertex positions.  The field
coefficients remain continuous material/mechanical correspondence authority even
when learned P is not the convex support lift.

A non-zero learned-P -> support-lift distance is therefore not a geometry failure.
Geometry authority remains the actual-candidate Stage19 source-fidelity + static
quality court.  No G1 threshold is relaxed here.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
    build_tessa_material_support_field_v1,
    validate_tessa_material_support_field_v1,
)


SCHEMA = "RealSaS.TESSALearnedGeometryG1AuthoritySeparationCourt.v1"
COURT_ID = "TESSA_LEARNED_GEOMETRY_G1_AUTHORITY_SEPARATION_V1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) == 0 or not np.isfinite(data).all():
        raise RuntimeError("COURT_DISTANCE_VECTOR_INVALID")
    return {
        "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50.0)),
        "p90": float(np.percentile(data, 90.0)),
        "p95": float(np.percentile(data, 95.0)),
        "p99": float(np.percentile(data, 99.0)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def _normalization_world_vertices(vertices_normalized: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    center = np.asarray(payload.get("center_xyz"), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise RuntimeError("COURT_NORMALIZATION_CENTER_INVALID")
    if "half_extent" in payload:
        scale = 2.0 * float(payload["half_extent"])
    elif "scale" in payload:
        scale = float(payload["scale"])
    else:
        raise RuntimeError("COURT_NORMALIZATION_SCALE_MISSING")
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("COURT_NORMALIZATION_SCALE_INVALID")
    vertices = np.asarray(vertices_normalized, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("COURT_TESSA_VERTEX_ARRAY_INVALID")
    return vertices * scale + center[None, :]


def run_court(*, surface_path: Path, normalization_path: Path, decoded_mesh_path: Path) -> dict[str, Any]:
    surface_payload = json.loads(surface_path.read_text(encoding="utf-8"))
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    surface = rigging_surface_from_dict(surface_payload)
    partition = build_structural_partition(surface)

    with np.load(decoded_mesh_path, allow_pickle=False) as payload:
        required = {"vertices_normalized", "faces", "vertex_component_indices"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"COURT_TESSA_DECODED_FIELDS_MISSING:{','.join(missing)}")
        vertices_normalized = np.asarray(payload["vertices_normalized"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
        decoded_component_indices = np.asarray(
            payload["vertex_component_indices"], dtype=np.int64
        ).reshape(-1)

    if len(vertices_normalized) != len(decoded_component_indices):
        raise RuntimeError("COURT_TESSA_COMPONENT_COUNT_DRIFT")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("COURT_TESSA_FACE_ARRAY_INVALID")
    if np.any(faces < 0) or np.any(faces >= len(vertices_normalized)):
        raise RuntimeError("COURT_TESSA_FACE_INDEX_INVALID")

    vertices_world = _normalization_world_vertices(vertices_normalized, normalization_payload)
    topology_sequence_hash = content_sha256(
        {
            "schema": "RealSaS.TESSACourtDecodedTopology.v1",
            "faces": faces.tolist(),
            "vertex_component_indices": decoded_component_indices.tolist(),
        }
    )
    proposal_vertex_ids = tuple(
        f"TESSA_V1:{index:08d}" for index in range(len(vertices_world))
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=vertices_world,
        decoded_component_indices=decoded_component_indices,
        surface=surface,
        partition=partition,
        topology_sequence_hash=topology_sequence_hash,
        proposal_vertex_ids=proposal_vertex_ids,
    )
    validate_tessa_material_support_field_v1(
        field,
        surface=surface,
        partition=partition,
        expected_vertex_ids=proposal_vertex_ids,
    )

    node_position = {
        str(node.surface_id): np.asarray(node.P, dtype=np.float64)
        for node in surface.surface_nodes
    }
    row_by_id = {str(row.proposal_vertex_id): row for row in field.rows}
    if set(row_by_id) != set(proposal_vertex_ids):
        raise RuntimeError("COURT_SUPPORT_VERTEX_ACCOUNTING_DRIFT")

    distances = []
    exact_count = 0
    simplex_max_abs_residual = 0.0
    cross_component_row_count = 0
    owner = {
        str(surface_id): str(component.component_id)
        for component in partition.components
        for surface_id in component.surface_ids
    }
    for index, proposal_vertex_id in enumerate(proposal_vertex_ids):
        row = row_by_id[proposal_vertex_id]
        coeffs = tuple((str(sid), float(weight)) for sid, weight in row.coefficients)
        total = float(sum(weight for _, weight in coeffs))
        simplex_max_abs_residual = max(simplex_max_abs_residual, abs(total - 1.0))
        if any(owner.get(sid) != str(row.mechanical_component_id) for sid, _ in coeffs):
            cross_component_row_count += 1
        lifted = sum(
            (weight * node_position[sid] for sid, weight in coeffs),
            np.zeros(3, dtype=np.float64),
        )
        distance = float(np.linalg.norm(vertices_world[index] - lifted))
        distances.append(distance)
        if distance <= 1.0e-9:
            exact_count += 1

    distance_array = np.asarray(distances, dtype=np.float64)
    support_valid = (
        simplex_max_abs_residual <= 1.0e-9
        and cross_component_row_count == 0
        and len(field.rows) == len(vertices_world)
    )
    authority_separation_observed = int(exact_count) < int(len(vertices_world))
    status = (
        "PASS__AUTHORITY_SEPARATION_REQUIRED"
        if support_valid and authority_separation_observed
        else "FAIL__COURT_DID_NOT_ESTABLISH_EXPECTED_SEPARATION"
    )

    report = {
        "schema": SCHEMA,
        "court_id": COURT_ID,
        "status": status,
        "product_authority": False,
        "threshold_relaxation_performed": False,
        "knight_specific_exemption": False,
        "interpretation": {
            "tessa_P_authority": "NOT_DECIDED_BY_THIS_COURT",
            "geometry_authority_owner": "STAGE19_ACTUAL_CANDIDATE_SOURCE_FIDELITY_PLUS_STATIC_QUALITY",
            "support_authority_owner": "TESSA_MATERIAL_SUPPORT_FIELD_V1",
            "support_coefficients_reconstruct_learned_P_claimed": False,
            "nonzero_support_lift_distance_is_geometry_failure": False,
        },
        "inputs": {
            "surface_sha256": _sha256(surface_path),
            "normalization_sha256": _sha256(normalization_path),
            "decoded_mesh_sha256": _sha256(decoded_mesh_path),
            "surface_geometry_lineage_hash": surface.geometry_lineage_hash,
            "partition_lineage_hash": partition.partition_lineage_hash,
            "topology_sequence_hash": topology_sequence_hash,
        },
        "measurements": {
            "tessa_vertex_count": int(len(vertices_world)),
            "tessa_face_count": int(len(faces)),
            "decoded_component_count": int(len(set(map(int, decoded_component_indices.tolist())))),
            "mechanical_partition_component_count": int(len(partition.components)),
            "support_row_count": int(len(field.rows)),
            "support_simplex_max_abs_residual": float(simplex_max_abs_residual),
            "cross_component_support_row_count": int(cross_component_row_count),
            "exact_position_equality_tolerance": 1.0e-9,
            "exact_position_equality_count": int(exact_count),
            "non_exact_position_count": int(len(vertices_world) - exact_count),
            "learned_P_to_current_support_lift_distance": _percentiles(distance_array),
            "field_lineage_hash": field.field_lineage_hash,
        },
    }
    report["report_hash"] = content_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--decoded-mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = run_court(
        surface_path=args.surface,
        normalization_path=args.normalization,
        decoded_mesh_path=args.decoded_mesh,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if not str(report["status"]).startswith("PASS__"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
