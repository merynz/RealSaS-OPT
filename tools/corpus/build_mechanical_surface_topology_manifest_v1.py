#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.mechanical_surface_topology_authority_v1 import (
    build_mechanical_surface_topology_authority_v1,
)

SCHEMA = "RealSaS.MechanicalSurfaceTopologyAuthority.v1"
STATUS = "PASS_MECHANICAL_SURFACE_TOPOLOGY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        a = np.asarray(value)
        if a.dtype == object or (np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all()):
            raise ValueError("MECHANICAL_SURFACE_INVALID_ARRAY:" + name)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(subject_id: str, geometry_response_npz: Path, output_dir: Path) -> dict:
    geometry_response_npz = geometry_response_npz.resolve()
    output_dir = output_dir.resolve()
    with np.load(geometry_response_npz, allow_pickle=False) as z:
        required = {"rest_vertices_source_world", "faces", "face_owner_privileged", "isolated_position"}
        if not required.issubset(z.files):
            raise ValueError("MECHANICAL_SURFACE_STAGE47_ARRAYS")
        rest = np.asarray(z["rest_vertices_source_world"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
        owner = np.asarray(z["face_owner_privileged"], np.int32)
        posed = np.asarray(z["isolated_position"], np.float64)

    authority = build_mechanical_surface_topology_authority_v1(rest, faces, owner, posed)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "MECHANICAL_SURFACE_TOPOLOGY_V1.npz"
    _write_npz(
        arrays_path,
        raw_to_mechanical_vertex=authority.raw_to_mechanical_vertex,
        mechanical_vertices_source_world=authority.mechanical_vertices_source_world.astype(np.float32),
        mechanical_faces=authority.mechanical_faces,
        face_owner=authority.face_owner,
        face_valid=authority.face_valid,
        surface_component_id=authority.surface_component_id,
        mechanical_vertex_owner_offsets=authority.mechanical_vertex_owner_offsets,
        mechanical_vertex_owner_values=authority.mechanical_vertex_owner_values,
    )
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "source": {"geometry_and_response": {"path": str(geometry_response_npz), "sha256": sha256(geometry_response_npz)}},
        "output": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
        "counts": {
            "raw_vertex_count": int(len(rest)),
            "mechanical_vertex_count": int(len(authority.mechanical_vertices_source_world)),
            "face_count": int(len(faces)),
            "valid_face_count": int(np.count_nonzero(authority.face_valid)),
            "surface_component_count": int(len(np.unique(authority.surface_component_id))),
            "qualified_cross_owner_seam_edge_count": int(authority.qualified_cross_owner_seam_count),
        },
        "qualified_cross_owner_seams": list(authority.qualified_cross_owner_seam_edges),
        "maximum_qualified_seam_trajectory_drift_over_scale": float(authority.maximum_qualified_seam_trajectory_drift_over_scale),
        "policy": {
            "baseline": "OWNER_PRESERVING_EXACT_POSITION_QUOTIENT",
            "cross_owner_union_requires": [
                "EXACT_FULL_BOUNDARY_EDGE_POSITION_MATCH",
                "EXACTLY_TWO_BOUNDARY_INCIDENCES",
                "DIFFERENT_AUTHORED_OWNERS",
                "REVERSE_EDGE_ORIENTATION",
                "SEALED_POSED_ENDPOINT_TRAJECTORY_DRIFT_LE_1E_9_SCALE",
            ],
            "point_only_contact_welded": False,
            "same_orientation_overlap_welded": False,
            "unknown_trajectory_welded": False,
            "appearance_addressing_changed": False,
            "render_vertex_identity_changed": False,
            "sdf_sign_authority_minted": False,
        },
        "authority": {
            "mechanical_surface_topology_candidate_only": True,
            "independent_court_required": True,
            "final_family_authority_minted": False,
            "training_authorized": False,
        },
        "topology_identity_sha256": authority.identity_sha256,
    }
    manifest["manifest_identity_sha256"] = digest(manifest)
    manifest_path = output_dir / "MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--geometry-response", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = build(a.subject_id, a.geometry_response, a.output_dir)
    print(json.dumps({"status": result["status"], "subject_id": result["subject_id"], "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
