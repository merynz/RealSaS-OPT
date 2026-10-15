from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from tools.corpus.fit8_f_qualified_fa_bundle_v1 import write_deterministic_npz_v1
from tools.corpus.fit8_f_qualified_fa_bundle_v2 import canonical_source_carrier_sha256_v1
from tools.corpus.fit8_polygon_triangulation_qualification_v1 import (
    qualify_polygon_triangulation_topology_v1,
)
from tools.corpus.fit8_static_ray_census_v1 import face_edge_components_v1
from tools.corpus.fit8_surface_topology_invariants_v1 import compute_surface_topology_invariants_v1

SCHEMA_VERSION = "RealSaS.FIT8FBXStaticCarrier.v1"
PROBE_SCHEMA = "RealSaS.UFBXStaticCensusProbe.v3"
EXPECTED_UFBX_REVISION = "5955c5c0b042ac2dc6f32955ab206aaec0620cfc"
OWNER_SCOPE = "DIAGNOSTIC_DOMINANT_NORMALIZED_RAW_SKIN_JOINT"
MATERIAL_SCOPE = "SOURCE_FACE_MATERIAL_PREFERS_NODE_INSTANCE_TABLE"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _require_sha(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"FIT8_FBX_CARRIER_SHA_INVALID:{name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"FIT8_FBX_CARRIER_SHA_NONHEX:{name}") from exc


def qualify_ufbx_probe_payload_v1(
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    probe_executable_sha256: str,
    build_receipt_sha256: str | None = None,
    build_receipt_verifies_expected_ufbx_revision: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _require_sha("source", source_sha256)
    _require_sha("probe_executable", probe_executable_sha256)
    if build_receipt_sha256 is not None:
        _require_sha("build_receipt", build_receipt_sha256)
    if build_receipt_verifies_expected_ufbx_revision and build_receipt_sha256 is None:
        raise ValueError("FIT8_FBX_CARRIER_REVISION_VERIFIED_WITHOUT_BUILD_RECEIPT")
    if payload.get("schema") != PROBE_SCHEMA:
        raise ValueError("FIT8_FBX_CARRIER_PROBE_SCHEMA_INVALID")
    if payload.get("triangulation_proposal") != "UFBX_TRIANGULATE_FACE_REQUIRES_REALSAS_QUALIFICATION":
        raise ValueError("FIT8_FBX_CARRIER_TRIANGULATION_SCOPE_INVALID")
    if payload.get("material_scope") != MATERIAL_SCOPE:
        raise ValueError("FIT8_FBX_CARRIER_MATERIAL_SCOPE_INVALID")
    if payload.get("owner_scope") != OWNER_SCOPE:
        raise ValueError("FIT8_FBX_CARRIER_OWNER_SCOPE_INVALID")

    vertices = np.asarray(payload.get("vertices"), dtype=np.float64)
    faces = np.asarray(payload.get("faces"), dtype=np.int64)
    triangle_polygon_id = np.asarray(payload.get("triangle_polygon_id"), dtype=np.int64)
    face_material_id = np.asarray(payload.get("face_material_id"), dtype=np.int32)
    face_owner_id = np.asarray(payload.get("face_owner_id"), dtype=np.int32)
    source_polygons_raw = payload.get("source_polygons")

    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("FIT8_FBX_CARRIER_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError("FIT8_FBX_CARRIER_FACES_INVALID")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("FIT8_FBX_CARRIER_FACE_INDEX_OOB")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])):
        raise ValueError("FIT8_FBX_CARRIER_DEGENERATE_FACE")
    for name, array in (
        ("triangle_polygon_id", triangle_polygon_id),
        ("face_material_id", face_material_id),
        ("face_owner_id", face_owner_id),
    ):
        if array.shape != (len(faces),):
            raise ValueError(f"FIT8_FBX_CARRIER_FACE_CHANNEL_SHAPE_INVALID:{name}")
    if not isinstance(source_polygons_raw, list) or not source_polygons_raw:
        raise ValueError("FIT8_FBX_CARRIER_SOURCE_POLYGONS_INVALID")
    if int(payload.get("vertex_count", -1)) != len(vertices):
        raise ValueError("FIT8_FBX_CARRIER_VERTEX_COUNT_MISMATCH")
    if int(payload.get("polygon_count", -1)) != len(source_polygons_raw):
        raise ValueError("FIT8_FBX_CARRIER_POLYGON_COUNT_MISMATCH")
    if np.any(triangle_polygon_id < 0) or np.any(triangle_polygon_id >= len(source_polygons_raw)):
        raise ValueError("FIT8_FBX_CARRIER_POLYGON_BINDING_OOB")

    source_polygons: list[tuple[int, ...]] = []
    for polygon in source_polygons_raw:
        row = tuple(int(v) for v in polygon)
        if len(row) < 3 or len(set(row)) != len(row) or min(row) < 0 or max(row) >= len(vertices):
            raise ValueError("FIT8_FBX_CARRIER_SOURCE_POLYGON_INVALID")
        source_polygons.append(row)

    triangles_by_polygon: dict[int, list[tuple[int, int, int]]] = {
        polygon_id: [] for polygon_id in range(len(source_polygons))
    }
    for triangle, polygon_id in zip(faces.tolist(), triangle_polygon_id.tolist()):
        triangles_by_polygon[int(polygon_id)].append(tuple(int(v) for v in triangle))
    triangulation = qualify_polygon_triangulation_topology_v1(source_polygons, triangles_by_polygon)
    if not triangulation.topology_qualified:
        raise ValueError("FIT8_FBX_CARRIER_TRIANGULATION_NOT_QUALIFIED")

    face_component_id = face_edge_components_v1(faces)
    topology = compute_surface_topology_invariants_v1(faces.tolist())
    triangulation_payload = asdict(triangulation)
    triangulation_sha256 = _canonical_json_sha256(triangulation_payload)
    topology_payload = asdict(topology)
    topology_sha256 = _canonical_json_sha256(topology_payload)

    arrays = {
        "vertices": vertices,
        "faces": faces,
        "face_component_id": face_component_id.astype(np.int32, copy=False),
        "face_material_id": face_material_id,
        "face_owner_id": face_owner_id,
        "triangle_polygon_id": triangle_polygon_id.astype(np.int32, copy=False),
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "probe_schema": PROBE_SCHEMA,
        "probe_executable_sha256": probe_executable_sha256,
        "expected_ufbx_revision": EXPECTED_UFBX_REVISION,
        "build_receipt_sha256": build_receipt_sha256,
        "expected_ufbx_revision_verified_by_build_receipt": bool(
            build_receipt_verifies_expected_ufbx_revision
        ),
        "vertex_count": int(len(vertices)),
        "source_polygon_count": int(len(source_polygons)),
        "face_count": int(len(faces)),
        "component_count": int(topology.component_count),
        "boundary_edge_count": int(topology.total_boundary_edges),
        "nonmanifold_edge_count": int(topology.total_nonmanifold_edges),
        "material_region_count": int(len(set(int(v) for v in face_material_id.tolist() if int(v) >= 0))),
        "diagnostic_mechanical_owner_count": int(len(set(int(v) for v in face_owner_id.tolist() if int(v) >= 0))),
        "owner_scope": OWNER_SCOPE,
        "material_scope": MATERIAL_SCOPE,
        "triangulation_qualification_sha256": triangulation_sha256,
        "triangulation_qualification": triangulation_payload,
        "topology_sha256": topology_sha256,
        "source_carrier_sha256": canonical_source_carrier_sha256_v1(vertices, faces),
        "native_motion_names": [str(v) for v in payload.get("native_motion_names", [])],
        "claims": {
            "triangulation_topology_qualified": True,
            "material_id_source_exact": True,
            "mechanical_owner_diagnostic_only": True,
            "semantic_owner_authority": False,
            "ufbx_revision_verified": bool(build_receipt_verifies_expected_ufbx_revision),
        },
    }
    return manifest, arrays


def run_ufbx_static_carrier_v1(
    *,
    source_path: Path,
    probe_executable: Path,
    build_receipt_sha256: str | None = None,
    build_receipt_verifies_expected_ufbx_revision: bool = False,
    timeout_seconds: int = 120,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_path = source_path.resolve()
    probe_executable = probe_executable.resolve()
    if not source_path.is_file():
        raise ValueError("FIT8_FBX_CARRIER_SOURCE_MISSING")
    if not probe_executable.is_file():
        raise ValueError("FIT8_FBX_CARRIER_PROBE_EXECUTABLE_MISSING")
    if timeout_seconds < 1:
        raise ValueError("FIT8_FBX_CARRIER_TIMEOUT_INVALID")
    completed = subprocess.run(
        [str(probe_executable), str(source_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FIT8_FBX_CARRIER_PROBE_FAILED:{completed.returncode}:{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("FIT8_FBX_CARRIER_PROBE_JSON_INVALID") from exc
    return qualify_ufbx_probe_payload_v1(
        payload,
        source_sha256=_sha256_file(source_path),
        probe_executable_sha256=_sha256_file(probe_executable),
        build_receipt_sha256=build_receipt_sha256,
        build_receipt_verifies_expected_ufbx_revision=build_receipt_verifies_expected_ufbx_revision,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--probe-executable", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--build-receipt-sha256")
    parser.add_argument("--build-receipt-verifies-expected-ufbx-revision", action="store_true")
    args = parser.parse_args()
    manifest, arrays = run_ufbx_static_carrier_v1(
        source_path=args.source,
        probe_executable=args.probe_executable,
        build_receipt_sha256=args.build_receipt_sha256,
        build_receipt_verifies_expected_ufbx_revision=args.build_receipt_verifies_expected_ufbx_revision,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    manifest["carrier_npz_sha256"] = write_deterministic_npz_v1(args.output_npz, arrays)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


__all__ = [
    "EXPECTED_UFBX_REVISION",
    "MATERIAL_SCOPE",
    "OWNER_SCOPE",
    "PROBE_SCHEMA",
    "SCHEMA_VERSION",
    "qualify_ufbx_probe_payload_v1",
    "run_ufbx_static_carrier_v1",
]

if __name__ == "__main__":
    main()
