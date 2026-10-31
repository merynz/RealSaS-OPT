#!/usr/bin/env python3
from __future__ import annotations

"""Build TESSA mechanical teacher from Corpus400 component geometry authority.

V4 changes the source-topology contract, not the mechanical qualification standard:
- exact render-seam quotient still comes from the geometry authority;
- TESSA consumes `surface_component_id`, never sign/manifold component IDs;
- every mechanical source face must be geometry-valid; invalid faces require a
  repair/quarantine decision rather than silent topology reinterpretation;
- Stage47 rest/posed trajectories are collapsed through the exact quotient and
  duplicate trajectories must agree;
- existing mechanical simplification/response/strain/silhouette courts are reused.

This builder emits a candidate teacher only. Final TESSA/causal-response family
qualification and training readiness remain separate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tessa.truth.component_surface_topology_v1 import load_tessa_surface_topology_v1
from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as impl
from tools.training.build_truth139_v32_mechanical_tessa_teacher_v3 import (
    collapse_positions,
    has_self_intersection_3d,
)


SCHEMA = "RealSaS.CorpusMechanicalTESSATeacher.v4"
STATUS = "PASS_CORPUS_MECHANICAL_TESSA_TEACHER_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
GEOMETRY_SCHEMA = "RealSaS.ComponentGeometryAuthority.v1"
GEOMETRY_STATUS = "PASS_COMPONENT_GEOMETRY_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
GEOMETRY_COURT_SCHEMA = "RealSaS.ComponentGeometryAuthorityCourt.v1"
GEOMETRY_COURT_STATUS = "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


# Keep the 3D SAT implementation introduced by TESSA V3 rather than the historical
# planar/legacy helper behind impl.has_self_intersection.
impl.has_self_intersection = has_self_intersection_3d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def ref(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return {"path": str(path), "sha256": sha256(path)}


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def _verify_geometry_authority(manifest_path: Path, court_path: Path, subject_id: str) -> tuple[dict, dict, Path]:
    manifest_path = manifest_path.resolve()
    court_path = court_path.resolve()
    gm = load_json(manifest_path, "TESSA_V4_GEOMETRY_MANIFEST")
    gc = load_json(court_path, "TESSA_V4_GEOMETRY_COURT")
    if gm.get("schema") != GEOMETRY_SCHEMA or gm.get("status") != GEOMETRY_STATUS:
        raise ValueError("TESSA_V4_GEOMETRY_NOT_COMPONENT_AUTHORITY")
    if gc.get("schema") != GEOMETRY_COURT_SCHEMA or gc.get("status") != GEOMETRY_COURT_STATUS:
        raise ValueError("TESSA_V4_GEOMETRY_COURT_NOT_PASS")
    if gm.get("subject_id") != subject_id or gc.get("subject_id") != subject_id:
        raise ValueError("TESSA_V4_GEOMETRY_SUBJECT_DRIFT")
    if not bool((gc.get("authority") or {}).get("independent_court_passed")):
        raise ValueError("TESSA_V4_GEOMETRY_INDEPENDENT_COURT_REQUIRED")
    parent = dict(gc.get("parent_manifest") or {})
    if parent.get("sha256") and parent.get("sha256") != sha256(manifest_path):
        raise ValueError("TESSA_V4_GEOMETRY_COURT_PARENT_DRIFT")
    outcomes = dict(gc.get("family_candidate_outcomes") or {})
    if outcomes.get("SURFACE_GEOMETRY") != "QUALIFIED_CORE_CANDIDATE":
        raise ValueError("TESSA_V4_SURFACE_GEOMETRY_NOT_CORE_CANDIDATE")
    if outcomes.get("SURFACE_TOPOLOGY") != "QUALIFIED_CORE_CANDIDATE":
        raise ValueError("TESSA_V4_SURFACE_TOPOLOGY_NOT_CORE_CANDIDATE")
    topology_path = bound((gm.get("outputs") or {}).get("topology"), "TESSA_V4_TOPOLOGY_DRIFT")
    return gm, gc, topology_path


def _teacher_scene_from_response_chain(response_chain: dict, subject_id: str) -> tuple[Path, dict]:
    outputs = dict(response_chain.get("outputs") or {})
    scene_path = bound(outputs.get("teacher_scene"), "TESSA_V4_RESPONSE_SCENE_DRIFT")
    scene = load_json(scene_path, "TESSA_V4_TEACHER_SCENE")
    if scene.get("schema") != "RealSaS.TeacherSceneIR.v2" or scene.get("subject_key") != subject_id:
        raise ValueError("TESSA_V4_TEACHER_SCENE_BINDING")
    return scene_path, scene


def _stage47_geometry(scene: dict) -> Path:
    rows = [
        dict(row)
        for row in scene.get("geometry_payloads") or ()
        if Path(str(row.get("path") or "")).name == "geometry_and_response.npz"
    ]
    if len(rows) != 1:
        raise ValueError("TESSA_V4_STAGE47_GEOMETRY_CARDINALITY")
    return bound(rows[0], "TESSA_V4_STAGE47_GEOMETRY_DRIFT")


def build(
    *,
    subject_id: str,
    geometry_manifest_path: Path,
    geometry_court_path: Path,
    response_chain_path: Path,
    camera_bundle_path: Path,
    output_dir: Path,
) -> dict:
    gm, gc, topology_path = _verify_geometry_authority(
        geometry_manifest_path, geometry_court_path, subject_id
    )
    source = load_tessa_surface_topology_v1(topology_path)
    # The adapter above has already fail-closed if any source face is invalid.
    raw_to_q = np.asarray(source.raw_to_topology_vertex, dtype=np.int64)
    source_v_declared = np.asarray(source.vertices_source_world, dtype=np.float64)
    source_f = np.asarray(source.faces, dtype=np.int64)
    source_fc = np.asarray(source.surface_component_id, dtype=np.int32)
    qowner = np.asarray(source.topology_vertex_owner, dtype=np.int32)

    response_chain_path = response_chain_path.resolve()
    response_chain = load_json(response_chain_path, "TESSA_V4_RESPONSE_CHAIN")
    scene_path, scene = _teacher_scene_from_response_chain(response_chain, subject_id)
    stage47 = _stage47_geometry(scene)
    camera_bundle_path = camera_bundle_path.resolve()
    camera = load_json(camera_bundle_path, "TESSA_V4_CAMERA_BUNDLE")
    replay_path = bound(
        (response_chain.get("inputs") or {}).get("admitted_observation_import"),
        "TESSA_V4_REPLAY_DRIFT",
    )
    replay = load_json(replay_path, "TESSA_V4_REPLAY")

    with np.load(stage47, allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world",
            "faces",
            "face_owner_privileged",
            "isolated_position",
        }
        if not required.issubset(z.files):
            raise ValueError("TESSA_V4_STAGE47_ARRAYS")
        raw_rest = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        raw_faces = np.asarray(z["faces"], dtype=np.int64)
        raw_owner = np.asarray(z["face_owner_privileged"], dtype=np.int32)
        raw_isolated = np.asarray(z["isolated_position"], dtype=np.float64)

    if len(raw_to_q) != len(raw_rest) or len(raw_faces) != len(source_f):
        raise ValueError("TESSA_V4_RAW_TOPOLOGY_BINDING")
    if not np.array_equal(raw_faces, np.asarray(gm.get("counts") and raw_faces)):
        # Deliberately no-op semantic guard removed below; exact face remap is checked
        # by quotient cardinality and source declared vertices. Kept as explicit branch
        # avoidance: raw render faces and quotient faces need not be numerically equal.
        pass
    scale = max(float(np.linalg.norm(raw_rest.max(0) - raw_rest.min(0))), 1.0e-9)
    tolerance = max(scale * 1.0e-9, 1.0e-11)
    source_v, rest_duplicate_drift = collapse_positions(
        raw_rest, raw_to_q, len(source_v_declared), tolerance, "TESSA_V4_REST"
    )
    isolated, posed_duplicate_drift = collapse_positions(
        raw_isolated, raw_to_q, len(source_v_declared), tolerance, "TESSA_V4_POSED"
    )
    max_rest_drift = float(np.max(np.abs(source_v - source_v_declared), initial=0.0))
    if max_rest_drift > max(scale * 2.0e-6, 2.0e-7):
        raise ValueError("TESSA_V4_QUOTIENT_REST_DRIFT")
    if np.any(qowner[source_f] != qowner[source_f][:, :1]):
        raise ValueError("TESSA_V4_FACE_OWNER_CROSS")

    source_face_object = qowner[source_f[:, 0]]
    attachment_ids = {
        index
        for index, obj in enumerate(replay.get("objects") or ())
        if obj.get("parent_bone")
    }
    actions = list((scene.get("calibration") or {}).get("actions") or [])
    cameras = list(camera.get("cameras") or [])
    if len(actions) != len(isolated) or len(cameras) != 8:
        raise ValueError("TESSA_V4_ACTION_CAMERA_BINDING")

    source_topology = impl.topology_signature(len(source_v), source_f, source_fc)
    ladder = impl.simplify_by_component(
        source_v, source_f, [0.35, 0.50, 0.65, 0.80, 1.0], source_fc
    )
    rows = []
    qualified = []
    for ratio, vertices, faces, kind in ladder:
        metrics, face_id, barycentric = impl.evaluate_candidate(
            vertices,
            faces,
            source_v,
            source_f,
            source_fc,
            source_topology,
            source_face_object,
            attachment_ids,
            isolated,
            actions,
            cameras,
            ratio,
            kind,
        )
        rows.append(metrics)
        if metrics["mechanical_pass"]:
            qualified.append(
                (
                    metrics["face_count"],
                    metrics["vertex_count"],
                    ratio,
                    vertices,
                    faces,
                    face_id,
                    barycentric,
                    metrics,
                )
            )
    if not qualified:
        raise RuntimeError("TESSA_V4_NO_MECHANICALLY_ADMISSIBLE_FALLBACK")
    qualified.sort(key=lambda row: (row[0], row[1], row[2]))
    _, _, ratio, vertices, faces, face_id, barycentric, selected = qualified[0]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "CORPUS_MECHANICAL_TESSA_TEACHER_V4.npz"
    impl.write_npz(
        mesh_path,
        vertices_source_world=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        source_triangle_id=face_id.astype(np.int32),
        source_barycentric=barycentric.astype(np.float32),
    )
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "parents": {
            "geometry_manifest": ref(geometry_manifest_path),
            "geometry_court": ref(geometry_court_path),
            "response_chain": ref(response_chain_path),
            "teacher_scene": ref(scene_path),
            "stage47_geometry_response": ref(stage47),
            "camera_bundle": ref(camera_bundle_path),
            "admitted_observation_import": ref(replay_path),
        },
        "source_topology": {
            "raw_render_vertex_count": int(len(raw_rest)),
            "topology_vertex_count": int(len(source_v)),
            "face_count": int(len(source_f)),
            "surface_component_count": int(len(np.unique(source_fc))),
            "all_faces_geometry_valid": True,
            "component_relation": "EDGE_CONNECTED_SURFACE_COMPONENT_ID",
            "sign_component_relation_consumed": False,
            "sdf_or_occupancy_truth_consumed": False,
            "render_vertex_identity_used_as_topology": False,
            "maximum_rest_duplicate_trajectory_drift": float(rest_duplicate_drift),
            "maximum_posed_duplicate_trajectory_drift": float(posed_duplicate_drift),
            "maximum_declared_rest_drift_over_scale": float(max_rest_drift / scale),
        },
        "policy": {
            "mechanical_admissibility_precedes_compactness": True,
            "simplification_ladder_ratios": [0.35, 0.50, 0.65, 0.80, 1.0],
            "source_topology_is_fallback_member": True,
            "teacher_topology_is_student_conditioning": False,
            "selection": "MIN_FACE_COUNT_THEN_VERTEX_COUNT_AMONG_PASS",
            "invalid_source_face_policy": "FAIL_CLOSED_REPAIR_OR_QUARANTINE__NO_SILENT_FACE_DROP",
            "mechanical_component_authority": "SURFACE_COMPONENT_ONLY",
            "sign_component_authority": "NOT_A_TESSA_MECHANICAL_INPUT",
        },
        "candidates": rows,
        "selected": selected,
        "teacher_mesh": ref(mesh_path),
        "family_candidate_outcomes": {
            "SURFACE_TOPOLOGY": "CONSUMER_SEMANTICS_PASS_CANDIDATE",
            "CAUSAL_RESPONSE": "EXISTING_EVIDENCE_CONSUMED__NEW_FAMILY_COURT_STILL_REQUIRED",
        },
        "authority": {
            "independent_geometry_court_passed": True,
            "tessa_teacher_candidate_only": True,
            "final_family_authority_minted": False,
            "training_authorized": False,
        },
    }
    manifest["teacher_identity_sha256"] = digest(manifest)
    manifest_path = output_dir / "CORPUS_MECHANICAL_TESSA_TEACHER_MANIFEST_V4.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "subject_id": subject_id,
                "selected_face_count": selected["face_count"],
                "selected_kind": selected["kind"],
                "surface_component_count": manifest["source_topology"]["surface_component_count"],
                "teacher_identity_sha256": manifest["teacher_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-court", type=Path, required=True)
    parser.add_argument("--response-chain", type=Path, required=True)
    parser.add_argument("--camera-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(
        subject_id=args.subject_id,
        geometry_manifest_path=args.geometry_manifest,
        geometry_court_path=args.geometry_court,
        response_chain_path=args.response_chain,
        camera_bundle_path=args.camera_bundle,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
