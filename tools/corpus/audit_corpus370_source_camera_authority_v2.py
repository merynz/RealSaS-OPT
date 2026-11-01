#!/usr/bin/env python3
from __future__ import annotations

"""Independently court Corpus370 SOURCE_IMPORT and CAMERA_OBSERVATION.

This V2 court separates source/import fidelity and camera/raster transport from
appearance visibility. The historical V1 admission additionally required the raw
geometry projected support mask to match RGBA alpha coverage. That comparison is
retained as presentation evidence elsewhere, but is not a source-import or camera
identity invariant.

The court reuses exact producer bytes; it never rerenders, repairs geometry, relaxes
tolerances, or mints graphics/surface/training authority. It is failure-preserving
and emits one current court per family per subject plus a family-evidence index.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF
from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "RealSaS.Corpus370SourceCameraAuthoritySweep.v2"
EVIDENCE_SCHEMA = "RealSaS.Corpus370FamilyEvidenceIndex.v1"
SOURCE_COURT_SCHEMA = "RealSaS.SourceImportAuthorityCourt.v2"
CAMERA_COURT_SCHEMA = "RealSaS.CameraObservationAuthorityCourt.v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def ref(path: Path, schema: str) -> dict:
    path = path.resolve()
    require(path.is_file(), "SOURCE_CAMERA_REF_MISSING:" + str(path))
    return {"path": str(path), "sha256": sha256(path), "schema": schema}


def packet_identity(binding: dict, subject_id: str, source_sha: str) -> str:
    payload = {"binding": binding, "subject_id": subject_id, "source_sha256": source_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def raw_geometry(source: Path, probe: Path) -> dict:
    if source.suffix.lower() in {".gltf", ".glb"}:
        return SourceGLTF(source).geometry()
    require(source.suffix.lower() == ".fbx", "SOURCE_CAMERA_UNSUPPORTED_FORMAT")
    value = json.loads(subprocess.check_output([str(probe), str(source)], text=True))
    require(isinstance(value, dict), "SOURCE_CAMERA_UFBX_PROBE_OBJECT")
    return value


def finite_points(value, code: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0, code + "_SHAPE")
    require(bool(np.isfinite(points).all()), code + "_FINITE")
    return points


def audit_source_import(
    *,
    sid: str,
    source: Path,
    source_sha: str,
    imported_path: Path,
    snapshot_path: Path,
    probe: Path,
    policy: dict,
) -> dict:
    require(source.is_file() and sha256(source) == source_sha, "SOURCE_IMPORT_RAW_HASH_DRIFT")
    require(imported_path.is_file(), "SOURCE_IMPORT_IMPORTED_GEOMETRY_MISSING")
    require(snapshot_path.is_file(), "SOURCE_IMPORT_SNAPSHOT_MISSING")

    raw = raw_geometry(source, probe)
    imported = load(imported_path)
    raw_vertices = finite_points(raw.get("vertices"), "SOURCE_IMPORT_RAW_VERTICES")
    imported_vertices = finite_points(imported.get("vertices"), "SOURCE_IMPORT_IMPORTED_VERTICES")
    diagonal = float(np.linalg.norm(np.ptp(raw_vertices, axis=0)))
    require(np.isfinite(diagonal) and diagonal > 0, "SOURCE_IMPORT_ZERO_DIAGONAL")
    raw_tree = cKDTree(raw_vertices)
    imported_tree = cKDTree(imported_vertices)
    raw_to_imported = float(imported_tree.query(raw_vertices)[0].max())
    imported_to_raw = float(raw_tree.query(imported_vertices)[0].max())
    relative = max(raw_to_imported, imported_to_raw) / diagonal
    maximum = float(policy["source_import"]["maximum_relative_symmetric_vertex_error"])
    require(relative <= maximum, f"SOURCE_IMPORT_BIND_VERTEX_DRIFT:{relative}")
    require(
        int(raw.get("bone_count", -1)) == int(imported.get("bone_count", -2)),
        "SOURCE_IMPORT_JOINT_CARDINALITY_DRIFT",
    )

    import_receipt = dict(imported.get("import") or {})
    require(
        import_receipt.get("source_sha256") == source_sha,
        "SOURCE_IMPORT_IMPORT_RECEIPT_SOURCE_DRIFT",
    )
    require(
        import_receipt.get("pose_policy") == "SOURCE_BIND_REST__NO_ANIMATION",
        "SOURCE_IMPORT_POSE_POLICY_DRIFT",
    )
    require(import_receipt.get("raw_source_mutated") is False, "SOURCE_IMPORT_RAW_MUTATION")

    snapshot = load(snapshot_path)
    require(snapshot.get("source_sha256") == source_sha, "SOURCE_IMPORT_SNAPSHOT_SOURCE_DRIFT")
    internal = validate_imported_scene_snapshot(snapshot, tolerance=1e-6)

    return {
        "relative_symmetric_vertex_error": relative,
        "maximum_allowed_relative_symmetric_vertex_error": maximum,
        "raw_vertex_count": int(len(raw_vertices)),
        "imported_vertex_count": int(len(imported_vertices)),
        "raw_bone_count": int(raw.get("bone_count", 0)),
        "imported_bone_count": int(imported.get("bone_count", 0)),
        "imported_scene_internal_consistency": internal,
    }


def unit(value, code: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (3,) and bool(np.isfinite(array).all()), code + "_VECTOR")
    norm = float(np.linalg.norm(array))
    require(norm > 0, code + "_ZERO")
    return array / norm


def audit_camera(
    *,
    sid: str,
    source_sha: str,
    observation_dir: Path,
    policy: dict,
) -> dict:
    candidate_path = observation_dir / "SOURCE_REST_RENDER_CANDIDATE.json"
    snapshot_path = observation_dir / "SOURCE_SCENE_SNAPSHOT.json"
    camera_path = observation_dir / "camera_bundle.json"
    require(candidate_path.is_file(), "CAMERA_CANDIDATE_MISSING")
    require(snapshot_path.is_file(), "CAMERA_SNAPSHOT_MISSING")
    require(camera_path.is_file(), "CAMERA_BUNDLE_MISSING")

    candidate = load(candidate_path)
    bundle = load(camera_path)
    camera_policy = policy["camera_observation"]
    require(candidate.get("schema") == "RealSaS.Controlled8ViewSourceRestCandidate.v2", "CAMERA_CANDIDATE_SCHEMA")
    require(candidate.get("subject_id") == sid, "CAMERA_SUBJECT_DRIFT")
    require(candidate.get("source_sha256") == source_sha, "CAMERA_SOURCE_DRIFT")
    require(candidate.get("pose_address") == "SOURCE_BIND_REST", "CAMERA_POSE_ADDRESS_DRIFT")
    require((candidate.get("import") or {}).get("pose_policy") == "SOURCE_BIND_REST__NO_ANIMATION", "CAMERA_POSE_POLICY_DRIFT")
    require(int(candidate.get("resolution", -1)) == int(camera_policy["resolution"]), "CAMERA_RESOLUTION_DRIFT")
    require(
        (candidate.get("source_scene_snapshot") or {}).get("sha256") == sha256(snapshot_path),
        "CAMERA_SNAPSHOT_HASH_DRIFT",
    )
    require((candidate.get("camera_bundle") or {}).get("sha256") == sha256(camera_path), "CAMERA_BUNDLE_HASH_DRIFT")
    require(bundle.get("schema") == "RealSaS.CameraProjectionBundle.v1", "CAMERA_BUNDLE_SCHEMA")
    require(bundle.get("per_view_recenter") is False, "CAMERA_PER_VIEW_RECENTER")
    require(bundle.get("per_view_pose_change") is False, "CAMERA_PER_VIEW_POSE_CHANGE")

    common_center = np.asarray(bundle.get("common_center_xyz"), dtype=np.float64)
    require(common_center.shape == (3,) and bool(np.isfinite(common_center).all()), "CAMERA_COMMON_CENTER")
    common_half = float(bundle.get("common_half_extent", 0.0))
    require(np.isfinite(common_half) and common_half > 0, "CAMERA_COMMON_HALF_EXTENT")

    views = list(candidate.get("views") or [])
    cameras = list(bundle.get("cameras") or [])
    count = int(camera_policy["view_count"])
    require(len(views) == len(cameras) == count, "CAMERA_VIEW_COUNT")
    metrics = []
    for expected_index, (view, camera) in enumerate(zip(views, cameras)):
        require(int(view.get("view_index", -1)) == expected_index, "CAMERA_VIEW_INDEX")
        require(int(camera.get("view_index", -1)) == expected_index, "CAMERA_PROJECTION_INDEX")
        require(camera.get("schema_version") == "RealSaS.FullSurfaceCameraProjection.v3", "CAMERA_PROJECTION_SCHEMA")
        require(int(camera.get("resolution", -1)) == int(camera_policy["resolution"]), "CAMERA_PROJECTION_RESOLUTION")
        require(abs(float(camera.get("half_extent", 0.0)) - common_half) <= 1e-9 * max(1.0, common_half), "CAMERA_PER_VIEW_SCALE_DRIFT")

        right = unit(camera.get("right"), "CAMERA_RIGHT")
        up = unit(camera.get("screen_up"), "CAMERA_UP")
        forward = unit(camera.get("forward"), "CAMERA_FORWARD")
        require(abs(float(right @ up)) <= 1e-6, "CAMERA_RIGHT_UP_NOT_ORTHOGONAL")
        require(abs(float(right @ forward)) <= 1e-6, "CAMERA_RIGHT_FORWARD_NOT_ORTHOGONAL")
        require(abs(float(up @ forward)) <= 1e-6, "CAMERA_UP_FORWARD_NOT_ORTHOGONAL")
        origin = np.asarray(camera.get("origin"), dtype=np.float64)
        require(origin.shape == (3,) and bool(np.isfinite(origin).all()), "CAMERA_ORIGIN")
        toward_center = unit(common_center - origin, "CAMERA_CENTER_DIRECTION")
        require(float(toward_center @ forward) >= 1.0 - 1e-6, "CAMERA_NOT_POINTING_AT_COMMON_CENTER")

        png = observation_dir / f"V{expected_index}.png"
        mask_path = observation_dir / f"V{expected_index}.mask.bin"
        require(png.is_file() and mask_path.is_file(), "CAMERA_RASTER_BYTES_MISSING")
        require(sha256(png) == (view.get("raster") or {}).get("sha256"), "CAMERA_RASTER_HASH_DRIFT")
        require(sha256(mask_path) == (view.get("foreground_mask") or {}).get("sha256"), "CAMERA_MASK_HASH_DRIFT")
        alpha = np.asarray(Image.open(png).convert("RGBA"))[:, :, 3] > 0
        raw_mask = mask_path.read_bytes()
        require(len(raw_mask) == alpha.size, "CAMERA_MASK_SIZE_DRIFT")
        declared = np.frombuffer(raw_mask, np.uint8).reshape(alpha.shape) > 0
        require(bool(np.array_equal(alpha, declared)), "CAMERA_ALPHA_MASK_DRIFT")
        require(bool(alpha.any()), "CAMERA_EMPTY_FOREGROUND")
        margin = int(view.get("border_margin_px", -1))
        require(margin >= int(camera_policy["minimum_border_margin_px"]), "CAMERA_BORDER_MARGIN_DRIFT")
        require(int(view.get("foreground_pixel_count", -1)) == int(declared.sum()), "CAMERA_FOREGROUND_COUNT_DRIFT")
        metrics.append(
            {
                "view_index": expected_index,
                "foreground_pixel_count": int(declared.sum()),
                "border_margin_px": margin,
                "raster_sha256": sha256(png),
                "foreground_mask_sha256": sha256(mask_path),
            }
        )

    return {
        "view_count": count,
        "resolution": int(camera_policy["resolution"]),
        "common_center_xyz": common_center.tolist(),
        "common_half_extent": common_half,
        "views": metrics,
        "raw_geometry_projection_vs_alpha_mask_compared": False,
        "raw_geometry_projection_vs_alpha_mask_scope": "PRESENTATION_EVIDENCE_ONLY",
    }


def court_ref(path: Path, schema: str, status: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "schema": schema, "status": status}


def family_claim(*, status: str, reasons: list[str], evidence: list[dict], court: dict, metadata: dict) -> dict:
    return {
        "status": status,
        "reason_codes": sorted(set(reasons)),
        "evidence_refs": evidence,
        "court": court,
        "producer_refs": [],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--parent-v2-index", type=Path, required=True)
    parser.add_argument("--composite-v3-index", type=Path, required=True)
    parser.add_argument("--parent-observation-root", type=Path, required=True)
    parser.add_argument("--rescue-observation-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "canonical/CORPUS370_SOURCE_CAMERA_AUTHORITY_POLICY_V2.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    roster_path = args.exact_source_roster.resolve()
    source_root = args.extracted_source_root.resolve()
    parent_path = args.parent_v2_index.resolve()
    composite_path = args.composite_v3_index.resolve()
    parent_root = args.parent_observation_root.resolve()
    rescue_root = args.rescue_observation_root.resolve()
    probe = args.probe.resolve()
    policy_path = args.policy.resolve()
    output_root = args.output_root.resolve()

    roster = load(roster_path)
    parent = load(parent_path)
    composite = load(composite_path)
    policy = load(policy_path)
    require(roster.get("schema") == "RealSaS.Corpus370ExactSourceRoster.v1", "SOURCE_CAMERA_ROSTER_SCHEMA")
    require(int(roster.get("subject_count", -1)) == 370, "SOURCE_CAMERA_ROSTER_COUNT")
    require(parent.get("schema") == "RealSaS.Corpus370PreparedObservationPackets.v2", "SOURCE_CAMERA_PARENT_SCHEMA")
    require(composite.get("schema") == "RealSaS.Corpus370PreparedObservationPackets.v3", "SOURCE_CAMERA_COMPOSITE_SCHEMA")
    require(policy.get("schema") == "RealSaS.Corpus370SourceCameraAuthorityPolicy.v2", "SOURCE_CAMERA_POLICY_SCHEMA")
    require((composite.get("parent") or {}).get("sha256") == sha256(parent_path), "SOURCE_CAMERA_PARENT_SHA_DRIFT")
    require(probe.is_file(), "SOURCE_CAMERA_PROBE_MISSING")

    rows = {str(row["subject_id"]): row for row in roster.get("subjects") or []}
    require(len(rows) == 370, "SOURCE_CAMERA_ROSTER_SUBJECT_SET")
    parent_binding = dict(parent.get("binding") or {})
    rescue_binding = dict((composite.get("rescue") or {}).get("worker_binding") or {})
    rescue_ids = set((composite.get("rescue") or {}).get("selected_subject_ids") or [])
    require(rescue_binding and parent_binding, "SOURCE_CAMERA_BINDING_MISSING")

    output_root.mkdir(parents=True, exist_ok=True)
    evidence_subjects = []
    rollup = {
        "SOURCE_IMPORT": {"PASS": 0, "FAIL": 0},
        "CAMERA_OBSERVATION": {"PASS": 0, "FAIL": 0},
    }
    sweep_rows = []

    for ordinal, sid in enumerate(sorted(rows), 1):
        row = rows[sid]
        source_sha = str(row["source_sha256"])
        source = (source_root / str(row["source_relative_path"])).resolve()
        require(source.is_relative_to(source_root), "SOURCE_CAMERA_SOURCE_ESCAPE")
        binding = rescue_binding if sid in rescue_ids else parent_binding
        root = rescue_root if sid in rescue_ids else parent_root
        identity = packet_identity(binding, sid, source_sha)
        object_root = root / "objects" / identity
        imported_path = object_root / "imported" / sid / "imported.json"
        observation_dir = object_root / "observations" / sid
        snapshot_path = observation_dir / "SOURCE_SCENE_SNAPSHOT.json"
        subject_root = output_root / "subjects" / sid
        source_court_path = subject_root / "SOURCE_IMPORT_AUTHORITY_COURT_V2.json"
        camera_court_path = subject_root / "CAMERA_OBSERVATION_AUTHORITY_COURT_V2.json"

        source_status = "PASS"
        source_reasons: list[str] = []
        source_metrics = {}
        source_traceback = None
        try:
            source_metrics = audit_source_import(
                sid=sid,
                source=source,
                source_sha=source_sha,
                imported_path=imported_path,
                snapshot_path=snapshot_path,
                probe=probe,
                policy=policy,
            )
        except Exception as exc:
            source_status = "FAIL"
            source_reasons = [str(exc).split(":", 1)[0] or type(exc).__name__]
            source_traceback = traceback.format_exc()

        source_court = {
            "schema": SOURCE_COURT_SCHEMA,
            "status": "PASS_SOURCE_IMPORT_AUTHORITY_V2" if source_status == "PASS" else "FAIL_SOURCE_IMPORT_AUTHORITY_V2",
            "court_status": source_status,
            "subject_id": sid,
            "source_sha256": source_sha,
            "source_format": row.get("source_format"),
            "packet_identity_sha256": identity,
            "measurements": source_metrics,
            "reason_codes": source_reasons,
            "traceback": source_traceback,
            "parents": {
                "raw_source": ref(source, "RealSaS.RawSourceBytes"),
                "imported_geometry": ref(imported_path, "RealSaS.ImportedBindGeometry.v1") if imported_path.is_file() else None,
                "source_scene_snapshot": ref(snapshot_path, "RealSaS.ImportedSourceSceneSnapshot.v2") if snapshot_path.is_file() else None,
                "policy": ref(policy_path, policy["schema"]),
            },
            "authority": {
                "family": "SOURCE_IMPORT",
                "appearance_authority_minted": False,
                "training_authorized": False,
            },
        }
        write(source_court_path, source_court)
        rollup["SOURCE_IMPORT"][source_status] += 1

        camera_status = "PASS"
        camera_reasons: list[str] = []
        camera_metrics = {}
        camera_traceback = None
        if source_status != "PASS":
            camera_status = "FAIL"
            camera_reasons = ["SOURCE_IMPORT_REQUIRED_FOR_SOURCE_BOUND_CAMERA"]
        else:
            try:
                camera_metrics = audit_camera(
                    sid=sid,
                    source_sha=source_sha,
                    observation_dir=observation_dir,
                    policy=policy,
                )
            except Exception as exc:
                camera_status = "FAIL"
                camera_reasons = [str(exc).split(":", 1)[0] or type(exc).__name__]
                camera_traceback = traceback.format_exc()

        candidate_path = observation_dir / "SOURCE_REST_RENDER_CANDIDATE.json"
        camera_path = observation_dir / "camera_bundle.json"
        camera_court = {
            "schema": CAMERA_COURT_SCHEMA,
            "status": "PASS_CAMERA_OBSERVATION_AUTHORITY_V2" if camera_status == "PASS" else "FAIL_CAMERA_OBSERVATION_AUTHORITY_V2",
            "court_status": camera_status,
            "subject_id": sid,
            "source_sha256": source_sha,
            "packet_identity_sha256": identity,
            "measurements": camera_metrics,
            "reason_codes": camera_reasons,
            "traceback": camera_traceback,
            "parents": {
                "source_import_court": ref(source_court_path, SOURCE_COURT_SCHEMA),
                "source_rest_candidate": ref(candidate_path, "RealSaS.Controlled8ViewSourceRestCandidate.v2") if candidate_path.is_file() else None,
                "camera_bundle": ref(camera_path, "RealSaS.CameraProjectionBundle.v1") if camera_path.is_file() else None,
                "policy": ref(policy_path, policy["schema"]),
            },
            "layer_separation": policy["layer_separation"],
            "authority": {
                "family": "CAMERA_OBSERVATION",
                "appearance_authority_minted": False,
                "visibility_authority_minted": False,
                "training_authorized": False,
            },
        }
        write(camera_court_path, camera_court)
        rollup["CAMERA_OBSERVATION"][camera_status] += 1

        source_evidence = [ref(source, "RealSaS.RawSourceBytes"), ref(source_court_path, SOURCE_COURT_SCHEMA)]
        if imported_path.is_file():
            source_evidence.append(ref(imported_path, "RealSaS.ImportedBindGeometry.v1"))
        camera_evidence = [ref(camera_court_path, CAMERA_COURT_SCHEMA), ref(source_court_path, SOURCE_COURT_SCHEMA)]
        if candidate_path.is_file():
            camera_evidence.append(ref(candidate_path, "RealSaS.Controlled8ViewSourceRestCandidate.v2"))
        if camera_path.is_file():
            camera_evidence.append(ref(camera_path, "RealSaS.CameraProjectionBundle.v1"))

        evidence_subjects.append(
            {
                "subject_id": sid,
                "source_sha256": source_sha,
                "families": {
                    "SOURCE_IMPORT": family_claim(
                        status=source_status,
                        reasons=source_reasons,
                        evidence=source_evidence,
                        court=court_ref(source_court_path, SOURCE_COURT_SCHEMA, source_status),
                        metadata={"packet_identity_sha256": identity, "source_format": row.get("source_format")},
                    ),
                    "CAMERA_OBSERVATION": family_claim(
                        status=camera_status,
                        reasons=camera_reasons,
                        evidence=camera_evidence,
                        court=court_ref(camera_court_path, CAMERA_COURT_SCHEMA, camera_status),
                        metadata={
                            "packet_identity_sha256": identity,
                            "raw_geometry_projection_vs_alpha_mask_compared": False,
                        },
                    ),
                },
            }
        )
        sweep_rows.append(
            {
                "subject_id": sid,
                "source_sha256": source_sha,
                "packet_identity_sha256": identity,
                "producer_generation": "RESCUE_V3" if sid in rescue_ids else "PARENT_V2",
                "SOURCE_IMPORT": source_status,
                "CAMERA_OBSERVATION": camera_status,
                "source_reason_codes": source_reasons,
                "camera_reason_codes": camera_reasons,
            }
        )
        print(
            json.dumps(
                {
                    "event": "CORPUS370_SOURCE_CAMERA_COURT_V2",
                    "index": ordinal,
                    "total": 370,
                    "subject_id": sid,
                    "source_import": source_status,
                    "camera_observation": camera_status,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": "CURRENT_SOURCE_CAMERA_COURTS_COMPLETE__FINAL_MATRIX_ADMISSION_SEPARATE",
        "producer": {"schema": SCHEMA, "policy_sha256": sha256(policy_path)},
        "subject_count": 370,
        "subjects": evidence_subjects,
        "rollup": rollup,
        "authority": {
            "final_family_matrix_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }
    evidence_path = output_root / "CORPUS370_SOURCE_CAMERA_FAMILY_EVIDENCE_V2.json"
    write(evidence_path, evidence)

    sweep = {
        "schema": SCHEMA,
        "status": "PASS_FAILURE_PRESERVING_SOURCE_CAMERA_SWEEP_V2",
        "subject_count": 370,
        "rollup": rollup,
        "subjects": sweep_rows,
        "parents": {
            "exact_source_roster": ref(roster_path, roster["schema"]),
            "parent_observation_index": ref(parent_path, parent["schema"]),
            "composite_observation_index": ref(composite_path, composite["schema"]),
            "policy": ref(policy_path, policy["schema"]),
        },
        "family_evidence_index": ref(evidence_path, EVIDENCE_SCHEMA),
        "training_authorized": False,
    }
    sweep_path = output_root / "CORPUS370_SOURCE_CAMERA_AUTHORITY_SWEEP_V2.json"
    write(sweep_path, sweep)
    print(json.dumps({"schema": SCHEMA, "rollup": rollup, "evidence_sha256": sha256(evidence_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
