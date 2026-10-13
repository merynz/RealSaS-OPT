from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.source import seal_source_bytes
from compiler.realsas_compiler_services.orchestrator.adapters.preflight_v2 import (
    audit_source_mechanics_stage,
    admit_full_subject_stage,
)
from compiler.realsas_compiler_services.orchestrator.adapters.source_scene_admission_v2 import (
    ADMISSION_MODE,
    PROFILE,
    validate_source_scene_admission,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def matrix(tx=0.0, ty=0.0, tz=0.0):
    value = np.eye(4)
    value[:3, 3] = [tx, ty, tz]
    return value.tolist()


def fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / "subject.raw"
    raw.write_bytes(b"exact raw subject bytes")
    source_sha = sha(raw)

    camera_payload = {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "view_order": [f"V{i}" for i in range(8)],
        "common_center_xyz": [0.0, 0.0, 0.0],
        "common_half_extent": 1.0,
        "per_view_recenter": False,
        "per_view_pose_change": False,
        "cameras": [
            {
                "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
                "view_id": f"V{i}", "view_index": i,
                "origin": [0.0, 0.0, -2.0], "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 1.0, 0.0], "forward": [0.0, 0.0, 1.0],
                "half_extent": 1.0, "resolution": 32,
            }
            for i in range(8)
        ],
    }
    camera = dump(tmp_path / "camera.json", camera_payload)

    rasters, masks, candidate_views, proof_views = [], [], [], []
    evidence_files = [("RAW_SOURCE", raw), ("CAMERA_BUNDLE", camera)]
    for i in range(8):
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[8:24, 8:24] = (60 + i, 90, 120, 255)
        raster = tmp_path / f"V{i}.png"
        Image.fromarray(rgba, "RGBA").save(raster)
        mask_array = np.zeros((32, 32), dtype=np.uint8)
        mask_array[8:24, 8:24] = 1
        mask = tmp_path / f"V{i}.mask.bin"
        mask.write_bytes(mask_array.tobytes())
        rasters.append({"view_index": i, "image": ref(raster)})
        masks.append({"view_index": i, "mask": ref(mask)})
        candidate_views.append({
            "view_index": i,
            "raster": {"sha256": sha(raster)},
            "foreground_mask": {"sha256": sha(mask)},
            "border_margin_px": 8,
        })
        proof_views.append({
            "view_index": i,
            "maximum_mask_gap_px": 1.0,
            "raster_sha256": sha(raster),
            "mask_sha256": sha(mask),
            "reference_sha256": hashlib.sha256(f"reference-{i}".encode()).hexdigest(),
            "border_margin_px": 8,
        })
        evidence_files.extend(((f"SOURCE_RASTER_V{i}", raster), (f"SOURCE_MASK_V{i}", mask)))

    root_world = matrix(3.0, 4.0, 5.0)
    child_local = matrix(-1.0, 2.0, 0.5)
    child_world = (np.asarray(root_world) @ np.asarray(child_local)).tolist()
    snapshot_payload = {
        "schema": "RealSaS.ImportedSourceSceneSnapshot.v2",
        "source_sha256": source_sha,
        "source_world_to_canonical": matrix(),
        "canonical_to_source_world": matrix(),
        "nodes": [
            {"node_id": "root", "parent_node_id": None, "parent_bone": None,
             "node_to_source_world": root_world, "effective_node_to_parent": root_world,
             "node_to_canonical": root_world, "active_action_count": 0,
             "active_nla_track_count": 0, "driver_count": 0},
            {"node_id": "equipment", "parent_node_id": "root", "parent_bone": "Grip",
             "node_to_source_world": child_world, "effective_node_to_parent": child_local,
             "node_to_canonical": child_world, "active_action_count": 0,
             "active_nla_track_count": 0, "driver_count": 0},
        ],
        "bones": [{"source_bone_name": "Grip"}],
    }
    snapshot = dump(tmp_path / "scene.json", snapshot_payload)
    candidate_payload = {
        "schema": "RealSaS.Controlled8ViewSourceRestCandidate.v2",
        "status": "ENGINEERING_MATERIALIZED__SOURCE_ADMISSION_PENDING",
        "subject_id": "fit8_fixture",
        "source_sha256": source_sha,
        "resolution": 1024,
        "import": {"raw_source_mutated": False},
        "pose_address": "SOURCE_BIND_REST",
        "native_frame0_is_rest_assumed": False,
        "source_scene_snapshot": {"sha256": sha(snapshot), "file": snapshot.name},
        "internal_consistency": {"animation_free_rest": True, "normalization_uniform": True},
        "camera_bundle": {"sha256": sha(camera), "file": camera.name},
        "views": candidate_views,
        "training_admission_claimed": False,
        "product_pass_claimed": False,
    }
    candidate = dump(tmp_path / "candidate.json", candidate_payload)
    proof_payload = {
        "schema": "RealSaS.IrisBindObservationAdmission.v1",
        "status": "PASS_IRIS_BIND_OBSERVATIONS",
        "subject_id": "fit8_fixture",
        "source_sha256": source_sha,
        "candidate_sha256": sha(candidate),
        "imported_geometry_sha256": hashlib.sha256(b"independent imported geometry").hexdigest(),
        "camera_sha256": sha(camera),
        "relative_vertex_error": 1e-7,
        "raw_vertex_count": 100,
        "imported_vertex_count": 100,
        "bone_count": 1,
        "native_motion_inventory": ["Idle"],
        "source_material_metadata": [],
        "views": proof_views,
        "scope": "IMAGE_DERIVED_FSTAR_BIND_OBSERVATIONS__NOT_FULL_MECHANICAL_TRUTH",
        "policy": {
            "schema": "RealSaS.IrisObservationPreparationPolicy.v1",
            "scope": "IMAGE_DERIVED_FSTAR_BIND_OBSERVATIONS__NOT_FULL_MECHANICAL_TRUTH",
            "resolution": 1024,
            "view_count": 8,
            "minimum_border_margin_px": 8,
            "maximum_relative_vertex_error": 1e-5,
            "maximum_mask_gap_px": 2.0,
            "reference_coverage": "TRIANGLE_UNIT_PIXEL_SQUARE_INTERSECTION",
            "raster_filter": "BOX_1PX",
            "pose": "SOURCE_BIND_REST",
            "native_frame0_is_bind_assumed": False,
            "native_lit_material_parity_claimed": False,
            "full_native_motion_truth_admitted": False,
            "go_stage_qualification_claimed": False,
        },
        "full_native_motion_truth_admitted": False,
        "native_lit_parity_claimed": False,
        "go_stage_qualification_claimed": False,
        "fit8_pass_claimed": False,
    }
    proof = dump(tmp_path / "IRIS_OBSERVATION_ADMISSION.json", proof_payload)
    evidence_files.extend((("SOURCE_SCENE_SNAPSHOT", snapshot), ("SOURCE_REST_CANDIDATE", candidate),
                           ("IRIS_OBSERVATION_ADMISSION", proof)))

    source_payload = {
        "schema": "RealSaS.SourceByteSeal.v1",
        "files": [{"role": role, "sha256": sha(path)} for role, path in evidence_files],
    }
    manifest = {
        "subject_id": "fit8_fixture",
        "source_files": [
            {"role": role, "path": str(path), "expected_sha256": sha(path),
             "expected_size_bytes": path.stat().st_size}
            for role, path in evidence_files
        ],
        "source_audit": {
            "profile": PROFILE,
            "subject_key": "fit8_fixture",
            "primary_source_role": "RAW_SOURCE",
            "minimum_border_margin_px": 8,
            "source_rest_candidate": ref(candidate),
            "source_scene_snapshot": ref(snapshot),
            "iris_observation_admission": ref(proof),
            "iris_observation_admission_required": True,
            "matrix_tolerance": 1e-6,
        },
        "admission": {"mode": ADMISSION_MODE},
        "observation": {"camera_bundle": ref(camera), "source_rasters": rasters,
                        "source_foreground_masks": masks},
    }
    return source_payload, manifest, {"raw": raw, "camera": camera, "snapshot": snapshot,
                                      "candidate": candidate, "proof": proof}


def test_source_scene_bind_admission_happy_path(tmp_path):
    source_payload, manifest, _ = fixture(tmp_path)
    ctx = {"run_manifest": manifest}
    result = validate_source_scene_admission(ctx, source_payload)
    assert result["profile"] == PROFILE
    assert result["node_count"] == 2
    assert result["bone_count"] == 1
    assert result["parent_relation_count"] == 1
    assert result["bone_parent_relation_count"] == 1
    assert result["relative_vertex_error"] == 1e-7
    assert result["maximum_mask_gap_px"] == 1.0
    assert result["native_motion_qualified"] is False
    assert result["appearance_qualified"] is False
    assert result["product_pass_claimed"] is False


def test_scene_transform_camera_view_and_exact_proof_drift_fail(tmp_path):
    source_payload, manifest, paths = fixture(tmp_path)
    ctx = {"run_manifest": manifest}

    snapshot = json.loads(paths["snapshot"].read_text())
    snapshot["nodes"][1]["node_to_source_world"][0][3] += 0.25
    dump(paths["snapshot"], snapshot)
    manifest["source_audit"]["source_scene_snapshot"]["sha256"] = sha(paths["snapshot"])
    for row in source_payload["files"]:
        if row["role"] == "SOURCE_SCENE_SNAPSHOT": row["sha256"] = sha(paths["snapshot"])
    with pytest.raises(QualificationError, match="SNAPSHOT_BINDING_DRIFT|PARENT_WORLD_DRIFT"):
        validate_source_scene_admission(ctx, source_payload)

    source_payload, manifest, paths = fixture(tmp_path / "proof")
    proof = json.loads(paths["proof"].read_text()); proof["views"][0]["maximum_mask_gap_px"] = 3.0
    dump(paths["proof"], proof)
    manifest["source_audit"]["iris_observation_admission"]["sha256"] = sha(paths["proof"])
    for row in source_payload["files"]:
        if row["role"] == "IRIS_OBSERVATION_ADMISSION": row["sha256"] = sha(paths["proof"])
    with pytest.raises(QualificationError, match="MASK_PARITY_FAIL"):
        validate_source_scene_admission({"run_manifest": manifest}, source_payload)

    source_payload, manifest, paths = fixture(tmp_path / "camera")
    camera = json.loads(paths["camera"].read_text()); camera["per_view_recenter"] = True
    dump(paths["camera"], camera)
    manifest["observation"]["camera_bundle"]["sha256"] = sha(paths["camera"])
    for row in source_payload["files"]:
        if row["role"] == "CAMERA_BUNDLE": row["sha256"] = sha(paths["camera"])
    with pytest.raises(QualificationError, match="CAMERA_BINDING_DRIFT|PROOF_CAMERA_DRIFT|PER_VIEW_CAMERA"):
        validate_source_scene_admission({"run_manifest": manifest}, source_payload)


def test_stage03_04_require_strong_profile_pair_and_stage1_sealed_evidence(tmp_path):
    _, manifest, _ = fixture(tmp_path)
    ctx = {"run_id": "SOURCE_SCENE_V2_TEST", "run_root": tmp_path / "run",
           "run_manifest": manifest, "ledger": {"stages": []}, "stage": {"id": "01_SOURCE_BYTES_SEALED"}}
    stage1 = seal_source_bytes(ctx)
    assert stage1["status"] == "PASS"
    ctx["ledger"]["stages"].append({"id": "01_SOURCE_BYTES_SEALED", "status": "PASS",
                                      "outputs": stage1["outputs"]})
    ctx["stage"] = {"id": "03_SOURCE_MECHANICAL_AUDIT"}
    stage3 = audit_source_mechanics_stage(ctx)
    assert stage3["status"] == "PASS"
    assert stage3["diagnostics"]["source_scene_bind_admission"] is True
    assert stage3["diagnostics"]["source_scene_parent_relations"] == 1
    assert stage3["diagnostics"]["source_scene_bone_parent_relations"] == 1
    ctx["ledger"]["stages"].append({"id": "03_SOURCE_MECHANICAL_AUDIT", "status": "PASS",
                                      "outputs": stage3["outputs"]})
    ctx["stage"] = {"id": "04_FULL_SUBJECT_ADMISSION"}
    stage4 = admit_full_subject_stage(ctx)
    assert stage4["status"] == "PASS"
    assert stage4["diagnostics"]["source_scene_bind_admitted"] is True

    mismatch = copy.deepcopy(ctx)
    mismatch["run_manifest"] = copy.deepcopy(manifest)
    mismatch["run_manifest"]["admission"]["mode"] = "CONTROLLED_FULL_SUBJECT_8VIEW_V1"
    result = admit_full_subject_stage(mismatch)
    assert result["status"] == "BLOCKED"
    assert result["blockers"] == ["FULL_SUBJECT_ADMISSION_MODE_PROFILE_MISMATCH"]


def test_proof_must_be_stage1_sealed(tmp_path):
    source_payload, manifest, _ = fixture(tmp_path)
    source_payload["files"] = [row for row in source_payload["files"]
                               if row["role"] != "IRIS_OBSERVATION_ADMISSION"]
    with pytest.raises(QualificationError, match="EVIDENCE_NOT_STAGE1_SEALED"):
        validate_source_scene_admission({"run_manifest": manifest}, source_payload)
