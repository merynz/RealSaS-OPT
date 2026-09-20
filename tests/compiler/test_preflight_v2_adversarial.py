from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.source import (
    seal_source_bytes,
    seal_source_license_provenance,
)
from compiler.realsas_compiler_services.orchestrator.adapters.observation_v2 import (
    qualify_camera_contract_stage,
)
from compiler.realsas_compiler_services.orchestrator.adapters.preflight_v2 import (
    audit_source_mechanics_stage,
    admit_full_subject_stage,
    materialize_observation_render_stage,
    qualify_observation_contract_stage,
    qualify_normalization_domain_stage,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install(ctx, stage_id: str, result: dict) -> None:
    assert result["status"] == "PASS", result
    ctx["ledger"]["stages"].append(
        {"id": stage_id, "status": "PASS", "outputs": list(result["outputs"])}
    )


def _camera_bundle(path: Path) -> Path:
    cameras = []
    for view in range(8):
        cameras.append(
            {
                "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
                "view_id": f"V{view}",
                "view_index": view,
                "origin": [0.0, 0.0, -2.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 1.0, 0.0],
                "forward": [0.0, 0.0, 1.0],
                "half_extent": 1.0,
                "resolution": 8,
            }
        )
    path.write_text(
        json.dumps(
            {"schema": "RealSaS.CameraProjectionBundle.v1", "cameras": cameras},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path):
    camera_path = _camera_bundle(tmp_path / "cameras.json")
    source_files = []
    raster_rows = []
    mask_rows = []
    for view in range(8):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[2:6, 2:6] = (40 + view, 80, 120, 255)
        raster = tmp_path / f"V{view}.png"
        Image.fromarray(rgba, "RGBA").save(raster)

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        mask_path = tmp_path / f"V{view}.mask"
        mask_path.write_bytes(mask.tobytes(order="C"))

        source_files.append(
            {
                "role": f"SOURCE_RASTER_V{view}",
                "path": str(raster),
                "expected_sha256": _sha(raster),
                "expected_size_bytes": raster.stat().st_size,
            }
        )
        raster_rows.append(
            {
                "view_index": view,
                "image": {"path": str(raster), "sha256": _sha(raster)},
            }
        )
        mask_rows.append(
            {
                "view_index": view,
                "mask": {"path": str(mask_path), "sha256": _sha(mask_path)},
            }
        )

    manifest = {
        "subject_id": "SUBJECT_FREE_PREFLIGHT_FIXTURE",
        "source_files": source_files,
        "source_license": {
            "source_pack": "subject-free-fixture",
            "license_name": "TEST_ONLY",
            "license_ref": "TEST_ONLY",
        },
        "source_audit": {"minimum_border_margin_px": 1},
        "admission": {"mode": "CONTROLLED_FULL_SUBJECT_8VIEW_V1"},
        "observation": {
            "camera_bundle": {
                "path": str(camera_path),
                "sha256": _sha(camera_path),
            },
            "source_rasters": raster_rows,
            "source_foreground_masks": mask_rows,
        },
        "normalization": {
            "center_xyz": [0.0, 0.0, 0.0],
            "half_extent": 1.0,
            "coordinate_frame": "REALSAS_OBJECT_FRAME",
        },
    }
    return {
        "run_id": "SUBJECT_FREE_PREFLIGHT",
        "run_root": tmp_path / "run",
        "run_manifest": manifest,
        "ledger": {"stages": []},
        "stage": {"id": ""},
    }


def _run(ctx, stage_id, fn):
    ctx["stage"] = {"id": stage_id}
    result = fn(ctx)
    if result["status"] == "PASS":
        _install(ctx, stage_id, result)
    return result


def test_source_sha_and_license_fail_closed(tmp_path):
    ctx = _fixture(tmp_path)
    ctx["run_manifest"]["source_files"][0]["expected_sha256"] = "0" * 64
    result = _run(ctx, "01_SOURCE_BYTES_SEALED", seal_source_bytes)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["SOURCE_SHA_MISMATCH:SOURCE_RASTER_V0"]

    ctx = _fixture(tmp_path / "license")
    ctx["run_manifest"]["source_license"]["license_ref"] = ""
    result = _run(ctx, "02_SOURCE_LICENSE_PROVENANCE", seal_source_license_provenance)
    assert result["status"] == "BLOCKED"
    assert result["blockers"][0].startswith("SOURCE_LICENSE_INCOMPLETE:")


def test_preflight_happy_path_and_smallest_authority_failures(tmp_path):
    ctx = _fixture(tmp_path)

    assert _run(ctx, "01_SOURCE_BYTES_SEALED", seal_source_bytes)["status"] == "PASS"
    assert _run(ctx, "03_SOURCE_MECHANICAL_AUDIT", audit_source_mechanics_stage)["status"] == "PASS"
    assert _run(ctx, "04_FULL_SUBJECT_ADMISSION", admit_full_subject_stage)["status"] == "PASS"
    assert _run(ctx, "05_CAMERA_CONTRACT_SOLVED", qualify_camera_contract_stage)["status"] == "PASS"
    assert _run(ctx, "06_OBSERVATION_RENDER_8VIEW", materialize_observation_render_stage)["status"] == "PASS"
    assert _run(ctx, "07_OBSERVATION_CONTRACT_QUALIFIED", qualify_observation_contract_stage)["status"] == "PASS"
    assert _run(ctx, "08_NORMALIZATION_DOMAIN_QUALIFIED", qualify_normalization_domain_stage)["status"] == "PASS"

    # Camera authority is exact-hash materialized.
    bad_camera = _fixture(tmp_path / "camera_drift")
    bad_camera["run_manifest"]["observation"]["camera_bundle"]["sha256"] = "f" * 64
    result = _run(
        bad_camera,
        "05_CAMERA_CONTRACT_SOLVED",
        qualify_camera_contract_stage,
    )
    assert result["status"] == "BLOCKED"
    assert result["blockers"] == ["CAMERA_BUNDLE_FILE_REF_INVALID"]

    # A border-truncated subject is an acquisition/admission failure, not negative evidence.
    border = _fixture(tmp_path / "border")
    _run(border, "01_SOURCE_BYTES_SEALED", seal_source_bytes)
    mask_ref = border["run_manifest"]["observation"]["source_foreground_masks"][0]["mask"]
    mask_path = Path(mask_ref["path"])
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 2:6] = 1
    mask_path.write_bytes(mask.tobytes(order="C"))
    mask_ref["sha256"] = _sha(mask_path)
    result = _run(border, "03_SOURCE_MECHANICAL_AUDIT", audit_source_mechanics_stage)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["SOURCE_SUBJECT_TOUCHES_OR_VIOLATES_BORDER_MARGIN"]

    # Observation mask authority may not drift after the Stage03 audit.
    drift = _fixture(tmp_path / "mask_drift")
    _run(drift, "01_SOURCE_BYTES_SEALED", seal_source_bytes)
    _run(drift, "03_SOURCE_MECHANICAL_AUDIT", audit_source_mechanics_stage)
    _run(drift, "04_FULL_SUBJECT_ADMISSION", admit_full_subject_stage)
    _run(drift, "05_CAMERA_CONTRACT_SOLVED", qualify_camera_contract_stage)
    _run(drift, "06_OBSERVATION_RENDER_8VIEW", materialize_observation_render_stage)
    row = drift["run_manifest"]["observation"]["source_foreground_masks"][0]
    new_mask = np.zeros((8, 8), dtype=np.uint8)
    new_mask[1:6, 1:6] = 1
    path = tmp_path / "mask_drift" / "V0_drift.mask"
    path.write_bytes(new_mask.tobytes(order="C"))
    row["mask"] = {"path": str(path), "sha256": _sha(path)}
    with pytest.raises(
        QualificationError,
        match="OBSERVATION_CONTRACT_FOREGROUND_AUTHORITY_DRIFT",
    ):
        _run(
            drift,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            qualify_observation_contract_stage,
        )

    # Normalization is exact object-frame authority; no implicit camera-space substitute.
    invalid_norm = _fixture(tmp_path / "normalization")
    _run(invalid_norm, "01_SOURCE_BYTES_SEALED", seal_source_bytes)
    _run(invalid_norm, "03_SOURCE_MECHANICAL_AUDIT", audit_source_mechanics_stage)
    _run(invalid_norm, "04_FULL_SUBJECT_ADMISSION", admit_full_subject_stage)
    _run(invalid_norm, "05_CAMERA_CONTRACT_SOLVED", qualify_camera_contract_stage)
    _run(invalid_norm, "06_OBSERVATION_RENDER_8VIEW", materialize_observation_render_stage)
    _run(invalid_norm, "07_OBSERVATION_CONTRACT_QUALIFIED", qualify_observation_contract_stage)
    invalid_norm["run_manifest"]["normalization"]["coordinate_frame"] = "CAMERA_FRAME"
    result = _run(
        invalid_norm,
        "08_NORMALIZATION_DOMAIN_QUALIFIED",
        qualify_normalization_domain_stage,
    )
    assert result["status"] == "BLOCKED"
    assert result["blockers"] == ["NORMALIZATION_COORDINATE_FRAME_INVALID"]
