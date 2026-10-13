from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.platform.prepare_fit8_source_fragments_v1 import prepare


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(root: Path, rel: str, data: bytes) -> dict:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": rel, "sha256": sha(path), "size_bytes": path.stat().st_size}


def test_builder_rehashes_bundle_and_emits_portable_strong_stage01_08_fragments(tmp_path):
    root = tmp_path / "bundle"
    rows = []
    corpus = put(root, "manifest/CORPUS_SEAL.json", b'{"schema":"corpus"}')
    rows.append(corpus)
    subjects = []
    selected = []
    for index in range(8):
        sid = f"fit8_fixture_{index}"
        raw = put(root, f"raw/{sid}/subject.bin", f"raw-{index}".encode())
        license_bytes = put(root, f"raw/{sid}/License.txt", b"CC0-1.0")
        page = put(root, f"provenance/{sid}.html", f"page-{index}".encode())
        candidate = put(root, f"observations/{sid}/SOURCE_REST_RENDER_CANDIDATE.json", b"candidate")
        snapshot = put(root, f"observations/{sid}/SOURCE_SCENE_SNAPSHOT.json", b"snapshot")
        proof = put(root, f"observations/{sid}/IRIS_OBSERVATION_ADMISSION.json", b"proof")
        camera_payload = {
            "schema": "RealSaS.CameraProjectionBundle.v1",
            "common_center_xyz": [float(index), 0.0, 1.0],
            "common_half_extent": 2.0,
            "per_view_recenter": False,
            "per_view_pose_change": False,
            "cameras": [{"view_index": view} for view in range(8)],
        }
        camera = put(root, f"observations/{sid}/camera_bundle.json",
                     (json.dumps(camera_payload, sort_keys=True) + "\n").encode())
        rows.extend((raw, license_bytes, page, candidate, snapshot, proof, camera))
        for view in range(8):
            rows.append(put(root, f"observations/{sid}/V{view}.png", f"png-{index}-{view}".encode()))
            rows.append(put(root, f"observations/{sid}/V{view}.mask.bin", bytes([view + 1])))
        subject = {
            "subject_id": sid,
            "pack_id": f"pack-{index}",
            "raw_source": raw,
            "license_bytes": license_bytes,
            "source_page_snapshot": page,
            "source_scene_evidence": {
                "source_rest_candidate": candidate,
                "source_scene_snapshot": snapshot,
                "iris_observation_admission": proof,
                "camera_bundle": camera,
            },
        }
        subjects.append(subject)
        selected.append({
            "subject_key": sid,
            "pack_id": subject["pack_id"],
            "source": {
                "corpus_relative_path": raw["path"],
                "sha256": raw["sha256"],
                "byte_size": raw["size_bytes"],
                "source_page_sha256": page["sha256"],
                "license": "CC0-1.0",
            },
        })

    manifest = {
        "schema": "RealSaS.Fit8PlatformStage01InputBundle.v1",
        "subject_count": 8,
        "subjects": subjects,
    }
    manifest_path = root / "BOOTSTRAP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "RealSaS.Fit8PlatformStage01InputBundleReceipt.v1",
        "file_count": len(rows),
        "files": rows,
    }
    receipt_path = root / "BUNDLE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    selection = {"schema": "RealSaS.FIT8DiversitySelection.v1", "selected": selected}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8")
    locator = {
        "schema": "RealSaS.Fit8PlatformStage01InputBundleLocator.v1",
        "subject_count": 8,
        "selection": {"path": "canonical/FIT8_DIVERSITY_SELECTION_V1.json",
                      "sha256": sha(selection_path)},
        "drive": {"sha256": "a" * 64, "file_id": "drive-fixture"},
        "bundle": {"manifest_sha256": sha(manifest_path), "receipt_sha256": sha(receipt_path)},
    }
    locator_path = tmp_path / "locator.json"
    locator_path.write_text(json.dumps(locator, sort_keys=True) + "\n", encoding="utf-8")

    result = prepare(root, locator_path, selection_path, tmp_path / "out")
    assert result["subject_count"] == 8
    assert result["selection_sha256"] == sha(selection_path)
    assert result["later_stage_manifest_sections_included"] is False
    assert result["fit8_pass_claimed"] is False
    for row in result["subjects"]:
        fragment = json.loads(Path(row["manifest_fragment"]["path"]).read_text())
        inventory = json.loads(Path(row["artifact_inventory"]["path"]).read_text())
        assert fragment["source_license"]["profile"] == "EXACT_SOURCE_PROVENANCE_V2"
        assert fragment["source_audit"]["profile"] == "SOURCE_SCENE_BIND_V2"
        assert fragment["admission"]["mode"] == "CONTROLLED_FULL_SUBJECT_8VIEW_V2_SOURCE_SCENE"
        assert fragment["iris_fit"]["training_profile"] == "SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"
        assert fragment["normalization"]["coordinate_frame"] == "REALSAS_OBJECT_FRAME"
        assert len(fragment["observation"]["source_rasters"]) == 8
        assert len(fragment["observation"]["source_foreground_masks"]) == 8
        assert len(fragment["source_files"]) == 24
        assert len(inventory["artifacts"]) == 24
        assert all(item["path"].startswith("${REALSAS_FIT8_INPUT_ROOT}/")
                   for item in fragment["source_files"])
        assert all("fit8_fixture_0" not in item["role"] for item in fragment["source_files"])

    drifted = json.loads(selection_path.read_text())
    drifted["selected"][0]["source"]["source_page_sha256"] = "f" * 64
    selection_path.write_text(json.dumps(drifted, sort_keys=True) + "\n", encoding="utf-8")
    locator["selection"]["sha256"] = sha(selection_path)
    locator_path.write_text(json.dumps(locator, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FIT8_SELECTION_SOURCE_PAGE_DRIFT"):
        prepare(root, locator_path, selection_path, tmp_path / "drifted")
