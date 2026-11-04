"""V3 family transport makes original admitted import bytes mandatory.

The old V2 transport remains immutable. Here we exercise one authentic-shaped
V4 PASS packet plus 369 explicit failures; none may be silently promoted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from tools.corpus.prepare_corpus370_platform_truth_bundle_v3 import prepare
from test_corpus370_platform_family_input_bundle_v2 import write_glb


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True) + "\n")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    extracted = root / "sources"
    packet = root / "v4-packet"
    packet.mkdir(parents=True, exist_ok=True)
    roster_rows = []
    source_rows = []
    provenance_rows = []
    failures = []
    license_file = root / "license.txt"
    page_file = root / "source.html"
    license_file.write_text("CC0")
    page_file.write_text("original-provenance-snapshot")
    for i in range(370):
        sid = f"subject_{i:03d}"
        rel = f"assets/{sid}.glb"
        source = extracted / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        write_glb(source, {"extras": {"subject_id": sid}})
        roster_rows.append({"subject_id": sid,
                            "source_relative_path": rel,
                            "source_sha256": sha(source)})
        source_rows.append({"subject_id": sid,
                            "source_relative_path": rel,
                            "source_sha256": sha(source)})
        provenance_rows.append({
            "subject_id": sid, "source_sha256": sha(source),
            "source_pack": "synthetic", "display_name": sid,
            "license_name": "CC0", "license_ref": "urn:test:cc0",
            "license_bytes": {"path": str(license_file), "sha256": sha(license_file)},
            "source_page_snapshot": {"path": str(page_file), "sha256": sha(page_file)},
        })
        if i:
            failure = write(root / "failures" / f"{sid}.json", {
                "subject_id": sid, "source_sha256": sha(source),
                "reason": "EXPLICIT_OLD_MASK_FAIL", "training_authorized": False,
            })
            failures.append({
                "subject_id": sid, "source_sha256": sha(source),
                "source_format": ".glb", "source_id": "synthetic",
                "error_type": "ValueError", "reason": "EXPLICIT_OLD_MASK_FAIL",
                "receipt": {"path": str(failure), "sha256": sha(failure)},
            })
    policy_path = Path(__file__).resolve().parents[2] / "canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json"
    policy = json.loads(policy_path.read_text())
    subject_id = "subject_000"
    source_sha = roster_rows[0]["source_sha256"]
    imported = write(packet / "imported.json", {
        "import": {"source_sha256": source_sha,
                   "pose_policy": "SOURCE_BIND_REST__NO_ANIMATION"},
        "vertices": [[0.0, 0.0, 0.0]],
    })
    snapshot = write(packet / "SOURCE_SCENE_SNAPSHOT.json",
                     {"source_sha256": source_sha})
    camera = write(packet / "camera_bundle.json", {
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "cameras": [{"view_index": i} for i in range(8)],
    })
    candidate_views, admission_views = [], []
    for i in range(8):
        image = packet / f"V{i}.png"
        mask = packet / f"V{i}.mask.bin"
        image.write_bytes(b"original-raster-" + bytes([i]))
        mask.write_bytes(b"original-mask-" + bytes([i]))
        candidate_views.append({"view_index": i,
                                "raster": {"sha256": sha(image)},
                                "foreground_mask": {"sha256": sha(mask)}})
        admission_views.append({"view_index": i,
                                "raster_sha256": sha(image),
                                "mask_sha256": sha(mask)})
    blender_sha = "b"*64
    candidate = write(packet / "SOURCE_REST_RENDER_CANDIDATE.json", {
        "subject_id": subject_id, "source_sha256": source_sha,
        "source_scene_snapshot": {"sha256": sha(snapshot)},
        "blender_binary_sha256": blender_sha, "resolution": policy["resolution"],
        "views": candidate_views,
    })
    admission = write(packet / "IRIS_OBSERVATION_ADMISSION.json", {
        "schema": "RealSaS.IrisBindObservationAdmission.v1",
        "status": "PASS_IRIS_BIND_OBSERVATIONS",
        "subject_id": subject_id, "source_sha256": source_sha,
        "candidate_sha256": sha(candidate), "camera_sha256": sha(camera),
        "imported_geometry_sha256": sha(imported),
        "views": admission_views,
    })
    files = {
        "admitted_import_geometry": imported,
        "observation_admission": admission,
        "source_rest_candidate": candidate,
        "source_scene_snapshot": snapshot,
        "camera_bundle": camera,
    }
    observation_record = {
        "subject_id": subject_id, "source_sha256": source_sha,
        "packet_path": str(packet), "packet_identity_sha256": "c"*64,
        "bindings": {name: {"path": str(path), "sha256": sha(path)}
                     for name, path in files.items()},
        "training_authorized": False,
    }
    roster = write(root / "roster.json", {
        "schema": "RealSaS.Corpus370ExactSourceRoster.v1",
        "subject_count": 370, "subjects": roster_rows,
        "training_authorized": False,
    })
    source_manifest = write(root / "source_manifest.json", {
        "schema": "RealSaS.ExpandedCorpusSourceInputBundle.v1",
        "subject_count": 370, "subjects": source_rows,
        "files": [], "training_authorized": False,
    })
    provenance = write(root / "provenance.json", {
        "schema": "RealSaS.Corpus370SourceProvenanceIndex.v1",
        "subject_count": 370, "subjects": provenance_rows,
        "training_authorized": False,
    })
    observations = write(root / "obs_v4.json", {
        "schema": "RealSaS.Corpus370PreparedObservationPackets.v4",
        "subject_count": 370, "prepared_subject_count": 1,
        "failure_count": 369, "subjects": [observation_record],
        "failures": failures,
        "binding": {
            "policy": {"path": str(policy_path), "sha256": sha(policy_path)},
            "blender_binary_sha256": blender_sha,
        },
        "training_authorized": False,
    })
    return roster, source_manifest, extracted, observations, provenance, packet


def pack(root):
    a, b, c, d, e, packet = fixture(root)
    result = prepare(a, b, c, d, e, root / "family-v3.zip")
    return result, packet


def test_complete_v3_family_transport_preserves_all_370_and_import_byte_identity(tmp_path):
    (manifest, receipt, archive), packet = pack(tmp_path)
    assert manifest["schema"] == "RealSaS.Corpus370PlatformFamilyInputBundle.v3"
    assert (manifest["subject_count"], manifest["observation_qualified_count"],
            manifest["observation_failed_count"]) == (370, 1, 369)
    subject = next(row for row in manifest["subjects"] if row["subject_id"] == "subject_000")
    admitted = subject["observation_preparation"]["admitted_import_geometry"]
    assert admitted["path"] == "observations/subject_000/imported.json"
    assert admitted["sha256"] == sha(packet / "imported.json")
    assert manifest["training_authorized"] is False
    assert manifest["rules"]["exact_admitted_import_bytes_required_for_each_observation_pass"] is True
    assert receipt["schema"] == "RealSaS.Corpus370PlatformFamilyInputBundleReceipt.v3"
    with zipfile.ZipFile(archive["path"]) as zipped:
        assert zipped.read(admitted["path"]) == (packet / "imported.json").read_bytes()
        assert zipped.read("observations/subject_003/PREPARATION_FAILURE.json")
        assert len(zipped.namelist()) == len(set(zipped.namelist()))


@pytest.mark.parametrize("corrupt", ["import-bytes", "raster-bytes", "index-sha", "policy-sha", "legacy-index"])
def test_v3_fails_closed_on_undocumented_admission_promotion(tmp_path, corrupt):
    args = fixture(tmp_path)
    roster, source_manifest, extracted, observations, provenance, packet = args
    if corrupt == "import-bytes":
        imported = packet / "imported.json"
        imported.write_bytes(imported.read_bytes() + b" ")
    elif corrupt == "raster-bytes":
        raster = packet / "V7.png"
        raster.write_bytes(raster.read_bytes() + b" ")
    else:
        data = json.loads(observations.read_text())
        if corrupt == "index-sha":
            data["subjects"][0]["bindings"]["admitted_import_geometry"]["sha256"] = "0"*64
        elif corrupt == "policy-sha":
            data["binding"]["policy"]["sha256"] = "0"*64
        else:
            data["schema"] = "RealSaS.Corpus370PreparedObservationPackets.v3"
        write(observations, data)
    with pytest.raises(ValueError, match="CORPUS370_PLATFORM_V3_"):
        prepare(roster, source_manifest, extracted, observations,
                provenance, tmp_path / "rejected.zip")
    assert not (tmp_path / "rejected.zip").exists()
