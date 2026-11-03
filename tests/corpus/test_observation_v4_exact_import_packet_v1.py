"""V4 replay checks *every* byte-bound view, not only receipt headings."""
import json
from pathlib import Path

import pytest

from tools.corpus.prepare_corpus370_observation_packets_v4 import (
    _packet_complete, sha256,
)


def packet_fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    sid = "subject"; source = "a"*64; blender = "b"*64
    imported = root / "imported.json"
    imported.write_text(json.dumps({"import": {"source_sha256": source},
                                    "vertices": [[0,0,0]], "faces": [], "bone_count": 0}))
    snapshot = root / "SOURCE_SCENE_SNAPSHOT.json"
    snapshot.write_text(json.dumps({"source_sha256": source}))
    camera = root / "camera_bundle.json"
    camera.write_text(json.dumps({
        "schema": "RealSaS.CameraProjectionBundle.v1",
        "cameras": [{"view_index": i} for i in range(8)]}))
    views, admitted = [], []
    for i in range(8):
        raster = root / f"V{i}.png"
        mask = root / f"V{i}.mask.bin"
        raster.write_bytes(b"exact-pixels" + bytes([i]))
        mask.write_bytes(b"exact-mask" + bytes([i]))
        views.append({"view_index": i,
                      "raster": {"sha256": sha256(raster)},
                      "foreground_mask": {"sha256": sha256(mask)}})
        admitted.append({"view_index": i,
                         "raster_sha256": sha256(raster),
                         "mask_sha256": sha256(mask)})
    candidate_path = root / "SOURCE_REST_RENDER_CANDIDATE.json"
    candidate_path.write_text(json.dumps({
        "subject_id": sid, "source_sha256": source,
        "blender_binary_sha256": blender, "resolution": 64,
        "source_scene_snapshot": {"sha256": sha256(snapshot)},
        "views": views}))
    admission = {
        "schema": "RealSaS.IrisBindObservationAdmission.v1",
        "status": "PASS_IRIS_BIND_OBSERVATIONS",
        "subject_id": sid, "source_sha256": source,
        "candidate_sha256": sha256(candidate_path),
        "camera_sha256": sha256(camera),
        "imported_geometry_sha256": sha256(imported),
        "views": admitted,
    }
    (root / "IRIS_OBSERVATION_ADMISSION.json").write_text(json.dumps(admission))
    return sid, source, blender, {"view_count": 8, "resolution": 64}


def test_v4_packet_requires_exact_original_admitted_import_bytes(tmp_path):
    sid, source, blender, policy = packet_fixture(tmp_path)
    assert _packet_complete(tmp_path, sid, source, policy, blender)
    p = tmp_path / "imported.json"
    p.write_text(p.read_text() + " ")
    assert not _packet_complete(tmp_path, sid, source, policy, blender)


def test_old_v2_packet_without_original_import_is_not_reused(tmp_path):
    sid, source, blender, policy = packet_fixture(tmp_path)
    (tmp_path / "imported.json").unlink()
    assert not _packet_complete(tmp_path, sid, source, policy, blender)


@pytest.mark.parametrize("name", [
    "V0.png", "V7.png", "V3.mask.bin",
    "camera_bundle.json", "SOURCE_SCENE_SNAPSHOT.json",
    "SOURCE_REST_RENDER_CANDIDATE.json",
])
def test_corrupted_hash_bound_parent_is_not_reused(tmp_path, name):
    sid, source, blender, policy = packet_fixture(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b" ")
    assert not _packet_complete(tmp_path, sid, source, policy, blender)


@pytest.mark.parametrize("changed", ["candidate-sha", "camera-sha", "mask-views", "view-order"])
def test_resigned_headings_cannot_skip_parent_checks(tmp_path, changed):
    sid, source, blender, policy = packet_fixture(tmp_path)
    p = tmp_path / "IRIS_OBSERVATION_ADMISSION.json"
    admission = json.loads(p.read_text())
    if changed == "candidate-sha":
        admission["candidate_sha256"] = "0"*64
    elif changed == "camera-sha":
        admission["camera_sha256"] = "0"*64
    elif changed == "mask-views":
        admission["views"][3]["mask_sha256"] = "0"*64
    else:
        admission["views"][0]["view_index"] = 3
    p.write_text(json.dumps(admission))
    assert not _packet_complete(tmp_path, sid, source, policy, blender)
