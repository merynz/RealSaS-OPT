import json
from pathlib import Path

from tools.corpus.prepare_corpus370_observation_packets_v3 import (
    _packet_complete, sha256,
)


def packet_fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    sid = "subject"; source = "a"*64; blender = "b"*64
    imported = root / "imported.json"
    imported.write_text(json.dumps({"import": {"source_sha256": source},
                                    "vertices": [[0,0,0]], "faces": [], "bone_count": 0}))
    admission = {"schema": "RealSaS.IrisBindObservationAdmission.v1",
                 "status": "PASS_IRIS_BIND_OBSERVATIONS",
                 "subject_id": sid, "source_sha256": source,
                 "imported_geometry_sha256": sha256(imported)}
    candidate = {"subject_id": sid, "source_sha256": source,
                 "blender_binary_sha256": blender, "resolution": 64}
    (root / "IRIS_OBSERVATION_ADMISSION.json").write_text(json.dumps(admission))
    (root / "SOURCE_REST_RENDER_CANDIDATE.json").write_text(json.dumps(candidate))
    (root / "SOURCE_SCENE_SNAPSHOT.json").write_text("{}")
    (root / "camera_bundle.json").write_text("{}")
    for i in range(8):
        (root / f"V{i}.png").write_bytes(b"src-pixel")
        (root / f"V{i}.mask.bin").write_bytes(b"src-mask")
    return sid, source, blender, {"view_count": 8, "resolution": 64}


def test_v3_packet_requires_original_admitted_import_bytes(tmp_path):
    sid, sha, blender, policy = packet_fixture(tmp_path)
    assert _packet_complete(tmp_path, sid, sha, policy, blender)
    original = tmp_path / "imported.json"
    original.write_text(original.read_text() + " ")
    assert not _packet_complete(tmp_path, sid, sha, policy, blender)


def test_old_v2_packet_without_original_import_is_not_reused(tmp_path):
    sid, sha, blender, policy = packet_fixture(tmp_path)
    (tmp_path / "imported.json").unlink()
    assert not _packet_complete(tmp_path, sid, sha, policy, blender)
