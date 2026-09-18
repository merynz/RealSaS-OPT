from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import experiments.playback_stack_v1.resolve_mage_full_assembly_runtime_v4_inputs_v1 as resolver


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def test_exact_resolver_requires_no_teacher_component_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    p1q = root / "03_P1Q_FIT2"
    mechanics = root / "mechanics"
    cameras = root / "cameras"
    observations = root / "observations"

    surface_bytes = b'{"surface":true}\n'
    skeleton_bytes = b'{"skeleton":true}\n'
    skin_bytes = b'{"skin":true}\n'
    monkeypatch.setattr(
        resolver,
        "EXPECTED",
        {
            "fit2_surface": {
                "filename": "CURRENT_FIT2_RIGGING_SURFACE_IR.json",
                "sha256": sha256(surface_bytes).hexdigest(),
            },
            "skeleton": {"filename": None, "sha256": sha256(skeleton_bytes).hexdigest()},
            "fit2_skin": {"filename": None, "sha256": sha256(skin_bytes).hexdigest()},
        },
    )
    monkeypatch.setattr(resolver, "EXPECTED_SURFACE_LINEAGE", "SURFACE")
    monkeypatch.setattr(resolver, "EXPECTED_SKELETON_LINEAGE", "SKELETON")
    monkeypatch.setattr(resolver, "EXPECTED_SKIN_LINEAGE", "SKIN")

    camera_payloads = tuple(f'{{"camera":{i}}}\n'.encode() for i in range(8))
    observation_payloads = tuple(b"PNG" + bytes([i]) for i in range(8))
    monkeypatch.setattr(
        resolver,
        "CAMERA_SHA256",
        tuple(sha256(payload).hexdigest() for payload in camera_payloads),
    )
    monkeypatch.setattr(
        resolver,
        "OBSERVATION_SHA256",
        tuple(sha256(payload).hexdigest() for payload in observation_payloads),
    )

    view_rows = []
    for i in range(8):
        files = {
            "mesh": f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_IR.json",
            "skin": f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_SKIN_IR.json",
            "appearance": f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_APPEARANCE_IR.json",
        }
        view_rows.append({"view": i, "files": files})
        for name in files.values():
            _write(p1q / name, b"{}")

    current_manifest = {
        "status": resolver.P1Q_STATUS,
        "teacher_truth_used": False,
        "source_component_truth_used": False,
        "current_gsa_lineage_hash": "SURFACE",
        "current_skeleton_lineage_hash": "SKELETON",
        "current_skin_lineage_hash": "SKIN",
        "views": view_rows,
    }
    manifest_path = p1q / resolver.P1Q_MANIFEST_NAME
    _write(
        manifest_path,
        (json.dumps(current_manifest, sort_keys=True) + "\n").encode(),
    )

    # A stale same-name manifest is present but must not compete because teacher truth
    # is explicitly forbidden by the current resolver policy.
    stale = dict(current_manifest)
    stale["teacher_truth_used"] = True
    _write(
        root / "stale" / resolver.P1Q_MANIFEST_NAME,
        (json.dumps(stale, sort_keys=True) + "\n").encode(),
    )

    surface = mechanics / resolver.EXPECTED["fit2_surface"]["filename"]
    skeleton = mechanics / "anonymous_g.json"
    skin = mechanics / "anonymous_skin.json"
    _write(surface, surface_bytes)
    _write(skeleton, skeleton_bytes)
    _write(skin, skin_bytes)

    camera_paths = []
    observation_paths = []
    for i, payload in enumerate(camera_payloads):
        path = cameras / f"camera_{i}.json"
        _write(path, payload)
        camera_paths.append(path)
    for i, payload in enumerate(observation_payloads):
        path = observations / f"view_{i}.png"
        _write(path, payload)
        observation_paths.append(path)

    # Giant product graph must still be skipped before anonymous JSON hashing.
    _write(root / "FIT2_CANONICAL_PUPPET_GRAPH_V3.json", b"x" * 1024)
    monkeypatch.setattr(resolver, "MAX_ANONYMOUS_JSON_BYTES", 128)

    result = resolver.resolve((root,))
    resolver.validate_resolution(result)

    assert result["status"] == "PASS__EXACT_MAGE_RUNTIME_V4_INPUTS_RESOLVED"
    assert Path(result["p1q_dir"]) == p1q.resolve()
    assert result["p1q_manifest_sha256"] == sha256(manifest_path.read_bytes()).hexdigest()
    assert Path(result["fit2_surface"]) == surface.resolve()
    assert Path(result["skeleton"]) == skeleton.resolve()
    assert Path(result["fit2_skin"]) == skin.resolve()
    assert tuple(map(Path, result["cameras"])) == tuple(path.resolve() for path in camera_paths)
    assert tuple(map(Path, result["observations"])) == tuple(path.resolve() for path in observation_paths)
    assert result["teacher_component_artifacts_resolved"] is False
    assert "foreground_dir" not in result
    assert "assembly_dir" not in result
    assert result["giant_product_graph_scanned"] is False
