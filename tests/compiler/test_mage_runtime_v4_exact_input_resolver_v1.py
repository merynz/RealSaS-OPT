from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import experiments.playback_stack_v1.resolve_mage_full_assembly_runtime_v4_inputs_v1 as resolver


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def test_exact_hash_resolver_ignores_wrong_named_alias_and_giant_json(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    p1q = root / "03_P1Q_FIT2"
    fg = root / "05_FOREGROUND_ATLAS"
    assembly = root / "06_COMPONENT_ASSEMBLY_FIT2"
    mechanics = root / "mechanics"
    cameras = root / "cameras"

    # Tiny deterministic substitutes keep the resolver test about selection semantics.
    manifest_bytes = {
        "p1q_manifest": b'{"p1q":true}\n',
        "foreground_manifest": b'{"foreground":true}\n',
        "assembly_manifest": b'{"assembly":true}\n',
        "fit2_surface": b'{"surface":true}\n',
        "skeleton": b'{"skeleton":true}\n',
        "fit2_skin": b'{"skin":true}\n',
    }
    specs = {}
    filenames = {
        "p1q_manifest": "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json",
        "foreground_manifest": "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json",
        "assembly_manifest": "FIT2_COMPONENT_ASSEMBLY_MANIFEST.json",
        "fit2_surface": "CURRENT_FIT2_RIGGING_SURFACE_IR.json",
        "skeleton": None,
        "fit2_skin": None,
    }
    for label, payload in manifest_bytes.items():
        specs[label] = {"filename": filenames[label], "sha256": sha256(payload).hexdigest()}

    monkeypatch.setattr(resolver, "EXPECTED", specs)
    camera_payloads = tuple(f'{{"camera":{i}}}\n'.encode() for i in range(8))
    monkeypatch.setattr(
        resolver,
        "CAMERA_SHA256",
        tuple(sha256(payload).hexdigest() for payload in camera_payloads),
    )

    _write(p1q / filenames["p1q_manifest"], manifest_bytes["p1q_manifest"])
    _write(fg / filenames["foreground_manifest"], manifest_bytes["foreground_manifest"])
    _write(assembly / filenames["assembly_manifest"], manifest_bytes["assembly_manifest"])
    surface = mechanics / filenames["fit2_surface"]
    _write(surface, manifest_bytes["fit2_surface"])
    skeleton = mechanics / "anonymous_g22.json"
    skin = mechanics / "anonymous_fit2_skin.json"
    _write(skeleton, manifest_bytes["skeleton"])
    _write(skin, manifest_bytes["fit2_skin"])
    camera_paths = []
    for i, payload in enumerate(camera_payloads):
        path = cameras / f"unknown_{i}.json"
        _write(path, payload)
        camera_paths.append(path)

    # Wrong same-name alias must never win over byte authority.
    _write(root / "stale" / filenames["p1q_manifest"], b'{"stale":true}\n')

    # Giant graph must be skipped before hashing. Size ceiling is monkeypatched small.
    giant = root / "FIT2_CANONICAL_PUPPET_GRAPH_V3.json"
    _write(giant, b"x" * 1024)
    monkeypatch.setattr(resolver, "MAX_ANONYMOUS_JSON_BYTES", 128)

    # Directory completeness witnesses.
    for i in range(8):
        _write(p1q / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_IR.json", b"{}")
        _write(p1q / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_SKIN_IR.json", b"{}")
        _write(p1q / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_APPEARANCE_IR.json", b"{}")
        _write(fg / f"V{i}_MAGE_PRODUCT_ATLAS.png", b"png")
    _write(assembly / "FIT2_QUALIFIED_COMPONENT_ASSEMBLY.json", b"{}")

    result = resolver.resolve((root,))

    assert result["status"] == "PASS__EXACT_MAGE_RUNTIME_V4_INPUTS_RESOLVED"
    assert Path(result["p1q_dir"]) == p1q.resolve()
    assert Path(result["foreground_dir"]) == fg.resolve()
    assert Path(result["assembly_dir"]) == assembly.resolve()
    assert Path(result["fit2_surface"]) == surface.resolve()
    assert Path(result["skeleton"]) == skeleton.resolve()
    assert Path(result["fit2_skin"]) == skin.resolve()
    assert tuple(map(Path, result["cameras"])) == tuple(path.resolve() for path in camera_paths)
    assert result["giant_product_graph_scanned"] is False
