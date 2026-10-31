from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.post139_source_inventory_v1 import inventory, representation_key


def test_representation_key_collapses_format_directories_without_collapsing_lanes():
    assert representation_key("Pack/FBX/Actors/Hero.fbx") == "pack/actors/hero"
    assert representation_key("Pack/glTF/Actors/Hero.gltf") == "pack/actors/hero"
    assert representation_key("Pack/FBX/Props/Hero.fbx") == "pack/props/hero"


def ns(root: Path, out: Path):
    return argparse.Namespace(source_root=str(root), out=str(out), source_id="source", registry=None)


def test_inventory_keeps_every_raw_file_and_selects_one_audit_representation(tmp_path: Path):
    root = tmp_path / "raw"
    (root / "Pack" / "FBX" / "Actors").mkdir(parents=True)
    (root / "Pack" / "glTF" / "Actors").mkdir(parents=True)
    (root / "Pack" / "OBJ" / "Actors").mkdir(parents=True)
    (root / "Pack" / "Textures").mkdir(parents=True)
    files = {
        "Pack/FBX/Actors/Hero.fbx": b"fbx",
        "Pack/glTF/Actors/Hero.gltf": b"{}",
        "Pack/OBJ/Actors/Hero.obj": b"obj",
        "Pack/Textures/Hero.png": b"png",
    }
    for relative, content in files.items():
        (root / relative).write_bytes(content)
    out = tmp_path / "out"
    inventory(ns(root, out))
    ledger = json.loads((out / "RAW_FILE_LEDGER.json").read_text())
    source_inventory = json.loads((out / "SOURCE_INVENTORY.json").read_text())
    manifest = json.loads((out / "CANDIDATE_MANIFEST.json").read_text())
    assert ledger["file_count"] == 4
    assert {row["path"] for row in ledger["files"]} == set(files)
    assert source_inventory["representation_group_count"] == 1
    assert source_inventory["audit_candidate_count"] == 1
    assert manifest["candidates"][0]["relative_path"] == "Pack/glTF/Actors/Hero.gltf"
    assert manifest["candidates"][0]["format_pair_count"] == 3


def test_ledger_hash_changes_with_any_raw_byte(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    source = root / "Actor.fbx"
    source.write_bytes(b"a")
    first = tmp_path / "first"
    inventory(ns(root, first))
    first_payload = json.loads((first / "RAW_FILE_LEDGER.json").read_text())
    source.write_bytes(b"b")
    second = tmp_path / "second"
    inventory(ns(root, second))
    second_payload = json.loads((second / "RAW_FILE_LEDGER.json").read_text())
    assert first_payload["ledger_sha256"] != second_payload["ledger_sha256"]
    assert first_payload["files"][0]["sha256"] == hashlib.sha256(b"a").hexdigest()
    assert second_payload["files"][0]["sha256"] == hashlib.sha256(b"b").hexdigest()
