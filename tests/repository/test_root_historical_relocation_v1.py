"""Actual root layout reduction, byte preservation and archive discoverability."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_research_roots_physically_archived_without_symlinks():
    manifest = json.loads(
        (ROOT / "historical/ROOT_COLLECTION_RELOCATION_V1.json").read_text()
    )
    assert manifest["schema"] == "RealSaS.RootHistoricalCollectionRelocation.v1"
    assert len(manifest["paths"]) == 17
    for row in manifest["paths"]:
        assert not (ROOT / row["old_path"]).exists(), row["old_path"]
        f = ROOT / row["new_path"]
        data = f.read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(data)).encode() + bytes([0]) + data
        ).hexdigest()
        assert actual == row["original_git_blob_sha1"], row["new_path"]
    for root in ("prereg", "prospective", "restoration", "research"):
        assert not (ROOT / root).exists(), root


def test_archived_prereg_stays_machine_discoverable():
    from tools.build_knowledge_artifact_catalog import is_high_signal as first
    from tools.audit_context_coverage import is_high_signal as second

    name = "historical/prereg/iris/G1_SINGLE_POSE_GEOMETRY_PREREG_V1.json"
    assert (ROOT / name).is_file()
    assert first(name)
    assert second(name)
