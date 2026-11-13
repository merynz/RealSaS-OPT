"""Historical model source relocation preserves bytes and public import IDs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_only_one_model_release_root_and_legacy_imports_still_resolve():
    assert (ROOT / "models/rss_vnext").is_dir()
    for legacy in ("arachne", "geppetto", "skin_field_codec"):
        assert not (ROOT / "models" / legacy).exists()
        assert (ROOT / "historical/model_sources" / legacy).is_dir()
        spec = importlib.util.find_spec("models." + legacy)
        assert spec is not None and spec.origin is not None
        assert "historical/model_sources/" + legacy in Path(spec.origin).as_posix()


def test_historical_model_files_preserve_original_git_blob_sha1():
    manifest = json.loads(
        (ROOT / "historical/model_sources/RELOCATION_MANIFEST_V1.json").read_text()
    )
    assert manifest["schema"] == "RealSaS.HistoricalModelSourceRelocation.v1"
    assert len(manifest["paths"]) == 55
    for row in manifest["paths"]:
        source = ROOT / row["new_path"]
        data = source.read_bytes()
        digest = hashlib.sha1(
            f"blob {len(data)}".encode("ascii") + bytes([0]) + data
        ).hexdigest()
        assert digest == row["git_blob_sha1"], row["new_path"]
