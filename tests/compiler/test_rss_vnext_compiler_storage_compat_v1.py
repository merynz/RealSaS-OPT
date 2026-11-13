"""RSS.vNext compiler specialized sources share one physical home."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_rss_vnext_compiler_source_layout_is_single_and_importable():
    obj = json.loads((ROOT / "compiler/rss_vnext/RELOCATION_MANIFEST_V1.json").read_text())
    assert obj["schema"] == "RealSaS.RSSvNextCompilerSourceRelocation.v1"
    assert len(obj["paths"]) == 20
    for row in obj["paths"]:
        assert not (ROOT / row["old_path"]).exists()
        dest = ROOT / row["new_path"]
        data = dest.read_bytes()
        assert hashlib.sha1(
            f"blob {len(data)}".encode("ascii") + bytes([0]) + data
        ).hexdigest() == row["git_blob_sha1"]
        module = row["old_path"][:-3].replace("/", ".")
        spec = importlib.util.find_spec(module)
        assert spec is not None, module
        assert Path(spec.origin).resolve() == dest.resolve(), module
