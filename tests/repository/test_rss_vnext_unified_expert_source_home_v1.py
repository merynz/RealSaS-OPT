"""Ensure one physical RSS.vNext expert home preserves stable import identities."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_all_four_experts_stored_beneath_unified_model_without_duplicates():
    manifest = json.loads((ROOT / "models/rss_vnext/EXPERT_SOURCE_RELOCATION_V1.json").read_text())
    assert manifest["schema"] == "RealSaS.RSSvNextUnifiedExpertStorage.v1"
    assert len(manifest["paths"]) == manifest["count"] == 178
    for row in manifest["paths"]:
        original, target = ROOT / row["old_path"], ROOT / row["new_path"]
        assert original.is_file(), row["old_path"]
        assert original.resolve() == target.resolve(), row["old_path"]
        assert (ROOT / "models" / Path(row["old_path"]).parts[1]).is_symlink()
        raw = target.read_bytes()
        h = hashlib.sha1(b"blob " + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()
        assert h == row["git_blob_sha1"], row["new_path"]
        assert row["old_path"].startswith("models/")
        assert row["new_path"] == row["old_path"].replace("models/", "models/rss_vnext/experts/", 1)
    for expert in ("iris", "tessa", "axis", "mira"):
        spec = importlib.util.find_spec("models." + expert)
        assert spec is not None and spec.origin
        assert Path(spec.origin).resolve() == (ROOT / ("models/rss_vnext/experts/" + expert + "/__init__.py")).resolve()
    assert (ROOT / "models/rss_vnext/unified_character_model.py").is_file()

def test_implementation_closure_hashes_actual_unified_expert_sources():
    from compiler.realsas_compiler_services.orchestrator.mainline import _local_module_path
    for expert,leaf in (
        ("iris","vnext/__init__.py"),
        ("tessa","v3/__init__.py"),
        ("axis","rig_program_model_v6.py"),
        ("mira","skin_fsq_codec_v6.py"),
    ):
        path = "models." + expert + "." + leaf.removesuffix(".py").replace("/", ".")
        if path.endswith(".__init__"):path=path[:-len(".__init__")]
        expected=ROOT / "models/rss_vnext/experts" / expert / leaf
        assert _local_module_path(path) == expected.resolve(), path
