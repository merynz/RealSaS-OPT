"""Cross-file links at the real render and CI entry points must stay executable."""
import ast
import json
from pathlib import Path
import re

import pytest

from tools.ops.knight_render_inputs_v1 import stage_payload, stage_row

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_render_entry_uses_the_sealed_single_m_consumer():
    entry = "tools/demo/render_knight_m_caa_v1.py"
    workflow = (ROOT / ".github/workflows/single_m_caa_canonical_gate.yml").read_text()
    assert f"python {entry} --output" in workflow
    tree = ast.parse((ROOT / entry).read_text())
    imports = {name.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "compiler.realsas_compiler_core.single_m_caa_v1"
        for name in node.names}
    assert {"render_single_m_caa", "load_material_bundle"} <= imports
    fixture = ROOT / "tests/fixtures/single_m_caa_knight_v1"
    baseline = json.loads((fixture / "baseline.json").read_text())
    assert baseline["inputs"]
    for filename in baseline["inputs"]:
        assert (fixture / filename).is_file(), filename


def test_visual_policy_workflow_test_selectors_exist():
    workflow = (ROOT / ".github/workflows/vf23_production_policy_e2e_bank.yml").read_text()
    selectors = re.findall(r"(tests/[\w/]+\.py)::([\w]+)", workflow)
    assert len(selectors) >= 2
    for filename, name in selectors:
        tree = ast.parse((ROOT / filename).read_text())
        names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert name in names, f"Non-executable CI test selector: {filename}::{name}"


def test_missing_and_ambiguous_stage_inputs_have_explicit_errors():
    with pytest.raises(RuntimeError, match="STAGE_ID_CARDINALITY::S::0"):
        stage_row({"stages": []}, "S")
    with pytest.raises(RuntimeError, match="STAGE_ID_CARDINALITY::S::2"):
        stage_row({"stages": [{"id": "S"}, {"id": "S"}]}, "S")
    with pytest.raises(RuntimeError, match="STAGE_SCHEMA_CARDINALITY::S::Required::0"):
        stage_payload({"stages": [{"id": "S", "outputs": []}]}, "S", "Required")
