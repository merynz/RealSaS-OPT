"""Cross-file links at the real render and CI entry points must stay executable."""
import ast
import json
from pathlib import Path
import re

import pytest

from tools.ops.knight_render_inputs_v1 import stage_payload, stage_row

ROOT = Path(__file__).resolve().parents[2]


def test_sealed_render_stage_references_exist_in_the_canonical_dag():
    plan = json.loads((ROOT / "canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())
    stages = {row["id"] for row in plan["stages"]}
    tree = ast.parse((ROOT / "tools/ops/render_knight_sealed_v6.py").read_text())
    references = {call.args[0].value for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "stage" and call.args and isinstance(call.args[0], ast.Constant)}
    assert references and not references - stages, sorted(references - stages)


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
