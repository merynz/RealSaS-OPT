from __future__ import annotations

import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.mainline import (
    status_text,
    validate_ledger,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_v2_plan_is_46_stage_subject_agnostic_contract():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    assert len(validate_plan(plan)) == 64
    assert plan["stage_count"] == 46
    assert plan["subject_specific_code_forbidden"] is True
    assert plan["stages"][15]["id"] == "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED"
    assert plan["stages"][17]["id"] == "18_CANONICAL_MESH_ADDRESSING_BUILD"
    assert plan["stages"][23]["id"] == "24_COMPLETE_APPEARANCE_QUALIFIED"
    assert plan["stages"][45]["id"] == "46_PRODUCT_CLOSURE_SEAL"
    assert plan["appearance_contract"]["first_knight_backend"] == "DETERMINISTIC_V1"


def test_v2_active_ledger_is_fresh_knight_lineage():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    ledger = load("canonical/ACTIVE_RUN_V2.json")
    validate_ledger(plan, ledger)
    text = status_text(plan, ledger)
    assert ledger["run_id"] == "SUBJECT2_KNIGHT_V2"
    assert "progress=0/46" in text
    assert "next=01_SOURCE_BYTES_SEALED" in text


def test_v2_mesh_birth_precedes_rig_and_caa():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    by = {row["id"]: row for row in plan["stages"]}
    assert by["18_CANONICAL_MESH_ADDRESSING_BUILD"]["depends_on"] == [
        "15_RIGGING_SURFACE_QUALIFIED",
        "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
        "17_MECHANICAL_PARTITION_QUALIFIED",
    ]
    assert "18_CANONICAL_MESH_ADDRESSING_BUILD" in by[
        "20_CAA_BACKEND_PREREGISTERED"
    ]["depends_on"]
    assert by["26_GEPPETTO_FIT_PREREGISTERED"]["depends_on"] == [
        "15_RIGGING_SURFACE_QUALIFIED"
    ]
    assert by["35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"]["adapter"].endswith(
        ":qualify_canonical_mesh_stage"
    )


def test_v2_repair_and_backend_policies_are_explicit():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    assert "NEW_STAGE18_LINEAGE" in plan["resume_contract"]["repair_semantics"]
    assert plan["appearance_contract"]["backend_axis"] == [
        "DETERMINISTIC_V1",
        "LEARNED_V2",
        "IM2SURFTEX_RESEARCH_ONLY",
    ]
    assert plan["appearance_contract"]["compiled_unobserved_exposure"] == (
        "FROZEN_BUDGET_GATE"
    )
