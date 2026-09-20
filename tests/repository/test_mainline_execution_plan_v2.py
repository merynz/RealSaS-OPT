from __future__ import annotations

import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.mainline import (
    _local_import_closure,
    status_text,
    validate_ledger,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_v2_plan_is_clean_46_stage_subject_agnostic_dag():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    assert len(validate_plan(plan)) == 64
    assert plan["stage_count"] == 46
    assert plan["subject_specific_code_forbidden"] is True
    assert plan["stages"][12]["id"] == "13_GEOMETRY_SUBSTRATE_QUALIFIED"
    assert plan["stages"][17]["id"] == "18_CANONICAL_MESH_ADDRESSING_BUILD"
    assert plan["stages"][23]["id"] == "24_COMPLETE_APPEARANCE_QUALIFIED"
    assert plan["stages"][44]["id"] == "45_DYNAMIC_VISUAL_INTEGRITY_PROOF"
    assert plan["stages"][45]["id"] == "46_PRODUCT_CLOSURE_SEAL"
    assert plan["appearance_contract"]["first_knight_backend"] == "DETERMINISTIC_V1"
    assert all(stage["adapter"] != "UNBOUND" for stage in plan["stages"])
    assert not any(
        ".adapters." in stage["adapter"] and "_v1:" in stage["adapter"]
        for stage in plan["stages"]
    )


def test_v2_active_ledger_is_implementation_assembly_not_premature_witness():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    ledger = load("canonical/ACTIVE_RUN_V2.json")
    validate_ledger(plan, ledger)
    text = status_text(plan, ledger)
    assert ledger["run_id"] == "V2_IMPLEMENTATION_ASSEMBLY"
    assert ledger["subject_id"] == "NONE"
    assert ledger["execution_enabled"] is False
    assert "progress=0/46" in text
    assert "01_SOURCE_BYTES_SEALED" in text
    assert "05_CAMERA_CONTRACT_SOLVED" in text


def test_v2_mesh_birth_precedes_parallel_mechanics_and_caa():
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


def test_v2_repair_appearance_and_visibility_policies_are_explicit():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    assert "NEW_STAGE18_LINEAGE" in plan["resume_contract"]["repair_semantics"]
    assert plan["appearance_contract"]["backend_axis"] == [
        "DETERMINISTIC_V1",
        "LEARNED_V2",
        "IM2SURFTEX_RESEARCH_ONLY",
    ]
    assert (
        plan["appearance_contract"]["compiled_unobserved_exposure"]
        == "FROZEN_BUDGET_GATE"
    )
    assert plan["product_quality_axes"] == {
        "geometry": "FIRST_CLASS",
        "mechanics": "FIRST_CLASS",
        "appearance": "FIRST_CLASS",
        "rule": "FAIL_ON_ANY_AXIS__NO_SILENT_CROSS_LAYER_COMPENSATION",
    }


def test_stage41_does_not_reintroduce_rest_unseen_appearance_failure():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    by = {row["id"]: row for row in plan["stages"]}
    stage41 = by["41_MOTION_DYNAMIC_PROOF"]
    assert "24_COMPLETE_APPEARANCE_QUALIFIED" not in stage41["depends_on"]
    stage45 = by["45_DYNAMIC_VISUAL_INTEGRITY_PROOF"]
    assert "24_COMPLETE_APPEARANCE_QUALIFIED" in stage45["depends_on"]


def test_current_v2_adapter_import_closure_excludes_obsolete_product_semantics():
    plan = load("canonical/MAINLINE_EXECUTION_PLAN_V2.json")
    modules = set()
    for stage in plan["stages"]:
        module_name = stage["adapter"].split(":", 1)[0]
        modules.update(name for name, _sha in _local_import_closure(module_name))
    forbidden = {
        "compiler.realsas_compiler_core.product_artifact_codec_v1",
        "compiler.realsas_compiler_core.product_appearance_v1",
        "compiler.realsas_compiler_core.product_composition_v1",
        "compiler.realsas_compiler_core.playback_runtime_v4",
        "compiler.realsas_compiler_core.runtime_projection_v1",
        "compiler.realsas_compiler_core.runtime_package_v1",
        "compiler.realsas_compiler_core.runtime_native_proof_v1",
        "compiler.realsas_compiler_core.motion_presentation_v1",
        "compiler.realsas_compiler_services.orchestrator.adapters.runtime_projection_v1",
        "compiler.realsas_compiler_services.orchestrator.adapters.runtime_package_v1",
        "compiler.realsas_compiler_services.orchestrator.adapters.runtime_native_v1",
        "compiler.realsas_compiler_services.orchestrator.adapters.product_closure_v1",
        "compiler.realsas_compiler_services.orchestrator.adapters.presentation_v1",
    }
    assert modules.isdisjoint(forbidden), sorted(modules & forbidden)
