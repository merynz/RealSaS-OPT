from __future__ import annotations

import json
from pathlib import Path

from tools.iris_vnext.build_fit8_direct_surface_presentation_queries_v1 import parser as direct_query_parser


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "canonical" / "STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json"
ADMISSION = ROOT / "canonical" / "STRUCTURAL_EVIDENCE_INPUT_ADMISSION_V2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_admitted_v2_artifact_is_exact_and_non_authorizing() -> None:
    admission = _load(ADMISSION)
    assert admission["status"] == "ADMITTED_TIE_SAFE_HASHED_EVIDENCE_ONLY__NO_SCIENTIFIC_PASS"
    assert admission["drive_roundtrip_verified"] is True
    assert admission["drive"]["manifest"]["sha256"] == "9accf4d2ca85e90e23a3c7ed2a490329676582c5e6759b5d4a1df84df998bb1f"
    assert admission["drive"]["receipt"]["sha256"] == "8323945b0b4f8d54dec44458d1e8b5d40e51a8e38ace13bb04b0e4533099fc01"
    assert admission["payload_totals"] == {"subject_count": 8, "oriented_pair_rows": 3499296}
    assert admission["execution_facts"]["optimizer_steps"] == 0
    assert admission["execution_facts"]["backward_steps"] == 0
    assert admission["scientific_execution_authorized"] is False
    assert admission["training_authorized"] is False


def test_ab_arms_are_matched_capacity_and_inference_safe() -> None:
    prereg = _load(PREREG)
    assert prereg["status"] == "LOCKED_PREOPTIMIZER_SOURCE_CONTRACT__USER_HANDOFF_REQUIRED"
    assert prereg["implementation_authority"]["presentation_binding_merge_sha"] == "b0a2ae68a7ef8652f0e319495def0304c1faa03e"
    arch = prereg["architecture"]
    assert (arch["scene_memory_tokens"], arch["surface_memory_tokens"], arch["presentation_memory_tokens"]) == (384, 384, 256)
    assert arch["total_conditioning_tokens"] == 1024
    assert prereg["arm_a"]["matched_capacity_control"] is True
    assert prereg["arm_b"]["teacher_geometry_allowed"] is False
    assert prereg["arm_b"]["teacher_topology_allowed"] is False
    assert prereg["arm_b"]["named_semantic_taxonomy_allowed"] is False
    assert prereg["evidence_input_authority"]["teacher_all_hit_evidence_allowed_as_inference_conditioning"] is False


def test_training_protocol_and_court_are_numerically_locked() -> None:
    prereg = _load(PREREG)
    protocol = prereg["training_protocol"]
    assert protocol["seed_set"] == [26091991]
    assert protocol["seed_count"] == 1
    assert protocol["optimizer"]["name"] == "AdamW"
    assert protocol["optimizer"]["learning_rate"] == 1e-4
    assert protocol["optimizer_steps_per_arm"] == 2400
    assert protocol["warmup_steps"] == 120
    assert protocol["checkpoint_interval_steps"] == 200
    court = prereg["independent_court"]
    assert court["presentation_primary"]["minimum_macro_absolute_gain_for_b"] == 0.01
    assert court["motion_primary"]["minimum_gap_tear_rate_relative_reduction_for_b"] == 0.10
    assert prereg["decision_rule"]["post_result_threshold_relaxation_allowed"] is False


def test_execution_remains_fail_closed_until_merge_binding_and_user_handoff() -> None:
    prereg = _load(PREREG)
    authority = prereg["implementation_authority"]
    execution = prereg["execution_authorization"]
    assert authority["exact_execution_merge_sha"] is None
    assert execution["notebook_build_authorized"] is True
    assert execution["notebook_execution_authorized"] is False
    assert execution["optimizer_execution_authorized"] is False
    assert execution["requires_final_source_ci_green"] is True
    assert execution["requires_exact_execution_merge_sha_binding"] is True
    assert execution["requires_user_handoff_before_execution"] is True


def test_live_source_contract_still_encodes_true_a_and_b_paths() -> None:
    bridge = (ROOT / "models" / "iris" / "vnext" / "structure_v4.py").read_text(encoding="utf-8")
    presentation = (ROOT / "models" / "iris" / "vnext" / "presentation_observations_v1.py").read_text(encoding="utf-8")
    direct_queries = (ROOT / "tools" / "iris_vnext" / "build_fit8_direct_surface_presentation_queries_v1.py").read_text(encoding="utf-8")
    assert "presentation=None" in bridge or "presentation: RoleFreePresentationEvidenceV1 | None = None" in bridge
    assert "null_presentation_evidence_v1" in bridge
    assert "MODEL_PRODUCED_SURFACE_ONLY" in presentation
    assert "teacher_geometry_allowed" in presentation and "False" in presentation
    assert "teacher_topology_allowed" in presentation and "False" in presentation

    parser_dests = {action.dest for action in direct_query_parser()._actions}
    assert "teacher_dir" not in parser_dests
    assert "teacher_mesh" not in parser_dests
    assert "teacher_topology" not in parser_dests

    # Recording forbidden inputs as explicit false-valued provenance is required,
    # and must not be confused with accepting those inputs at the CLI boundary.
    assert '"teacher_mesh_input": False' in direct_queries
    assert '"teacher_topology_input": False' in direct_queries
