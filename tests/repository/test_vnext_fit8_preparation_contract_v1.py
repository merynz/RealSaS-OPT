from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBJECT_IDS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_vnext_fit8_exact_input_binding_is_complete_and_fresh_optimizer_only():
    value = json.loads(_text("canonical/VNEXT_FIT8_INPUT_BINDING_V1.json"))
    assert value["schema"] == "RealSaS.VNextFit8InputBinding.v1"
    assert value["legacy_optimizer_state_reused"] is False
    assert tuple(row["subject_id"] for row in value["subjects"]) == SUBJECT_IDS
    assert len({row["legacy_subject_id"] for row in value["subjects"]}) == 8
    assert len(value["stage01_bundle"]["sha256"]) == 64
    for row in value["subjects"]:
        for key in (
            "legacy_shard_sha256",
            "raw_source_sha256",
            "observation_admission_sha256",
            "camera_bundle_sha256",
            "source_rest_candidate_sha256",
        ):
            assert len(row[key]) == 64


def test_teacher_and_cohort_materialization_are_behind_fail_closed_binding():
    common = _text("tools/iris_vnext/input_binding_v1.py")
    teacher = _text("tools/iris_vnext/export_fit8_authored_teachers_v1.py")
    prepare = _text("tools/iris_vnext/prepare_fit8_vnext_cohort_v1.py")
    builder = _text("tools/iris_vnext/build_fit8_vnext_shards_v2.py")
    topology = _text("models/iris/vnext/topology_teacher_v1.py")
    assert "VNEXT_STAGE01_BUNDLE" in common
    assert "VNEXT_LEGACY_SHARD" in common
    assert "build_input_receipt_v1" in teacher
    assert "build_input_receipt_v1" in prepare
    assert "TEACHER_BATCH_RECEIPT.json" in prepare
    assert "PREPARATION_RECEIPT.json" in prepare
    assert "build_fit8_vnext_shards_v2.py" in prepare
    assert "canonicalize_vnext_topology_teacher_v1" in builder
    assert "VNEXT_TESSA_UNRESOLVED_POSITION_COLLISION" in builder
    assert "MOTION_EQUIVALENT_EXACT_SEAM_WELD_THEN_BOUNDED_TRAINING_ONLY_QUANTIZED_IDENTITY_LIFT_THEN_ZERO_RESIDUAL_COLLISIONS_REQUIRED" in builder
    assert "quantized_identity_lift" in topology
    assert "product_geometry_authority" in topology
    assert '"legacy_optimizer_state_reused": False' in builder
    assert '"legacy_5400_checkpoint_reused": False' in builder


def test_windowed_v2_is_the_single_active_fit8_trainer_contract():
    v2 = _text("models/iris/vnext/cohort_v2.py")
    test = _text("tests/iris/test_iris_vnext_fit8_cohort_v1.py")
    assert "V2 is the sole vNext Fit8 training authority" in v2
    assert "IRISVNextFit8TrainerV2" in test
    assert "topology_window_offsets" in v2
    assert "topology_window_count" in v2
    assert "sequence_position_offset=topology_offsets.long()" in v2
    assert "ONE_VALID_WINDOW_PER_SUBJECT__BATCH8_DECODER_FORWARD" in v2


def test_a100_preflight_cannot_claim_or_seed_scientific_fit():
    source = _text("tools/iris_vnext/a100_preflight_v1.py")
    scientific = _text("tools/iris_vnext/run_fit8_vnext_scientific_v1.py")
    assert "VNEXT_A100_DEVICE_REQUIRED" in source
    assert '"preflight_checkpoint_reusable": False' in source
    assert '"scientific_fit_started": False' in source
    assert '"scientific_fit_pass_claimed": False' in source
    assert "TemporaryDirectory" in source
    assert "ANALYTIC_ENGINEERING_COURT_ONLY" in source
    assert "RealSaS.IRISVNextFit8Manifest.v4" in source
    assert "RealSaS.IRISVNextFit8PreparationReceipt.v2" in source
    assert "VNEXT_FIT8_PREFLIGHT_TOPOLOGY_GRAMMAR_DRIFT" in scientific
    assert "scientific_fit_pass_claimed" in scientific
