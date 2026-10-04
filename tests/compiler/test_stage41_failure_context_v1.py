import pytest

from compiler.realsas_compiler_services.proof.stage41_failure_context_v1 import (
    build_stage41_failure_attribution_context_v1,
)


BINDINGS = {
    "mechanical_state_binding_hash": "PRODUCT",
    "skeleton_binding_hash": "SK",
    "mesh_binding_hash": "MESH",
    "mesh_skin_binding_hash": "SKIN",
    "qualified_motion_binding_hash": "MOTION",
    "constraint_set_binding_hash": "CONSTRAINTS",
    "presentation_binding_hash": "PRESENTATION",
}


def _context(**overrides):
    args = {
        "measurements": {
            "clip_id": "run",
            "time_seconds": 0.25,
            "max_triangle_condition_number": 22.0,
        },
        "bindings": BINDINGS,
        "mesh_policy_hash": "POLICY",
        "camera_binding_hashes": tuple(f"CAM:{i}" for i in range(8)),
        "observation_set_hash": "OBS",
        "evaluator_semantic_version": "EVAL:V1",
    }
    args.update(overrides)
    return build_stage41_failure_attribution_context_v1(**args)


def test_stage41_attribution_context_is_deterministic_and_fail_closed():
    a = _context()
    b = _context()
    assert a == b
    assert a["proof_probe_fingerprint"]
    assert a["baseline_measurement_report_hash"]
    assert a["causal_owner_attribution"] == "NOT_PERFORMED"
    assert a["automatic_repair_eligible"] is False
    assert a["same_probe_counterfactual_required"] is True
    assert a["failure_signature_alone_is_not_owner_evidence"] is True


def test_stage41_probe_fingerprint_changes_with_any_mechanical_probe_authority():
    base = _context()["proof_probe_fingerprint"]

    changed_policy = _context(mesh_policy_hash="POLICY:2")
    assert changed_policy["proof_probe_fingerprint"] != base

    changed_motion = _context(
        bindings={**BINDINGS, "qualified_motion_binding_hash": "MOTION:2"}
    )
    assert changed_motion["proof_probe_fingerprint"] != base

    changed_camera = _context(
        camera_binding_hashes=tuple(
            "CAM:CHANGED" if i == 3 else f"CAM:{i}" for i in range(8)
        )
    )
    assert changed_camera["proof_probe_fingerprint"] != base

    changed_observation = _context(observation_set_hash="OBS:2")
    assert changed_observation["proof_probe_fingerprint"] != base


def test_measurement_identity_changes_without_changing_probe_identity():
    a = _context()
    b = _context(
        measurements={
            "clip_id": "run",
            "time_seconds": 0.25,
            "max_triangle_condition_number": 23.0,
        }
    )
    assert a["proof_probe_fingerprint"] == b["proof_probe_fingerprint"]
    assert (
        a["baseline_measurement_report_hash"]
        != b["baseline_measurement_report_hash"]
    )


def test_stage41_context_rejects_incomplete_bindings_or_camera_set():
    with pytest.raises(ValueError, match="BINDING_MISSING"):
        _context(bindings={k: v for k, v in BINDINGS.items() if k != "mesh_binding_hash"})
    with pytest.raises(ValueError, match="CAMERA_BINDING_INVALID"):
        _context(camera_binding_hashes=("CAM:0",))
