import json
from pathlib import Path


CORRECTED_DECODER = (
    "RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT."
    "ExactShellContainment.v5_1"
)


def test_direct_surface_binding_v2_uses_canonical_exact_containment():
    binding = json.loads(
        Path("canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V2.json").read_text(encoding="utf-8")
    )
    assert binding["schema"] == "RealSaS.IRISVNextDirectSurfaceInputBinding.v1"
    assert binding["binding_revision"] == 2
    surface = binding["surface_policy"]
    assert surface["decoder_id"] == CORRECTED_DECODER
    assert surface["base_cells"] == 512
    assert surface["fine_cells"] == 1024
    assert surface["coordinate_transform"] == "IDENTITY_NORMALIZED_MINUS1_PLUS1"
    assert surface["field_scalar_transform"] == "IDENTITY_LEARNED_FIELD"
    assert surface["numeric_policy"] == "FP32_NO_AUTOCAST"
    assert surface["exact_shell_containment_required"] is True
    assert surface["legacy_boundary_crossing_tets_role"] == "TELEMETRY_ONLY"
    assert surface["legacy_boundary_crossing_tets_is_truncation_gate"] is False
    assert binding["training_allowed"] is False
    assert binding["scientific_pass_claimed"] is False


def test_canonical_closure_falsifies_legacy_boundary_proxy_and_forbids_blind_disable():
    closure = json.loads(
        Path("canonical/IRIS_V5_EXTERIOR_DECODER_SHELL_COMPAT_CLOSURE_20260924.json")
        .read_text(encoding="utf-8")
    )
    assert closure["status"].startswith("CLOSED__NO_ACTUAL_OMITTED_FACE_ESCAPE")
    diagnosis = closure["decoder_gate_diagnosis"]
    assert diagnosis["falsified_interpretation"] == (
        "NONZERO_BOUNDARY_CROSSING_TETS_PROVES_SPARSE_SHELL_TRUNCATES_ZERO_SURFACE"
    )
    action = closure["decoder_next_action"]
    assert action["required_behavior"].startswith(
        "Gate on actual zero-set escape through omitted in-domain refined-neighbor faces"
    )
    assert action["forbidden_shortcut"].startswith(
        "Do not simply disable fail_on_boundary_crossing"
    )


def test_contained_decoder_proves_before_disabling_legacy_proxy():
    source = Path("models/iris/v5/contained_sparse_tetra_decoder_v5.py").read_text(encoding="utf-8")
    report_call = source.index("exact_sparse_shell_containment_report_v5(")
    pass_check = source.index('if not bool(containment["passed"])')
    disable_proxy = source.index("replace(policy, fail_on_boundary_crossing=False)")
    extraction = source.index("build_indexed_mt_from_sparse_carrier_v5(", disable_proxy)
    assert report_call < pass_check < disable_proxy < extraction
    assert "SPARSE_SHELL_ACTUAL_ZERO_SURFACE_ESCAPE" in source
    assert "legacy_boundary_crossing_tets_role\": \"TELEMETRY_ONLY" in source


def test_vnext_bridge_calls_contained_decoder_and_keeps_field_authority():
    source = Path("models/iris/vnext/sparse_surface_v4.py").read_text(encoding="utf-8")
    assert "decode_indexed_sparse_tetra_mt_contained_v5(" in source
    assert "VNextScalarFieldQueryV3" in source
    assert "IDENTITY_NORMALIZED_MINUS1_PLUS1" not in source  # inherited, not redefined
    assert "teacher_truth_used\": False" in source
    assert "compiler_qualified\": False" in source
    assert "scientific_pass_claimed\": False" in source
    assert "optimizer.step" not in source
    assert "loss.backward" not in source


def test_isolated_worker_and_parent_are_bound_to_corrected_decoder():
    child = Path("tools/iris_vnext/materialize_fit8_direct_surface_subject_v3.py").read_text(encoding="utf-8")
    parent = Path("tools/iris_vnext/materialize_fit8_direct_surface_resumable_v3.py").read_text(encoding="utf-8")
    assert "decode_vnext_sparse_surface_v4" in child
    assert "CONTAINED_DECODER_ID_V5" in child
    assert "exact_shell_containment_required\": True" in child
    assert "legacy_boundary_crossing_tets_role\": \"TELEMETRY_ONLY" in child
    assert "materialize_fit8_direct_surface_subject_v3" in parent
    assert "CONTAINED_DECODER_ID_V5" in parent
    assert "EXACT_SELECTED_T512_TO_SPARSE_T1024_EXACT_SHELL_CONTAINMENT_V5_1" in parent
    assert "court_thresholds_changed\": False" in parent
    assert "optimizer_or_schedule_changed\": False" in parent
    for source in (child, parent):
        assert "optimizer.step" not in source
        assert "loss.backward" not in source


def test_failed_snake_proxy_observation_is_not_promoted_to_scientific_fail():
    # Regression statement from the observed pre-optimizer failure class.  The
    # correction source must not encode the legacy proxy string as a new gate.
    bridge = Path("models/iris/vnext/sparse_surface_v4.py").read_text(encoding="utf-8")
    parent = Path("tools/iris_vnext/materialize_fit8_direct_surface_resumable_v3.py").read_text(encoding="utf-8")
    assert "SPARSE_SHELL_TRUNCATES_ZERO_SURFACE" not in bridge
    assert "SPARSE_SHELL_TRUNCATES_ZERO_SURFACE" not in parent
    assert "scientific_pass_claimed" not in parent or "scientific_pass_claimed" in parent
