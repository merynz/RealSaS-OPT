"""Synthetic contract tests only; NOT real original-source qualification."""
import copy
import pytest
from tools.corpus.admit_corpus370_original_cat_v3_graphics_readiness_v0 import (
    build_from_recomputed_court_v0, INDEPENDENT_COURT_SHA, RENDER_SHA,
    RENDER_MANIFEST_SHA,
)


def observed_fixture():
    return {
        "schema": "RealSaS.Corpus370IndependentOriginalCatPoseUVTransportCourt.v3",
        "status": "PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__FIXED_SCREEN_CLASS_VIOLATED",
        "source_original_blender_physical_sample_sha256": RENDER_SHA,
        "source_original_blender_physical_render_manifest_sha256": RENDER_MANIFEST_SHA,
        "actual_source_original_full_physical_renders": 64,
        "source_original_pose_scope": "AUTHORED_TEACHER_ORDINAL_0_AND_14_ONLY",
        "source_original_edited_atlas_operation_scope": "THREE_FIXED_ANONYMOUS_UV_ADDRESSES",
        "source_original_shader_domain": "ADMITTED_UNLIT_LINKED_TEXTURE_ONLY",
        "source_uv_surface_attached_edit_native_source_status":
            "SATISFIED_PHYSICAL_ORIGINAL_SOURCE_WITNESS_ONLY",
        "source_missing_evidence_obligations": [],
        "source_original_pose_world_max_displacement": 0.2230362676971853,
        "source_edit_observed_pose0_by_operation": [59303, 57005, 1207],
        "source_edit_observed_pose14_by_operation": [57800, 57952, 1207],
        "measured_original_physical_source": {
            "fixed_screen_pose14_total_mismatched_sample_incidents": 40878,
        },
        "original_drawable_artist_partition": "UNKNOWN_NOT_MEASURED",
        "graphics_owner_authorized": False,
        "iris_training_authorized": False,
        "tessa_training_authorized": False,
        "materialization_authorized": False,
        "training_authorized": False,
    }


def test_v3_source_observed_pose_only_is_not_full_puppet_capability():
    r = build_from_recomputed_court_v0(
        observed_fixture(), court_sha=INDEPENDENT_COURT_SHA)
    source = r["graphics_readiness"]["per_operation_source_observability"]
    assert source["POSE_AWARE_SOURCE_UV"] == "OBSERVED_IN_ADMITTED_SOURCE_SCOPE"
    for name in ("INDEPENDENT_ATLAS_EDITS", "LOCAL_DEPTH_SOURCE_OVER",
                 "HIDDEN_EDIT_STATE_PERSISTENCE"):
        assert source[name] == "UNKNOWN"
        assert r["graphics_readiness"]["missing_obligations"][name]
    assert r["compiled_puppet_physical_parity"] == "UNKNOWN"
    assert r["model_inference_input_admitted"] is False
    assert r["student_training_authorized"] is False


@pytest.mark.parametrize(("key", "value"), (
    ("source_original_blender_physical_sample_sha256", "0" * 64),
    ("source_original_blender_physical_render_manifest_sha256", "0" * 64),
    ("source_original_pose_scope", "ANY_POSE"),
    ("source_uv_surface_attached_edit_native_source_status", "UNKNOWN"),
    ("source_original_pose_world_max_displacement", 0),
    ("source_edit_observed_pose0_by_operation", [0, 0, 0]),
    ("source_edit_observed_pose14_by_operation", [0, 0, 0]),
    ("actual_source_original_full_physical_renders", 0),
    ("graphics_owner_authorized", True),
    ("original_drawable_artist_partition", "SOURCE_ARTIST_LAYER"),
))
def test_source_replay_provenance_or_consequence_tamper_rejected(key, value):
    r = observed_fixture()
    r[key] = value
    with pytest.raises(ValueError, match="CAT_GRAPHICS_READINESS_V0_"):
        build_from_recomputed_court_v0(r, court_sha=INDEPENDENT_COURT_SHA)


def test_court_hash_forgery_rejected():
    with pytest.raises(ValueError, match="COURT_HASH"):
        build_from_recomputed_court_v0(observed_fixture(), court_sha="0" * 64)
