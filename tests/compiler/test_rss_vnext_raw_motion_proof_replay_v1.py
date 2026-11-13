from __future__ import annotations

"""Full raw two-image product flow, through deliberate Proof refusal.

These are synthetic neural proposals on real immutable CAS bytes. A passing
preflight replay does NOT mint a product, source RGB authority or visibility.
"""

import json

import pytest

from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_graphics_capability_v1,
    execute_motion_capability_v2,
    execute_proof_capability_v1,
)
from tests.compiler.test_rss_vnext_raw_graphics_consumer_v1 import (
    _full_parent_fixture,
)


def _complete_product_preflight(tmp_path, *, interventions=True):
    cas, graphics_req, mechanics, skin = _full_parent_fixture(
        tmp_path, active_mechanics_courts=interventions,
    )
    graphics = execute_graphics_capability_v1(
        graphics_req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert graphics["status"] == "PASS"
    parents = list(graphics_req["input_artifacts"]) + graphics["outputs"]
    motion = execute_motion_capability_v2({
        "kind": "MODULE",
        "capability_id": "product.motion",
        "capability_metadata": {
            "executor_contract": "RealSaS.ProductMotionCompile.v2",
            "required_models": [],
            "default_rest_hold_policy": "DETERMINISTIC_ZERO_TRACK_REST_HOLD_V1",
        },
        "model_releases": [],
        "input_artifacts": parents,
    }, read_object=cas.read_object, put_bytes=cas.put_bytes)
    assert motion["status"] == "PASS"
    assert motion["diagnostics"]["motion_source_kind"] == (
        "COMPILER_POLICY_REST_HOLD_NOT_ANIMATED_IDLE"
    )
    assert not motion["diagnostics"]["product_revision_minted"]
    return cas, {
        "kind": "MODULE",
        "capability_id": "product.proof",
        "capability_metadata": {
            "executor_contract": "RealSaS.ProductProofCompile.v1",
            "required_models": [],
            "mechanics_intervention_required": interventions,
            "runtime_graphics_required": True,
            "dynamic_integrity_court_required": interventions,
        },
        "model_releases": [],
        "input_artifacts": parents + motion["outputs"],
    }, motion, graphics


def _proof(cas, request):
    return execute_proof_capability_v1(
        request, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )


def test_two_original_images_through_motion_and_independent_proof_source_replay(
    tmp_path,
):
    cas, request, motion, graphics = _complete_product_preflight(tmp_path)
    result = _proof(cas, request)
    assert result["status"] == "FAIL"
    assert result["outputs"] == []
    assert result["failure"]["code"] == "PRODUCT_PROOF_INDEPENDENT_WITNESSES_MISSING"
    d = result["diagnostics"]
    assert d["typed_canonical_mesh_rig_skin_reopened"]
    assert d["typed_motion_reopened"]
    assert d["persisted_graphics_runtime_replayed"]
    assert d["dynamic_integrity_recomputed_in_proof"]
    assert d["raw_two_image_replay_completed"]
    assert d["raw_original_source_graphics_replay"][
        "original_rgb_source_pixel_face_candidates_recomputed"
    ] is True
    assert d["raw_original_source_graphics_replay"][
        "mesh_uv_texture_atlas_recomputed_independently"
    ] is True
    assert d["raw_original_source_graphics_replay"][
        "mesh_uv_texture_surface_owner_source_qualified"
    ] is False
    assert d["raw_original_source_graphics_replay"][
        "original_rgb_image_plane_sample_count"
    ] > 0
    assert d["raw_original_source_graphics_replay"][
        "original_rgb_surface_owner_source_qualified"
    ] is False
    assert d["raw_original_source_graphics_replay"] == {
        **d["raw_original_source_graphics_replay"],
        "original_two_image_source_binding_replayed": True,
        "original_alpha_static_mesh_qualification_recomputed": True,
        "same_checkpoint_typed_graphics_recomputed": True,
        "original_source_rgb_appearance_qualified": False,
        "source_camera_observed": False,
        "dynamic_source_visibility_qualified": False,
        "source_attachment_contact_qualified": False,
    }
    assert "SOURCE_APPEARANCE_AND_POSE_VISIBILITY" in d["missing_independent_witnesses"]
    assert "NATIVE_GRAPHICS_RENDERER_PARITY" in d["missing_independent_witnesses"]
    assert "CONTACT_AND_ATTACHMENT_BEHAVIOR" in d["missing_independent_witnesses"]
    assert "EDIT_CONSEQUENCE_AND_RUNTIME_RSS_REOPEN" in d["missing_independent_witnesses"]
    assert "SOURCE_AND_MODEL_RELEASE_ADMISSION" in d["missing_independent_witnesses"]
    assert d["product_pass_authority"] is False
    assert d["product_revision_minted"] is False


def test_raw_proof_rejects_rehashed_altered_graphics_candidate_receipt(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    old = next(
        ref for ref in request["input_artifacts"]
        if ref["artifact_type"] == "RealSaS.RSSvNextGraphicsNativeCandidateReceipt"
    )
    receipt = json.loads(cas.read_object(old))
    receipt["source_hidden_surface_qualified"] = True
    from compiler.realsas_compiler_core.hashing import content_sha256
    receipt.pop("receipt_hash")
    receipt["receipt_hash"] = content_sha256(receipt)
    forged = {
        **cas.put_bytes(json.dumps(receipt, sort_keys=True).encode()),
        "artifact_type": old["artifact_type"],
        "schema_version": "v1",
    }
    request["input_artifacts"] = [
        forged if ref is old else ref for ref in request["input_artifacts"]
    ]
    with pytest.raises(RuntimeError, match="RAW_GRAPHICS_RECEIPT_REPLAY_MISMATCH"):
        _proof(cas, request)


def test_raw_proof_rejects_changed_original_rgb_even_if_rehashed_CAS(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    old = next(
        row for row in request["input_artifacts"]
        if row["artifact_type"] == "RealSaS.RSSvNextSourceRGBImagePlaneCandidates"
    )
    payload = bytearray(cas.read_object(old))
    payload[len(payload) // 2] ^= 1
    forged = {
        **cas.put_bytes(bytes(payload)),
        "artifact_type": old["artifact_type"],
        "schema_version": "v1",
    }
    request["input_artifacts"] = [
        forged if row is old else row for row in request["input_artifacts"]
    ]
    with pytest.raises(Exception, match="SOURCE_RGB|INVALID_ARCHIVE"):
        _proof(cas, request)


def test_raw_proof_requires_rgb_candidate_companion_on_active_product_path(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    request["input_artifacts"] = [
        row for row in request["input_artifacts"]
        if row["artifact_type"] != "RealSaS.RSSvNextSourceRGBImagePlaneCandidates"
    ]
    with pytest.raises(RuntimeError, match="RAW_SOURCE_RGB_CAS_CARDINALITY"):
        _proof(cas, request)


def test_raw_proof_rejects_malicious_mesh_uv_atlas_without_source_replay(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    old = next(
        r for r in request["input_artifacts"]
        if r["artifact_type"] == "RealSaS.RSSvNextMeshUVTextureAtlasCandidate"
    )
    import io
    import numpy as np
    from compiler.rss_vnext.workers.rss_vnext_uv_texture_atlas_v1 import (
        _archive, reopen_rss_vnext_uv_texture_atlas_v1,
    )
    blob = cas.read_object(old)
    with np.load(io.BytesIO(blob), allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    actual = np.argwhere(arrays["texture_source_sample_mask_u8"] == 1)
    assert len(actual) > 0
    view, y, x = actual[0]
    arrays["texture_rgba_u8"][view, y, x, 0] ^= 2
    forged = _archive(arrays)
    # Fully rehashed and parseable source color, but original CAS still wins.
    reopen_rss_vnext_uv_texture_atlas_v1(forged)
    bad = {
        **cas.put_bytes(forged),
        "artifact_type": old["artifact_type"],
        "schema_version": "v1",
    }
    request["input_artifacts"] = [
        bad if r is old else r for r in request["input_artifacts"]
    ]
    with pytest.raises(RuntimeError, match="RAW_UV_TEXTURE_ATLAS_SOURCE_REPLAY_MISMATCH"):
        _proof(cas, request)


def test_raw_proof_refuses_missing_mesh_uv_atlas_before_release(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    request["input_artifacts"] = [
        r for r in request["input_artifacts"]
        if r["artifact_type"] != "RealSaS.RSSvNextMeshUVTextureAtlasCandidate"
    ]
    with pytest.raises(RuntimeError, match="RAW_UV_TEXTURE_ATLAS_CARDINALITY"):
        _proof(cas, request)


def test_raw_proof_rejects_changed_original_source_cas(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    source = next(
        ref for ref in request["input_artifacts"]
        if ref["artifact_type"] == "RealSaS.RSSvNextUncalibratedTwoImageSource"
    )
    altered = {
        **cas.put_bytes(b"not-original-source"),
        "artifact_type": source["artifact_type"],
        "schema_version": "v1",
    }
    request["input_artifacts"] = [
        altered if ref is source else ref for ref in request["input_artifacts"]
    ]
    with pytest.raises(Exception):
        _proof(cas, request)


def test_raw_proof_requires_graphics_compilation_receipt_even_with_matching_hashes(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    request["input_artifacts"] = [
        ref for ref in request["input_artifacts"]
        if ref["artifact_type"] != "RealSaS.RSSvNextGraphicsNativeCandidateReceipt"
    ]
    with pytest.raises(RuntimeError, match="RAW_GRAPHICS_PARENT_CARDINALITY"):
        _proof(cas, request)


def test_raw_proof_refuses_source_and_receipt_downgrade_to_legacy(
    tmp_path,
):
    cas, request, _, _ = _complete_product_preflight(tmp_path)
    request["input_artifacts"] = [
        row for row in request["input_artifacts"]
        if row["artifact_type"] not in {
            "RealSaS.RSSvNextUncalibratedTwoImageSource",
            "RealSaS.RSSvNextGraphicsNativeCandidateReceipt",
        }
    ]
    with pytest.raises(RuntimeError, match="ORPHAN_RAW_UV_TEXTURE_ATLAS"):
        _proof(cas, request)
