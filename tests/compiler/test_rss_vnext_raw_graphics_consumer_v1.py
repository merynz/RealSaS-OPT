from __future__ import annotations

import json
import pytest

from compiler.realsas_compiler_services.platform_worker.mechanics_executor_v1 import (
    execute_mechanics_capability_v1,
)
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_graphics_capability_v1,
)
from tests.compiler.test_rss_vnext_raw_canonical_skin_v4 import (
    _fixture as skin_fixture, _run as run_skin,
)


def _full_parent_fixture(tmp_path, *, active_mechanics_courts=False):
    cas, skin_req, _, _ = skin_fixture(tmp_path)
    skin = run_skin(cas, skin_req)
    assert skin["status"] == "PASS"
    inherited = list(skin_req["input_artifacts"]) + skin["outputs"]
    mesh_rig_skin = [
        row for row in inherited
        if row["artifact_type"] in (
            "RealSaS.CanonicalMeshIR", "RealSaS.CanonicalRigIR",
            "RealSaS.CanonicalSkinIR",
        )
    ]
    mechanics_meta = {
        "executor_contract": "RealSaS.ProductMechanicsCompile.v1",
        "required_models": [],
        "rest_qualification": {
            "max_position_drift": 0.00001,
            "max_constraint_residual": 0.0001,
        },
    }
    if active_mechanics_courts:
        mechanics_meta["control_intervention_policy"] = {
            "enabled": True, "max_probed_controls": 128,
            "max_full_mesh_deformation_probes": 4,
            "relative_translation": 0.02,
            "max_constraint_residual": 0.001,
        }
        mechanics_meta["dynamic_integrity_policy"] = {
            "enabled": True, "max_faces": 1024,
            "max_controls": 128,
            "relative_translation": 0.02,
            "max_constraint_residual": 0.001,
        }
    mechanics = execute_mechanics_capability_v1({
        "kind": "MODULE", "capability_id": "product.mechanics",
        "capability_metadata": mechanics_meta,
        "model_releases": [], "input_artifacts": mesh_rig_skin,
    }, read_object=cas.read_object, put_bytes=cas.put_bytes)
    assert mechanics["status"] == "PASS"
    req = {
        "kind": "MODULE", "capability_id": "product.graphics",
        "capability_metadata": {
            "executor_contract": "RealSaS.ProductGraphicsCompile.v1",
            "required_models": [],
            "mechanics_intervention_required": active_mechanics_courts,
        },
        "model_releases": [],
        "input_artifacts": inherited + mechanics["outputs"],
    }
    return cas, req, mechanics, skin


def test_two_original_images_neural_shape_mesh_rig_skin_mechanics_graphics_runtime(
    tmp_path,
):
    cas, req, mechanics, skin = _full_parent_fixture(tmp_path)
    out = execute_graphics_capability_v1(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert out["status"] == "PASS"
    assert len(out["outputs"]) == 4
    assert set(x["artifact_type"] for x in out["outputs"]) == {
        "RealSaS.CompiledGraphics", "RealSaS.RuntimeGraphics",
        "RealSaS.RSSvNextGraphicsNativeCandidateReceipt",
        "RealSaS.RSSvNextObservedRGBHypothesisSamples",
    }
    diagnostics = out["diagnostics"]
    assert diagnostics["source_appearance_qualified"] is False
    assert diagnostics["source_visibility_qualified"] is False
    assert diagnostics["pose_conditioned_occlusion_qualified"] is False
    graphics = json.loads(cas.read_object(out["outputs"][0]))
    assert len(graphics["fragments"]) >= 1
    assert graphics["metadata"]["origin"] == (
        "RAW_TWO_IMAGE_UNIFIED_H_TYPED_GRAPHICS_CANDIDATES"
    )
    assert graphics["metadata"]["source_static_silhouette_passed"] is True
    assert graphics["metadata"]["qualified_source_occlusion"] is False
    assert graphics["metadata"]["qualified_source_attachments"] is False
    receipt = json.loads(cas.read_object(out["outputs"][2]))
    assert receipt["source_hidden_surface_qualified"] is False
    assert receipt["pose_conditioned_visibility_qualified"] is False
    assert receipt["product_revision_minted"] is False
    rgb = json.loads(cas.read_object(out["outputs"][3]))
    assert rgb["schema"] == "RealSaS.RSSvNext.OriginalRGBProjectedCandidate.v1"
    assert rgb["source_rgb_pixels_observed"] is True
    assert rgb["source_rgb_to_canonical_face_ownership_qualified"] is False
    assert rgb["source_appearance_release_qualified"] is False
    assert rgb["product_revision_minted"] is False
    assert len(rgb["views"]) == 2
    assert rgb["sample_count"] > 0
    for view in rgb["views"]:
        assert view["original_source_rgb_pixels_observed"] is True
        assert view["projected_face_correspondence_is_original_source_truth"] is False
        assert view["sampled_source_opaque_interior_count"] > 0
        assert all(
            sample["rgba8_observed"] == [180, 35, 15, 255]
            and sample["candidate_face_index"] == 0
            and sample["candidate_face_owner_is_source_truth"] is False
            for sample in view["samples"]
        )
    assert receipt["source_original_rgb_candidate_hash"] == rgb["candidate_hash"]


def test_raw_graphics_consumes_real_mechanics_nonrest_intervention_and_dynamic_court(
    tmp_path,
):
    cas, req, mechanics, skin = _full_parent_fixture(
        tmp_path, active_mechanics_courts=True,
    )
    types = {row["artifact_type"] for row in mechanics["outputs"]}
    assert "RealSaS.MechanicsInterventionCourt" in types
    assert "RealSaS.MechanicsDynamicIntegrityCourt" in types
    result = execute_graphics_capability_v1(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert result["status"] == "PASS"
    assert result["diagnostics"]["mechanics_intervention_court_hash"]
    assert result["diagnostics"]["mechanical_source_contact_qualified"] is False
    graphics = json.loads(cas.read_object(result["outputs"][0]))
    assert graphics["metadata"]["mechanics_intervention_court_hash"]
    assert graphics["metadata"]["source_static_silhouette_passed"] is True


def test_raw_graphics_rejects_different_scene_parent_even_if_source_cas_was_reused(
    tmp_path,
):
    cas, req, _, _ = _full_parent_fixture(tmp_path)
    source = next(
        r for r in req["input_artifacts"]
        if r["artifact_type"] == "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier"
    )
    forged = {
        **cas.put_bytes(b"replaced_shape_scene"),
        "artifact_type": source["artifact_type"],
        "schema_version": "v1",
    }
    req["input_artifacts"] = [
        forged if row is source else row for row in req["input_artifacts"]
    ]
    with pytest.raises(RuntimeError, match="SOURCE_OR_CHECKPOINT_PARENT_DRIFT"):
        execute_graphics_capability_v1(
            req, read_object=cas.read_object, put_bytes=cas.put_bytes,
        )
