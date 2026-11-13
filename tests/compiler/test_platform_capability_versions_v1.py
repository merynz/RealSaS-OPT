from __future__ import annotations

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_capability_versions import (
    capability_implementations, capability_specs, product_model_requirements,
)
from tools.platform_release_snapshot import snapshot


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def test_product_capability_implementations_are_complete_and_semantic():
    specs = capability_specs()
    implementations = capability_implementations()
    ids = {
        "product.input", "product.shape", "product.mesh", "product.rig",
        "product.skin", "product.mechanics", "product.graphics",
        "product.motion", "product.proof",
    }
    assert set(specs) == set(implementations) == ids
    assert specs["product.input"]["parameters"]["responsibility_ids"] == [
        "input.model_observation_lowering",
    ]
    assert specs["product.shape"]["parameters"]["responsibility_ids"] == [
        "shape.shared_scene_evidence", "shape.topology_proposal",
    ]
    assert specs["product.rig"]["parameters"]["responsibility_ids"] == [
        "rig.canonical_identity_mint", "rig.structure_proposal",
        "rig.structure_qualification",
    ]
    assert specs["product.skin"]["parameters"]["responsibility_ids"] == [
        "skin.canonical_influence_mint", "skin.carrier_binding_qualification",
        "skin.influence_legality_qualification", "skin.influence_proposal",
    ]
    input_spec = specs["product.input"]
    assert input_spec["parameters"]["executor_contract"] == "RealSaS.ProductInputTwoView.v3"
    assert input_spec["parameters"]["original_observed_view_count"] == 2
    assert input_spec["parameters"]["source_copy_policy"] == "BYTE_IDENTICAL_CAS_REUSE"
    assert input_spec["parameters"]["model_inference_in_input_forbidden"] is True

    shape = specs["product.shape"]
    assert shape["parameters"]["executor_contract"] == "RealSaS.ProductShapeInference.v4"
    assert shape["parameters"]["required_models"] == [{
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
    }]
    assert shape["parameters"]["independent_iris_tessa_forbidden"] is True
    assert len(shape["parameters"]["auxiliary_outputs"]) == 7

    mesh = specs["product.mesh"]
    assert mesh["policy"]["dependencies"] == ["product.shape"]
    assert mesh["parameters"]["executor_contract"] == "RealSaS.ProductMeshCompile.v3"
    assert mesh["parameters"]["output_artifact_type"] == "RealSaS.CanonicalMeshIR"
    assert mesh["parameters"]["output_schema_version"] == "v1"
    assert mesh["parameters"]["requires_shape_parent_binding"] is True
    assert mesh["parameters"]["canonical_mesh_required_for_product_execution"] is True
    assert mesh["parameters"]["required_models"] == []
    assert _is_sha256(mesh["parameters"]["two_view_static_policy_sha256"])
    assert _is_sha256(mesh["parameters"]["mesh_policy_document_sha256"])

    rig = specs["product.rig"]
    assert rig["policy"]["dependencies"] == ["product.shape", "product.mesh"]
    assert rig["entry_module"] == "compiler.rss_vnext.workers.rss_vnext_rig_executor_v4"
    assert rig["parameters"]["executor_contract"] == "RealSaS.ProductRigInference.v4"
    assert rig["parameters"]["required_models"] == []
    assert rig["parameters"]["single_checkpoint_inference_reused"] is True
    assert rig["parameters"]["independent_axis_model_release_allowed"] is False
    assert rig["parameters"]["compiler_control_synthesis_allowed"] is False
    assert rig["parameters"]["compiler_constraint_synthesis_allowed"] is False
    assert rig["parameters"]["output_artifact_type"] == "RealSaS.CanonicalRigIR"
    assert rig["parameters"]["output_schema_version"] == "v2"

    skin = specs["product.skin"]
    assert skin["policy"]["dependencies"] == [
        "product.shape", "product.mesh", "product.rig",
    ]
    assert skin["entry_module"] == "compiler.rss_vnext.workers.rss_vnext_skin_executor_v4"
    assert skin["parameters"]["executor_contract"] == "RealSaS.ProductSkinInference.v4"
    assert skin["parameters"]["required_models"] == []
    assert skin["parameters"]["single_checkpoint_inference_reused"] is True
    assert skin["parameters"]["exact_embedded_mira_decoder_replayed"] is True
    assert skin["parameters"]["independent_mira_model_release_allowed"] is False
    assert skin["parameters"]["compiler_code_search_allowed"] is False
    assert skin["parameters"]["post_hoc_skin_transfer_allowed"] is False
    assert skin["parameters"]["input_skin_code_artifact_type"] == "RealSaS.SkinCodeProposalIR"
    assert skin["parameters"]["output_artifact_type"] == "RealSaS.CanonicalSkinIR"

    assert product_model_requirements() == [{
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
    }]
    for capability_id, implementation in implementations.items():
        assert set(implementation) == {"implementation_sha256"}, capability_id
        assert _is_sha256(implementation["implementation_sha256"]), capability_id
    assert len({
        implementations[id]["implementation_sha256"]
        for id in ("product.input", "product.shape", "product.mesh",
                   "product.rig", "product.skin")
    }) == 5


def test_release_snapshot_v2_carries_implementation_and_model_requirements():
    plan = mainline.load_json(mainline.PLAN_PATH)
    request = snapshot(
        plan, {"subject_id": "not-release-authority"},
        name="capability-release-v2", purpose="RESEARCH", created_by="ci",
    )
    assert request["manifest"]["contract_version"] == "RealSaS.EngineReleaseManifest.v2"
    assert request["capability_implementations"] == capability_implementations()
    assert request["manifest"]["capability_implementations"] == {
        cid: row["implementation_sha256"]
        for cid, row in request["capability_implementations"].items()
    }
    assert "capability_versions" not in request
    for row in request["capability_implementations"].values():
        assert set(row) == {"implementation_sha256"}
        assert "policy_sha256" not in row
        assert "parameters_sha256" not in row
    assert request["manifest"]["model_requirements"] == [{
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
        "channel": "RESEARCH",
    }]
    assert "subject_id" not in str(request)
