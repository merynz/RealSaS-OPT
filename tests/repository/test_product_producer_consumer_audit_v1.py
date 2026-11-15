from __future__ import annotations

from copy import deepcopy

from tools.product_producer_consumer_audit_v1 import (
    EDGES, V2_MODEL_MODULES, evaluate, read_snapshot,
)


def test_registered_product_contracts_and_executors_are_consistent_on_main():
    report = evaluate(read_snapshot())
    assert report["registered_contract_status"] == "PASS", report["registered_issues"]
    assert report["registered_issues"] == []
    ids = {row["capability_id"] for row in report["registered_capabilities"]}
    assert ids == {"product.input", "product.shape", "product.mesh", "product.rig", "product.skin", "product.mechanics", "product.graphics", "product.motion", "product.proof"}
    assert len(report["registered_edges"]) == 25
    assert {row["status"] for row in report["registered_edges"]} == {"MATCH", "LINEAGE_ONLY"}
    assert sum(row["status"] == "LINEAGE_ONLY" for row in report["registered_edges"]) == 5


def test_current_sources_do_not_overclaim_an_e2e_product():
    report = evaluate(read_snapshot())
    assert report["product_e2e_qualified"] is False
    assert report["product_e2e_proof"] == "NOT_EXECUTED_OR_MEASURED_BY_THIS_AUDIT"
    assert report["v2_candidate"]["motion_source_present"]
    # This audit must follow the exact source tree: V6 model runtime sources are
    # now available, but their presence cannot grant model/product authority.
    assert report["v2_candidate"]["missing_model_runtime_sources"] == [
        path for path in V2_MODEL_MODULES if not read_snapshot()["source_files"][path]
    ]
    assert report["runtime_consumption"]["legacy_stage40_43_46_product_route_present"]
    assert report["runtime_consumption"]["frame_baked_render_consumer_present"]
    assert not report["runtime_consumption"]["canonical_motion_v2_render_bridge_verified"]
    assert report["editor"]["compile_available_literal"] is False
    assert report["editor"]["edits_require_compiler_requalification"] is True
    blockers = report["product_completion_blockers"]
    for critical in (
        "CANONICAL_MOTION_V2_NOT_CONSUMED_BY_PRODUCT_RENDER",
        "EDITOR_COMPILE_NOT_CONNECTED",
        "EDIT_LAYER_NOT_REQUALIFIED_AS_PRODUCT",
    ):
        assert critical in blockers
    assert report["authority"].endswith("__NO_PRODUCT_OR_SCIENTIFIC_PASS")


def test_detects_producer_consumer_schema_drift_even_if_both_have_valid_names():
    snapshot = deepcopy(read_snapshot())
    snapshot["capabilities"]["product.skin"]["metadata"]["input_rig_schema_version"] = "v1"
    report = evaluate(snapshot)
    assert report["registered_contract_status"] == "FAIL"
    assert "REGISTERED_ARTIFACT_SCHEMA_MISMATCH:product.rig->product.skin" in report["registered_issues"]
    assert next(
        row for row in report["registered_edges"]
        if row["producer"] == "product.rig" and row["consumer"] == "product.skin"
    )["status"] == "MISMATCH"


def test_detects_missing_real_worker_route_not_just_importable_python_module():
    snapshot = deepcopy(read_snapshot())
    snapshot["worker"] = snapshot["worker"].replace(
        '"RealSaS.ProductSkinInference.v4"', '"RealSaS.ProductSkinInference.test_removed"', 1
    )
    report = evaluate(snapshot)
    assert "REGISTERED_EXECUTOR_DISPATCH_MISSING:product.skin" in report["registered_issues"]


def test_detects_go_provider_omission_despite_json_descriptor():
    snapshot = deepcopy(read_snapshot())
    assert "RSSvNextSkinCandidateProvider{}" in snapshot["go_catalog"]
    snapshot["go_catalog"] = snapshot["go_catalog"].replace("RSSvNextSkinCandidateProvider{},", "", 1)
    report = evaluate(snapshot)
    assert "REGISTERED_GO_JSON_CATALOG_DRIFT" in report["registered_issues"]


def test_detects_missing_actual_mesh_secondary_artifact_emission():
    snapshot = deepcopy(read_snapshot())
    assert 'artifact_type="RealSaS.TESSAMaterialSupportFieldIR"' in snapshot["mesh_executor"]
    snapshot["mesh_executor"] = snapshot["mesh_executor"].replace(
        'artifact_type="RealSaS.TESSAMaterialSupportFieldIR"',
        'artifact_type="RealSaS.RemovedMaterialSupport"', 1,
    )
    report = evaluate(snapshot)
    assert "MESH_SECONDARY_OUTPUT_MISSING:RealSaS.TESSAMaterialSupportFieldIR" in report["registered_issues"]


def test_registered_mechanics_cannot_forge_v2_parent_versions():
    snapshot = deepcopy(read_snapshot())
    snapshot["capabilities"]["product.mechanics"]["metadata"]["input_rig_schema_version"] = "v1"
    snapshot["capabilities"]["product.mechanics"]["metadata"]["input_skin_schema_version"] = "v1"
    report = evaluate(snapshot)
    assert "REGISTERED_ARTIFACT_SCHEMA_MISMATCH:product.rig->product.mechanics" in report["registered_issues"]
    assert "REGISTERED_ARTIFACT_SCHEMA_MISMATCH:product.skin->product.mechanics" in report["registered_issues"]
    assert report["product_e2e_qualified"] is False


def test_registered_proof_cannot_claim_an_executable_product_from_candidate_motion():
    snapshot = deepcopy(read_snapshot())
    snapshot["capabilities"]["product.proof"]["metadata"]["input_motion_schema_version"] = "v1"
    report = evaluate(snapshot)
    assert "REGISTERED_ARTIFACT_SCHEMA_MISMATCH:product.motion->product.proof" in report["registered_issues"]
    assert "CANONICAL_MOTION_V2_NOT_CONSUMED_BY_PRODUCT_RENDER" in report["product_completion_blockers"]


def test_missing_v2_runtime_source_remains_an_explicit_product_blocker():
    snapshot = deepcopy(read_snapshot())
    snapshot["source_files"][V2_MODEL_MODULES[0]] = False
    report = evaluate(snapshot)
    assert V2_MODEL_MODULES[0] in report["v2_candidate"]["missing_model_runtime_sources"]
    assert "V2_RIG_SKIN_MODEL_RUNTIME_SOURCE_DEPENDENCIES_MISSING" in report["product_completion_blockers"]
    assert report["product_e2e_qualified"] is False


def test_contract_edge_recipes_cover_known_steady_product_prefix():
    assert EDGES["product.skin"]["product.rig"] == ("input_rig_artifact_type", "input_rig_schema_version")
    assert EDGES["product.motion"]["product.mechanics"] == ("input_mechanics_artifact_type", "input_mechanics_schema_version")



def test_active_raw_shape_scene_is_the_primary_producer_not_old_inference_fanout():
    snapshot = deepcopy(read_snapshot())
    shape = snapshot["capabilities"]["product.shape"]["metadata"]
    mesh = snapshot["capabilities"]["product.mesh"]["metadata"]
    assert shape["output_artifact_type"] == mesh["input_shape_artifact_type"]
    assert shape["output_artifact_type"] == (
        "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier"
    )
    assert len(shape["auxiliary_outputs"]) == 4
    assert len(shape["calibrated_auxiliary_outputs"]) == 7
    for role in ("product.rig", "product.skin", "product.graphics"):
        assert snapshot["capabilities"][role]["metadata"]["raw_scene_artifact_type"] == (
            shape["output_artifact_type"]
        )
    shape["output_artifact_type"] = "RealSaS.RSSvNextInferenceEvidence"
    report = evaluate(snapshot)
    assert "REGISTERED_ARTIFACT_SCHEMA_MISMATCH:product.shape->product.mesh" in (
        report["registered_issues"]
    )
    assert "ACTIVE_RAW_AND_CALIBRATED_SHAPE_PRIMARY_CONTRACT_DRIFT" in (
        report["registered_issues"]
    )


def test_active_shape_cannot_hide_missing_raw_embedded_mira_child():
    snapshot = deepcopy(read_snapshot())
    shape = snapshot["capabilities"]["product.shape"]["metadata"]
    shape["auxiliary_outputs"] = [
        x for x in shape["auxiliary_outputs"]
        if x["role"] != "rss_uncalibrated_embedded_mira_decoder"
    ]
    report = evaluate(snapshot)
    assert "ACTIVE_RAW_SHAPE_FIVE_CAS_OUTPUT_CONTRACT_DRIFT" in report["registered_issues"]


def test_calibrated_shape_fanout_is_explicit_alternative_not_raw_source_authority():
    snapshot = deepcopy(read_snapshot())
    snapshot["capabilities"]["product.skin"]["metadata"][
        "input_fanout_artifact_type"
    ] = "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier"
    report = evaluate(snapshot)
    assert "CALIBRATED_SHAPE_FANOUT_CONSUMER_DRIFT:product.skin" in (
        report["registered_issues"]
    )
    assert report["product_e2e_qualified"] is False



def test_single_rss_vnext_product_graph_and_authority_cannot_split_into_parallel_models():
    """The existing Go/JSON catalog is THE authority, not another freeze manifest.

    This protects live product selections, not training quality or release PASS.
    Earlier research modules may coexist without becoming additional providers.
    """
    capabilities = read_snapshot()["capabilities"]
    expected = {
        "product.input", "product.shape", "product.mesh", "product.rig",
        "product.skin", "product.mechanics", "product.graphics",
        "product.motion", "product.proof",
    }
    assert set(capabilities) == expected

    model = [{
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
    }]
    assert capabilities["product.shape"]["metadata"]["required_models"] == model
    assert capabilities["product.shape"]["metadata"]["implementation_class"] == "MODEL"
    for role in expected - {"product.shape"}:
        assert capabilities[role]["metadata"].get("required_models", []) == []
        assert capabilities[role]["metadata"]["implementation_class"] != "MODEL"

    pass_owners = {
        role for role in expected
        if capabilities[role]["policy"].get("product_pass_authority") is True
    }
    assert pass_owners == {"product.proof"}
    assert capabilities["product.proof"]["metadata"][
        "product_revision_requires_independent_admission"
    ] is True
    assert capabilities["product.proof"]["metadata"][
        "can_mint_product_revision"
    ] is False

    # A cycle or disconnected research entrypoint cannot masquerade as the
    # nine-role source-to-proof production DAG.
    visiting = set()
    ancestors = {}
    def preceding(role):
        if role in visiting:
            raise AssertionError("RSS_VNEXT_ACTIVE_PRODUCT_DAG_CYCLE")
        if role not in ancestors:
            visiting.add(role)
            deps = capabilities[role]["policy"].get("dependencies", [])
            assert len(deps) == len(set(deps)), role
            assert set(deps).issubset(expected), role
            ancestors[role] = set(deps)
            for dependency in deps:
                ancestors[role].update(preceding(dependency))
            visiting.remove(role)
        return ancestors[role]

    assert capabilities["product.input"]["policy"].get("dependencies", []) == []
    for role in expected:
        preceding(role)
    assert ancestors["product.proof"] == expected - {"product.proof"}
    assert all(
        "product.input" in ancestors[role]
        for role in expected - {"product.input"}
    )

    shape = capabilities["product.shape"]["metadata"]
    mesh = capabilities["product.mesh"]["metadata"]
    rig = capabilities["product.rig"]["metadata"]
    skin = capabilities["product.skin"]["metadata"]
    mechanics = capabilities["product.mechanics"]["metadata"]
    graphics = capabilities["product.graphics"]["metadata"]
    motion = capabilities["product.motion"]["metadata"]
    assert shape["input_artifact_type"] == capabilities["product.input"]["metadata"]["output_artifact_type"]
    assert shape["output_artifact_type"] == mesh["input_shape_artifact_type"]
    assert mesh["output_artifact_type"] == rig["input_mesh_artifact_type"]
    assert mesh["output_artifact_type"] == skin["input_mesh_artifact_type"]
    assert rig["output_artifact_type"] == skin["input_rig_artifact_type"]
    assert rig["single_checkpoint_inference_reused"] is True
    assert skin["single_checkpoint_inference_reused"] is True
    assert rig["independent_axis_model_release_allowed"] is False
    assert skin["independent_mira_model_release_allowed"] is False
    assert skin["exact_embedded_mira_decoder_replayed"] is True
    assert mechanics["dynamic_integrity_not_source_contact_or_secondary_motion"] is True
    assert graphics["source_appearance_qualified_by_graphics_evidence"] is False
    assert motion["default_rest_hold_policy"] == "DETERMINISTIC_ZERO_TRACK_REST_HOLD_V1"
