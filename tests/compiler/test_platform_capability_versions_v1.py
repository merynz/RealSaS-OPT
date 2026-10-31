from __future__ import annotations

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_capability_versions import (
    capability_implementations,
    capability_specs,
    product_model_requirements,
)
from tools.platform_release_snapshot import snapshot


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def test_product_capability_implementations_are_complete_and_semantic():
    specs = capability_specs()
    implementations = capability_implementations()
    assert set(implementations) == {
        "product.input", "product.shape", "product.mesh", "product.rig", "product.skin",
        "product.mechanics", "product.graphics", "product.motion", "product.proof",
    }
    assert set(specs) == set(implementations)
    assert specs["product.input"]["parameters"]["responsibility_ids"] == ["input.model_observation_lowering"]
    assert specs["product.shape"]["parameters"]["responsibility_ids"] == [
        "shape.shared_scene_evidence", "shape.topology_proposal",
    ]
    assert specs["product.mesh"]["parameters"]["responsibility_ids"] == [
        "mesh.canonical_identity_mint", "mesh.source_fidelity_qualification", "mesh.static_topology_qualification",
    ]
    assert specs["product.rig"]["parameters"]["responsibility_ids"] == [
        "rig.canonical_identity_mint", "rig.structure_proposal", "rig.structure_qualification",
    ]
    assert specs["product.skin"]["parameters"]["responsibility_ids"] == [
        "skin.canonical_influence_mint", "skin.carrier_binding_qualification",
        "skin.influence_legality_qualification", "skin.influence_proposal",
    ]
    assert specs["product.mechanics"]["parameters"]["responsibility_ids"] == [
        "mechanics.constraint_solution", "mechanics.deformation_domain_qualification",
        "mechanics.post_bind_frame_observability_qualification",
    ]
    assert specs["product.graphics"]["parameters"]["responsibility_ids"] == [
        "graphics.appearance_authority_binding", "graphics.presentation_relation_evidence",
        "graphics.presentation_relation_solution",
    ]
    assert specs["product.motion"]["parameters"]["responsibility_ids"] == [
        "motion.motion_solution", "motion.source_motion_admission",
        "motion.target_body_constraint_solution",
    ]
    assert specs["product.proof"]["parameters"]["responsibility_ids"] == ["proof.product_pass_decision"]
    assert "rig.frame_observability_qualification" not in str(specs["product.rig"])

    product_input = specs["product.input"]
    assert product_input["parameters"]["camera_output_artifact_type"] == "RealSaS.QualifiedCameraSetIR"
    assert product_input["parameters"]["observation_output_artifact_type"] == "RealSaS.QualifiedObservationSetIR"
    assert product_input["parameters"]["normalization_output_artifact_type"] == "RealSaS.NormalizationDomainIR"
    assert product_input["parameters"]["source_qualification_owner"] == "PRODUCT_INPUT"
    assert product_input["parameters"]["legacy_compiler_stage_bridge_required"] is False

    assert specs["product.shape"]["parameters"]["required_models"] == [
        {"family": "IRIS", "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3"},
        {"family": "TESSA", "architecture_id": "RealSaS.TESSA.Structuralizer.v3"},
    ]

    mesh = specs["product.mesh"]
    assert mesh["policy"]["dependencies"] == ["product.input", "product.shape"]
    assert not any(str(dep).startswith("compiler.stage/") for dep in mesh["policy"]["dependencies"])
    assert mesh["parameters"]["input_camera_artifact_type"] == "RealSaS.QualifiedCameraSetIR"
    assert mesh["parameters"]["input_observation_artifact_type"] == "RealSaS.QualifiedObservationSetIR"
    assert mesh["parameters"]["input_normalization_artifact_type"] == "RealSaS.NormalizationDomainIR"
    assert mesh["parameters"]["source_qualification_parent"] == "product.input"
    assert mesh["parameters"]["legacy_compiler_stage_bridge_required"] is False
    assert mesh["parameters"]["required_models"] == []
    assert mesh["parameters"]["fit_allowed_during_product_execution"] is False
    assert mesh["parameters"]["legacy_ir_fit_stage_dependency"] is False
    assert mesh["parameters"]["surface_born_inside_mesh_court"] is True
    assert mesh["parameters"]["surface_is_canonical_mesh_authority"] is False
    assert mesh["parameters"]["shape_surface_court_policy_sha256"] == "e182023bd59d079ecb747d0812af4e83de7415593c4bca54c753d0f7abfa6f51"
    assert mesh["parameters"]["source_fidelity_policy_sha256"] == "8edf9934f1d14f45dcb719fcf39691d69c5797f7410c958ae3005ef639a3c214"
    assert mesh["parameters"]["surface_adequacy_policy_sha256"] == "bc98d3aa3cd419ca9ae43010e890fc554ab62e57268933afc630aae20c38139b"

    rig = specs["product.rig"]
    assert rig["policy"]["dependencies"] == ["product.mesh"]
    assert rig["parameters"]["executor_contract"] == "RealSaS.ProductRigInference.v2"
    assert rig["parameters"]["feature_evidence_source"] == "PRODUCT_MESH_OUTPUT"
    assert rig["parameters"]["legacy_stage15_dependency"] is False
    assert rig["parameters"]["output_schema_version"] == "v2"
    assert rig["parameters"]["skin_code_output_type"] == "RealSaS.SkinCodeProposalIR"
    assert rig["parameters"]["compiler_parent_reselection_allowed"] is False
    assert rig["parameters"]["compiler_control_synthesis_allowed"] is False
    assert rig["parameters"]["compiler_constraint_synthesis_allowed"] is False
    assert rig["parameters"]["required_models"] == [
        {"family": "AXIS", "architecture_id": "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0"}
    ]

    skin = specs["product.skin"]
    assert skin["policy"]["dependencies"] == ["product.mesh", "product.rig"]
    assert skin["parameters"]["executor_contract"] == "RealSaS.ProductSkinInference.v2"
    assert skin["parameters"]["input_rig_schema_version"] == "v2"
    assert skin["parameters"]["input_skin_code_artifact_type"] == "RealSaS.SkinCodeProposalIR"
    assert skin["parameters"]["output_schema_version"] == "v2"
    assert skin["parameters"]["required_models"] == [
        {"family": "MIRA", "architecture_id": "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0"}
    ]
    assert skin["parameters"]["canonical_parents"] == ["CANONICAL_MESH", "CANONICAL_RIG_DEFORM_STRUCTURE"]
    assert skin["parameters"]["full_control_program_is_skin_parent"] is False
    assert skin["parameters"]["axis_rerun_allowed"] is False
    assert skin["parameters"]["compiler_code_search_allowed"] is False
    assert skin["parameters"]["compiler_code_repair_allowed"] is False
    assert skin["parameters"]["carrier_skin_qualification"]["max_influences"] == 256

    mechanics = specs["product.mechanics"]
    assert mechanics["policy"]["dependencies"] == ["product.mesh", "product.rig", "product.skin"]
    assert mechanics["parameters"]["executor_contract"] == "RealSaS.ProductMechanicsCompile.v1"
    assert mechanics["parameters"]["runtime_and_compile_solver_shared"] is True
    assert mechanics["parameters"]["rig_program_mutation_allowed"] is False

    graphics = specs["product.graphics"]
    assert graphics["policy"]["dependencies"] == [
        "product.shape", "product.mesh", "product.skin", "product.mechanics",
    ]
    assert graphics["entry_module"] == "compiler.realsas_compiler_services.platform_worker.graphics_executor_v1"
    assert graphics["parameters"]["executor_contract"] == "RealSaS.ProductGraphicsCompile.v1"
    assert graphics["parameters"]["required_models"] == [
        {"family": "IRIS", "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3"}
    ]
    assert graphics["parameters"]["input_shape_artifact_type"] == "RealSaS.ShapeProposal"
    assert graphics["parameters"]["shared_scene_h_reused_without_scene_reencode"] is True
    assert graphics["parameters"]["graphics_owns_all_visual_presentation"] is True
    assert graphics["parameters"]["caa_is_graphics_appearance_sublayer"] is True
    assert graphics["parameters"]["visibility_occlusion_owner"] == "GRAPHICS"
    assert graphics["parameters"]["draw_order_owner"] == "GRAPHICS"
    assert graphics["parameters"]["compiler_perception_synthesis_allowed"] is False
    assert graphics["parameters"]["mechanics_draw_order_authority_allowed"] is False
    assert graphics["parameters"]["motion_draw_order_authority_allowed"] is False

    motion = specs["product.motion"]
    assert motion["policy"]["dependencies"] == ["product.rig", "product.mechanics"]
    assert motion["parameters"]["executor_contract"] == "RealSaS.ProductMotionCompile.v2"
    assert motion["parameters"]["editable_motion_space"] == "ANIMATOR_CONTROLS"
    assert motion["parameters"]["direct_joint_track_authority"] is False
    assert motion["parameters"]["constraint_execution_owner"] == "MECHANICS"

    proof = specs["product.proof"]
    assert proof["policy"]["product_pass_authority"] is True
    assert proof["policy"]["dependencies"] == [
        "product.mesh", "product.rig", "product.skin", "product.mechanics",
        "product.graphics", "product.motion",
    ]
    assert proof["parameters"]["executor_contract"] == "RealSaS.ProductProofCompile.v1"
    assert proof["parameters"]["output_artifact_type"] == "RealSaS.ProductProofBundleIR"
    assert proof["parameters"]["legacy_puppet_projection_required"] is False
    assert proof["parameters"]["product_revision_minting_owned_here"] is False

    assert product_model_requirements() == [
        {"family": "AXIS", "architecture_id": "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0"},
        {"family": "IRIS", "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3"},
        {"family": "MIRA", "architecture_id": "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0"},
        {"family": "TESSA", "architecture_id": "RealSaS.TESSA.Structuralizer.v3"},
    ]
    for capability_id, implementation in implementations.items():
        assert set(implementation) == {"implementation_sha256"}, capability_id
        assert _is_sha256(implementation["implementation_sha256"]), capability_id
    assert len({implementations[x]["implementation_sha256"] for x in implementations}) == 9


def test_release_snapshot_v2_carries_implementation_and_model_requirements():
    plan = mainline.load_json(mainline.PLAN_PATH)
    request = snapshot(
        plan, {"subject_id": "not-release-authority"}, name="capability-release-v2",
        purpose="RESEARCH", created_by="ci",
    )
    assert request["manifest"]["contract_version"] == "RealSaS.EngineReleaseManifest.v2"
    assert request["capability_implementations"] == capability_implementations()
    assert request["manifest"]["capability_implementations"] == {
        capability_id: row["implementation_sha256"]
        for capability_id, row in request["capability_implementations"].items()
    }
    assert "capability_versions" not in request
    for row in request["capability_implementations"].values():
        assert set(row) == {"implementation_sha256"}
        assert "policy_sha256" not in row and "parameters_sha256" not in row
    assert request["manifest"]["model_requirements"] == [
        {"family": "AXIS", "architecture_id": "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0", "channel": "RESEARCH"},
        {"family": "IRIS", "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3", "channel": "RESEARCH"},
        {"family": "MIRA", "architecture_id": "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0", "channel": "RESEARCH"},
        {"family": "TESSA", "architecture_id": "RealSaS.TESSA.Structuralizer.v3", "channel": "RESEARCH"},
    ]
    assert "subject_id" not in str(request)
