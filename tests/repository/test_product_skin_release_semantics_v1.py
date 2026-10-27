from tools.platform_capability_versions import capability_specs, product_model_requirements


def test_skin_is_first_class_release_semantics_without_legacy_stage15():
    specs = capability_specs()
    assert "product.skin" in specs
    skin = specs["product.skin"]
    assert skin["entry_module"] == "compiler.realsas_compiler_services.platform_worker.skin_executor_v1"
    assert skin["policy"]["allowed_modes"] == ["DEVELOPER"]
    assert skin["policy"]["product_pass_authority"] is False
    assert skin["policy"]["dependencies"] == ["product.mesh", "product.rig"]
    assert "compiler.stage/15_RIGGING_SURFACE_QUALIFIED" not in skin["policy"]["dependencies"]
    assert skin["parameters"]["feature_evidence_source"] == "PRODUCT_MESH_OUTPUT"
    assert skin["parameters"]["material_support_source"] == "PRODUCT_MESH_OUTPUT"
    assert skin["parameters"]["legacy_stage15_dependency"] is False
    assert skin["parameters"]["model_release_admission_owner"] == "PLATFORM_MODEL_REQUIREMENTS"
    assert skin["parameters"]["operational_model_state_embedded_in_source"] is False


def test_skin_adds_mira_to_release_model_envelope():
    rows = {row["family"]: row["architecture_id"] for row in product_model_requirements()}
    assert rows["MIRA"] == "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"
    assert set(rows) == {"AXIS", "IRIS", "MIRA", "TESSA"}
