import inspect

from compiler.realsas_compiler_core.mesh.quality import FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
import experiments.mage_demo_fit1_v5_p1.materialize_fit2_dense_zero_surface_body_v3_product_mesh_quality as body_v3
import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v4_motion_bound as product_v4
import experiments.mage_demo_fit1_v5_p1.seal_fit2_dense_zero_surface_body_binding_v2_motion_bound as derivation_v2


def test_dense_body_v3_filters_then_remeasures_with_official_product_mesh_quality():
    source = inspect.getsource(body_v3)
    assert "FIT2_PRODUCT_MESH_QUALITY_POLICY_V1" in source
    assert "_static_quality_mask" in source
    assert "mesh_raster_quality_report" in source
    assert "raster_quality_gate_failures" in source
    assert '"static_product_mesh_quality_gate_applied": True' in source
    assert '"static_product_mesh_quality_all_views_passed": True' in source
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    assert policy.min_raster_triangle_angle_deg == 0.25
    assert policy.max_raster_triangle_aspect_ratio == 250.0


def test_motion_bound_derivation_requires_file_backed_static_mesh_quality_witnesses():
    source = inspect.getsource(derivation_v2)
    assert "DENSE_DERIVATION_V2_STATIC_MESH_QUALITY_GATE_MISSING" in source
    assert "DENSE_DERIVATION_V2_MESH_QUALITY_FILE_SHA_DRIFT" in source
    assert "DENSE_DERIVATION_V2_MESH_QUALITY_POLICY_DRIFT" in source
    assert "product_mesh_quality_witnesses" in source
    assert "topology_W_motion_and_mesh_quality_share_one_persisted_authority_chain" in source


def test_final_product_remains_bound_to_strengthened_derivation_and_no_product_pass_shortcut():
    source = inspect.getsource(product_v4)
    assert "DERIVATION_FILE = derivation_v2.DERIVATION_FILE" in source
    assert "dense_binding_derivation_v2_manifest_sha256" in source
    assert '"P1_or_P1Q_topology_used": False' in source
    assert '"product_pass_claimed": False' in source
