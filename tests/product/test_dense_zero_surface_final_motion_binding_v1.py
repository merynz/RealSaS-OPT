import inspect

import experiments.mage_demo_fit1_v5_p1.materialize_fit2_dense_zero_surface_body_v2_historical_motion as body_v2
import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v4_motion_bound as product_v4
import experiments.mage_demo_fit1_v5_p1.seal_fit2_dense_zero_surface_body_binding_v2_motion_bound as derivation_v2


def test_dense_body_admission_uses_final_historical_motion_builder():
    source = inspect.getsource(body_v2)
    assert "build_mage_historical_phase_motion" in source
    assert "compile_motion_quality" in source
    assert 'MOTION_AUTHORITY = "HISTORICAL_PHASE_QUALITY_MOTION_STATE"' in source
    assert '"motion_state_hash": motion["motion_state_hash"]' in source
    assert '"historical_motion_engine_recovered": True' in source


def test_dense_derivation_binds_each_dynamic_witness_to_one_motion_hash():
    source = inspect.getsource(derivation_v2)
    assert "DENSE_DERIVATION_V2_DYNAMIC_MOTION_HASH_DRIFT" in source
    assert "dynamic_witness_file_sha256" in source
    assert "topology_W_and_motion_share_one_persisted_authority_chain" in source
    assert derivation_v2.EXPECTED_MOTION_AUTHORITY == "HISTORICAL_PHASE_QUALITY_MOTION_STATE"


def test_final_product_refuses_body_or_derivation_motion_hash_drift():
    source = inspect.getsource(product_v4)
    assert "FIT2_DENSE_V4_BODY_FINAL_MOTION_HASH_DRIFT" in source
    assert "FIT2_DENSE_V4_DERIVATION_FINAL_MOTION_HASH_DRIFT" in source
    assert "FIT2_DENSE_V4_DYNAMIC_WITNESS_MOTION_HASH_DRIFT" in source
    assert '"dynamic_face_admission_motion_state_hash": motion_hash' in source
    assert '"product_pass_claimed": False' in source


def test_final_product_uses_file_backed_v2_derivation_and_no_p1_topology():
    source = inspect.getsource(product_v4)
    assert "DERIVATION_FILE = derivation_v2.DERIVATION_FILE" in source
    assert "dense_binding_derivation_v2_manifest_sha256" in source
    assert '"P1_or_P1Q_topology_used": False' in source
    assert "persist=False" in source
