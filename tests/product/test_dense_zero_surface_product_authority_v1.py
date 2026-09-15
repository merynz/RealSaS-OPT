from pathlib import Path
import inspect

import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v2 as dense_v2
import experiments.mage_demo_fit1_v5_p1.materialize_product_state_fit2_dense_v3_evidence_bound as dense_v3
import experiments.mage_demo_fit1_v5_p1.seal_fit2_dense_zero_surface_body_binding_v1 as derivation


def test_dense_product_final_authority_is_file_backed_derivation():
    source = inspect.getsource(dense_v3)
    assert "FIT2_DENSE_ZERO_SURFACE_BODY_BINDING_DERIVATION_MANIFEST.json" in source
    assert "direct_binding_manifest_sha256=derivation_sha" in source
    assert "binding_derivation_file_backed" in source
    assert "P1_or_P1Q_topology_used\": False" in source
    assert "persist=False" in source


def test_dense_derivation_forbids_historical_and_new_adjacency_authority():
    source = inspect.getsource(derivation)
    assert "PASS__DENSE_TO_GSA_EXACT_COMPACTION_ROWS_BOUND_TO_FRESH_FIT2_W" in source
    assert '"historical_weight_transfer_used": False' in source
    assert '"new_adjacency_created": False' in source
    assert '"P1_or_P1Q_topology_used": False' in source
    assert '"topology_and_W_share_exact_compaction_address": True' in source


def test_dense_v3_cli_constructs_its_own_argument_parser():
    source = inspect.getsource(dense_v3.parse_args)
    assert "argparse.ArgumentParser" in source
    assert "dense_v2.parse_args()" not in source


def test_dense_v2_remains_non_product_pass_intermediate():
    source = inspect.getsource(dense_v2)
    assert '"product_pass_claimed":False' in source or '"product_pass_claimed": False' in source
