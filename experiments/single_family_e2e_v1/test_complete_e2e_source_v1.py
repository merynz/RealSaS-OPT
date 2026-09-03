from pathlib import Path

import pytest

from compiler.realsas_compiler_core.proof_engine import evaluate_product_proof
from compiler.realsas_compiler_core.types import QualificationError
from experiments.single_family_e2e_v1.inference_firewall_v1 import assert_inference_callable_firewall_v1, assert_source_path_firewall_v1
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import run_complete_e2e_v1


def test_complete_e2e_real_compiler_proof_runtime(tmp_path):
    assert_inference_callable_firewall_v1(run_complete_e2e_v1)
    assert_source_path_firewall_v1(Path(__file__).with_name('run_complete_e2e_v1.py'))
    out = run_complete_e2e_v1(tmp_path / 'bundle')
    assert out['status'] == 'PASS_COMPLETE_SYNTHETIC_E2E_V1'
    assert out['proof'].overall_status == 'PASS' and out['runtime_report']['status'] == 'PASS_REFERENCE_V4_CONSUMPTION'
    assert out['bundle_manifest']['direction_count'] == 8 and out['truth_paths_consumed'] is False and out['scientific_fit_steps'] == 0
    assert len(out['directional_binding_hash']) == 64
    assert len(out['qualified_motion_provider_hash']) == 64
    assert len(out['motion_bake_hash']) == 64
    assert out['synthetic_binding_policy_only'] is True
    motion_report = next(r for r in out['proof'].domain_reports if r.proof_domain == 'MOTION')
    assert motion_report.status == 'PASS'
    assert motion_report.metadata['directional_joint_view_binding_required'] is True
    assert motion_report.metadata['arbitrary_motion_provider_callable_forbidden'] is True
    assert motion_report.metadata['directional_binding_set_hash'] == out['directional_binding_hash']
    assert motion_report.metadata['qualified_motion_provider_hash'] == out['qualified_motion_provider_hash']
    assert motion_report.metadata['qualification_owned_bake_hash'] == out['motion_bake_hash']
    assert (tmp_path / 'bundle' / 'bundle_manifest.json').is_file()

    with pytest.raises(QualificationError, match='MOTION_BAKE_PROVIDER_NOT_QUALIFIED'):
        evaluate_product_proof(out['product'], motion_bake_provider=lambda *_args, **_kwargs: None)
