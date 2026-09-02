from pathlib import Path
from experiments.single_family_e2e_v1.inference_firewall_v1 import assert_inference_callable_firewall_v1, assert_source_path_firewall_v1
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import run_complete_e2e_v1

def test_complete_e2e_real_compiler_proof_runtime(tmp_path):
    assert_inference_callable_firewall_v1(run_complete_e2e_v1)
    assert_source_path_firewall_v1(Path(__file__).with_name('run_complete_e2e_v1.py'))
    out=run_complete_e2e_v1(tmp_path/'bundle')
    assert out['status']=='PASS_COMPLETE_SYNTHETIC_E2E_V1'
    assert out['proof'].overall_status=='PASS' and out['runtime_report']['status']=='PASS_REFERENCE_V4_CONSUMPTION'
    assert out['bundle_manifest']['direction_count']==8 and out['truth_paths_consumed'] is False and out['scientific_fit_steps']==0
    assert (tmp_path/'bundle'/'bundle_manifest.json').is_file()
