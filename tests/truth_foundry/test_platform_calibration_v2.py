import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import released_execution_plan
from tools.platform_release_snapshot import snapshot


ROOT = Path(__file__).resolve().parents[2]


def test_v2_measurement_target_closes_source_without_entering_training():
    load = lambda name: json.loads((ROOT / 'canonical' / name).read_text())
    contract = load('FIT8_TRUTH_CALIBRATION_CAMPAIGN_V2.json')
    plan = load('TRUTH_CALIBRATION_EXECUTION_PLAN_V2.json')
    request = load('TRUTH_CALIBRATION_OPERATOR_REQUEST_V2.json')
    old = load('FIT8_TRUTH_CALIBRATION_CAMPAIGN_V1.json')
    assert contract['cohort_subject_keys'] == old['cohort_subject_keys']
    assert contract['regression_subject_keys'] == old['regression_subject_keys']
    assert plan['stage_count'] == 9
    assert not any(stage['policy']['product_pass_authority'] for stage in plan['stages'])
    target = contract['required_target_stage_id']
    assert target == '47_SOURCE_RESPONSE_CALIBRATION'
    stages = {row['id']: row for row in plan['stages']}
    ancestors = set()
    def visit(stage):
        if stage not in ancestors:
            ancestors.add(stage)
            for parent in stages[stage]['depends_on']:
                visit(parent)
    visit(target)
    assert '01_SOURCE_BYTES_SEALED' in ancestors and '08_NORMALIZATION_DOMAIN_QUALIFIED' in ancestors
    assert '09_IRIS_FIT_PREREGISTERED' not in ancestors
    assert '46_PRODUCT_CLOSURE_SEAL' not in ancestors
    assert stages[target]['policy']['product_pass_authority'] is False
    assert request['training_allowed'] is False and request['native_baseline_authority_for_this_rest'] is False
    assert stages[target]['adapter'].endswith('truth_foundry_v2:measure_teacher_scene_stage')
    release = snapshot(plan, name='observation-rest-v2-source-test', purpose='RESEARCH', created_by='source-test')
    base = mainline.load_json(mainline.PLAN_PATH)
    restored = released_execution_plan(base, release['manifest']['dag'], mode='RESEARCH')
    assert mainline.validate_plan(restored, require_product_pass_authority=False) == mainline.validate_plan(plan, require_product_pass_authority=False)


def test_v2_preserves_the_calibration_ladder_and_root_control():
    a = json.loads((ROOT / 'canonical/TEACHER_KINEMATIC_CALIBRATION_POLICY_V1.json').read_text())
    b = json.loads((ROOT / 'canonical/TEACHER_KINEMATIC_CALIBRATION_POLICY_V2.json').read_text())
    assert b['rest_contract'] == 'EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY'
    for key in ['angle_amplitudes_radians', 'root_translation_source_units', 'signs', 'action_measure']:
        assert a[key] == b[key]
    assert b['training_allowed'] is False and b['qualification_minted'] is False
