#!/usr/bin/env python3
"""Prepare/submit the released same-cohort teacher calibration through Platform.

This coordinator creates an Experiment, released campaign and immutable input
bindings before measurement. It never launches Blender or training directly.
The legacy learned stage stays WAIT_EXTERNAL; no learned evidence is supplied.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from tools import platform_cohort_bootstrap_v1 as bootstrap
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256, _bound_path


def _require(value,code):
    if not value:raise ValueError(code)


def immutable_json(root,name,value):
    raw=bootstrap.json_bytes(value);digest=bootstrap.digest_bytes(raw)
    path=Path(root)/digest/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():_require(path.read_bytes()==raw,'TRUTH_OPERATOR_IMMUTABLE_BYTES_DRIFT')
    else:
        with path.open('xb') as f:f.write(raw)
    return {'path':str(path.resolve()),'sha256':digest}


def prepare(request, *, repo, platform_root, authority_root, expected_sha, output, api_root, rclone_remote, submit):
    _require(request.get('schema')=='RealSaS.TeacherCalibrationOperatorRequest.v1'
             and request.get('training_allowed') is False and request.get('new_calibration_actions_allowed') is True,
             'TRUTH_OPERATOR_SCOPE_REQUIRED')
    repo,platform,authority,output=map(lambda p:Path(p).resolve(),(repo,platform_root,authority_root,output))
    _require(json.loads((platform/'deployment.json').read_text())['code_sha']==expected_sha,'TRUTH_OPERATOR_DEPLOYED_MAIN_DRIFT')
    _require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected_sha,'TRUTH_OPERATOR_CHECKOUT_DRIFT')
    _require(subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/main'],text=True).split()[0]==expected_sha,'TRUTH_OPERATOR_REMOTE_MAIN_DRIFT')
    # A recovered cohort is input, not a mutable cache alias. Verify every byte
    # before creating new scientific execution state.
    baseline_rows=[]
    for row in request['native_baselines']:
        path=_bound_path(authority,row['relative'])
        _require(path.is_file() and sha256(path)==row['sha256'],'TRUTH_OPERATOR_NATIVE_BASELINE_BYTES_DRIFT')
        baseline_rows.append({'subject_key':row['subject_key'],'native_teacher':{'path':str(path),'sha256':row['sha256']}})
    equivalence_path=_bound_path(authority,request['source_equivalence_report']['relative'])
    _require(equivalence_path.is_file() and sha256(equivalence_path)==request['source_equivalence_report']['sha256'],
             'TRUTH_OPERATOR_EQUIVALENCE_REPORT_DRIFT')
    report=json.loads(equivalence_path.read_text())
    _require(report['status']=='EXACT_SEALED_REPLAY' and all(r['exact_arrays_equal'] for r in report['subjects']),
             'TRUTH_OPERATOR_NATIVE_REPLAY_NOT_EQUIVALENT')
    _require({r['subject_id'] for r in report['subjects']}=={r['subject_key'] for r in baseline_rows},'TRUTH_OPERATOR_BASELINE_COHORT_DRIFT')
    executable=Path(request['evaluator_executable']).expanduser().resolve()
    version=subprocess.check_output([str(executable),'--version'],text=True).splitlines()[0]
    _require(version==request['evaluator_version'],'TRUTH_OPERATOR_EVALUATOR_VERSION_DRIFT')
    binary_sha=sha256(executable)
    policy_path=_bound_path(repo,request['policy_repo_path']);policy=json.loads(policy_path.read_text())
    _require(policy['schema']=='RealSaS.TeacherKinematicCalibrationPolicy.v1','TRUTH_OPERATOR_POLICY_SCHEMA')
    durable=authority/'truth_foundry_inputs'
    policy_ref=immutable_json(durable,'POLICY.json',policy)
    baseline_ref=immutable_json(durable,'NATIVE_BASELINES.json',{'schema':'RealSaS.NativeTeacherBaselineBindings.v1','subjects':baseline_rows})
    cfg={'mode':'SOURCE_KINEMATIC_CALIBRATION_ONLY','policy':policy_ref,'native_baselines':baseline_ref,
         'source_equivalence_report':{'path':str(equivalence_path),'sha256':request['source_equivalence_report']['sha256']},
         'evaluator_binary':{'path':str(executable),'sha256':binary_sha},'evaluator_version':version,
         'maximum_seconds_per_subject':request['maximum_seconds_per_subject'],'producer_source_commit':expected_sha}
    descriptor=json.loads(_bound_path(repo,request['input_descriptor_repo_path']).read_text())
    descriptor['campaign_contract']=request['campaign_contract_repo_path'];descriptor['campaign_name']=request['campaign_name']
    descriptor['actor']='truth-foundry-calibration-coordinator'
    descriptor['additional_manifest_sections']={'truth_foundry':cfg}
    stage_parameters=json.loads(_bound_path(repo,request['stage_parameters_repo_path']).read_text())
    _require(stage_parameters.get('schema')=='RealSaS.StageReleaseParameters.v1','TRUTH_OPERATOR_STAGE_PARAMETER_SCHEMA')
    descriptor['additional_stage_parameters']=stage_parameters['stages']
    descriptor['regression_additional_manifest_sections']={'truth_foundry':{
        'mode':'SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS','new_truth_claimed':False}}
    descriptor_ref=immutable_json(durable,'BOOTSTRAP_DESCRIPTOR.json',descriptor)
    output.mkdir(parents=True,exist_ok=True)
    bootstrap_path=output/'CAMPAIGN_BOOTSTRAP.json'
    experiment_request={'slug':request['experiment_slug'],'name':request['experiment_name'],
        'research_question':request['research_question'],'hypothesis':request['hypothesis'],'initial_state':'PREREGISTERED',
        'created_by':descriptor['actor'],'metadata':{'scope':'SOURCE_KINEMATIC_AND_SAMPLER_INPUT_CALIBRATION_ONLY',
             'policy_sha256':policy_ref['sha256'],'native_baseline_index_sha256':baseline_ref['sha256'],
             'source_equivalence_report_sha256':request['source_equivalence_report']['sha256'],
             'training_allowed':False,'new_truth_qualification_from_stage_success':False}}
    experiment=bootstrap.api(api_root,'POST','/v1/experiments',experiment_request)['experiment']
    # Keep noisy bootstrap detail in its exact log; the coordinator reports only
    # bound identities and progress to the operator.
    with (output/'bootstrap.log').open('w') as log, contextlib.redirect_stdout(log):
        bootstrap.main(SimpleNamespace(repo=repo,descriptor=Path(descriptor_ref['path']),platform_root=platform,
            authority_root=authority,api=api_root,rclone_remote=rclone_remote,out=bootstrap_path,no_submit=True))
    prepared=json.loads(bootstrap_path.read_text());campaign_id=prepared['campaign_id']
    status=bootstrap.api(api_root,'GET','/v1/research/campaigns/'+campaign_id)
    bootstrap.api(api_root,'POST','/v1/experiments/link',{'experiment_id':experiment['id'],'target_kind':'CAMPAIGN',
                    'target_id':campaign_id,'actor':descriptor['actor']})
    for member in status['members']:
        bootstrap.api(api_root,'POST','/v1/experiments/link',{'experiment_id':experiment['id'],'target_kind':'ATTEMPT',
                      'target_id':member['attempt_id'],'actor':descriptor['actor']})
    result={'schema':'RealSaS.TeacherCalibrationCampaignPreparation.v1','source_code_sha':expected_sha,
            'experiment_id':experiment['id'],'campaign_id':campaign_id,'engine_release_id':prepared['release_id'],
            'evaluator_binary_sha256':binary_sha,'policy':policy_ref,'baseline_index':baseline_ref,
            'bootstrap_receipt':{'path':str(bootstrap_path),'sha256':sha256(bootstrap_path)},
            'members':status['members'],'training_submitted':False,'truth_qualification_minted':False,
            'measurement_submitted':False,'campaign_full_closure_claimed':False}
    result_path=output/'PREPARATION.json';result_path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    if submit:
        _require(experiment['lifecycle_state'] in {'PREREGISTERED','RUNNING'},'TRUTH_OPERATOR_EXPERIMENT_NOT_RUNNABLE')
        if experiment['lifecycle_state']=='PREREGISTERED':
            bootstrap.api(api_root,'POST','/v1/experiments/transition',{'experiment_id':experiment['id'],'to_state':'RUNNING','actor':descriptor['actor']})
        if status['state']=='READY':
            receipt=bootstrap.api(api_root,'POST','/v1/research/campaigns/run',{'campaign_id':campaign_id,
                        'members':prepared['run_bindings'],'pipeline_plan_sha256':prepared['pipeline_plan_sha256'],'requested_by':descriptor['actor']})
            result['campaign_run']=receipt
        else:
            _require(status['state']=='RUNNING','TRUTH_OPERATOR_CAMPAIGN_NEEDS_EXPLICIT_RECOVERY')
            result['existing_campaign_observed']=status
        result['measurement_submitted']=True
        result_path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('experiment_id','campaign_id','engine_release_id','measurement_submitted','training_submitted')},sort_keys=True),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','repo','platform-root','authority-root','expected-main-sha','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--api-root',default='http://127.0.0.1:8080');p.add_argument('--rclone-remote',default='gdrive')
    p.add_argument('--submit',action='store_true');a=p.parse_args()
    prepare(json.loads(Path(a.request).read_text()),repo=a.repo,platform_root=a.platform_root,authority_root=a.authority_root,
            expected_sha=a.expected_main_sha,output=a.output,api_root=a.api_root,rclone_remote=a.rclone_remote,submit=a.submit)
