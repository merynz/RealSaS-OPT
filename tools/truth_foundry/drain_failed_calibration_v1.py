#!/usr/bin/env python3
"""Drain only a pinned, idle regression after its calibration cohort failed.

This is the explicit operator drain required by DEVELOPER_HOST.md. Temporal
records termination of the exact idle run; the existing Go reconciler owns the
Attempt transition. No SQL state write, evidence substitution or guard bypass.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from tools.platform_cohort_bootstrap_v1 import api
from tools.platform_deploy import runtime_environment
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256,_bound_path
from tools.truth_foundry.observe_calibration_v1 import write_json


def require(value,code):
    if not value:raise ValueError(code)


async def drain(request,preparation, *, client,api_root,output,reconcile):
    require(request['schema']=='RealSaS.FailedCalibrationDrainRequest.v1'
        and request['terminate_idle_regression_allowed'] is True and request['new_measurements_allowed'] is False
        and request['training_allowed'] is False,'CALIBRATION_DRAIN_SCOPE_REQUIRED')
    require(preparation['schema']=='RealSaS.TeacherCalibrationCampaignPreparation.v1'
        and preparation['campaign_id']==request['campaign_id'],'CALIBRATION_DRAIN_PREPARATION_SCOPE_DRIFT')
    campaign=api(api_root,'GET','/v1/research/campaigns/'+request['campaign_id'])
    original={(r['subject_id'],r['attempt_id'],r['role']) for r in preparation['members']}
    require({(r['subject_id'],r['attempt_id'],r['role']) for r in campaign['members']}==original,
            'CALIBRATION_DRAIN_CAMPAIGN_MEMBER_DRIFT')
    require(campaign['candidate_engine_release_id']==preparation['engine_release_id'],
            'CALIBRATION_DRAIN_RELEASE_DRIFT')
    cohort=[r for r in campaign['members'] if r['role']=='COHORT']
    require(cohort and all(r['final_state']=='FAILED' for r in cohort),'CALIBRATION_DRAIN_COHORT_NOT_TERMINAL_FAILED')
    regression=[r for r in campaign['members'] if r['role']=='REGRESSION' and r['attempt_id']==request['attempt_id']]
    require(len(regression)==1,'CALIBRATION_DRAIN_EXACT_REGRESSION_REQUIRED')
    attempt=api(api_root,'GET','/v1/attempts/'+request['attempt_id'])
    require(attempt['attempt_id']==request['attempt_id'],'CALIBRATION_DRAIN_ATTEMPT_DRIFT')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={'schema':'RealSaS.FailedCalibrationDrainReceipt.v1','campaign_id':request['campaign_id'],
        'attempt_id':request['attempt_id'],'reason':request['reason'],'actor':request['actor'],
        'termination_requested':False,'new_execution_submitted':False,'training_submitted':False,
        'qualification_minted':False,'before':attempt}
    receipt=output/'DRAIN_RECEIPT.json';write_json(receipt,result)
    if attempt['state']!='OPEN':
        require(attempt['state'] in {'FAILED','COMPLETED'},'CALIBRATION_DRAIN_UNKNOWN_TERMINAL_STATE')
        result['already_terminal']=True;write_json(receipt,result);return result
    last=attempt['events'][-1] if attempt['events'] else {}
    require(last.get('type')=='STAGE_WAITING_EXTERNAL_EVIDENCE'
        and last.get('payload',{}).get('stage_id')==request['expected_wait_stage_id'],
        'CALIBRATION_DRAIN_NOT_AT_DECLARED_WAIT')
    bindings=[r for r in preparation['campaign_run']['receipts'] if r['AttemptID']==request['attempt_id']]
    require(len(bindings)==1 and bindings[0]['SubjectID']==regression[0]['subject_id'],
            'CALIBRATION_DRAIN_COMMAND_BINDING_DRIFT')
    workflow_id='realsas:compile:'+bindings[0]['CommandID']
    description=await client.get_workflow_handle(workflow_id).describe()
    require(description.id==workflow_id and description.workflow_type=='realsas.compile.v1'
            and bool(description.run_id),'CALIBRATION_DRAIN_TEMPORAL_SCOPE_DRIFT')
    status=description.status.name
    require(status in {'RUNNING','FAILED','CANCELED','TERMINATED','TIMED_OUT'},'CALIBRATION_DRAIN_WORKFLOW_STATE_UNSUPPORTED')
    result.update(workflow_id=workflow_id,workflow_run_id=description.run_id,workflow_status_before=status)
    write_json(receipt,result)
    handle=client.get_workflow_handle(workflow_id,run_id=description.run_id)
    if status=='RUNNING':
        require(not description.raw_description.pending_activities,'CALIBRATION_DRAIN_ACTIVE_ACTIVITY_FORBIDDEN')
        # Re-read the Attempt immediately before the scoped operator operation.
        require(api(api_root,'GET','/v1/attempts/'+request['attempt_id'])==attempt,
                'CALIBRATION_DRAIN_ATTEMPT_CHANGED_DURING_PREFLIGHT')
        await handle.terminate(reason=request['actor']+': '+request['reason'])
        result['termination_requested']=True;write_json(receipt,result)
    closed=await handle.describe()
    require(closed.status.name in {'FAILED','CANCELED','TERMINATED','TIMED_OUT'} and closed.close_time is not None,
            'CALIBRATION_DRAIN_TERMINAL_WORKFLOW_PROOF_REQUIRED')
    result.update(workflow_status_after=closed.status.name,workflow_closed_at=closed.close_time.isoformat())
    write_json(receipt,result)
    reconcile()  # Existing Go owner checks the single command/run/close-time proof.
    started=time.monotonic()
    while True:
        after=api(api_root,'GET','/v1/attempts/'+request['attempt_id'])
        if after['state']!='OPEN':break
        require(time.monotonic()-started<30,'CALIBRATION_DRAIN_GO_RECONCILIATION_PENDING')
        await asyncio.sleep(1)
    require(after['state']=='FAILED' and any(e['type']=='ATTEMPT_WORKFLOW_FAILED' for e in after['events']),
            'CALIBRATION_DRAIN_GO_TERMINAL_PROOF_REQUIRED')
    result['after']=after;write_json(receipt,result)
    print(json.dumps({'attempt_id':request['attempt_id'],'state':after['state'],'new_execution_submitted':False}),flush=True)
    return result


async def main(args):
    from temporalio.client import Client
    request=json.loads(Path(args.request).read_text());require(request['enabled'] is True,'CALIBRATION_DRAIN_DISABLED')
    preparation_path=_bound_path(Path(args.authority_root),request['preparation_relative_path'])
    require(sha256(preparation_path)==request['preparation_sha256'],'CALIBRATION_DRAIN_PREPARATION_BYTES_DRIFT')
    preparation=json.loads(preparation_path.read_text());root=Path(args.platform_root)
    client=await Client.connect('127.0.0.1:7233',namespace='default')
    def reconcile():
        env={**os.environ,**runtime_environment(root,Path(args.authority_root),os.environ.get('LD_LIBRARY_PATH',''))}
        subprocess.run([str(root/'current/bin/realsas-control-worker'),'--reconcile-once'],env=env,check=True,timeout=30)
    await drain(request,preparation,client=client,api_root='http://127.0.0.1:8080',output=args.output,reconcile=reconcile)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','platform-root','authority-root','output'):p.add_argument('--'+name,required=True)
    asyncio.run(main(p.parse_args()))
