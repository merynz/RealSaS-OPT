#!/usr/bin/env python3
"""Read-only campaign observation and Registry-bound CAS evidence export.

This program never submits, retries, cancels, qualifies or promotes an Attempt.
A bounded observation timeout leaves the original campaign running in Platform.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import time

from tools.platform_cohort_bootstrap_v1 import api
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256

STAGE='47_SOURCE_RESPONSE_CALIBRATION'
TERMINAL={'FAILED','FAIL','ABSTAIN','CANCELED','CANCELLED','ABORTED','COMPLETED','PASS'}


def require(value,code):
    if not value:raise ValueError(code)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def verified_cas(root,ref):
    key=PurePosixPath(ref['storage_key'])
    require(not key.is_absolute() and '..' not in key.parts,'TRUTH_OBSERVE_CAS_PATH_ESCAPE')
    path=(Path(root)/key).resolve()
    require(path.is_relative_to(Path(root).resolve()),'TRUTH_OBSERVE_CAS_PATH_ESCAPE')
    require(path.is_file() and path.stat().st_size==ref['size_bytes']
            and sha256(path)==ref['content_sha256'],'TRUTH_OBSERVE_CAS_BYTES_DRIFT')
    return path


def export_member(member, *, api_root, artifact_root, output):
    # Select the exact attempt binding, never a latest-subject artifact alias.
    records=api(api_root,'GET',f"/v1/subjects/{member['subject_id']}/artifacts?artifact_type=RealSaS.StageResultManifest&schema_version=v1&limit=500")
    rows=[r for r in records if r['source_kind']=='ATTEMPT' and r['source_id']==member['attempt_id'] and r['role']=='stage:'+STAGE]
    require(len(rows)<=1,'TRUTH_OBSERVE_DUPLICATE_ATTEMPT_STAGE_BINDING')
    if not rows:return None
    binding=rows[0];record=api(api_root,'GET','/v1/artifacts/'+binding['artifact_id'])
    require(record['id']==binding['artifact_id'] and record['content_sha256']==binding['content_sha256']
            and record['producer_contract']==STAGE,'TRUTH_OBSERVE_REGISTRY_BINDING_DRIFT')
    manifest_path=verified_cas(artifact_root,record);manifest=json.loads(manifest_path.read_text())
    require(manifest['schema']=='RealSaS.StageResultManifest.v1' and manifest['stage_id']==STAGE
            and manifest['compiler_status']=='PASS','TRUTH_OBSERVE_STAGE_MANIFEST_DRIFT')
    target=Path(output)/member['attempt_id'];target.mkdir(parents=True,exist_ok=True)
    write_json(target/'REGISTRY_BINDING.json',{'binding':binding,'artifact':record})
    shutil.copyfile(manifest_path,target/'STAGE_RESULT.json')
    exported=[]
    for ref in manifest['outputs']:
        relative=PurePosixPath(ref['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==ref['relative_path'],
                'TRUTH_OBSERVE_OUTPUT_PATH_ESCAPE')
        source=verified_cas(artifact_root,ref);destination=target/'outputs'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+'.partial')
        shutil.copyfile(source,temporary)
        require(sha256(temporary)==ref['content_sha256'],'TRUTH_OBSERVE_COPY_BYTES_DRIFT')
        temporary.replace(destination)
        exported.append({'path':str(destination),'sha256':ref['content_sha256'],'size_bytes':ref['size_bytes'],
                         'schema':ref['payload_schema'],'authority_class':ref['authority_class']})
    return {'attempt_id':member['attempt_id'],'subject_id':member['subject_id'],'role':member['role'],
            'registry_stage_artifact_id':binding['artifact_id'],'outputs':exported,
            'measurement_committed':True,'truth_qualification_inferred':False}


def observe(preparation, *, api_root, artifact_root, output, maximum_seconds=900, interval_seconds=30):
    require(preparation['schema']=='RealSaS.TeacherCalibrationCampaignPreparation.v1','TRUTH_OBSERVE_PREPARATION_SCHEMA')
    require(0<interval_seconds<=60 and maximum_seconds>=0,'TRUTH_OBSERVE_POLL_BOUNDS')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    initial={r['attempt_id']:r for r in preparation['members']};exports={};started=time.monotonic()
    while True:
        campaign=api(api_root,'GET','/v1/research/campaigns/'+preparation['campaign_id'])
        require({r['attempt_id'] for r in campaign['members']}==set(initial),'TRUTH_OBSERVE_CAMPAIGN_MEMBERS_DRIFT')
        states=[];errors=[]
        for member in campaign['members']:
            saved=initial[member['attempt_id']]
            require(all(member[k]==saved[k] for k in ('subject_id','role')),'TRUTH_OBSERVE_MEMBER_SCOPE_DRIFT')
            attempt=api(api_root,'GET','/v1/attempts/'+member['attempt_id'])
            require(attempt['attempt_id']==member['attempt_id'],'TRUTH_OBSERVE_ATTEMPT_ID_DRIFT')
            write_json(output/member['attempt_id']/'ATTEMPT.json',attempt)
            states.append({'attempt_id':member['attempt_id'],'role':member['role'],'state':attempt['state'],
                           'last_event':attempt['events'][-1] if attempt['events'] else None})
            if member['attempt_id'] not in exports:
                try:
                    exported=export_member(member,api_root=api_root,artifact_root=artifact_root,output=output)
                    if exported is not None:exports[member['attempt_id']]=exported
                except (OSError,ValueError,KeyError) as error:
                    errors.append({'attempt_id':member['attempt_id'],'error':str(error)})
        complete=len(exports)==len(initial)
        elapsed=time.monotonic()-started
        stop=complete or bool(errors) or any(s['state'] in TERMINAL for s in states) or elapsed>=maximum_seconds
        result={'schema':'RealSaS.TeacherCalibrationObservation.v1','campaign_id':preparation['campaign_id'],
                'experiment_id':preparation['experiment_id'],'source_code_sha':preparation['source_code_sha'],
                'campaign':campaign,'attempts':states,'exports':list(exports.values()),'errors':errors,
                'all_measurements_committed_and_exported':complete,'observation_elapsed_seconds':elapsed,
                'observation_timed_out':not complete and not errors and elapsed>=maximum_seconds,
                'execution_resubmitted':False,'training_submitted':False,'truth_qualification_inferred':False}
        write_json(output/'OBSERVATION.json',result)
        print(f'[TRUTH_OBSERVE] committed={len(exports)}/{len(initial)} elapsed_s={elapsed:.1f} errors={len(errors)}',flush=True)
        if stop:return result
        time.sleep(min(interval_seconds,max(0,maximum_seconds-elapsed)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','artifact-root','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--api-root',default='http://127.0.0.1:8080');p.add_argument('--maximum-seconds',type=float,default=900)
    a=p.parse_args();result=observe(json.loads(Path(a.preparation).read_text()),api_root=a.api_root,
        artifact_root=a.artifact_root,output=a.output,maximum_seconds=a.maximum_seconds)
    if not result['all_measurements_committed_and_exported']:
        raise SystemExit('TRUTH_CALIBRATION_NOT_YET_EXPORTED__ORIGINAL_PLATFORM_CAMPAIGN_PRESERVED')
