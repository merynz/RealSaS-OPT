#!/usr/bin/env python3
"""Read-only export of exact Platform-bound inputs to an external compute worker.

The capsule preserves absolute file addresses so embedded JSON hashes do not need
rewriting. It contains no credentials, database, mutable ledger or authorization.
Notebook execution returns raw evidence; it never writes the Platform Registry.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

from models.materialization_contract_v1 import require, sha256
from tools.platform_cohort_bootstrap_v1 import api

SCHEMA='RealSaS.Corpus370ComputeCapsule.v1'
PLAN='canonical/CORPUS370_FAMILY_TRUTH_EXECUTION_PLAN_V2.json'
TARGET='49_SUBJECT_TRUTH_FAMILIES'


def references(value):
    if isinstance(value,list):
        for item in value:yield from references(item)
    elif isinstance(value,dict):
        for key,path in value.items():
            digest=(value.get('expected_sha256') or value.get('sha256')) if key=='path' else value.get(key[:-5]+'_sha256') if key.endswith('_path') else None
            if isinstance(path,str) and re.fullmatch('[0-9a-f]{64}',str(digest or '')):
                yield path,digest
        for item in value.values():yield from references(item)


def build(preparation_path, expected_preparation_sha, *, authority_root, platform_root, repo, output, api_root):
    require(sha256(preparation_path)==expected_preparation_sha,'COMPUTE_PREPARATION_DRIFT')
    preparation=json.loads(preparation_path.read_text())
    bootstrap_ref=preparation['bootstrap_receipt'];bootstrap_path=Path(bootstrap_ref['path'])
    require(sha256(bootstrap_path)==bootstrap_ref['sha256'],'COMPUTE_BOOTSTRAP_DRIFT')
    bootstrap=json.loads(bootstrap_path.read_text())
    campaign=api(api_root,'GET','/v1/research/campaigns/'+preparation['campaign_id'])
    members={m['subject_id']:m for m in campaign['members']}
    submitted={r['SubjectID']:r['AttemptID'] for r in preparation['campaign_run']['receipts']}
    require(len(members)==preparation['member_count']==294,'COMPUTE_MEMBER_SCOPE')
    require({k:v['attempt_id'] for k,v in members.items()}==submitted,'COMPUTE_ATTEMPT_SET_DRIFT')
    require(preparation['candidate_subject_count']==370 and preparation['excluded_subject_count']==76,'COMPUTE_UNIVERSE_SCOPE')
    code_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    allowed_roots=[str(Path(p).resolve()) for p in (authority_root,platform_root,repo)]
    files={};objects={};provenance=[];runtimes=set();subjects=[]
    output.mkdir(parents=True,exist_ok=True)
    archive_path=output/'CORPUS370_COMPUTE_CAPSULE_V1.zip'
    temporary=archive_path.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        def add(path, expected=None, *, recurse=True):
            path=Path(os.path.abspath(path))
            require(any(path.is_relative_to(Path(root)) for root in allowed_roots),'COMPUTE_INPUT_PATH_OUTSIDE_ROOTS:'+str(path))
            prior=files.get(str(path))
            if prior:
                require(expected is None or prior['sha256']==expected,'COMPUTE_REFERENCE_CONFLICT')
                return prior
            require(path.is_file(),'COMPUTE_INPUT_MISSING:'+str(path))
            digest=sha256(path)
            require(expected is None or digest==expected,'COMPUTE_INPUT_HASH_DRIFT:'+str(path))
            row={'path':str(path),'sha256':digest,'size_bytes':path.stat().st_size,
                 'object':'objects/'+digest,'executable':bool(path.stat().st_mode&0o111)}
            files[str(path)]=row
            if digest not in objects:
                archive.write(path,row['object']);objects[digest]=row['size_bytes']
            if recurse and path.suffix.lower()=='.json':walk(json.loads(path.read_text()),provenance_only=True)
            return row
        def walk(value, *, provenance_only=False):
            for raw,digest in references(value):
                path=Path(raw)
                # Some source snapshots retain historical acquisition addresses.
                # Keep them as provenance, never silently substitute another file.
                local=path.is_absolute() and any(path.is_relative_to(Path(root)) for root in allowed_roots)
                if provenance_only and (not local or not path.is_file()):
                    provenance.append({'path':raw,'sha256':digest,'reason':'HISTORICAL_REFERENCE_NOT_EXECUTION_INPUT'})
                else:add(path,digest)
        for number,binding in enumerate(bootstrap['run_bindings'],1):
            member=members.get(binding['subject_id'])
            require(member is not None,'COMPUTE_UNKNOWN_MEMBER')
            manifest_path=Path(binding['run_manifest_path'])
            manifest=json.loads(manifest_path.read_text())
            require(manifest['run_id']==Path(binding['run_ledger_path']).parent.name,'COMPUTE_MANIFEST_RUN_DRIFT')
            executable=Path(manifest['truth_foundry']['evaluator_binary']['path'])
            runtime_root=executable.parent
            if not any(executable.is_relative_to(Path(root)) for root in allowed_roots):
                from tools.observation.pinned_evaluator_runtime_v1 import PATCHED_BINARY_SHA
                require(executable.is_relative_to(Path('/home/monster/.cache/realsas/blender'))
                        and executable.name=='blender-glibc239' and runtime_root.name=='numeric-runtime-239'
                        and sha256(executable)==PATCHED_BINARY_SHA,'COMPUTE_PINNED_EVALUATOR_LOCATION')
                allowed_roots.append(str(runtime_root))
            walk(manifest)
            if str(runtime_root) not in runtimes:
                runtimes.add(str(runtime_root))
                # The pinned install links its4.0 resources and several libraries
                # into the official distribution. Dereference their bytes while
                # retaining each executable-relative address in the capsule.
                for directory,dirs,names in os.walk(runtime_root,followlinks=True):
                    dirs.sort();names.sort()
                    for name in names:
                        path=Path(directory)/name
                        require(path.resolve().is_relative_to(runtime_root.parent.resolve()),'COMPUTE_RUNTIME_LINK_ESCAPE')
                        if path.is_file():add(path,recurse=False)
            subjects.append({'subject_id':manifest['subject_id'],'platform_subject_id':binding['subject_id'],
                'subject_input_id':binding['subject_input_id'],'parent_attempt_id':member['attempt_id'],
                'parent_state_at_export':member['final_state'],'manifest':add(manifest_path,recurse=False)})
            if number%10==0 or number==len(members):
                print(json.dumps({'event':'COMPUTE_INPUT_EXPORT','subjects':number,'total':len(members),
                                  'unique_payload_bytes':sum(objects.values())}),flush=True)
        plan_path=repo/PLAN
        seal={'schema':SCHEMA,'status':'SEALED_INPUTS__EXTERNAL_COMPUTE_ONLY',
            'code_sha':code_sha,'parent_code_sha':preparation['source_code_sha'],
            'parent_campaign_id':preparation['campaign_id'],'parent_engine_release_id':preparation['engine_release_id'],
            'preparation_sha256':expected_preparation_sha,'plan':add(plan_path,recurse=False),
            'target_stage_id':TARGET,'candidate_subject_count':370,'worker_subject_count':len(subjects),
            'excluded_subject_count':76,'excluded_subjects':bootstrap['excluded_subjects'],
            'subjects':subjects,'files':sorted(files.values(),key=lambda r:r['path']),
            'allowed_original_roots':allowed_roots,'historical_references':provenance,
            'resource_policy':{'concurrent_subjects':1,'native_threads_per_process':1,'recommended_ram_gib':24},
            'platform_registration_required':True,'qualification_minted':False,'training_authorized':False}
        raw=(json.dumps(seal,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        archive.writestr('COMPUTE_CAPSULE.json',raw)
    temporary.replace(archive_path)
    seal_path=output/'COMPUTE_CAPSULE.json';seal_path.write_bytes(raw)
    receipt={'schema':'RealSaS.Corpus370ComputeCapsuleReceipt.v1','archive_sha256':sha256(archive_path),
             'archive_size_bytes':archive_path.stat().st_size,'seal_sha256':sha256(seal_path),
             'code_sha':code_sha,'worker_subject_count':len(subjects),'candidate_subject_count':370,
             'training_authorized':False}
    (output/'COMPUTE_CAPSULE_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','authority-root','platform-root','repo','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preparation-sha256',required=True);p.add_argument('--api',default='http://127.0.0.1:8080')
    a=p.parse_args();print(json.dumps(build(a.preparation,a.preparation_sha256,authority_root=a.authority_root,
        platform_root=a.platform_root,repo=a.repo,output=a.output,api_root=a.api)))


if __name__=='__main__':main()
