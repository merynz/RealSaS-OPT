#!/usr/bin/env python3
"""External, serial, resumable compute. Registry import/qualification is separate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import subprocess
import sys
import time
import zipfile

from models.materialization_contract_v1 import require, sha256
from tools.corpus.export_corpus370_compute_capsule_v1 import SCHEMA,TARGET
from compiler.realsas_compiler_services.process_log_v1 import follow_process_log


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    staged=path.with_suffix(path.suffix+'.partial')
    staged.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    staged.replace(path)


def safe_member(name):
    p=PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name,'COMPUTE_ARCHIVE_PATH_ESCAPE')
    return p


def hydrate(archive_path, expected_sha, cache):
    require(sha256(archive_path)==expected_sha,'COMPUTE_ARCHIVE_HASH_DRIFT')
    cache.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        require(len(names)==len(set(names)),'COMPUTE_ARCHIVE_DUPLICATE')
        for entry in entries:
            safe_member(entry.filename)
            require((entry.external_attr>>16)&0o170000!=0o120000,'COMPUTE_ARCHIVE_SYMLINK')
        seal=json.loads(archive.read('COMPUTE_CAPSULE.json'))
        require(seal.get('schema')==SCHEMA and seal.get('training_authorized') is False
                and seal.get('qualification_minted') is False,'COMPUTE_CAPSULE_SCHEMA')
        expected_names={'COMPUTE_CAPSULE.json'}|{r['object'] for r in seal['files']}
        require(set(names)==expected_names,'COMPUTE_ARCHIVE_COVERAGE')
        roots=[Path(p) for p in seal['allowed_original_roots']]
        require(all(p.is_absolute() and str(p).startswith('/home/monster/') and len(p.parts)>3 for p in roots),
                'COMPUTE_APPROVED_ORIGINAL_ROOT_REQUIRED')
        for index,row in enumerate(seal['files'],1):
            destination=Path(row['path'])
            require(destination.is_absolute() and '..' not in destination.parts
                    and any(destination.is_relative_to(root) and destination.resolve().is_relative_to(root.resolve())
                            for root in roots),'COMPUTE_DESTINATION_ESCAPE')
            require(row['object']=='objects/'+row['sha256'],'COMPUTE_OBJECT_NAME_DRIFT')
            obj=cache/row['object']
            if not obj.is_file() or sha256(obj)!=row['sha256']:
                obj.parent.mkdir(parents=True,exist_ok=True)
                staged=obj.with_suffix('.partial')
                with archive.open(row['object']) as source,staged.open('wb') as target:
                    shutil.copyfileobj(source,target,length=8<<20)
                require(staged.stat().st_size==row['size_bytes'] and sha256(staged)==row['sha256'],'COMPUTE_OBJECT_BYTES_DRIFT')
                staged.replace(obj)
            if destination.exists():
                require(destination.is_file() and sha256(destination)==row['sha256'],'COMPUTE_EXISTING_INPUT_DRIFT:'+str(destination))
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(obj,destination)
            if row['executable']:destination.chmod(destination.stat().st_mode|0o111)
            if index%250==0:print(f'INPUT_VERIFY {index}/{len(seal["files"])}',flush=True)
    write(cache/'COMPUTE_CAPSULE.json',seal)
    return seal


def verify_result(receipt,seal,subject):
    require(receipt.get('schema')=='RealSaS.Corpus370ComputeSubjectReceipt.v1'
            and receipt.get('capsule_identity')==seal['identity']
            and receipt.get('subject_id')==subject['subject_id']
            and receipt.get('parent_attempt_id')==subject['parent_attempt_id']
            and receipt.get('code_sha')==seal['code_sha']
            and receipt.get('qualification_minted') is False
            and receipt.get('training_authorized') is False,'COMPUTE_RESUME_SCOPE_DRIFT')
    for row in receipt['files']:
        path=Path(row['path'])
        require(path.is_file() and sha256(path)==row['sha256'],'COMPUTE_RESUME_PAYLOAD_DRIFT')


def run_subject(seal,subject,output):
    from compiler.realsas_compiler_services.orchestrator import mainline
    plan=json.loads(Path(seal['plan']['path']).read_text())
    require(sha256(Path(seal['plan']['path']))==seal['plan']['sha256'],'COMPUTE_PLAN_DRIFT')
    mainline.validate_plan(plan,require_product_pass_authority=False)
    manifest_path=Path(subject['manifest']['path'])
    require(sha256(manifest_path)==subject['manifest']['sha256'],'COMPUTE_MANIFEST_DRIFT')
    manifest=json.loads(manifest_path.read_text())
    # A distinct compute namespace never overwrites a durable Attempt's ledger.
    run_id='COMPUTE_'+seal['identity'][:16]+'_'+subject['platform_subject_id']
    root=mainline.authority_root()/'runs'/run_id
    root.mkdir(parents=True,exist_ok=True)
    manifest={**manifest,'run_id':run_id}
    worker_manifest=root/'run_manifest.json';write(worker_manifest,manifest)
    ledger=mainline.build_fresh_run_ledger(plan,run_id=run_id,subject_id=subject['subject_id'],
        manifest_ref=str(worker_manifest),architecture_scope='EXTERNAL_COMPUTE_ONLY__PLATFORM_IMPORT_REQUIRED',
        require_product_pass_authority=False)
    closure=mainline._target_closure(plan,(TARGET,));started=time.monotonic()
    ledger_path=root/'COMPUTE_LEDGER.json'
    for stage in plan['stages']:
        sid=stage['id']
        if sid not in closure:continue
        if mainline.dependency_failure_ids(plan,ledger,sid):continue
        print(json.dumps({'event':'WORKER_STAGE','subject':subject['subject_id'],'stage':sid}),flush=True)
        stage_started=time.monotonic()
        mainline._run_stage(plan=plan,ledger=ledger,manifest=manifest,manifest_path=worker_manifest,
                            run_id=run_id,stage_id=sid,ledger_path=ledger_path)
        print(json.dumps({'event':'WORKER_STAGE_FINISHED','subject':subject['subject_id'],'stage':sid,
            'status':mainline._ledger_map(ledger)[sid]['status'],
            'elapsed_seconds':round(time.monotonic()-stage_started,3)}),flush=True)
    mainline.validate_ledger(plan,ledger,require_product_pass_authority=False)
    target=mainline._ledger_map(ledger)[TARGET]
    state='COMPUTED_UNQUALIFIED' if target['status'] in mainline.PASS_STATUSES else 'EXECUTION_FAILED'
    files=[{'path':str(path),'sha256':sha256(path),'size_bytes':path.stat().st_size}
           for path in sorted(root.rglob('*')) if path.is_file() and not path.name.endswith('.partial')]
    result={'schema':'RealSaS.Corpus370ComputeSubjectReceipt.v1','status':state,'subject_id':subject['subject_id'],
        'parent_attempt_id':subject['parent_attempt_id'],'parent_campaign_id':seal['parent_campaign_id'],
        'subject_input_id':subject['subject_input_id'],'code_sha':seal['code_sha'],'capsule_identity':seal['identity'],
        'elapsed_seconds':time.monotonic()-started,'worker_run_id':run_id,
        'peak_rss_kib':max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        'failed_stages':[{'id':r['id'],'blockers':r['blockers']} for r in ledger['stages'] if r['status'] in mainline.FAIL_STATUSES],
        'files':files,'platform_registration_required':True,'qualification_minted':False,'training_authorized':False}
    # Persist each subject's full bytes immediately, not only at the end of a Colab session.
    archive=output/(subject['subject_id']+'.zip');staged=archive.with_suffix('.partial.zip')
    with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as zipped:
        for row in files:zipped.write(row['path'],Path(row['path']).relative_to(mainline.authority_root()).as_posix())
    staged.replace(archive)
    result['archive']={'path':str(archive),'sha256':sha256(archive),'size_bytes':archive.stat().st_size}
    write(output/(subject['subject_id']+'.json'),result)
    return result


def run(seal,output,cache):
    from compiler.realsas_compiler_core.hashing import content_sha256
    require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==seal['code_sha'],'COMPUTE_CODE_DRIFT')
    seal={**seal,'identity':content_sha256(seal)}
    output.mkdir(parents=True,exist_ok=True);write(cache/'WORKER_SEAL.json',seal)
    rows=[];started=time.monotonic()
    for index,subject in enumerate(seal['subjects'],1):
        path=output/(subject['subject_id']+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            # Resume checks the persistent archive, even after local Colab disk vanished.
            require(result.get('capsule_identity')==seal['identity'] and result.get('code_sha')==seal['code_sha']
                    and result.get('parent_attempt_id')==subject['parent_attempt_id'],'COMPUTE_RESUME_SCOPE_DRIFT')
            require(sha256(Path(result['archive']['path']))==result['archive']['sha256'],'COMPUTE_RESUME_ARCHIVE_DRIFT')
        else:
            log=output/(subject['subject_id']+'.log')
            with log.open('w') as stream:
                process=subprocess.Popen([sys.executable,'-u','-m',__name__,'--subject',subject['subject_id'],
                    '--seal',str(cache/'WORKER_SEAL.json'),'--output',str(output)],stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True)
                try:
                    follow_process_log(process,log,context={'event':'COMPUTE_HEARTBEAT',
                        'subject':subject['subject_id'],'subject_index':index,
                        'subject_count':len(seal['subjects'])},heartbeat_seconds=30.)
                except BaseException:
                    import signal
                    os.killpg(process.pid,signal.SIGKILL);process.wait();raise
            require(process.returncode==0 and path.is_file(),'COMPUTE_PROCESS_FAILED__SEE_LOG:'+str(log))
            result=json.loads(path.read_text());verify_result(result,seal,subject)
            require(sha256(Path(result['archive']['path']))==result['archive']['sha256'],'COMPUTE_PERSISTED_ARCHIVE_DRIFT')
            from compiler.realsas_compiler_services.orchestrator import mainline
            expected_run='COMPUTE_'+seal['identity'][:16]+'_'+subject['platform_subject_id']
            require(result['worker_run_id']==expected_run,'COMPUTE_CLEANUP_SCOPE_DRIFT')
            # Raw bytes are verified in Drive before releasing this reproducible local working set.
            shutil.rmtree(mainline.authority_root()/'runs'/expected_run)
        rows.append({k:result[k] for k in ('subject_id','status','elapsed_seconds','peak_rss_kib','failed_stages','archive')})
        summary={'schema':'RealSaS.Corpus370ComputeProgress.v1','capsule_identity':seal['identity'],
            'candidate_subject_count':370,'worker_subject_count':len(seal['subjects']),'accounted_subject_count':len(rows),
            'excluded_subjects':seal['excluded_subjects'],'subjects':rows,'elapsed_seconds':time.monotonic()-started,
            'platform_registration_required':True,'qualification_minted':False,'training_authorized':False}
        write(output/'COMPUTE_PROGRESS.json',summary)
        print(f'COMPUTE {index}/{len(seal["subjects"])} {result["status"]}; {result["elapsed_seconds"]:.1f}s; peak {result["peak_rss_kib"]/1024:.0f} MiB',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject');p.add_argument('--seal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();seal=json.loads(a.seal.read_text())
    require(seal['training_authorized'] is False and seal['qualification_minted'] is False,'COMPUTE_SCOPE')
    subjects=[r for r in seal['subjects'] if r['subject_id']==a.subject];require(len(subjects)==1,'COMPUTE_SUBJECT_SCOPE')
    run_subject(seal,subjects[0],a.output)


if __name__=='__main__':main()
