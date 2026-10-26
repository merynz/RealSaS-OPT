#!/usr/bin/env python3
"""Reproduce sealed native teacher arrays with the exact historical transport.

This is the allowed sealed-evidence reproducibility lane, not an intervention
builder or new research authority. It does not change the teacher recipe,
normalization, actions or frames. All subjects are reported, including failures.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256, _bound_path


def compare_arrays(expected, actual):
    with np.load(expected,allow_pickle=False) as a, np.load(actual,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):
            raise ValueError('NATIVE_REPLAY_ARRAY_SET_DRIFT')
        rows=[]
        for key in sorted(a.files):
            x,y=a[key],b[key]
            if x.dtype!=y.dtype or x.shape!=y.shape:
                raise ValueError('NATIVE_REPLAY_ARRAY_LAYOUT_DRIFT:'+key)
            exact=np.array_equal(x,y)
            maximum=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if x.size else 0.
            rows.append({'array':key,'shape':list(x.shape),'dtype':x.dtype.str,'exact_equal':bool(exact),'max_abs_difference':maximum})
        return {'exact_arrays_equal':all(r['exact_equal'] for r in rows),'arrays':rows}


def replay(request, *, repo, authority_root, blender, output):
    if (request.get('schema')!='RealSaS.NativeTeacherReplayRequest.v1'
            or request.get('lane')!='SEALED_EVIDENCE_REPRODUCIBILITY_ONLY'
            or request.get('training_allowed') is not False
            or request.get('new_interventions_allowed') is not False):
        raise ValueError('NATIVE_REPLAY_REQUEST_SCOPE')
    repo,authority,output=map(lambda p:Path(p).resolve(),(repo,authority_root,output))
    output.mkdir(parents=True,exist_ok=False)
    for relative,digest in request['source_closure_sha256'].items():
        path=_bound_path(repo,relative)
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError('NATIVE_REPLAY_HISTORICAL_SOURCE_DRIFT:'+relative)
    version=subprocess.check_output([str(blender),'--version'],text=True).splitlines()[0]
    if version!=request['blender_version']:
        raise ValueError('NATIVE_REPLAY_BLENDER_VERSION_DRIFT:'+version)
    results=[]
    for index,row in enumerate(request['subjects']):
        print(f"[NATIVE_REPLAY] {index+1}/{len(request['subjects'])} {row['subject_id']}",flush=True)
        began=time.monotonic();result={'subject_id':row['subject_id'],'status':'CRASH'}
        folder=_bound_path(output,row['subject_id']);folder.mkdir()
        try:
            source=_bound_path(authority,row['source']['relative'])
            admission=_bound_path(authority,row['observation_admission']['relative'])
            expected=_bound_path(authority,row['expected_teacher']['relative'])
            receipt_path=_bound_path(authority,row['expected_receipt']['relative'])
            for path,ref in ((source,row['source']),(admission,row['observation_admission']),(expected,row['expected_teacher']),(receipt_path,row['expected_receipt'])):
                if not path.is_file() or sha256(path)!=ref['sha256']:
                    raise ValueError('NATIVE_REPLAY_INPUT_BYTES_DRIFT:'+path.name)
            receipt=json.loads(receipt_path.read_text())
            if (row['max_frames_per_clip']!=receipt['max_frames_per_clip'] or row['subject_id']!=receipt['subject_id']
                    or row['source']['sha256']!=receipt['source']['sha256']
                    or row['observation_admission']['sha256']!=receipt['observation_admission_sha256']
                    or row['expected_teacher']['sha256']!=receipt['output']['sha256']):
                raise ValueError('NATIVE_REPLAY_SEALED_RECIPE_DRIFT')
            target=folder/'replay.npz'
            cmd=[str(blender),'-b','--factory-startup','--python',str(repo/'tools/iris_vnext/blender_authored_teacher_export_v3.py'),'--',
                 '--source',str(source),'--expected-source-sha256',row['source']['sha256'],
                 '--subject-id',row['subject_id'],'--observation-admission',str(admission),
                 '--output',str(target),'--max-frames-per-clip',str(row['max_frames_per_clip'])]
            with (folder/'blender.log').open('w') as log:
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
                try:
                    while True:
                        try:
                            code=process.wait(timeout=30);break
                        except subprocess.TimeoutExpired:
                            elapsed=time.monotonic()-began
                            print(f"[NATIVE_REPLAY_HEARTBEAT] {row['subject_id']} elapsed_seconds={elapsed:.1f}",flush=True)
                            if elapsed>request['maximum_seconds_per_subject']:
                                raise TimeoutError('NATIVE_REPLAY_SUBJECT_TIMEOUT')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
            if code!=0:
                raise RuntimeError('NATIVE_REPLAY_BLENDER_EXIT:'+str(code))
            comparison=compare_arrays(expected,target)
            result.update(comparison,status='EXACT_SEALED_ARRAY_REPLAY' if comparison['exact_arrays_equal'] else 'ARRAY_DRIFT',
                          replay_payload_sha256=sha256(target),expected_payload_sha256=row['expected_teacher']['sha256'])
        except Exception as exc:
            result.update(exception_type=type(exc).__name__,message=str(exc))
        result['elapsed_seconds']=round(time.monotonic()-began,3)
        results.append(result)
        print(json.dumps({k:result[k] for k in ('subject_id','status','elapsed_seconds')},sort_keys=True),flush=True)
    report={'schema':'RealSaS.NativeTeacherReplayReport.v1','status':'EXACT_SEALED_REPLAY' if all(r['status']=='EXACT_SEALED_ARRAY_REPLAY' for r in results) else 'NOT_EQUIVALENT',
            'subjects':results,'blender_version':version,'qualification_minted':False,'new_interventions_executed':False,
            'training_executed':False,'source_raster_equivalence_claimed':False,'causal_truth_qualified':False}
    (output/'REPLAY_REPORT.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','repo','authority-root','blender','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    report=replay(json.loads(Path(a.request).read_text()),repo=a.repo,authority_root=a.authority_root,blender=a.blender,output=a.output)
    return 0 if report['status']=='EXACT_SEALED_REPLAY' else 2


if __name__=='__main__':sys.exit(main())
