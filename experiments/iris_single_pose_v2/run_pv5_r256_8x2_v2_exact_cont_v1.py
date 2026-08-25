from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

PARENT_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_CERTIFICATION_V2'
PARENT_COMPLETE='RUN_COMPLETE_P_V5_R256_8X2_V2.json'
PARENT_DEC='P_V5_R256_8X2_V2_DECISION.json'
PARENT_CP='BEST_CHECKPOINT.pt'
PARENT_COMPLETE_SHA='339b6a2f929e732e9ac0bb5c540e85b3b473e86acc6508391de9807b20dbb77b'
PARENT_DEC_SHA='77626d41cf87facac5e5d5247a20268bf3654cd94ab24ab16ff8031d8d37678c'
PARENT_CP_SHA='8c3872412124644885b66aea1b00413a8da5ab91a7471e83fcae001d43544049'
TRUTH_SHA={
'asset_551ea351b43a1787d0f55536':'f32a90c5b15422f84a2ef051ff8908553eb2a31801dee83554d7a493f8356751',
'asset_36fb02305846592b1ecdf3d4':'18c69d343f7624061af2c45dca0c49ae64768f903593d27ef566ccca391d807e',
'asset_0679fdef64f19a4832a6d521':'d95dd8cb13b7e85d6193fa585f9046458b5f36303940e0ff9d8e75381f3eef6c',
'asset_76313e4bd82b82fcd1659c70':'034a11a5efc076f2d5615aad33ae7743fd59f0c61e69b1ff59a2f237a81b5945',
'asset_6f086a5b1a66378ffe04d7e4':'ea9723e082d76aab7bdf2030c623447b31534f261a52f6e197ccf403d3b83a69',
'asset_425122d500ecf5767404f9c0':'49bcbd25c832e45198d9763b5c1f8ea9cf1a7bebedb9c00b943e01ba9ac2adf3',
'asset_5a19f8c5254be7bf30c504f5':'e50075ccccb35b6ddbc60ffd4e42d587b285147d214fab0abab46b6efbe23b85',
'asset_f8a40d6c5d815fe79c8b5e42':'15f5b7d7174c399366734854558963afb57cd06d89aa0bff20f50453b4ec74c2',
}

def sha(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def cmd(x): print('+',' '.join(map(str,x)),flush=True); subprocess.check_call([str(q) for q in x])

def cmd_logged(x,log_path):
    print('+',' '.join(map(str,x)),flush=True); log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
    with open(log_path,'w',encoding='utf-8') as log:
        p=subprocess.Popen([str(q) for q in x],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout: print(line,end='',flush=True); log.write(line); log.flush()
        rc=p.wait()
    if rc!=0: raise subprocess.CalledProcessError(rc,[str(q) for q in x])

def exact_json(path,expected):
    if not path.is_file(): raise FileNotFoundError(path)
    got=sha(path)
    if got!=expected: raise RuntimeError(f'authority SHA drift {path}: {got} != {expected}')
    return json.load(open(path,encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--persist-dir',required=True); a=ap.parse_args()
    root=Path(a.root); here=Path(__file__).resolve().parent; work=Path(a.work_dir); persist=Path(a.persist_dir)
    if persist.exists(): raise RuntimeError(f'persist target exists: {persist}')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)

    pr=root/'runs'/PARENT_RUN; pc=pr/PARENT_COMPLETE; pd=pr/PARENT_DEC; pcp=pr/PARENT_CP
    co=exact_json(pc,PARENT_COMPLETE_SHA); de=exact_json(pd,PARENT_DEC_SHA)
    if sha(pcp)!=PARENT_CP_SHA: raise RuntimeError('parent checkpoint SHA drift')
    if co.get('status')!='P_V5_R256_8X2_V2_CERTIFICATION_INSUFFICIENT' or co.get('selected_label')!='TAIL_4096' or co.get('selected_total_optimizer_steps')!=6144 or co.get('selected_pass_count')!=14: raise RuntimeError('parent run-complete identity drift')
    if de.get('status')!=co.get('status') or de.get('selected_worst_cell_P_p95')!=co.get('selected_worst_cell_P_p95'): raise RuntimeError('parent decision/run-complete mismatch')
    if co.get('checkpoint_optimizer_state_included') is not True or co.get('patchmatch_admitted') is not False or co.get('architecture_changed') is not False or co.get('sealed_splits_opened') is not False or co.get('camera_json_consumed') is not False: raise RuntimeError('parent firewall drift')

    cpu=work/'CPU_PREFLIGHT_EXACT_CONT.json'; gpu=work/'GPU_PREFLIGHT_EXACT_CONT.json'
    cmd([sys.executable,here/'pv5_r256_8x2_v2_exact_cont_cpu_preflight.py','--parent-checkpoint',pcp,'--out',cpu])
    cmd([sys.executable,here/'pv5_r256_8x2_v2_gpu_preflight.py','--out',gpu])
    cpo=json.load(open(cpu,encoding='utf-8')); gpo=json.load(open(gpu,encoding='utf-8'))
    if cpo.get('status')!='PASS' or cpo.get('scientific_optimizer_steps')!=0 or cpo.get('fresh_optimizer_moments') is not False: raise RuntimeError('CPU exact-resume preflight drift')
    if gpo.get('status')!='PASS' or gpo.get('scientific_optimizer_steps')!=0 or gpo.get('effective_cells_per_optimizer_step')!=16: raise RuntimeError('GPU preflight drift')

    stage=work/'stage'; cache=work/'cache'; train=work/'training'; mem=here/'P_V5_R256_8X2_MEMBERSHIP_V1.json'; prereg=here/'P_V5_R256_8X2_V2_EXACT_CONTINUATION_PREREG_20260825.md'; parent_auth=here/'P_V5_R256_8X2_V2_EXACT_CONT_PARENT_AUTHORITY.json'
    cmd([sys.executable,here/'stage_pv5_r256_8x2_v1.py','--root',root,'--membership',mem,'--out',stage]); sm=stage/'STAGE_MANIFEST.json'; so=json.load(open(sm,encoding='utf-8'))
    if so.get('record_count')!=16 or so.get('asset_count')!=8 or so.get('camera_json_consumed') is not False: raise RuntimeError('stage firewall')
    cmd([sys.executable,here/'prepare_pv5_r256_8x2_cache_v1.py','--stage-manifest',sm,'--out',cache,'--samples-per-view','4096']); cm=cache/'CACHE_MANIFEST.json'; ca=json.load(open(cm,encoding='utf-8'))
    if ca.get('record_count')!=16 or ca.get('asset_count')!=8 or ca.get('shared_truth_loci_across_styles') is not True or ca.get('samples_per_view')!=4096 or ca.get('camera_json_consumed') is not False: raise RuntimeError('cache authority')
    if ca.get('truth_sha256_by_asset')!=TRUTH_SHA: raise RuntimeError(f'frozen truth SHA mapping drift: {ca.get("truth_sha256_by_asset")}')

    pre={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoV2ExactContinuationPreOptimizerAuthority.v1','status':'FROZEN__EXACT_PARENT_STATE_CONTINUATION_AUTHORIZED_ONLY','parent_run_complete_sha256':sha(pc),'parent_decision_sha256':sha(pd),'parent_checkpoint_sha256':sha(pcp),'prereg_sha256':sha(prereg),'membership_sha256':sha(mem),'parent_authority_sha256':sha(parent_auth),'cpu_preflight_sha256':sha(cpu),'gpu_preflight_sha256':sha(gpu),'stage_manifest_sha256':sha(sm),'cache_manifest_sha256':sha(cm),'truth_sha256_by_asset':TRUTH_SHA,'asset_count':8,'asset_style_cells':16,'output_field_hw':256,'effective_cells_per_optimizer_step':16,'microbatch_cells':8,'accumulation_steps':2,'parent_total_optimizer_steps':6144,'additional_steps':1024,'lr':3e-5,'fresh_optimizer_moments':False,'optimizer_state_restore_required':True,'scaler_state_restore_required':True,'rng_state_restore_required':True,'patchmatch_admitted':False,'architecture_changed':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'scientific_optimizer_steps_at_freeze':0}
    atomic(work/'PREOPT_AUTHORITY_EXACT_CONT.json',pre)

    cmd_logged([sys.executable,here/'train_pv5_r256_8x2_v2_exact_cont_v1.py','--cache-manifest',cm,'--parent-checkpoint',pcp,'--parent-decision',pd,'--out-dir',train],work/'TRAIN_PROCESS_EXACT_CONT.log')
    dec=train/'P_V5_R256_8X2_V2_EXACT_CONTINUATION_DECISION.json'; d=json.load(open(dec,encoding='utf-8'))
    complete={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoV2ExactContinuationRunComplete.v1','status':d['status'],'parent_status':co['status'],'parent_label':'TAIL_4096','parent_total_optimizer_steps':6144,'parent_checkpoint_sha256':sha(pcp),'decision_sha256':sha(dec),'best_checkpoint_sha256':sha(train/'BEST_CONTINUATION_CHECKPOINT.pt'),'preopt_authority_sha256':sha(work/'PREOPT_AUTHORITY_EXACT_CONT.json'),'selected_label':d['selected_label'],'selected_continuation_step':d['selected_continuation_step'],'selected_total_optimizer_steps':d['selected_total_optimizer_steps'],'selected_worst_cell_P_p95':d['selected_worst_cell_P_p95'],'selected_aggregate_P_p95':d['selected_aggregate_P_p95'],'selected_pass_count':d['selected_pass_count'],'selected_per_cell':d['selected_per_cell'],'threshold_P_p95_per_cell':d['threshold_P_p95_per_cell'],'asset_count':8,'asset_style_cells':16,'additional_steps':1024,'lr':3e-5,'fresh_optimizer_moments':False,'optimizer_state_restored':True,'scaler_state_restored':True,'rng_state_restored':True,'patchmatch_admitted':False,'architecture_changed':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':d['next_policy']}
    atomic(work/'RUN_COMPLETE_P_V5_R256_8X2_V2_EXACT_CONTINUATION_V1.json',complete)

    persist.parent.mkdir(parents=True,exist_ok=True); tmp=persist.with_name(persist.name+'.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    for p in (cpu,gpu,work/'PREOPT_AUTHORITY_EXACT_CONT.json',work/'RUN_COMPLETE_P_V5_R256_8X2_V2_EXACT_CONTINUATION_V1.json',work/'TRAIN_PROCESS_EXACT_CONT.log',dec,train/'BEST_CONTINUATION_CHECKPOINT.pt',train/'CONTINUATION_TRAIN_AUTHORITY.json',train/'CONTINUATION_HISTORY.json',train/'eval'/'PARENT_REPRO.json',prereg,mem,parent_auth,here/'P_V5_R256_8X2_V2_RESULT_20260825.md'):
        shutil.copy2(p,tmp/p.name)
    shutil.copytree(train/'eval',tmp/'eval')
    os.replace(tmp,persist); print(json.dumps(complete,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
