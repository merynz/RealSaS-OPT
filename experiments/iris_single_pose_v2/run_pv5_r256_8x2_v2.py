from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ORIG_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_JOINT_FIT_V1'
ORIG_FILE='RUN_COMPLETE_P_V5_R256_8X2_V1.json'
ORIG_SHA='1d31697eec4ee41e2b13384330708c8cd37bcb68c6cfa93eaf125fe9f944d867'
CONT_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_CONTINUATION_LOCALIZATION_V1'
CONT_FILE='RUN_COMPLETE_P_V5_R256_8X2_CONTINUATION_V1.json'
CONT_SHA='ab0bba107020c65968b7440d35446413d18278faa68b57a279d42cb4e772f921'
HARD_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_36FB_TWO_STYLE_SUFFICIENCY_V1'
HARD_FILE='RUN_COMPLETE_P_V5_R256_36FB_TWO_STYLE_V1.json'
HARD_SHA='bbcb57a64b3326007f2b17ae3e4249890d29d3650773ae17f792521f22609c97'

def sha(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def cmd(x):
    print('+',' '.join(map(str,x)),flush=True); subprocess.check_call([str(q) for q in x])

def cmd_logged(x,log_path):
    print('+',' '.join(map(str,x)),flush=True); log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
    with open(log_path,'w',encoding='utf-8') as log:
        proc=subprocess.Popen([str(q) for q in x],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line,end='',flush=True); log.write(line); log.flush()
        rc=proc.wait()
    if rc!=0: raise subprocess.CalledProcessError(rc,[str(q) for q in x])

def load_exact(path,expected_sha):
    if not path.is_file(): raise FileNotFoundError(path)
    got=sha(path)
    if got!=expected_sha: raise RuntimeError(f'authority SHA drift {path}: {got} != {expected_sha}')
    return json.load(open(path,encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--persist-dir',required=True); a=ap.parse_args()
    here=Path(__file__).resolve().parent; root=Path(a.root); work=Path(a.work_dir); persist=Path(a.persist_dir)
    if persist.exists(): raise RuntimeError(f'persist target exists: {persist}')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)

    orig=load_exact(root/'runs'/ORIG_RUN/ORIG_FILE,ORIG_SHA)
    if orig.get('status')!='P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT' or orig.get('selected_pass_count')!=8: raise RuntimeError('original V1 FAIL authority drift')
    cont=load_exact(root/'runs'/CONT_RUN/CONT_FILE,CONT_SHA)
    if cont.get('status')!='P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_NOT_SUFFICIENT' or cont.get('selected_pass_count')!=13: raise RuntimeError('continuation FAIL authority drift')
    hard=load_exact(root/'runs'/HARD_RUN/HARD_FILE,HARD_SHA)
    if hard.get('status')!='P_V5_R256_36FB_TWO_STYLE_SUFFICIENCY_PASS' or hard.get('asset_style_cells')!=2 or float(hard.get('selected_worst_cell_P_p95',1))>0.005: raise RuntimeError('36fb PASS authority drift')
    if hard.get('tune_consumed') is not False or hard.get('sealed_splits_opened') is not False or hard.get('camera_json_consumed') is not False: raise RuntimeError('hard-asset firewall drift')
    if cont.get('patchmatch_admitted') is not False or cont.get('architecture_changed') is not False: raise RuntimeError('continuation architecture firewall drift')

    cpu=work/'CPU_PREFLIGHT_V2.json'; gpu=work/'GPU_PREFLIGHT_V2.json'
    cmd([sys.executable,here/'pv5_r256_8x2_v2_cpu_preflight.py','--out',cpu])
    cmd([sys.executable,here/'pv5_r256_8x2_v2_gpu_preflight.py','--out',gpu])
    go=json.load(open(gpu,encoding='utf-8'))
    if go.get('status')!='PASS' or go.get('effective_cells_per_optimizer_step')!=16 or go.get('scientific_optimizer_steps')!=0 or go.get('resumable_checkpoint_schema')!='PASS': raise RuntimeError('GPU preflight drift')

    stage=work/'stage'; cache=work/'cache'; train=work/'training'
    mem=here/'P_V5_R256_8X2_MEMBERSHIP_V1.json'; prereg=here/'P_V5_R256_8X2_V2_PREREG_20260825.md'; parent_auth=here/'P_V5_R256_8X2_V2_PARENT_AUTHORITY.json'
    cmd([sys.executable,here/'stage_pv5_r256_8x2_v1.py','--root',root,'--membership',mem,'--out',stage])
    sm=stage/'STAGE_MANIFEST.json'; so=json.load(open(sm,encoding='utf-8'))
    if so.get('record_count')!=16 or so.get('asset_count')!=8 or so.get('camera_json_consumed') is not False: raise RuntimeError('stage firewall')
    cmd([sys.executable,here/'prepare_pv5_r256_8x2_cache_v1.py','--stage-manifest',sm,'--out',cache,'--samples-per-view','4096'])
    cm=cache/'CACHE_MANIFEST.json'; ca=json.load(open(cm,encoding='utf-8'))
    if ca.get('record_count')!=16 or ca.get('asset_count')!=8 or ca.get('shared_truth_loci_across_styles') is not True or ca.get('samples_per_view')!=4096: raise RuntimeError('cache authority')

    pre={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoV2PreOptimizerAuthority.v1','status':'FROZEN__OPTIMIZER_AUTHORIZED_FOR_FRESH_8X2_V2_ONLY','original_v1_run_sha256':sha(root/'runs'/ORIG_RUN/ORIG_FILE),'continuation_run_sha256':sha(root/'runs'/CONT_RUN/CONT_FILE),'hardasset_36fb_run_sha256':sha(root/'runs'/HARD_RUN/HARD_FILE),'prereg_sha256':sha(prereg),'membership_sha256':sha(mem),'parent_authority_sha256':sha(parent_auth),'cpu_preflight_sha256':sha(cpu),'gpu_preflight_sha256':sha(gpu),'stage_manifest_sha256':sha(sm),'cache_manifest_sha256':sha(cm),'asset_count':8,'asset_style_cells':16,'output_field_hw':256,'effective_cells_per_optimizer_step':16,'microbatch_cells':8,'accumulation_steps':2,'planned_optimizer_steps_total':6144,'fresh_model_initialization':True,'checkpoint_reuse':False,'patchmatch_admitted':False,'architecture_changed':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'scientific_optimizer_steps_at_freeze':0}
    atomic(work/'PREOPT_AUTHORITY_V2.json',pre)

    cmd_logged([sys.executable,here/'train_pv5_r256_8x2_v2.py','--cache-manifest',cm,'--out-dir',train],work/'TRAIN_PROCESS_V2.log')
    dec=train/'P_V5_R256_8X2_V2_DECISION.json'; d=json.load(open(dec,encoding='utf-8'))
    complete={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoRunComplete.v2','status':d['status'],'original_v1_status':orig['status'],'continuation_status':cont['status'],'hardasset_36fb_status':hard['status'],'decision_sha256':sha(dec),'best_checkpoint_sha256':sha(train/'BEST_CHECKPOINT.pt'),'preopt_authority_sha256':sha(work/'PREOPT_AUTHORITY_V2.json'),'selected_label':d['selected_label'],'selected_total_optimizer_steps':d['selected_total_optimizer_steps'],'selected_worst_cell_P_p95':d['selected_worst_cell_P_p95'],'selected_aggregate_P_p95':d['selected_aggregate_P_p95'],'selected_pass_count':d['selected_pass_count'],'selected_per_cell':d['selected_per_cell'],'threshold_P_p95_per_cell':d['threshold_P_p95_per_cell'],'asset_count':8,'asset_style_cells':16,'planned_optimizer_steps_total':6144,'checkpoint_optimizer_state_included':True,'fresh_model_initialization':True,'checkpoint_reuse':False,'patchmatch_admitted':False,'architecture_changed':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':d['next_policy']}
    atomic(work/'RUN_COMPLETE_P_V5_R256_8X2_V2.json',complete)

    persist.parent.mkdir(parents=True,exist_ok=True); tmp=persist.with_name(persist.name+'.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    for p in (cpu,gpu,work/'PREOPT_AUTHORITY_V2.json',work/'RUN_COMPLETE_P_V5_R256_8X2_V2.json',work/'TRAIN_PROCESS_V2.log',dec,train/'BEST_CHECKPOINT.pt',train/'TRAIN_AUTHORITY.json',train/'TRAIN_HISTORY.json',prereg,mem,parent_auth): shutil.copy2(p,tmp/p.name)
    shutil.copytree(train/'eval',tmp/'eval')
    os.replace(tmp,persist)
    print(json.dumps(complete,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
