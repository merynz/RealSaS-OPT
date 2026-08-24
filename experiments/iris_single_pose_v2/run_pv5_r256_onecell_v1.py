from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.check_call([str(x) for x in cmd])

def verify_parent_field_closure(root):
    p=root/'runs'/'IRIS_SINGLE_POSE_V2_P_V5_FIELD_REPRESENTATION_CLOSURE_V1'/'RUN_COMPLETE_P_V5_FIELD_REPRESENTATION_CLOSURE_V1.json'
    if not p.is_file(): raise FileNotFoundError(f'field-closure parent authority missing: {p}')
    o=json.load(open(p,encoding='utf-8'))
    required={'status':'P_V5_FIELD_REPRESENTATION_CLOSED','current_R2_certified':False,'full_R_certified':True,'smallest_certified_field_hw':256,'neural_optimizer_steps':0,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False}
    for k,v in required.items():
        if o.get(k)!=v: raise RuntimeError(f'field-closure parent drift {k}: {o.get(k)!r} != {v!r}')
    if o.get('certified_field_hw')!=[256]: raise RuntimeError(f'certified field set drift: {o.get("certified_field_hw")}')
    return p,o

def main():
    ap=argparse.ArgumentParser(description='Canonical R256 one-asset/one-style P-V5 learner overfit runner'); ap.add_argument('--root',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--persist-dir',required=True); a=ap.parse_args()
    here=Path(__file__).resolve().parent; root=Path(a.root); work=Path(a.work_dir); persist=Path(a.persist_dir)
    if persist.exists(): raise RuntimeError(f'persist target already exists; refusing overwrite: {persist}')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True); parent_path,parent=verify_parent_field_closure(root)
    cpu=work/'CPU_PREFLIGHT.json'; run([sys.executable,here/'pv5_r256_onecell_cpu_preflight.py','--out',cpu]); co=json.load(open(cpu,encoding='utf-8'))
    if co.get('status')!='PASS' or co['checks'].get('scientific_optimizer_steps')!=0: raise RuntimeError('CPU preflight failed')
    gpu=work/'GPU_PREFLIGHT.json'; run([sys.executable,here/'pv5_r256_onecell_gpu_preflight.py','--out',gpu]); go=json.load(open(gpu,encoding='utf-8'))
    if go.get('status')!='PASS' or go.get('scientific_optimizer_steps')!=0 or go.get('output_field_hw')!=256: raise RuntimeError('GPU R256 preflight failed')
    stage=work/'stage'; cache=work/'cache'; train=work/'training'; mem=here/'P_V5_R256_ONE_CELL_MEMBERSHIP_V1.json'; prereg=here/'P_V5_R256_ONE_CELL_PREREG_20260824.md'
    run([sys.executable,here/'stage_pv5_r256_onecell_v1.py','--root',root,'--membership',mem,'--out',stage]); sm=stage/'STAGE_MANIFEST.json'; so=json.load(open(sm,encoding='utf-8'))
    if so.get('record_count')!=1 or so.get('asset_style_cells')!=1 or so.get('camera_json_consumed') is not False: raise RuntimeError('stage firewall failed')
    run([sys.executable,here/'prepare_pv5_r256_onecell_cache_v1.py','--stage-manifest',sm,'--out',cache,'--samples-per-view','4096']); cm=cache/'CACHE_MANIFEST.json'; ca=json.load(open(cm,encoding='utf-8'))
    if ca.get('record_count')!=1 or ca.get('asset_style_cells')!=1 or ca.get('camera_json_consumed') is not False: raise RuntimeError('cache firewall failed')
    preauth={'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellPreOptimizerAuthority.v1','status':'FROZEN__OPTIMIZER_AUTHORIZED_FOR_R256_ONE_CELL_ONLY','parent_field_closure_sha256':sha256_file(parent_path),'parent_status':parent['status'],'prereg_sha256':sha256_file(prereg),'membership_sha256':sha256_file(mem),'cpu_preflight_sha256':sha256_file(cpu),'gpu_preflight_sha256':sha256_file(gpu),'stage_manifest_sha256':sha256_file(sm),'cache_manifest_sha256':sha256_file(cm),'fit_assets':1,'asset_style_cells':1,'output_field_hw':256,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'scientific_optimizer_steps_at_freeze':0}; atomic_json(work/'PREOPT_AUTHORITY.json',preauth)
    run([sys.executable,here/'train_pv5_r256_onecell_v1.py','--cache-manifest',cm,'--out-dir',train]); decision=train/'P_V5_R256_ONE_CELL_DECISION.json'; d=json.load(open(decision,encoding='utf-8'))
    if d.get('tune_consumed') is not False or d.get('sealed_splits_opened') is not False or d.get('camera_json_consumed') is not False: raise RuntimeError('final firewall drift')
    if d.get('output_field_hw')!=256: raise RuntimeError('final output-field drift')
    complete={'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellRunComplete.v1','status':d['status'],'parent_status':parent['status'],'decision_sha256':sha256_file(decision),'best_checkpoint_sha256':sha256_file(train/'BEST_CHECKPOINT.pt'),'preopt_authority_sha256':sha256_file(work/'PREOPT_AUTHORITY.json'),'optimizer_steps':d['optimizer_steps'],'selected_step':d['selected_step'],'selected_P_p95':d['selected_P_p95'],'threshold_P_p95':d['threshold_P_p95'],'output_field_hw':256,'asset_id':d['asset_id'],'style':d['style'],'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':d['next_policy']}; atomic_json(work/'RUN_COMPLETE_P_V5_R256_ONE_CELL_V1.json',complete)
    persist.parent.mkdir(parents=True,exist_ok=True); tmp=persist.with_name(persist.name+'.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    for p in (cpu,gpu,work/'PREOPT_AUTHORITY.json',work/'RUN_COMPLETE_P_V5_R256_ONE_CELL_V1.json',decision,train/'BEST_CHECKPOINT.pt',train/'TRAIN_AUTHORITY.json',train/'TRAIN_HISTORY.json',prereg,mem): shutil.copy2(p,tmp/p.name)
    shutil.copytree(train/'eval',tmp/'eval'); os.replace(tmp,persist); print(json.dumps(complete,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
