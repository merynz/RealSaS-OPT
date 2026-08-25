from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

CONT_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_CONTINUATION_LOCALIZATION_V1'
CONT_DEC='P_V5_R256_8X2_CONTINUATION_DECISION.json'
OLD_TWO='IRIS_SINGLE_POSE_V2_P_V5_R256_TWO_STYLE_OVERFIT_V1'
OLD_TWO_COMPLETE='RUN_COMPLETE_P_V5_R256_TWO_STYLE_V1.json'

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,path)

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.check_call([str(x) for x in cmd])

def verify_parents(root):
    cp=root/'runs'/CONT_RUN/CONT_DEC
    old=root/'runs'/OLD_TWO/OLD_TWO_COMPLETE
    if not cp.is_file() or not old.is_file():
        raise FileNotFoundError(f'missing parent authority: {cp} / {old}')
    c=json.load(open(cp)); o=json.load(open(old))
    if c.get('status')!='P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_NOT_SUFFICIENT' or c.get('selected_label')!='CONT_2048' or c.get('selected_pass_count')!=13:
        raise RuntimeError('continuation authority drift')
    if abs(float(c.get('selected_worst_cell_P_p95'))-0.005740759451873588)>1e-12:
        raise RuntimeError('continuation metric drift')
    if o.get('status')!='P_V5_R256_TWO_STYLE_OVERFIT_PASS' or o.get('selected_label')!='TAIL_0512':
        raise RuntimeError('prior two-style PASS authority drift')
    if o.get('tune_consumed') is not False or o.get('sealed_splits_opened') is not False or o.get('camera_json_consumed') is not False:
        raise RuntimeError('old two-style firewall drift')
    return cp,c,old,o

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--work-dir',required=True)
    ap.add_argument('--persist-dir',required=True)
    a=ap.parse_args()
    here=Path(__file__).resolve().parent; root=Path(a.root); work=Path(a.work_dir); persist=Path(a.persist_dir)
    if persist.exists(): raise RuntimeError(f'persist target already exists; refusing overwrite: {persist}')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    cp,c,old,o=verify_parents(root)

    cpu=work/'CPU_PREFLIGHT.json'
    run([sys.executable,here/'pv5_r256_36fb_twostyle_cpu_preflight.py','--out',cpu])
    co=json.load(open(cpu))
    if co.get('status')!='PASS' or co['checks'].get('scientific_optimizer_steps')!=0:
        raise RuntimeError('CPU preflight failed')

    gpu=work/'GPU_PREFLIGHT.json'
    run([sys.executable,here/'pv5_r256_36fb_twostyle_gpu_preflight.py','--out',gpu])
    go=json.load(open(gpu))
    if go.get('status')!='PASS' or go.get('scientific_optimizer_steps')!=0 or go.get('output_field_hw')!=256:
        raise RuntimeError('GPU preflight failed')

    stage=work/'stage'; cache=work/'cache'; train=work/'training'
    mem=here/'P_V5_R256_36FB_TWO_STYLE_MEMBERSHIP_V1.json'
    prereg=here/'P_V5_R256_36FB_TWO_STYLE_PREREG_20260825.md'
    run([sys.executable,here/'stage_pv5_r256_hardasset_twostyle_v1.py','--root',root,'--membership',mem,'--out',stage])
    sm=stage/'STAGE_MANIFEST.json'; so=json.load(open(sm))
    if so.get('record_count')!=2 or so.get('asset_style_cells')!=2 or so.get('camera_json_consumed') is not False:
        raise RuntimeError('stage firewall drift')
    run([sys.executable,here/'prepare_pv5_r256_hardasset_twostyle_cache_v1.py','--stage-manifest',sm,'--out',cache,'--samples-per-view','4096'])
    cm=cache/'CACHE_MANIFEST.json'; ca=json.load(open(cm))
    if ca.get('record_count')!=2 or ca.get('shared_truth_loci_across_styles') is not True:
        raise RuntimeError('cache authority drift')

    pre={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStylePreOptimizerAuthority.v1',
         'status':'FROZEN__OPTIMIZER_AUTHORIZED_FOR_36FB_X_2_STYLES_ONLY',
         'continuation_decision_sha256':sha256_file(cp),'continuation_status':c['status'],
         'continuation_selected_label':c['selected_label'],
         'prior_two_style_complete_sha256':sha256_file(old),'prior_two_style_status':o['status'],
         'prereg_sha256':sha256_file(prereg),'membership_sha256':sha256_file(mem),
         'cpu_preflight_sha256':sha256_file(cpu),'gpu_preflight_sha256':sha256_file(gpu),
         'stage_manifest_sha256':sha256_file(sm),'cache_manifest_sha256':sha256_file(cm),
         'asset_id':'asset_36fb02305846592b1ecdf3d4','asset_style_cells':2,'output_field_hw':256,
         'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
         'scientific_optimizer_steps_at_freeze':0}
    atomic_json(work/'PREOPT_AUTHORITY.json',pre)

    run([sys.executable,here/'train_pv5_r256_36fb_twostyle_v1.py','--cache-manifest',cm,'--out-dir',train])
    decision=train/'P_V5_R256_36FB_TWO_STYLE_DECISION.json'; d=json.load(open(decision))
    complete={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleRunComplete.v1',
              'status':d['status'],'decision_sha256':sha256_file(decision),
              'best_checkpoint_sha256':sha256_file(train/'BEST_CHECKPOINT.pt'),
              'preopt_authority_sha256':sha256_file(work/'PREOPT_AUTHORITY.json'),
              'selected_label':d['selected_label'],'selected_total_optimizer_steps':d['selected_total_optimizer_steps'],
              'selected_worst_cell_P_p95':d['selected_worst_cell_P_p95'],'selected_aggregate_P_p95':d['selected_aggregate_P_p95'],
              'selected_per_cell':d['selected_per_cell'],'threshold_P_p95_per_cell':d['threshold_P_p95_per_cell'],
              'asset_style_cells':2,'output_field_hw':256,
              'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
              'next_policy':d['next_policy']}
    atomic_json(work/'RUN_COMPLETE_P_V5_R256_36FB_TWO_STYLE_V1.json',complete)

    persist.parent.mkdir(parents=True,exist_ok=True)
    tmp=persist.with_name(persist.name+'.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    for p in (cpu,gpu,work/'PREOPT_AUTHORITY.json',work/'RUN_COMPLETE_P_V5_R256_36FB_TWO_STYLE_V1.json',
              decision,train/'BEST_CHECKPOINT.pt',train/'TRAIN_AUTHORITY.json',train/'TRAIN_HISTORY.json',prereg,mem):
        shutil.copy2(p,tmp/p.name)
    shutil.copytree(train/'eval',tmp/'eval')
    os.replace(tmp,persist)
    print(json.dumps(complete,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
