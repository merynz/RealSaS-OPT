from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

PARENT_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_JOINT_FIT_V1'
PERSIST_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_8X2_CONTINUATION_LOCALIZATION_V1'
EXPECTED_PARENT_SHA='0228c8c939490c9ac33cee6ca360c9228a5b4ee462b30b9774888c12129f3776'
EXPECTED_ORIGINAL_STATUS='P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT'
EXPECTED_TRUTH={
'asset_0679fdef64f19a4832a6d521':'d95dd8cb13b7e85d6193fa585f9046458b5f36303940e0ff9d8e75381f3eef6c',
'asset_36fb02305846592b1ecdf3d4':'18c69d343f7624061af2c45dca0c49ae64768f903593d27ef566ccca391d807e',
'asset_425122d500ecf5767404f9c0':'49bcbd25c832e45198d9763b5c1f8ea9cf1a7bebedb9c00b943e01ba9ac2adf3',
'asset_551ea351b43a1787d0f55536':'f32a90c5b15422f84a2ef051ff8908553eb2a31801dee83554d7a493f8356751',
'asset_5a19f8c5254be7bf30c504f5':'e50075ccccb35b6ddbc60ffd4e42d587b285147d214fab0abab46b6efbe23b85',
'asset_6f086a5b1a66378ffe04d7e4':'ea9723e082d76aab7bdf2030c623447b31534f261a52f6e197ccf403d3b83a69',
'asset_76313e4bd82b82fcd1659c70':'034a11a5efc076f2d5615aad33ae7743fd59f0c61e69b1ff59a2f237a81b5945',
'asset_f8a40d6c5d815fe79c8b5e42':'15f5b7d7174c399366734854558963afb57cd06d89aa0bff20f50453b4ec74c2',
}

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def run(cmd,cwd,log=None):
    print('+',' '.join(map(str,cmd)),flush=True)
    if log is None: subprocess.run(cmd,cwd=cwd,check=True); return
    with open(log,'w',encoding='utf-8') as f:
        p=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=False)
    if p.returncode!=0: raise RuntimeError(f'child failed {p.returncode}; log={log}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--persist-dir',required=True); a=ap.parse_args()
    root=Path(a.root); work=Path(a.work_dir); persist=Path(a.persist_dir); here=Path(__file__).resolve().parent
    parent_dir=root/'runs'/PARENT_RUN; parent_cp=parent_dir/'BEST_CHECKPOINT.pt'; parent_dec=parent_dir/'P_V5_R256_8X2_DECISION.json'
    if not parent_cp.is_file() or not parent_dec.is_file(): raise FileNotFoundError('canonical parent 8x2 run artifacts missing')
    pd=json.load(open(parent_dec,encoding='utf-8'))
    if pd.get('status')!=EXPECTED_ORIGINAL_STATUS or pd.get('selected_label')!='TAIL_0512' or int(pd.get('selected_total_optimizer_steps',-1))!=2560:
        raise RuntimeError('parent decision identity/status drift')
    psha=sha256_file(parent_cp)
    if psha!=EXPECTED_PARENT_SHA: raise RuntimeError(f'parent checkpoint SHA drift {psha} != {EXPECTED_PARENT_SHA}')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True); stage=work/'stage'; cache=work/'cache'; train=work/'train'
    membership=here/'P_V5_R256_8X2_MEMBERSHIP_V1.json'
    run([sys.executable,str(here/'stage_pv5_r256_8x2_v1.py'),'--root',str(root),'--membership',str(membership),'--out',str(stage)],here)
    run([sys.executable,str(here/'prepare_pv5_r256_8x2_cache_v1.py'),'--stage-manifest',str(stage/'STAGE_MANIFEST.json'),'--out',str(cache),'--samples-per-view','4096'],here)
    cm=json.load(open(cache/'CACHE_MANIFEST.json',encoding='utf-8'))
    if cm.get('truth_sha256_by_asset')!=EXPECTED_TRUTH: raise RuntimeError('reconstructed truth SHA authority drift')
    pre={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoContinuationPreOptimizerAuthority.v1','status':'FROZEN__CONTINUATION_OPTIMIZER_AUTHORIZED_ONLY','scientific_role':'LOCALIZATION_ONLY','original_8x2_status_immutable':EXPECTED_ORIGINAL_STATUS,'parent_checkpoint_sha256':psha,'parent_checkpoint_label':'TAIL_0512','parent_total_optimizer_steps':2560,'asset_count':8,'asset_style_cells':16,'microbatch_cells':8,'accumulation_steps':2,'effective_cells_per_optimizer_step':16,'samples_per_view':4096,'truth_sha256_by_asset':EXPECTED_TRUTH,'patchmatch_admitted':False,'architecture_changed':False,'scientific_continuation_optimizer_steps_at_freeze':0,'camera_json_consumed':False,'tune_consumed':False,'sealed_splits_opened':False}
    atomic_json(work/'CONTINUATION_PREOPT_AUTHORITY.json',pre)
    log=work/'CONTINUATION_TRAIN_PROCESS.log'
    run([sys.executable,str(here/'train_pv5_r256_8x2_continuation_localization_v1.py'),'--cache-manifest',str(cache/'CACHE_MANIFEST.json'),'--init-checkpoint',str(parent_cp),'--expected-parent-sha256',EXPECTED_PARENT_SHA,'--out-dir',str(train)],here,log)
    if persist.exists(): shutil.rmtree(persist)
    persist.mkdir(parents=True)
    for p in train.rglob('*'):
        if p.is_file():
            q=persist/p.relative_to(train); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
    for p in [work/'CONTINUATION_PREOPT_AUTHORITY.json',log,membership,here/'P_V5_R256_8X2_CONTINUATION_LOCALIZATION_PREREG_20260825.md']:
        shutil.copy2(p,persist/p.name)
    dec=json.load(open(train/'P_V5_R256_8X2_CONTINUATION_DECISION.json',encoding='utf-8'))
    complete={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoContinuationRunComplete.v1','status':dec['status'],'original_8x2_status_immutable':EXPECTED_ORIGINAL_STATUS,'parent_checkpoint_sha256':psha,'selected_label':dec['selected_label'],'selected_continuation_optimizer_steps':dec['selected_continuation_optimizer_steps'],'selected_total_optimizer_steps_including_parent':dec['selected_total_optimizer_steps_including_parent'],'selected_worst_cell_P_p95':dec['selected_worst_cell_P_p95'],'selected_aggregate_P_p95':dec['selected_aggregate_P_p95'],'selected_pass_count':dec['selected_pass_count'],'all_cells_pass_0p005':dec['all_cells_pass_0p005'],'patchmatch_admitted':False,'architecture_changed':False,'next_policy':dec['next_policy']}
    atomic_json(persist/'RUN_COMPLETE_P_V5_R256_8X2_CONTINUATION_V1.json',complete)
    print(json.dumps(complete,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
