from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
EXPECTED_REP_SEED_SHA='f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af';EXPECTED_MEMBERSHIP_CANONICAL_SHA='4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055';EXPECTED_PANEL_DIGEST='366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961';ROLE_COUNTS={'FIT_TRAIN':128,'FIT_SELECT':32,'TUNE_FINAL':26}
def sha256_file(path,chunk=8<<20):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(chunk),b''):h.update(b)
 return h.hexdigest()
def canonical_json_sha(path):
 o=json.load(open(path,encoding='utf-8'));return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def run(cmd):print('+',' '.join(map(str,cmd)),flush=True);subprocess.check_call([str(x) for x in cmd])
def script_hashes(here):
 names=['MINI_EXTRACTABILITY_MEMBERSHIP_V1.json','build_mini_seed_v1.py','stage_source.py','prepare_cache.py','audit_staged_cache_v2.py','dataset.py','losses.py','model.py','coords.py','geometry.py','matcher.py','metrics.py','qualification.py','evaluate_v2.py','evaluate_mini_v2.py','gpu_training_preflight_v1.py','train_mini_v2.py','finalize_mini_v2.py','compact_mini_handoff_v1.py','run_mini_extractability_v1.py']
 return {n:sha256_file(here/n) for n in names}
def verify_gpu(path):
 o=json.load(open(path,encoding='utf-8'));req={'schema':'RealSaS.IRISSinglePoseV2.MiniGPUCapacityPreflight.v1','status':'PASS','input_resolution':256,'microbatch':1,'full_loss_enabled':True,'forward_backward_executed':True,'adamw_moment_memory_accounted_without_step':True,'scientific_optimizer_steps':0}
 for k,v in req.items():
  if o.get(k)!=v:raise RuntimeError(f'GPU preflight failed {k}: {o.get(k)!r}')
 return o
def filter_manifest(full_path,seed,role,out):
 full=json.load(open(full_path,encoding='utf-8'));by={r['asset_id']:r for r in full['records']};ids=[r['asset_id'] for r in seed['records'] if r['mini_role']==role];rows=[]
 for aid in ids:
  if aid not in by:raise RuntimeError(f'role asset absent {role}:{aid}')
  rows.append(by[aid])
 if len(rows)!=ROLE_COUNTS[role]:raise RuntimeError(f'role count drift {role}:{len(rows)}')
 obj={**{k:v for k,v in full.items() if k!='records'},'record_count':len(rows),'records':rows,'mini_role':role,'role_asset_ids':ids,'sealed_splits_opened':False};atomic_json(out,obj);return obj
def main():
 ap=argparse.ArgumentParser(description='Frozen image->observable-evidence mini with minimal membership-only runtime seed and hard TUNE staging firewall');ap.add_argument('--root',required=True);ap.add_argument('--membership',default=str(Path(__file__).with_name('MINI_EXTRACTABILITY_MEMBERSHIP_V1.json')));ap.add_argument('--gpu-preflight-json',required=True);ap.add_argument('--work-dir',required=True);a=ap.parse_args();root=Path(a.root);here=Path(__file__).resolve().parent;work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
 msh=canonical_json_sha(a.membership)
 if msh!=EXPECTED_MEMBERSHIP_CANONICAL_SHA:raise RuntimeError(f'mini membership canonical SHA drift {msh}')
 mem=json.load(open(a.membership,encoding='utf-8'))
 if mem.get('source_representation_seed_sha256')!=EXPECTED_REP_SEED_SHA or mem.get('source_panel_asset_id_digest')!=EXPECTED_PANEL_DIGEST or mem.get('sealed_splits_opened') is not False:raise RuntimeError('membership provenance drift')
 capacity=verify_gpu(a.gpu_preflight_json);scripts=script_hashes(here)
 if canonical_json_sha(here/'MINI_EXTRACTABILITY_MEMBERSHIP_V1.json')!=EXPECTED_MEMBERSHIP_CANONICAL_SHA:raise RuntimeError('bundled membership semantic drift')
 seed=work/'MINI_EXTRACTABILITY_RUNTIME_SEED_V2.json';fit_stage=work/'stage_fit_v2_256';fit_cache=work/'cache_fit_v2_256';fit_audit=work/'FIT_STAGE_CACHE_AUDIT_V2_256.json';roles=work/'role_manifests';learner=work/'learner';tune_stage=work/'stage_tune_v2_256';tune_cache=work/'cache_tune_v2_256';tune_audit=work/'TUNE_STAGE_CACHE_AUDIT_V2_256.json';compact=work/'MINI_COMPACT_HANDOFF_V1.json'
 run([sys.executable,here/'build_mini_seed_v1.py','--membership',a.membership,'--out',seed]);ms=json.load(open(seed,encoding='utf-8'))
 if ms.get('panel_asset_id_digest')!=EXPECTED_PANEL_DIGEST or ms.get('record_count')!=186 or ms.get('representation_seed_consumed_at_runtime') is not False or ms.get('runtime_tune_fields')!=['asset_id','split','mini_role']:raise RuntimeError('minimal mini runtime seed drift')
 run([sys.executable,here/'stage_source.py','--root',root,'--seed-manifest',seed,'--out',fit_stage,'--input-resolution','256','--splits','FIT']);fsm=fit_stage/'STAGE_MANIFEST.json'
 run([sys.executable,here/'prepare_cache.py','--stage-manifest',fsm,'--out',fit_cache,'--geom-samples','1024','--anchors-per-view','512','--max-tracks','2048','--radius-px','3','--max-surface-error','.003']);fcm=fit_cache/'CACHE_MANIFEST.json'
 run([sys.executable,here/'audit_staged_cache_v2.py','--stage-manifest',fsm,'--cache-manifest',fcm,'--seed-manifest',seed,'--out',fit_audit,'--progress-every','16']);fao=json.load(open(fit_audit,encoding='utf-8'))
 if fao.get('status')!='PASS' or fao.get('fatal_asset_count')!=0 or fao.get('asset_count')!=160 or fao.get('split_counts')!={'FIT':160}:raise RuntimeError(f'FIT stage/cache audit failed {fao.get("status")} {fao.get("split_counts")}')
 roles.mkdir(parents=True,exist_ok=True);train=roles/'FIT_TRAIN_CACHE_MANIFEST.json';select=roles/'FIT_SELECT_CACHE_MANIFEST.json';filter_manifest(fcm,ms,'FIT_TRAIN',train);filter_manifest(fcm,ms,'FIT_SELECT',select)
 if tune_stage.exists() or tune_cache.exists():raise RuntimeError('TUNE stage/cache exists before checkpoint freeze')
 pre={'schema':'RealSaS.IRISSinglePoseV2.MiniPreOptimizerAuthority.v3','status':'FROZEN__OPTIMIZER_AUTHORIZED_FOR_THIS_MINI_ONLY','representation_gate':'P_GEOMETRY_SUFFICIENT','source_representation_seed_sha256_provenance_only':EXPECTED_REP_SEED_SHA,'representation_seed_consumed_at_runtime':False,'panel_asset_id_digest':EXPECTED_PANEL_DIGEST,'membership_canonical_sha256':EXPECTED_MEMBERSHIP_CANONICAL_SHA,'membership_artifact_byte_sha256':sha256_file(a.membership),'runtime_seed_sha256':sha256_file(seed),'runtime_tune_fields':['asset_id','split','mini_role'],'fit_stage_manifest_sha256':sha256_file(fsm),'fit_cache_manifest_sha256':sha256_file(fcm),'fit_stage_cache_audit_sha256':sha256_file(fit_audit),'role_manifest_sha256':{'FIT_TRAIN':sha256_file(train),'FIT_SELECT':sha256_file(select)},'gpu_capacity_preflight_sha256':sha256_file(a.gpu_preflight_json),'gpu_capacity':capacity,'script_sha256':scripts,'input_resolution':256,'role_counts':ROLE_COUNTS,'normal_policy':'local raster-observation orientation auxiliary target only; forbidden from correspondence ranking and checkpoint selection','tune_policy':'TUNE_FINAL membership IDs frozen, no TUNE content/metadata from CI104 seed consumed at runtime, and no TUNE image/truth stage/cache exists before FIT checkpoint freeze','tune_stage_cache_exists_at_optimizer_open':False,'sealed_splits_opened':False,'optimizer_steps_at_freeze':0};atomic_json(work/'MINI_PREOPT_AUTHORITY.json',pre)
 run([sys.executable,here/'train_mini_v2.py','--train-cache',train,'--select-cache',select,'--out-dir',learner]);freeze=learner/'CHECKPOINT_SELECTION_FROZEN.json'
 if not freeze.is_file():raise RuntimeError('checkpoint freeze missing; TUNE may not open')
 fro=json.load(open(freeze,encoding='utf-8'))
 if fro.get('tune_seen_during_selection') is not False or fro.get('tune_stage_cache_exists_during_selection') is not False:raise RuntimeError('checkpoint freeze TUNE firewall failed')
 run([sys.executable,here/'stage_source.py','--root',root,'--seed-manifest',seed,'--out',tune_stage,'--input-resolution','256','--splits','TUNE']);tsm=tune_stage/'STAGE_MANIFEST.json'
 run([sys.executable,here/'prepare_cache.py','--stage-manifest',tsm,'--out',tune_cache,'--geom-samples','1024','--anchors-per-view','512','--max-tracks','2048','--radius-px','3','--max-surface-error','.003']);tcm=tune_cache/'CACHE_MANIFEST.json'
 run([sys.executable,here/'audit_staged_cache_v2.py','--stage-manifest',tsm,'--cache-manifest',tcm,'--seed-manifest',seed,'--out',tune_audit,'--progress-every','8']);tao=json.load(open(tune_audit,encoding='utf-8'))
 if tao.get('status')!='PASS' or tao.get('fatal_asset_count')!=0 or tao.get('asset_count')!=26 or tao.get('split_counts')!={'TUNE':26}:raise RuntimeError(f'TUNE stage/cache audit failed {tao.get("status")} {tao.get("split_counts")}')
 tune=roles/'TUNE_FINAL_CACHE_MANIFEST.json';filter_manifest(tcm,ms,'TUNE_FINAL',tune)
 postfreeze={'schema':'RealSaS.IRISSinglePoseV2.MiniTuneOpenAuthority.v2','checkpoint_freeze_sha256':sha256_file(freeze),'best_checkpoint_sha256':sha256_file(learner/'BEST_CHECKPOINT.pt'),'tune_stage_manifest_sha256':sha256_file(tsm),'tune_cache_manifest_sha256':sha256_file(tcm),'tune_stage_cache_audit_sha256':sha256_file(tune_audit),'tune_role_manifest_sha256':sha256_file(tune),'tune_opened_after_checkpoint_freeze':True,'tune_used_for_checkpoint_selection':False,'representation_seed_consumed_at_runtime':False,'sealed_splits_opened':False};atomic_json(work/'TUNE_OPEN_AUTHORITY.json',postfreeze)
 run([sys.executable,here/'finalize_mini_v2.py','--tune-cache',tune,'--out-dir',learner]);run([sys.executable,here/'compact_mini_handoff_v1.py','--run-dir',learner,'--out',compact]);r=json.load(open(learner/'MINI_RESULT.json',encoding='utf-8'));h=json.load(open(compact,encoding='utf-8'))
 if h['status']!=r['status'] or r.get('sealed_splits_opened') is not False or r.get('tune_used_for_checkpoint_selection') is not False or r.get('tune_stage_started_after_checkpoint_freeze') is not True:raise RuntimeError('mini result discipline drift')
 final={'schema':'RealSaS.IRISSinglePoseV2.MiniRunComplete.v3','status':r['status'],'representation_gate':'P_GEOMETRY_SUFFICIENT','source_panel_asset_id_digest':EXPECTED_PANEL_DIGEST,'source_representation_seed_sha256_provenance_only':EXPECTED_REP_SEED_SHA,'representation_seed_consumed_at_runtime':False,'membership_canonical_sha256':EXPECTED_MEMBERSHIP_CANONICAL_SHA,'preoptimizer_authority_sha256':sha256_file(work/'MINI_PREOPT_AUTHORITY.json'),'tune_open_authority_sha256':sha256_file(work/'TUNE_OPEN_AUTHORITY.json'),'selected_epoch':r['selected_epoch'],'optimizer_steps':r['optimizer_steps'],'best_checkpoint_sha256':sha256_file(learner/'BEST_CHECKPOINT.pt'),'mini_result_sha256':sha256_file(learner/'MINI_RESULT.json'),'compact_handoff_sha256':sha256_file(compact),'sealed_splits_opened':False,'tune_used_for_checkpoint_selection':False,'tune_stage_started_after_checkpoint_freeze':True,'normal_correspondence_authority':False,'next_authority':'Interpret frozen mini result; do not alter thresholds after TUNE.'};atomic_json(work/'RUN_COMPLETE_MINI_V1.json',final);print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__':main()
