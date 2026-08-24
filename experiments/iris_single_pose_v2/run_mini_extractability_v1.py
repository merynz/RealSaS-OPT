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
 names=['MINI_EXTRACTABILITY_MEMBERSHIP_V1.json','build_mini_seed_v1.py','stage_source.py','prepare_cache.py','audit_staged_cache_v2.py','dataset.py','losses.py','model.py','coords.py','geometry.py','matcher.py','metrics.py','qualification.py','evaluate_v2.py','evaluate_mini_v2.py','gpu_training_preflight_v1.py','train_mini_v2.py','compact_mini_handoff_v1.py','run_mini_extractability_v1.py']
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
 ap=argparse.ArgumentParser(description='Frozen image->observable-evidence mini: FIT train/select -> untouched TUNE final');ap.add_argument('--root',required=True);ap.add_argument('--representation-seed',required=True);ap.add_argument('--membership',default=str(Path(__file__).with_name('MINI_EXTRACTABILITY_MEMBERSHIP_V1.json')));ap.add_argument('--gpu-preflight-json',required=True);ap.add_argument('--work-dir',required=True);a=ap.parse_args();root=Path(a.root);here=Path(__file__).resolve().parent;work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
 if sha256_file(a.representation_seed)!=EXPECTED_REP_SEED_SHA:raise RuntimeError('CI104 representation seed SHA drift')
 msh=canonical_json_sha(a.membership)
 if msh!=EXPECTED_MEMBERSHIP_CANONICAL_SHA:raise RuntimeError(f'mini membership canonical SHA drift {msh}')
 capacity=verify_gpu(a.gpu_preflight_json);scripts=script_hashes(here)
 if canonical_json_sha(here/'MINI_EXTRACTABILITY_MEMBERSHIP_V1.json')!=EXPECTED_MEMBERSHIP_CANONICAL_SHA:raise RuntimeError('bundled membership semantic drift')
 seed=work/'MINI_EXTRACTABILITY_SEED_V1.json';stage=work/'stage_v2_256';cache=work/'cache_v2_256';audit=work/'STAGE_CACHE_AUDIT_V2_256.json';roles=work/'role_manifests';learner=work/'learner';compact=work/'MINI_COMPACT_HANDOFF_V1.json'
 run([sys.executable,here/'build_mini_seed_v1.py','--representation-seed',a.representation_seed,'--membership',a.membership,'--out',seed]);ms=json.load(open(seed,encoding='utf-8'))
 if ms.get('panel_asset_id_digest')!=EXPECTED_PANEL_DIGEST or ms.get('record_count')!=186:raise RuntimeError('mini seed drift')
 run([sys.executable,here/'stage_source.py','--root',root,'--seed-manifest',seed,'--out',stage,'--input-resolution','256','--splits','FIT,TUNE']);sm=stage/'STAGE_MANIFEST.json'
 run([sys.executable,here/'prepare_cache.py','--stage-manifest',sm,'--out',cache,'--geom-samples','1024','--anchors-per-view','512','--max-tracks','2048','--radius-px','3','--max-surface-error','.003']);cm=cache/'CACHE_MANIFEST.json'
 run([sys.executable,here/'audit_staged_cache_v2.py','--stage-manifest',sm,'--cache-manifest',cm,'--seed-manifest',seed,'--out',audit,'--progress-every','16']);ao=json.load(open(audit,encoding='utf-8'))
 if ao.get('status')!='PASS' or ao.get('fatal_asset_count')!=0 or ao.get('asset_count')!=186 or ao.get('split_counts')!={'FIT':160,'TUNE':26}:raise RuntimeError(f'mini stage/cache audit failed {ao.get("status")} {ao.get("split_counts")}')
 roles.mkdir(parents=True,exist_ok=True);train=roles/'FIT_TRAIN_CACHE_MANIFEST.json';select=roles/'FIT_SELECT_CACHE_MANIFEST.json';tune=roles/'TUNE_FINAL_CACHE_MANIFEST.json';filter_manifest(cm,ms,'FIT_TRAIN',train);filter_manifest(cm,ms,'FIT_SELECT',select);filter_manifest(cm,ms,'TUNE_FINAL',tune)
 pre={'schema':'RealSaS.IRISSinglePoseV2.MiniPreOptimizerAuthority.v1','status':'FROZEN__OPTIMIZER_AUTHORIZED_FOR_THIS_MINI_ONLY','representation_gate':'P_GEOMETRY_SUFFICIENT','representation_seed_sha256':EXPECTED_REP_SEED_SHA,'panel_asset_id_digest':EXPECTED_PANEL_DIGEST,'membership_canonical_sha256':EXPECTED_MEMBERSHIP_CANONICAL_SHA,'membership_artifact_byte_sha256':sha256_file(a.membership),'mini_seed_sha256':sha256_file(seed),'stage_manifest_sha256':sha256_file(sm),'cache_manifest_sha256':sha256_file(cm),'stage_cache_audit_sha256':sha256_file(audit),'role_manifest_sha256':{'FIT_TRAIN':sha256_file(train),'FIT_SELECT':sha256_file(select),'TUNE_FINAL':sha256_file(tune)},'gpu_capacity_preflight_sha256':sha256_file(a.gpu_preflight_json),'gpu_capacity':capacity,'script_sha256':scripts,'input_resolution':256,'role_counts':ROLE_COUNTS,'normal_policy':'local raster-observation orientation auxiliary target only; forbidden from correspondence ranking and checkpoint selection','sealed_splits_opened':False,'optimizer_steps_at_freeze':0};atomic_json(work/'MINI_PREOPT_AUTHORITY.json',pre)
 run([sys.executable,here/'train_mini_v2.py','--train-cache',train,'--select-cache',select,'--tune-cache',tune,'--out-dir',learner]);run([sys.executable,here/'compact_mini_handoff_v1.py','--run-dir',learner,'--out',compact]);r=json.load(open(learner/'MINI_RESULT.json',encoding='utf-8'));h=json.load(open(compact,encoding='utf-8'))
 if h['status']!=r['status'] or r.get('sealed_splits_opened') is not False or r.get('tune_used_for_checkpoint_selection') is not False:raise RuntimeError('mini result discipline drift')
 final={'schema':'RealSaS.IRISSinglePoseV2.MiniRunComplete.v1','status':r['status'],'representation_gate':'P_GEOMETRY_SUFFICIENT','source_panel_asset_id_digest':EXPECTED_PANEL_DIGEST,'membership_canonical_sha256':EXPECTED_MEMBERSHIP_CANONICAL_SHA,'preoptimizer_authority_sha256':sha256_file(work/'MINI_PREOPT_AUTHORITY.json'),'selected_epoch':r['selected_epoch'],'optimizer_steps':r['optimizer_steps'],'best_checkpoint_sha256':sha256_file(learner/'BEST_CHECKPOINT.pt'),'mini_result_sha256':sha256_file(learner/'MINI_RESULT.json'),'compact_handoff_sha256':sha256_file(compact),'sealed_splits_opened':False,'tune_used_for_checkpoint_selection':False,'normal_correspondence_authority':False,'next_authority':'Interpret frozen mini result; do not alter thresholds after TUNE.'};atomic_json(work/'RUN_COMPLETE_MINI_V1.json',final);print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__':main()
