from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
PARENT_RUN='IRIS_SINGLE_POSE_V2_P_V5_R256_TWO_STYLE_OVERFIT_V1'; PARENT_STATUS='P_V5_R256_TWO_STYLE_OVERFIT_PASS'; PARENT_DEC='P_V5_R256_TWO_STYLE_DECISION.json'; PARENT_DEC_SHA='2250f7c21076c0ae0b04093b5e711d2a2c0d26ababb2f2960ea0759a007cb5eb'
def sha(path,chunk=8<<20):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(chunk),b''): h.update(x)
 return h.hexdigest()
def atomic(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def cmd(x): print('+',' '.join(map(str,x)),flush=True); subprocess.check_call([str(q) for q in x])
def cmd_logged(x,log_path):
    print('+',' '.join(map(str,x)),flush=True)
    log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
    with open(log_path,'w',encoding='utf-8') as log:
        proc=subprocess.Popen([str(q) for q in x],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line,end='',flush=True); log.write(line); log.flush()
        rc=proc.wait()
    if rc!=0:
        raise subprocess.CalledProcessError(rc,[str(q) for q in x])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--persist-dir',required=True); a=ap.parse_args(); here=Path(__file__).resolve().parent; root=Path(a.root); work=Path(a.work_dir); persist=Path(a.persist_dir)
 if persist.exists(): raise RuntimeError(f'persist target exists: {persist}')
 if work.exists(): shutil.rmtree(work)
 work.mkdir(parents=True)
 pd=root/'runs'/PARENT_RUN/PARENT_DEC
 if not pd.is_file(): raise FileNotFoundError(pd)
 po=json.load(open(pd));
 if po.get('status')!=PARENT_STATUS or sha(pd)!=PARENT_DEC_SHA or po.get('all_cells_pass_0p005') is not True: raise RuntimeError('parent two-style authority drift')
 if po.get('tune_consumed') is not False or po.get('sealed_splits_opened') is not False or po.get('camera_json_consumed') is not False: raise RuntimeError('parent firewall drift')
 cpu=work/'CPU_PREFLIGHT.json'; gpu=work/'GPU_PREFLIGHT.json'; cmd([sys.executable,here/'pv5_r256_8x2_cpu_preflight.py','--out',cpu]); cmd([sys.executable,here/'pv5_r256_8x2_gpu_preflight.py','--out',gpu])
 go=json.load(open(gpu));
 if go.get('status')!='PASS' or go.get('effective_cells_per_optimizer_step')!=16 or go.get('scientific_optimizer_steps')!=0: raise RuntimeError('GPU preflight drift')
 stage=work/'stage'; cache=work/'cache'; train=work/'training'; mem=here/'P_V5_R256_8X2_MEMBERSHIP_V1.json'; prereg=here/'P_V5_R256_8X2_PREREG_20260825.md'
 cmd([sys.executable,here/'stage_pv5_r256_8x2_v1.py','--root',root,'--membership',mem,'--out',stage]); sm=stage/'STAGE_MANIFEST.json'; so=json.load(open(sm));
 if so.get('record_count')!=16 or so.get('asset_count')!=8 or so.get('camera_json_consumed') is not False: raise RuntimeError('stage firewall')
 cmd([sys.executable,here/'prepare_pv5_r256_8x2_cache_v1.py','--stage-manifest',sm,'--out',cache,'--samples-per-view','4096']); cm=cache/'CACHE_MANIFEST.json'; ca=json.load(open(cm));
 if ca.get('record_count')!=16 or ca.get('asset_count')!=8 or ca.get('shared_truth_loci_across_styles') is not True: raise RuntimeError('cache authority')
 pre={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoPreOptimizerAuthority.v1','status':'FROZEN__OPTIMIZER_AUTHORIZED_FOR_8X2_ONLY','parent_status':po['status'],'parent_decision_sha256':sha(pd),'prereg_sha256':sha(prereg),'membership_sha256':sha(mem),'cpu_preflight_sha256':sha(cpu),'gpu_preflight_sha256':sha(gpu),'stage_manifest_sha256':sha(sm),'cache_manifest_sha256':sha(cm),'asset_count':8,'asset_style_cells':16,'output_field_hw':256,'effective_cells_per_optimizer_step':16,'microbatch_cells':8,'accumulation_steps':2,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'scientific_optimizer_steps_at_freeze':0}; atomic(work/'PREOPT_AUTHORITY.json',pre)
 cmd_logged([sys.executable,here/'train_pv5_r256_8x2_v1.py','--cache-manifest',cm,'--out-dir',train],work/'TRAIN_PROCESS.log'); dec=train/'P_V5_R256_8X2_DECISION.json'; d=json.load(open(dec));
 complete={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoRunComplete.v1','status':d['status'],'parent_status':po['status'],'decision_sha256':sha(dec),'best_checkpoint_sha256':sha(train/'BEST_CHECKPOINT.pt'),'preopt_authority_sha256':sha(work/'PREOPT_AUTHORITY.json'),'selected_label':d['selected_label'],'selected_total_optimizer_steps':d['selected_total_optimizer_steps'],'selected_worst_cell_P_p95':d['selected_worst_cell_P_p95'],'selected_aggregate_P_p95':d['selected_aggregate_P_p95'],'selected_pass_count':d['selected_pass_count'],'selected_per_cell':d['selected_per_cell'],'threshold_P_p95_per_cell':d['threshold_P_p95_per_cell'],'asset_count':8,'asset_style_cells':16,'output_field_hw':256,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':d['next_policy']}; atomic(work/'RUN_COMPLETE_P_V5_R256_8X2_V1.json',complete)
 persist.parent.mkdir(parents=True,exist_ok=True); tmp=persist.with_name(persist.name+'.partial');
 if tmp.exists(): shutil.rmtree(tmp)
 tmp.mkdir()
 for p in (cpu,gpu,work/'PREOPT_AUTHORITY.json',work/'RUN_COMPLETE_P_V5_R256_8X2_V1.json',work/'TRAIN_PROCESS.log',dec,train/'BEST_CHECKPOINT.pt',train/'TRAIN_AUTHORITY.json',train/'TRAIN_HISTORY.json',prereg,mem): shutil.copy2(p,tmp/p.name)
 shutil.copytree(train/'eval',tmp/'eval'); os.replace(tmp,persist); print(json.dumps(complete,indent=2,sort_keys=True),flush=True)
if __name__=='__main__': main()
