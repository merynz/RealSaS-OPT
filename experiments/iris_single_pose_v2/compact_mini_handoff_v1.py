from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
def sha256_file(path,chunk=8<<20):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(chunk),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();rd=Path(a.run_dir);rp=rd/'MINI_RESULT.json';sp=rd/'CHECKPOINT_SELECTION.json';tp=rd/'TUNE_FINAL_EVAL.json';hp=rd/'TRAIN_HISTORY.json';bp=rd/'BEST_CHECKPOINT.pt'
 for p in (rp,sp,tp,hp,bp):
  if not p.is_file():raise FileNotFoundError(p)
 r=json.load(open(rp,encoding='utf-8'))
 if r.get('sealed_splits_opened') is not False or r.get('tune_used_for_checkpoint_selection') is not False:raise RuntimeError('mini split discipline failed')
 out={'schema':'RealSaS.IRISSinglePoseV2.MiniExtractabilityCompactHandoff.v1','status':r['status'],'selected_epoch':r['selected_epoch'],'optimizer_steps':r['optimizer_steps'],'FIT_SELECT':r['FIT_SELECT'],'TUNE_FINAL':r['TUNE_FINAL'],'random_init_FIT_SELECT':r['random_init_FIT_SELECT'],'thresholds_frozen_pre_optimizer':r['thresholds_frozen_pre_optimizer'],'normal_gate':r['normal_gate'],'best_checkpoint_sha256':sha256_file(bp),'selection_sha256':sha256_file(sp),'tune_eval_sha256':sha256_file(tp),'history_sha256':sha256_file(hp),'sealed_splits_opened':False,'tune_used_for_checkpoint_selection':False,'non_claims':['256-input mini does not establish final 1024 localization quality','source-track-free dense autonomous evidence construction remains open','SurfaceBuilder/Geppetto/Arachne downstream sufficiency remains open','artist-domain robustness remains open']};atomic_json(a.out,out);print(json.dumps({k:out[k] for k in ('status','selected_epoch','optimizer_steps','FIT_SELECT','TUNE_FINAL')},indent=2))
if __name__=='__main__':main()
