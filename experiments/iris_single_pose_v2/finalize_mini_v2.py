from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from dataset import IRISV2Dataset
from evaluate_mini_v2 import atomic_json,build_report
from model import IRISSinglePoseV2,IRISV2Config

TRACK_SAMPLES_EVAL=512; QUERY_LIMIT_FINAL=24; QUAL_LIMIT_FINAL=8
TARGET_P_P95=.005; TARGET_ZC_TOP8=.90; TARGET_ZF_P95_PX=16.0

def load_checkpoint(path,device):
 ck=torch.load(path,map_location='cpu');m=IRISSinglePoseV2(IRISV2Config());m.load_state_dict(ck['model_state'],strict=True);m.to(device).eval();return m,ck

def classify(s,t):
 select_core=s['P_p95']<=TARGET_P_P95 and s['Zc_top8']>=TARGET_ZC_TOP8 and s['oracle_Zf_top1_p95_native_px']<=TARGET_ZF_P95_PX
 tune_core=t['P_p95']<=TARGET_P_P95 and t['Zc_top8']>=TARGET_ZC_TOP8 and t['family_Zc_top8_p10']>=.75 and t['oracle_Zf_top1_p95_native_px']<=TARGET_ZF_P95_PX and t['min_style_Zc_top8']>=.85 and t['max_style_P_p95']<=.0065
 gap=t['P_p95']<=max(.0065,1.5*s['P_p95']) and t['Zc_top8']>=s['Zc_top8']-.10
 if select_core and tune_core and gap:return 'EXTRACTABILITY_GENERALIZATION_PASS'
 if select_core:return 'EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP'
 return 'LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT'

def main():
 ap=argparse.ArgumentParser(description='Frozen mini phase B: evaluate one frozen checkpoint on TUNE_FINAL only after FIT checkpoint freeze.')
 ap.add_argument('--tune-cache',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 out=Path(a.out_dir);freeze=out/'CHECKPOINT_SELECTION_FROZEN.json';selection_path=out/'CHECKPOINT_SELECTION.json';best_path=out/'BEST_CHECKPOINT.pt';init_path=out/'eval'/'FIT_SELECT_INIT.json'
 for p in (freeze,selection_path,best_path,init_path,out/'TRAIN_HISTORY.json'):
  if not p.is_file():raise FileNotFoundError(p)
 fr=json.load(open(freeze,encoding='utf-8'));sel=json.load(open(selection_path,encoding='utf-8'))
 if fr.get('tune_seen_during_selection') is not False or fr.get('tune_stage_cache_exists_during_selection') is not False:raise RuntimeError('checkpoint freeze TUNE firewall failed')
 if sel.get('tune_seen_during_selection') is not False or sel.get('tune_stage_cache_exists_during_selection') is not False:raise RuntimeError('selection TUNE firewall failed')
 tm=json.load(open(a.tune_cache,encoding='utf-8'))
 if tm.get('record_count')!=26 or tm.get('mini_role')!='TUNE_FINAL' or tm.get('sealed_splits_opened') is not False:raise RuntimeError('TUNE_FINAL cache contract drift')
 if len(IRISV2Dataset(a.tune_cache,split='TUNE',track_samples=1,style_mode='cel_clean'))!=26:raise RuntimeError('TUNE_FINAL cardinality drift')
 device=torch.device('cuda');model,ck=load_checkpoint(best_path,device);tune=build_report(model,ck,a.tune_cache,'TUNE',device,QUERY_LIMIT_FINAL,QUAL_LIMIT_FINAL,TRACK_SAMPLES_EVAL);tune['checkpoint']=str(best_path);atomic_json(out/'TUNE_FINAL_EVAL.json',tune)
 init=json.load(open(init_path,encoding='utf-8'));s=sel['selected_metrics'];label=classify(s,tune['selection_metrics'])
 result={'schema':'RealSaS.IRISSinglePoseV2.MiniExtractabilityResult.v2','status':label,'input_resolution':256,'train_assets':128,'fit_select_assets':32,'tune_final_assets':26,'selected_epoch':sel['selected_epoch'],'optimizer_steps':int(sel['selected_optimizer_steps']),'FIT_SELECT':s,'TUNE_FINAL':tune['selection_metrics'],'random_init_FIT_SELECT':init['selection_metrics'],'thresholds_frozen_pre_optimizer':{'P_p95_max':.005,'Zc_top8_min':.90,'family_Zc_top8_p10_min':.75,'oracle_Zf_top1_p95_native_px_max':16.0,'min_style_Zc_top8_min':.85,'max_style_P_p95_max':.0065,'generalization_P_ratio_max':1.5,'generalization_Zc_drop_max':.10},'normal_gate':'diagnostic_only_in_this_mini; correspondence authority and checkpoint-selection authority explicitly forbidden','sealed_splits_opened':False,'tune_used_for_checkpoint_selection':False,'tune_stage_started_after_checkpoint_freeze':True}
 atomic_json(out/'MINI_RESULT.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
