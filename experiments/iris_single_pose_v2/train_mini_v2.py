from __future__ import annotations
import argparse,json,math,os,random,shutil
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset import IRISV2Dataset
from evaluate_mini_v2 import atomic_json,build_report
from losses import total_loss
from model import IRISSinglePoseV2,IRISV2Config,count_parameters

SEED=20260824; EPOCHS=16; WARMUP_EPOCHS=3; CANDIDATE_EPOCHS=(4,8,12,16); LR=3e-4; WEIGHT_DECAY=1e-4; GRAD_ACCUM=4
TRACK_SAMPLES_TRAIN=128; TRACK_SAMPLES_EVAL=512; QUERY_LIMIT_SELECT=18; QUAL_LIMIT_SELECT=6
TARGET_P_P95=.005; TARGET_ZC_TOP8=.90; TARGET_ZF_P95_PX=16.0

def set_seed(seed=SEED):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def to_device(batch,device): return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def selection_score(m):
 vals=(m.get('P_p95'),m.get('Zc_top8'),m.get('oracle_Zf_top1_p95_native_px'),m.get('family_Zc_top8_p10'),m.get('family_P_p95_p90'),m.get('min_style_Zc_top8'),m.get('max_style_P_p95'))
 if any(x is None or not math.isfinite(float(x)) for x in vals): return float('inf')
 p,z,f,fz,fp,sz,sp=map(float,vals)
 return p/.005+max(0,1-z)/.10+f/16+max(0,.75-fz)/.25+fp/.0065+max(0,.85-sz)/.15+sp/.0065

def save_checkpoint(path,model,epoch,steps):
 torch.save({'schema':'RealSaS.IRISSinglePoseV2.MiniCheckpoint.v1','model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':int(epoch),'optimizer_steps':int(steps),'architecture':'IRISSinglePoseV2 default config','input_resolution':256,'normal_semantics':'local raster-observation orientation only; no correspondence ranking authority'},path)

def main():
 ap=argparse.ArgumentParser(description='Frozen IRIS V2 mini learner phase A: FIT_TRAIN -> FIT_SELECT -> checkpoint freeze. No TUNE path accepted.')
 ap.add_argument('--train-cache',required=True);ap.add_argument('--select-cache',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 set_seed();device=torch.device('cuda');out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True);(out/'eval').mkdir(exist_ok=True)
 train_ds=IRISV2Dataset(a.train_cache,split='FIT',track_samples=TRACK_SAMPLES_TRAIN,style_mode='random')
 if len(train_ds)!=128:raise RuntimeError(f'expected 128 training assets, got {len(train_ds)}')
 if len(IRISV2Dataset(a.select_cache,split='FIT',track_samples=1,style_mode='cel_clean'))!=32:raise RuntimeError('FIT_SELECT cardinality drift')
 model=IRISSinglePoseV2(IRISV2Config()).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(.9,.95));scaler=torch.cuda.amp.GradScaler(enabled=True)
 authority={'schema':'RealSaS.IRISSinglePoseV2.MiniTrainingAuthority.v2','phase':'FIT_ONLY_SELECTION','seed':SEED,'input_resolution':256,'epochs':EPOCHS,'warmup_epochs':WARMUP_EPOCHS,'candidate_epochs':list(CANDIDATE_EPOCHS),'optimizer':{'name':'AdamW','lr':LR,'weight_decay':WEIGHT_DECAY,'betas':[.9,.95]},'microbatch':1,'grad_accum':GRAD_ACCUM,'track_samples_train':TRACK_SAMPLES_TRAIN,'track_samples_eval':TRACK_SAMPLES_EVAL,'style_train':'deterministic per-asset/per-epoch random cel_clean or ink_cel','style_eval':'both styles, equal macro','augmentations':'none beyond frozen style alternation','checkpoint_selection':{'score':'P_p95/.005 + max(0,1-Zc_top8)/.10 + oracle_Zf_p95/16 + max(0,.75-family_Zc_p10)/.25 + family_P_p95_p90/.0065 + max(0,.85-min_style_Zc)/.15 + max_style_P_p95/.0065','lower_is_better':True,'evaluated_only_on':'FIT_SELECT','candidate_epochs':list(CANDIDATE_EPOCHS)},'tune_policy':'no TUNE argument/path is accepted by this phase; TUNE staging starts only after CHECKPOINT_SELECTION_FROZEN.json exists','normal_policy':'N local raster-observation orientation only; excluded from correspondence ranking and checkpoint selection','parameters':count_parameters(model),'tune_seen':False}
 atomic_json(out/'TRAIN_AUTHORITY.json',authority)
 init=build_report(model,{'epoch':0,'optimizer_steps':0},a.select_cache,'FIT',device,QUERY_LIMIT_SELECT,QUAL_LIMIT_SELECT,TRACK_SAMPLES_EVAL);atomic_json(out/'eval'/'FIT_SELECT_INIT.json',init)
 history=[];candidates=[];steps=0
 for epoch in range(1,EPOCHS+1):
  train_ds.set_epoch(epoch);gen=torch.Generator();gen.manual_seed(SEED+epoch);loader=DataLoader(train_ds,batch_size=1,shuffle=True,generator=gen,num_workers=0,pin_memory=True);model.train();optimizer.zero_grad(set_to_none=True);sums={};count=0
  for bi,batch in enumerate(loader,1):
   batch=to_device(batch,device)
   with torch.autocast(device_type='cuda',dtype=torch.float16):
    outputs=model(batch['images'],batch['yaw_deg']);parts=total_loss(outputs,batch,epoch=epoch-1,warmup_epochs=WARMUP_EPOCHS);loss=parts['total']/GRAD_ACCUM
   scaler.scale(loss).backward();do_step=(bi%GRAD_ACCUM==0) or bi==len(loader)
   if do_step:
    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
   for k,v in parts.items():
    if torch.is_tensor(v):sums[k]=sums.get(k,0.0)+float(v.detach().float().cpu())
   count+=1
   if bi%16==0 or bi==len(loader):print(f'[mini-train] epoch={epoch}/{EPOCHS} asset={bi}/{len(loader)} opt_steps={steps}',flush=True)
  history.append({'epoch':epoch,'optimizer_steps':steps,'mean':{k:v/max(1,count) for k,v in sums.items()}});atomic_json(out/'TRAIN_HISTORY.json',{'epochs':history})
  if epoch in CANDIDATE_EPOCHS:
   cp=out/'checkpoints'/f'epoch_{epoch:02d}.pt';save_checkpoint(cp,model,epoch,steps);model.eval();rep=build_report(model,{'epoch':epoch,'optimizer_steps':steps},a.select_cache,'FIT',device,QUERY_LIMIT_SELECT,QUAL_LIMIT_SELECT,TRACK_SAMPLES_EVAL);rep['checkpoint']=str(cp);rep['selection_score']=selection_score(rep['selection_metrics']);atomic_json(out/'eval'/f'FIT_SELECT_EPOCH_{epoch:02d}.json',rep);candidates.append({'epoch':epoch,'optimizer_steps':steps,'checkpoint':str(cp),'selection_score':rep['selection_score'],'selection_metrics':rep['selection_metrics']});print(json.dumps({'candidate_epoch':epoch,'score':rep['selection_score'],**rep['selection_metrics']},indent=2),flush=True)
 if not candidates:raise RuntimeError('no candidate checkpoints')
 best=min(candidates,key=lambda x:(x['selection_score'],x['epoch']));frozen=out/'BEST_CHECKPOINT.pt';shutil.copy2(best['checkpoint'],frozen)
 selection={'schema':'RealSaS.IRISSinglePoseV2.MiniCheckpointSelection.v2','selection_policy':authority['checkpoint_selection'],'candidates':candidates,'selected_epoch':best['epoch'],'selected_optimizer_steps':best['optimizer_steps'],'selected_score':best['selection_score'],'selected_metrics':best['selection_metrics'],'best_checkpoint':str(frozen),'tune_seen_during_selection':False,'tune_stage_cache_exists_during_selection':False}
 atomic_json(out/'CHECKPOINT_SELECTION.json',selection)
 atomic_json(out/'CHECKPOINT_SELECTION_FROZEN.json',{'schema':'RealSaS.IRISSinglePoseV2.MiniCheckpointFreeze.v1','selected_epoch':best['epoch'],'selected_optimizer_steps':best['optimizer_steps'],'selected_score':best['selection_score'],'tune_seen_during_selection':False,'tune_stage_cache_exists_during_selection':False,'next_authority':'Only now may TUNE_FINAL image/truth staging begin.'})
 for p in (out/'checkpoints').glob('epoch_*.pt'):p.unlink(missing_ok=True)
 print(json.dumps({'status':'FIT_CHECKPOINT_FROZEN','selected_epoch':best['epoch'],'optimizer_steps':best['optimizer_steps'],'tune_seen_during_selection':False},indent=2),flush=True)

if __name__=='__main__':main()
