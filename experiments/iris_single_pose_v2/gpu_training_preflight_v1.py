from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import torch
from losses import total_loss
from model import IRISSinglePoseV2,IRISV2Config,count_parameters

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)

def synthetic_batch(device,resolution=256,geom_samples=256,tracks=128):
 g=torch.Generator(device='cpu');g.manual_seed(20260824);images=torch.rand((1,8,4,resolution,resolution),generator=g).to(device);images[:,:,3]=1.;yaw=(torch.arange(8,dtype=torch.float32)[None]*45.).to(device);geom_xy=(torch.rand((1,8,geom_samples,2),generator=g)*1.8-.9).to(device);geom_p=(torch.rand((1,8,geom_samples,3),generator=g)-.5).to(device);geom_n=torch.nn.functional.normalize(torch.randn((1,8,geom_samples,3),generator=g).to(device),dim=-1);geom_mask=torch.ones((1,8,geom_samples),dtype=torch.bool,device=device);track_p=(torch.rand((1,tracks,3),generator=g)-.5).to(device);track_p[:,1::8]=track_p[:,0::8][:,:track_p[:,1::8].shape[1]]+.001;track_xy=(torch.rand((1,tracks,8,2),generator=g)*1.8-.9).to(device);vis=torch.ones((1,tracks,8),dtype=torch.bool,device=device);vis[:,::5,3]=False;vis[:,1::7,6]=False
 return {'images':images,'yaw_deg':yaw,'geom_xy':geom_xy,'geom_p':geom_p,'geom_n':geom_n,'geom_mask':geom_mask,'track_p':track_p,'track_xy':track_xy,'track_visible':vis}

def main():
 ap=argparse.ArgumentParser(description='No-step production-width mini training capacity probe');ap.add_argument('--out',required=True);ap.add_argument('--resolution',type=int,default=256);a=ap.parse_args()
 if a.resolution!=256:raise RuntimeError('mini prereg freezes capacity probe at 256')
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for mini capacity preflight')
 dev=torch.device('cuda');torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(dev);t0=time.time();status='PASS';error=None;model=None;state_bytes=0
 try:
  model=IRISSinglePoseV2(IRISV2Config()).to(dev).train();batch=synthetic_batch(dev,a.resolution)
  with torch.autocast(device_type='cuda',dtype=torch.float16):out=model(batch['images'],batch['yaw_deg']);parts=total_loss(out,batch,epoch=3,warmup_epochs=3);loss=parts['total']
  loss.backward();buffers=[]
  for p in model.parameters():
   if p.requires_grad:buffers.extend([torch.zeros_like(p,dtype=torch.float32,device=dev),torch.zeros_like(p,dtype=torch.float32,device=dev)]);state_bytes+=2*p.numel()*4
  torch.cuda.synchronize(dev);finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
  if not finite or not torch.isfinite(loss).item():raise RuntimeError('non-finite full-loss/backward capacity probe')
  _=sum(x.numel() for x in buffers);shapes={k:list(v.shape) for k,v in out.items()};scalar=float(loss.detach().float().cpu())
 except torch.cuda.OutOfMemoryError as exc:status='OOM';error=str(exc);shapes=None;scalar=None
 report={'schema':'RealSaS.IRISSinglePoseV2.MiniGPUCapacityPreflight.v1','status':status,'gpu':torch.cuda.get_device_name(dev),'input_resolution':a.resolution,'microbatch':1,'grad_accum_prereg':4,'full_loss_enabled':True,'forward_backward_executed':status=='PASS','adamw_moment_memory_accounted_without_step':status=='PASS','adamw_moment_bytes':state_bytes,'scientific_optimizer_steps':0,'parameters':count_parameters(model) if model is not None else None,'peak_allocated_GB':torch.cuda.max_memory_allocated(dev)/1e9,'peak_reserved_GB':torch.cuda.max_memory_reserved(dev)/1e9,'elapsed_sec':time.time()-t0,'loss':scalar,'shapes':shapes,'error':error};atomic_json(a.out,report);print(json.dumps(report,indent=2),flush=True)
 if status!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
