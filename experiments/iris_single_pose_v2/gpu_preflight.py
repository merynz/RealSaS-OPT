from __future__ import annotations
import argparse,json,time
import torch
from model import IRISSinglePoseV2,IRISV2Config,count_parameters

def run_resolution(model,resolution,device):
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);x=torch.zeros(1,8,4,resolution,resolution,device=device,dtype=torch.float16);yaw=torch.arange(8,device=device,dtype=torch.float32)[None]*45.;t0=time.time()
 try:
  with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):out=model(x,yaw)
  torch.cuda.synchronize(device);return {"status":"PASS","elapsed_sec":time.time()-t0,"peak_allocated_GB":torch.cuda.max_memory_allocated(device)/1e9,"peak_reserved_GB":torch.cuda.max_memory_reserved(device)/1e9,"shapes":{k:list(v.shape) for k,v in out.items()}}
 except torch.cuda.OutOfMemoryError as e:
  torch.cuda.empty_cache();return {"status":"OOM","error":str(e)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--resolutions",default="256,512,1024");a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA required; this is a no-optimizer capacity preflight")
 dev=torch.device("cuda");model=IRISSinglePoseV2(IRISV2Config()).to(dev).eval();report={"schema":"RealSaS.IRISSinglePoseV2.GPUPreflight.v1","gpu":torch.cuda.get_device_name(dev),"parameters":count_parameters(model),"optimizer_steps":0,"results":{}}
 for r in [int(x) for x in a.resolutions.split(",") if x.strip()]:report["results"][str(r)]=run_resolution(model,r,dev);print(json.dumps({str(r):report["results"][str(r)]},indent=2),flush=True)
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
