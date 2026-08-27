from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from realsas_mapanything_ortho.config import OrthoConfig
from realsas_mapanything_ortho.dataset import OrthoFamilyDataset, collate_one_family
from realsas_mapanything_ortho.model import MapAnythingOrthoIRIS
from realsas_mapanything_ortho.losses import geometry_loss, geometry_metrics


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def stage_for_epoch(epoch0,cfg):
    if epoch0<cfg.epochs_adapter:return "adapter"
    if epoch0<cfg.epochs_adapter+cfg.epochs_geometry:return "geometry"
    return "full"


def build_optimizer(model,cfg):
    groups=model.parameter_groups()
    for g in groups:g["weight_decay"]=cfg.weight_decay
    return torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8)


def move_target(target,device):return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in target.items()}


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();rows=[]
    for batch in loader:
        images=batch["images"].to(device,non_blocking=True);target=move_target(batch["target"],device)
        with torch.autocast("cuda",dtype=torch.bfloat16):outputs=model(images)
        m=geometry_metrics(outputs,target);m["family_id"]=int(target["family_id"][0].item());rows.append(m)
    keys=["P_mean","P50","P90","P95","N_deg_median","N_deg_p95"];agg={k:float(np.mean([r[k] for r in rows])) for k in keys}
    if rows:agg["family_P95_p90"]=float(np.quantile([r["P95"] for r in rows],.90));agg["family_P95_worst"]=float(max(r["P95"] for r in rows))
    return {"aggregate":agg,"families":rows}


def main():
    ap=argparse.ArgumentParser(description="RealSaS IRIS MapAnything-Apache orthographic fine-tuning")
    ap.add_argument("--manifest",required=True);ap.add_argument("--preflight-json",required=True);ap.add_argument("--out",required=True);ap.add_argument("--train-split",default="FIT");ap.add_argument("--val-split",default="TUNE");ap.add_argument("--resume");args=ap.parse_args()
    cfg=OrthoConfig()
    if cfg.require_cuda and not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    if cfg.require_bf16 and not torch.cuda.is_bf16_supported():raise RuntimeError("BF16_REQUIRED")
    device=torch.device("cuda");torch.set_float32_matmul_precision("high");random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    pre=json.loads(Path(args.preflight_json).read_text())
    if not pre.get("PASS") or int(pre.get("optimizer_steps",-1))!=0:raise RuntimeError("VALID_ZERO_STEP_PREFLIGHT_REQUIRED")
    if pre.get("config",{}).get("upstream_model_revision")!=cfg.upstream_model_revision:raise RuntimeError("PREFLIGHT_MODEL_REVISION_MISMATCH")
    required_stage="full" if cfg.epochs_full>0 else "geometry" if cfg.epochs_geometry>0 else "adapter"
    if pre.get("stage")!=required_stage:raise RuntimeError(f"PREFLIGHT_STAGE_MISMATCH required={required_stage} got={pre.get('stage')}")
    train_ds=OrthoFamilyDataset(args.manifest,args.train_split,cfg.native_size,cfg.target_xy_reference_size);val_ds=OrthoFamilyDataset(args.manifest,args.val_split,cfg.native_size,cfg.target_xy_reference_size)
    train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=2,pin_memory=True,collate_fn=collate_one_family);val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,collate_fn=collate_one_family)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);model=MapAnythingOrthoIRIS(cfg).to(device)
    start_epoch=0;global_step=0;optimizer_steps=0;history=[];resume_optimizer_state=None;resume_stage=None
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
        if ck["schema"]!="RealSaS.IRIS.MapAnythingOrtho.Checkpoint.v1":raise RuntimeError(ck["schema"])
        if ck.get("config",{}).get("upstream_model_revision")!=cfg.upstream_model_revision:raise RuntimeError("RESUME_MODEL_REVISION_MISMATCH")
        model.load_state_dict(ck["model"],strict=True);start_epoch=int(ck["epoch"]);global_step=int(ck["global_step"]);optimizer_steps=int(ck["optimizer_steps"]);resume_optimizer_state=ck.get("optimizer");resume_stage=ck.get("stage")
        hp=out/"TRAIN_HISTORY.json"
        if hp.is_file():history=json.loads(hp.read_text())
    current_stage=None;opt=None
    for epoch0 in range(start_epoch,cfg.total_epochs):
        stage=stage_for_epoch(epoch0,cfg)
        if stage!=current_stage:
            model.set_train_stage(stage);opt=build_optimizer(model,cfg)
            if resume_optimizer_state is not None and resume_stage==stage:opt.load_state_dict(resume_optimizer_state);resume_optimizer_state=None
            current_stage=stage;print(json.dumps({"event":"stage_start","epoch":epoch0+1,"stage":stage,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"groups":[{"name":g.get("name"),"lr":g["lr"],"tensors":len(g["params"])} for g in opt.param_groups]}),flush=True)
        model.train();opt.zero_grad(set_to_none=True);sums={};micro=0;t0=time.time()
        for batch_idx,batch in enumerate(train_loader):
            images=batch["images"].to(device,non_blocking=True);target=move_target(batch["target"],device)
            with torch.autocast("cuda",dtype=torch.bfloat16):outputs=model(images);loss,parts=geometry_loss(outputs,target,cfg);scaled=loss/cfg.grad_accum_steps
            if not torch.isfinite(loss):raise RuntimeError(f"NONFINITE_LOSS epoch={epoch0+1} batch={batch_idx}")
            scaled.backward();micro+=1;global_step+=1
            for k,v in parts.items():sums[k]=sums.get(k,0.0)+float(v.detach().float().item())
            if micro%cfg.grad_accum_steps==0 or batch_idx+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg.grad_clip);opt.step();opt.zero_grad(set_to_none=True);optimizer_steps+=1
            if (batch_idx+1)%10==0:print(json.dumps({"event":"train_progress","epoch":epoch0+1,"stage":stage,"family_batches":batch_idx+1,"global_step":global_step,"optimizer_steps":optimizer_steps,"loss":float(loss.detach().float().item()),"cuda_gb":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        train_mean={k:v/max(micro,1) for k,v in sums.items()};val=evaluate(model,val_loader,device);rec={"epoch":epoch0+1,"stage":stage,"global_step":global_step,"optimizer_steps":optimizer_steps,"train":train_mean,"val":val,"elapsed_min":(time.time()-t0)/60.0};history.append(rec);print(json.dumps(rec,sort_keys=True),flush=True)
        ck={"schema":"RealSaS.IRIS.MapAnythingOrtho.Checkpoint.v1","epoch":epoch0+1,"global_step":global_step,"optimizer_steps":optimizer_steps,"stage":stage,"config":cfg.to_dict(),"model":model.state_dict(),"optimizer":opt.state_dict(),"record":rec,"rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}}
        torch.save(ck,out/"LATEST_CHECKPOINT.pt");(out/"TRAIN_HISTORY.json").write_text(json.dumps(history,indent=2,sort_keys=True,allow_nan=False)+"\n")
    torch.save({"schema":"RealSaS.IRIS.MapAnythingOrtho.Checkpoint.v1","epoch":cfg.total_epochs,"global_step":global_step,"optimizer_steps":optimizer_steps,"stage":"full","config":cfg.to_dict(),"model":model.state_dict()},out/"FINAL_CHECKPOINT.pt")
    complete={"schema":"RealSaS.IRIS.MapAnythingOrtho.RunComplete.v1","status":"COMPLETE","scientific_optimizer_steps":optimizer_steps,"global_step":global_step,"epochs":cfg.total_epochs,"device":torch.cuda.get_device_name(0),"final_checkpoint_sha256":sha256_file(out/"FINAL_CHECKPOINT.pt"),"upstream_model_id":cfg.upstream_model_id,"upstream_model_revision":cfg.upstream_model_revision,"upstream_code_commit":cfg.upstream_code_commit,"manifest_sha256":sha256_file(args.manifest),"preflight_sha256":sha256_file(args.preflight_json)}
    (out/"RUN_COMPLETE.json").write_text(json.dumps(complete,indent=2,sort_keys=True)+"\n");print(json.dumps(complete,indent=2,sort_keys=True),flush=True)

if __name__=="__main__":main()
