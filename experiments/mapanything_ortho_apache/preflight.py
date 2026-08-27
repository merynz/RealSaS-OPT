from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys, traceback
from pathlib import Path
import torch

from realsas_mapanything_ortho.config import OrthoConfig
from realsas_mapanything_ortho.dataset import load_native_rgba, load_target_npz
from realsas_mapanything_ortho.model import MapAnythingOrthoIRIS
from realsas_mapanything_ortho.losses import geometry_loss


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def grad_sum(module):
    return float(sum(p.grad.detach().abs().sum().item() for p in module.parameters() if p.grad is not None))


def main():
    ap=argparse.ArgumentParser(description="Zero-optimizer-step executable preflight for MapAnything-Ortho IRIS")
    ap.add_argument("--pose-a-dir",required=True); ap.add_argument("--target-npz",required=True); ap.add_argument("--expected-family-id",type=int,required=True)
    ap.add_argument("--stage",choices=["adapter","geometry","full"],default="geometry"); ap.add_argument("--output",required=True); args=ap.parse_args()
    cfg=OrthoConfig(); result={"schema":"RealSaS.IRIS.MapAnythingOrtho.Preflight.v1","PASS":False,"optimizer_steps":0,"stage":args.stage,"config":cfg.to_dict(),"python":sys.version,"platform":platform.platform()}
    try:
        if cfg.require_cuda and not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
        device=torch.device("cuda")
        if cfg.require_bf16 and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16_REQUIRED")
        result["device"]=torch.cuda.get_device_name(0); torch.set_float32_matmul_precision("high")
        import mapanything
        upstream_root=Path(mapanything.__file__).resolve().parents[1]
        actual_commit=subprocess.check_output(["git","-C",str(upstream_root),"rev-parse","HEAD"],text=True).strip()
        result["upstream_import_root"]=str(upstream_root); result["upstream_code_commit_actual"]=actual_commit
        if actual_commit!=cfg.upstream_code_commit: raise RuntimeError(f"UPSTREAM_CODE_COMMIT_MISMATCH expected={cfg.upstream_code_commit} actual={actual_commit}")

        images=load_native_rgba(args.pose_a_dir,cfg.native_size).unsqueeze(0).to(device)
        target=load_target_npz(args.target_npz,cfg.target_xy_reference_size)
        if target["family_id"]!=args.expected_family_id: raise RuntimeError((target["family_id"],args.expected_family_id))
        target_gpu={k:(v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v] if k=="xy_source_mode" else v) for k,v in target.items() if k!="family_id"}
        target_gpu["family_id"]=torch.tensor([target["family_id"]],device=device)
        from realsas_mapanything_ortho.camera import forward_depth_from_points
        d_truth=forward_depth_from_points(target_gpu["P"],target_gpu["XY01"]); visible_d=d_truth[target_gpu["V"].bool()]; max_abs_d=float(visible_d.abs().max().item()); result["target_depth_max_abs"]=max_abs_d
        if max_abs_d>=cfg.depth_limit*.95: raise RuntimeError(f"DEPTH_LIMIT_TOO_TIGHT target={max_abs_d:.6f} limit={cfg.depth_limit}")

        model=MapAnythingOrthoIRIS(cfg).to(device); model.set_train_stage(args.stage); model.train(); model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda",dtype=torch.bfloat16): outputs=model(images); loss,parts=geometry_loss(outputs,target_gpu,cfg)
        if not torch.isfinite(loss): raise RuntimeError("NONFINITE_LOSS")
        loss.backward(); grads={"refiner":grad_sum(model.refiner),"base":grad_sum(model.base)}
        if grads["refiner"]<=0: raise RuntimeError(f"NO_REFINER_GRAD {grads}")
        if args.stage!="adapter" and grads["base"]<=0: raise RuntimeError(f"NO_BASE_GRAD {grads}")
        if args.stage=="adapter" and grads["base"]!=0: raise RuntimeError(f"FROZEN_BASE_RECEIVED_GRAD {grads}")
        result.update({"PASS":True,"family_id":target["family_id"],"xy_source_mode":target["xy_source_mode"],"input_shape":list(images.shape),"output_shapes":{k:list(v.shape) for k,v in outputs.items() if torch.is_tensor(v)},"loss_parts":{k:float(v.detach().float().item()) for k,v in parts.items()},"gradient_l1":grads,"trainable_tensors":sum(int(p.requires_grad) for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"total_parameters":sum(p.numel() for p in model.parameters()),"peak_cuda_bytes":int(torch.cuda.max_memory_allocated()),"target_npz_sha256":sha256_file(args.target_npz),"checkpointing_flags_touched":getattr(model,"_checkpointing_flags_touched",0)})
    except Exception as exc:
        result["error"]=repr(exc); result["traceback"]=traceback.format_exc()
        if torch.cuda.is_available(): result["peak_cuda_bytes"]=int(torch.cuda.max_memory_allocated())
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"); print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),flush=True)
    if not result["PASS"]: raise SystemExit(2)

if __name__=="__main__": main()
