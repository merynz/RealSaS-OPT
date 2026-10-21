#!/usr/bin/env python3
from __future__ import annotations

"""H-GEO-P1: retrain only the existing V8 local carrier field with exact triangle truth."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from models.iris.vnext.metric_geometry_v1 import metric_geometry_objective_v1
from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS
from tools.iris_vnext.h_geo_p1_runtime_v1 import (
    build_subject_runtime_v1,
    create_aux_head_v1,
    load_binding_v1,
    load_prereg_v1,
    set_h_geo_p1_trainable_v1,
)
from tools.iris_vnext.h_rep_probe_authority_v1 import (
    load_exact_v8_model,
    module_state_sha256_v1,
    require_a100_runtime_v1,
    require_exact_clean_source_v1,
    sha256_file,
)

SCHEMA = "RealSaS.HGeoP1TrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.HGeoP1Checkpoint.v1"


def _atomic_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(tmp,path)


def _atomic_torch(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(value,tmp); os.replace(tmp,path)


def _lr(step: int, *, total: int, warmup: int, peak: float, minimum: float) -> float:
    if step < warmup: return peak * float(step + 1) / float(warmup)
    t=(step-warmup)/max(1,total-warmup-1)
    return minimum + 0.5*(peak-minimum)*(1.0+math.cos(math.pi*t))


def train(args: argparse.Namespace):
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower()
    require_exact_clean_source_v1(repo,source_sha)
    binding_path=Path(args.binding).resolve(); prereg_path=Path(args.preregistration).resolve()
    binding=load_binding_v1(binding_path); prereg=load_prereg_v1(prereg_path)
    gpu=require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"]))
    binding_sha=sha256_file(binding_path); prereg_sha=sha256_file(prereg_path)
    preflight_path=Path(args.preflight_receipt).resolve(); preflight=json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS_H_GEO_P1_A100_PREFLIGHT__CARRIER_TRAINING_AUTHORIZED":
        raise RuntimeError("H_GEO_P1_PREFLIGHT_NOT_AUTHORIZED")
    for key,value in {"execution_source_sha":source_sha,"binding_sha256":binding_sha,"preregistration_sha256":prereg_sha}.items():
        if preflight.get(key) != value: raise RuntimeError("H_GEO_P1_PREFLIGHT_AUTHORITY_DRIFT:"+key)

    device=torch.device("cuda")
    model=load_exact_v8_model(Path(args.v8_checkpoint),device)
    v8_scene_hash=module_state_sha256_v1(model.scene_core)
    if preflight.get("v8_scene_core_sha256") != v8_scene_hash:
        raise RuntimeError("H_GEO_P1_V8_SCENE_HASH_DRIFT")

    runtime=build_subject_runtime_v1(
        model=model,
        prepared_root=Path(args.prepared_root), prepared_manifest=Path(args.prepared_manifest),
        dataset_root=Path(args.dataset_root), dataset_receipt=Path(args.dataset_receipt),
        binding=binding, binding_sha256=binding_sha, preregistration_sha256=prereg_sha,
        split="train", device=device, feature_chunk=int(args.feature_chunk), cache_baseline=True,
    )
    existing_params, trainable_names=set_h_geo_p1_trainable_v1(model)
    aux=create_aux_head_v1(device)
    trainable_count=sum(p.numel() for p in existing_params)
    aux_count=sum(p.numel() for p in aux.parameters() if p.requires_grad)
    if trainable_count != int(preflight["existing_trainable_parameter_count"]) or aux_count != int(preflight["training_aux_parameter_count"]):
        raise RuntimeError("H_GEO_P1_TRAINABLE_COUNT_DRIFT")

    cfg=prereg["training"]; total=int(cfg["carrier_steps"]); batch_size=int(cfg["batch_size"])
    optimizer=torch.optim.AdamW(list(existing_params)+list(aux.parameters()),lr=float(cfg["peak_learning_rate"]),betas=(0.9,0.95),weight_decay=0.01)
    output=Path(args.output_dir).resolve(); mirror=Path(args.checkpoint_mirror_dir).resolve(); output.mkdir(parents=True,exist_ok=True); mirror.mkdir(parents=True,exist_ok=True)
    local_ckpt=output/"checkpoint_latest.pt"; remote_ckpt=mirror/"checkpoint_latest.pt"
    if not local_ckpt.is_file() and remote_ckpt.is_file(): shutil.copy2(remote_ckpt,local_ckpt)
    dataset_sha=sha256_file(Path(args.dataset_receipt).resolve()); preflight_sha=sha256_file(preflight_path)
    authority={"execution_source_sha":source_sha,"binding_sha256":binding_sha,"preregistration_sha256":prereg_sha,"dataset_receipt_sha256":dataset_sha,"preflight_receipt_sha256":preflight_sha,"v8_scene_core_sha256":v8_scene_hash}
    completed=0
    if local_ckpt.is_file():
        payload=torch.load(local_ckpt,map_location="cpu",weights_only=True)
        if payload.get("schema") != CHECKPOINT_SCHEMA: raise RuntimeError("H_GEO_P1_RESUME_SCHEMA_DRIFT")
        for key,value in authority.items():
            if payload.get(key) != value: raise RuntimeError("H_GEO_P1_RESUME_AUTHORITY_DRIFT:"+key)
        if tuple(payload.get("trainable_existing_parameter_names") or ()) != tuple(trainable_names): raise RuntimeError("H_GEO_P1_RESUME_CENSUS_DRIFT")
        model.scene_core.field.load_state_dict(payload["field_state"],strict=True); aux.load_state_dict(payload["aux_state"],strict=True); optimizer.load_state_dict(payload["optimizer_state"])
        completed=int(payload["completed_steps"])
        if completed<0 or completed>total: raise RuntimeError("H_GEO_P1_RESUME_STEP_INVALID")
        print("H_GEO_P1_RESUME",completed,flush=True)

    weights=cfg["loss_weights"]; started=time.perf_counter(); model.scene_core.field.train(); aux.train()
    for step in range(completed,total):
        subject_id=SUBJECT_IDS[step%len(SUBJECT_IDS)]; item=runtime[subject_id]; arrays=item["arrays"]; n=len(arrays["xyz_iris"])
        generator=torch.Generator(device="cpu"); generator.manual_seed(27030000+step)
        index=torch.randint(0,n,(min(batch_size,n),),generator=generator)
        xyz=arrays["xyz_iris"][index].to(device=device,dtype=torch.float32)
        target={k:v[index].to(device=device) for k,v in arrays.items() if k not in ("baseline_H","baseline_field")}
        baseline_h=arrays["baseline_H"][index].to(device=device,dtype=torch.float32); baseline_field=arrays["baseline_field"][index].to(device=device,dtype=torch.float32)
        lr=_lr(step,total=total,warmup=int(cfg["warmup_steps"]),peak=float(cfg["peak_learning_rate"]),minimum=float(cfg["minimum_learning_rate"]))
        for group in optimizer.param_groups: group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        local=model.scene_core.query(item["scene_planes"],xyz[None])
        h=local["H"].squeeze(0); field=local["field"].squeeze(0); prediction=aux(h)
        losses=metric_geometry_objective_v1(
            prediction,distance_iris=target["distance_iris"],offset_to_surface_iris=target["offset_to_surface_iris"],normal_unit=target["normal_unit"],normal_valid=target["normal_valid"].bool(),near_label=target["near_label"].bool(),offset_valid_distance_iris=float(prereg["truth"]["offset_valid_distance_iris"]),
        )
        h_ret=F.smooth_l1_loss(h,baseline_h,beta=0.05); field_ret=F.smooth_l1_loss(field,baseline_field,beta=0.02)
        total_loss=(float(weights["distance"])*losses["distance"]+float(weights["offset"])*losses["offset"]+float(weights["normal"])*losses["normal"]+float(weights["near"])*losses["near"]+float(weights["h_retention"])*h_ret+float(weights["field_retention"])*field_ret)
        if not bool(torch.isfinite(total_loss).item()): raise RuntimeError("H_GEO_P1_NONFINITE_LOSS")
        total_loss.backward(); grad=torch.nn.utils.clip_grad_norm_(list(existing_params)+list(aux.parameters()),float(cfg["gradient_clip_norm"]))
        if not bool(torch.isfinite(torch.as_tensor(grad)).item()): raise RuntimeError("H_GEO_P1_NONFINITE_GRAD")
        optimizer.step(); done=step+1
        if done==1 or done%20==0 or done==total:
            print("H_GEO_P1_STEP",f"step={done}/{total}",f"subject={subject_id}",f"lr={lr:.8g}",f"total={float(total_loss.detach()):.6f}",f"d={float(losses['distance'].detach()):.6f}",f"off={float(losses['offset'].detach()):.6f}",f"n={float(losses['normal'].detach()):.6f}",f"near={float(losses['near'].detach()):.6f}",f"hret={float(h_ret.detach()):.6f}",f"fret={float(field_ret.detach()):.6f}",flush=True)
        if done%int(cfg["checkpoint_interval"])==0 or done==total:
            payload={"schema":CHECKPOINT_SCHEMA,"completed_steps":done,**authority,"trainable_existing_parameter_names":list(trainable_names),"field_state":model.scene_core.field.state_dict(),"aux_state":aux.state_dict(),"optimizer_state":optimizer.state_dict(),"sampler_schema":"RealSaS.HGeoP1AbsoluteStepSeededSampler.v1"}
            _atomic_torch(payload,local_ckpt); tmp=remote_ckpt.with_suffix(".pt.tmp"); shutil.copy2(local_ckpt,tmp); os.replace(tmp,remote_ckpt)
            _atomic_json({"schema":"RealSaS.HGeoP1CheckpointPointer.v1","completed_steps":done,"checkpoint_sha256":sha256_file(remote_ckpt),**authority},mirror/"LATEST.json")
            print("H_GEO_P1_CHECKPOINT",done,sha256_file(remote_ckpt)[:16],flush=True)

    result={"schema":SCHEMA,"status":"COMPLETED_H_GEO_P1_CARRIER_3600_STEPS__AWAITING_INDEPENDENT_COURT",**authority,"gpu":gpu,"carrier_optimizer_steps":total,"trainable_existing_parameter_count":trainable_count,"training_aux_parameter_count":aux_count,"trainable_existing_parameter_names":list(trainable_names),"checkpoint_sha256":sha256_file(local_ckpt),"final_field_state_sha256":module_state_sha256_v1(model.scene_core.field),"elapsed_seconds":time.perf_counter()-started,"teacher_truth_in_inference_forward":False,"tessa_invoked":False,"scientific_pass_claimed":False,"next_action":"RUN_INDEPENDENT_H_GEO_P1_FRESH_PROBE_COURT"}
    _atomic_json(result,output/"H_GEO_P1_TRAINING_RESULT.json"); print("H_GEO_P1_TRAINING_COMPLETE",result["checkpoint_sha256"],flush=True); return result


def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo-root",required=True); p.add_argument("--expected-source-sha",required=True); p.add_argument("--binding",default="canonical/H_GEO_P1_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P1_PREREGISTRATION_V1.json"); p.add_argument("--preflight-receipt",required=True); p.add_argument("--dataset-root",required=True); p.add_argument("--dataset-receipt",required=True); p.add_argument("--prepared-root",required=True); p.add_argument("--prepared-manifest",required=True); p.add_argument("--v8-checkpoint",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint-mirror-dir",required=True); p.add_argument("--feature-chunk",type=int,default=4096); train(p.parse_args()); return 0

if __name__=="__main__": raise SystemExit(main())
