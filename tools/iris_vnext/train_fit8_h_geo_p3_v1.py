#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,os,shutil,time
from pathlib import Path
import torch
from models.iris.vnext.metric_geometry_v1 import metric_geometry_objective_v1
from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS
from tools.iris_vnext.h_geo_p3_runtime_v1 import build_subject_runtime_v1,create_aux_head_v1,create_explicit_geometry_carrier_v1,load_binding_v1,load_prereg_v1,set_h_geo_p3_trainable_v1,shared_v8_hashes_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_exact_v8_model,module_state_sha256_v1,require_a100_runtime_v1,require_exact_clean_source_v1,sha256_file

SCHEMA="RealSaS.HGeoP3TrainingResult.v1"; CHECKPOINT_SCHEMA="RealSaS.HGeoP3Checkpoint.v1"
def atomic_json(v,p): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
def atomic_torch(v,p): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); torch.save(v,t); os.replace(t,p)
def lr(step,total,warmup,peak,minimum):
    if step<warmup:return peak*float(step+1)/float(warmup)
    x=(step-warmup)/max(1,total-warmup-1); return minimum+0.5*(peak-minimum)*(1.0+math.cos(math.pi*x))

def train(args):
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower(); require_exact_clean_source_v1(repo,source_sha); bp=Path(args.binding).resolve(); pp=Path(args.preregistration).resolve(); binding=load_binding_v1(bp); prereg=load_prereg_v1(pp); bsha=sha256_file(bp); psha=sha256_file(pp); gpu=require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"])); device=torch.device("cuda")
    pf_path=Path(args.preflight_receipt).resolve(); pf=json.loads(pf_path.read_text()); dr=Path(args.dataset_receipt).resolve()
    if pf.get("status")!="PASS_H_GEO_P3_A100_PREFLIGHT__EXPLICIT_CARRIER_TRAINING_AUTHORIZED": raise RuntimeError("H_GEO_P3_PREFLIGHT_NOT_AUTHORIZED")
    for k,v in {"execution_source_sha":source_sha,"binding_sha256":bsha,"preregistration_sha256":psha,"dataset_receipt_sha256":sha256_file(dr)}.items():
        if pf.get(k)!=v: raise RuntimeError("H_GEO_P3_PREFLIGHT_AUTHORITY_DRIFT:"+k)
    v8=load_exact_v8_model(Path(args.v8_checkpoint),device)
    for p in v8.parameters(): p.requires_grad_(False)
    shared_before=shared_v8_hashes_v1(v8)
    if pf.get("shared_v8_hashes_before")!=shared_before or pf.get("shared_v8_hashes_after")!=shared_before: raise RuntimeError("H_GEO_P3_SHARED_V8_PREFLIGHT_HASH_DRIFT")
    carrier=create_explicit_geometry_carrier_v1(v8,device); groups,names=set_h_geo_p3_trainable_v1(carrier); aux=create_aux_head_v1(device)
    if tuple(names)!=tuple(pf["trainable_parameter_names"]): raise RuntimeError("H_GEO_P3_TRAINABLE_CENSUS_DRIFT")
    runtime=build_subject_runtime_v1(v8_model=v8,prepared_root=Path(args.prepared_root),prepared_manifest=Path(args.prepared_manifest),dataset_root=Path(args.dataset_root),dataset_receipt=dr,binding=binding,split="train",device=device)
    cfg=prereg["training"]; total=int(cfg["carrier_steps"]); batch=int(cfg["batch_size"])
    opt=torch.optim.AdamW([{"params":groups["scene"],"lr":float(cfg["geometry_scene_peak_learning_rate"]),"tag":"geometry_scene"},{"params":groups["field"],"lr":float(cfg["geometry_field_peak_learning_rate"]),"tag":"geometry_field"},{"params":groups["refiner"]+groups["position"]+list(aux.parameters()),"lr":float(cfg["new_geometry_modules_peak_learning_rate"]),"tag":"geometry_new_aux"}],betas=(0.9,0.95),weight_decay=0.01)
    out=Path(args.output_dir).resolve(); mirror=Path(args.checkpoint_mirror_dir).resolve(); out.mkdir(parents=True,exist_ok=True); mirror.mkdir(parents=True,exist_ok=True); local=out/"checkpoint_latest.pt"; remote=mirror/"checkpoint_latest.pt"
    if not local.is_file() and remote.is_file(): shutil.copy2(remote,local)
    auth={"execution_source_sha":source_sha,"binding_sha256":bsha,"preregistration_sha256":psha,"dataset_receipt_sha256":sha256_file(dr),"preflight_receipt_sha256":sha256_file(pf_path),"v8_checkpoint_sha256":binding["v8_initialization"]["checkpoint"]["sha256"],"shared_v8_hashes":shared_before}; completed=0
    if local.is_file():
        ck=torch.load(local,map_location="cpu",weights_only=True)
        if ck.get("schema")!=CHECKPOINT_SCHEMA: raise RuntimeError("H_GEO_P3_RESUME_SCHEMA_DRIFT")
        for k,v in auth.items():
            if ck.get(k)!=v: raise RuntimeError("H_GEO_P3_RESUME_AUTHORITY_DRIFT:"+k)
        if tuple(ck.get("trainable_parameter_names") or ())!=tuple(names): raise RuntimeError("H_GEO_P3_RESUME_CENSUS_DRIFT")
        carrier.load_state_dict(ck["carrier_state"],strict=True); aux.load_state_dict(ck["aux_state"],strict=True); opt.load_state_dict(ck["optimizer_state"]); completed=int(ck["completed_steps"]); print("H_GEO_P3_RESUME",completed,flush=True)
    weights=cfg["loss_weights"]; started=time.perf_counter(); carrier.train(); aux.train(); trainable=[p for p in carrier.parameters() if p.requires_grad]+list(aux.parameters())
    for step in range(completed,total):
        subject=SUBJECT_IDS[step%len(SUBJECT_IDS)]; item=runtime[subject]; a=item["arrays"]; g=torch.Generator(device="cpu"); g.manual_seed(27100000+step); idx=torch.randint(0,len(a["xyz_iris"]),(min(batch,len(a["xyz_iris"])),),generator=g); xyz=a["xyz_iris"][idx].to(device); target={k:v[idx].to(device) for k,v in a.items()}; memory=item["frozen_view_memory"]
        slr=lr(step,total,int(cfg["warmup_steps"]),float(cfg["geometry_scene_peak_learning_rate"]),float(cfg["geometry_scene_minimum_learning_rate"])); flr=lr(step,total,int(cfg["warmup_steps"]),float(cfg["geometry_field_peak_learning_rate"]),float(cfg["geometry_field_minimum_learning_rate"])); nlr=lr(step,total,int(cfg["warmup_steps"]),float(cfg["new_geometry_modules_peak_learning_rate"]),float(cfg["new_geometry_modules_minimum_learning_rate"])); opt.param_groups[0]["lr"]=slr; opt.param_groups[1]["lr"]=flr; opt.param_groups[2]["lr"]=nlr; opt.zero_grad(set_to_none=True)
        planes=carrier.encode_geometry(memory); h_geo=carrier.query_geometry(planes,xyz[None])["H_geo"].squeeze(0); pred=aux(h_geo); losses=metric_geometry_objective_v1(pred,distance_iris=target["distance_iris"],offset_to_surface_iris=target["offset_to_surface_iris"],normal_unit=target["normal_unit"],normal_valid=target["normal_valid"].bool(),near_label=target["near_label"].bool(),offset_valid_distance_iris=float(prereg["truth"]["offset_valid_distance_iris"])); loss=float(weights["distance"])*losses["distance"]+float(weights["offset"])*losses["offset"]+float(weights["normal"])*losses["normal"]+float(weights["near"])*losses["near"]
        if not bool(torch.isfinite(loss).item()): raise RuntimeError("H_GEO_P3_NONFINITE_LOSS")
        loss.backward()
        if any(p.grad is not None for p in v8.parameters()): raise RuntimeError("H_GEO_P3_SHARED_V8_RECEIVED_GRADIENT")
        grad=torch.nn.utils.clip_grad_norm_(trainable,float(cfg["gradient_clip_norm"]));
        if not bool(torch.isfinite(torch.as_tensor(grad)).item()): raise RuntimeError("H_GEO_P3_NONFINITE_GRAD")
        opt.step(); done=step+1
        if done==1 or done%20==0 or done==total: print("H_GEO_P3_STEP",f"step={done}/{total}",f"subject={subject}",f"scene_lr={slr:.8g}",f"field_lr={flr:.8g}",f"new_lr={nlr:.8g}",f"total={float(loss.detach()):.6f}",f"d={float(losses['distance']):.6f}",f"off={float(losses['offset']):.6f}",f"n={float(losses['normal']):.6f}",f"near={float(losses['near']):.6f}",flush=True)
        if done%int(cfg["checkpoint_interval"])==0 or done==total:
            ck={"schema":CHECKPOINT_SCHEMA,"completed_steps":done,**auth,"trainable_parameter_names":list(names),"carrier_state":carrier.state_dict(),"aux_state":aux.state_dict(),"optimizer_state":opt.state_dict(),"sampler_schema":"RealSaS.HGeoP3AbsoluteStepSeededSampler.v1"}; atomic_torch(ck,local); tmp=remote.with_suffix(".pt.tmp"); shutil.copy2(local,tmp); os.replace(tmp,remote); atomic_json({"schema":"RealSaS.HGeoP3CheckpointPointer.v1","completed_steps":done,"checkpoint_sha256":sha256_file(remote),**auth},mirror/"LATEST.json"); print("H_GEO_P3_CHECKPOINT",done,sha256_file(remote)[:16],flush=True)
    shared_after=shared_v8_hashes_v1(v8)
    if shared_after!=shared_before: raise RuntimeError("H_GEO_P3_SHARED_V8_HASH_DRIFT")
    result={"schema":SCHEMA,"status":"COMPLETED_H_GEO_P3_EXPLICIT_HIRES_4800_STEPS__AWAITING_INDEPENDENT_COURT",**auth,"gpu":gpu,"carrier_optimizer_steps":total,"trainable_parameter_counts":{k:sum(p.numel() for p in v) for k,v in groups.items()},"training_aux_parameter_count":sum(p.numel() for p in aux.parameters()),"trainable_parameter_names":list(names),"checkpoint_sha256":sha256_file(local),"final_carrier_state_sha256":module_state_sha256_v1(carrier),"shared_v8_hashes_before":shared_before,"shared_v8_hashes_after":shared_after,"p2_checkpoint_used_for_initialization":False,"elapsed_seconds":time.perf_counter()-started,"teacher_truth_in_inference_forward":False,"tessa_invoked":False,"scientific_pass_claimed":False,"next_action":"RUN_INDEPENDENT_H_GEO_P3_P2_V8_FRESH_PROBE_COURT"}; atomic_json(result,out/"H_GEO_P3_TRAINING_RESULT.json"); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--expected-source-sha",required=True); p.add_argument("--binding",default="canonical/H_GEO_P3_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P3_PREREGISTRATION_V1.json"); p.add_argument("--preflight-receipt",required=True); p.add_argument("--dataset-root",required=True); p.add_argument("--dataset-receipt",required=True); p.add_argument("--prepared-root",required=True); p.add_argument("--prepared-manifest",required=True); p.add_argument("--v8-checkpoint",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint-mirror-dir",required=True); train(p.parse_args()); return 0
if __name__=="__main__": raise SystemExit(main())
