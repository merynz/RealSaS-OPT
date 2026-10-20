#!/usr/bin/env python3
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from models.iris.vnext.metric_geometry_v1 import metric_geometry_objective_v1
from tools.iris_vnext.h_geo_p3_runtime_v1 import build_subject_runtime_v1,create_aux_head_v1,create_explicit_geometry_carrier_v1,load_binding_v1,load_prereg_v1,set_h_geo_p3_trainable_v1,shared_v8_hashes_v1,verify_p1_dataset_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_exact_v8_model,require_a100_runtime_v1,require_exact_clean_source_v1,sha256_file

SCHEMA="RealSaS.HGeoP3A100Preflight.v1"
def _count(params): return sum(p.numel() for p in params)

def preflight(args):
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower(); require_exact_clean_source_v1(repo,source_sha)
    bp=Path(args.binding).resolve(); pp=Path(args.preregistration).resolve(); binding=load_binding_v1(bp); prereg=load_prereg_v1(pp); bsha=sha256_file(bp); psha=sha256_file(pp)
    if prereg["architecture"]["p2_checkpoint_initialization_allowed"] is not False: raise RuntimeError("H_GEO_P3_P2_INIT_PREREG_DRIFT")
    gpu=require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"])); device=torch.device("cuda")
    dr=Path(args.dataset_receipt).resolve(); receipt,_=verify_p1_dataset_v1(dataset_root=Path(args.dataset_root),dataset_receipt=dr,binding=binding)
    v8=load_exact_v8_model(Path(args.v8_checkpoint),device)
    for p in v8.parameters(): p.requires_grad_(False)
    shared_before=shared_v8_hashes_v1(v8); carrier=create_explicit_geometry_carrier_v1(v8,device); groups,names=set_h_geo_p3_trainable_v1(carrier); aux=create_aux_head_v1(device)
    runtime=build_subject_runtime_v1(v8_model=v8,prepared_root=Path(args.prepared_root),prepared_manifest=Path(args.prepared_manifest),dataset_root=Path(args.dataset_root),dataset_receipt=dr,binding=binding,split="train",device=device)
    item=runtime["Snake"]; memory=item["frozen_view_memory"]
    with torch.no_grad():
        base=carrier.geometry_scene(memory); hi=carrier.highres_refiner(base); b,three,c,h,w=base.shape
        expected=F.interpolate(base.reshape(b*three,c,h,w),scale_factor=2.0,mode="bilinear",align_corners=False).reshape_as(hi)
        init_refine_error=float((hi-expected).abs().max().item()); init_position_absmax=float(carrier.position_projection.weight.abs().max().item())
    if init_refine_error!=0.0 or init_position_absmax!=0.0: raise RuntimeError("H_GEO_P3_INITIAL_ZERO_RESIDUAL_CONTRACT_FAIL")
    if int(hi.shape[-1])!=int(prereg["architecture"]["high_resolution_geometry_plane_resolution"]): raise RuntimeError("H_GEO_P3_HIRES_RESOLUTION_DRIFT")
    a=item["arrays"]; idx=torch.arange(0,min(64,len(a["xyz_iris"]))); xyz=a["xyz_iris"][idx].to(device)
    carrier.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True); planes=carrier.encode_geometry(memory); pred=aux(carrier.query_geometry(planes,xyz[None])["H_geo"].squeeze(0))
    losses=metric_geometry_objective_v1(pred,distance_iris=a["distance_iris"][idx].to(device),offset_to_surface_iris=a["offset_to_surface_iris"][idx].to(device),normal_unit=a["normal_unit"][idx].to(device),normal_valid=a["normal_valid"][idx].to(device).bool(),near_label=a["near_label"][idx].to(device).bool(),offset_valid_distance_iris=float(prereg["truth"]["offset_valid_distance_iris"]))
    weights=prereg["training"]["loss_weights"]; geometry_loss=float(weights["distance"])*losses["distance"]+float(weights["offset"])*losses["offset"]+float(weights["normal"])*losses["normal"]+float(weights["near"])*losses["near"]
    if not bool(torch.isfinite(geometry_loss).item()): raise RuntimeError("H_GEO_P3_PREFLIGHT_NONFINITE_GEOMETRY_LOSS")
    geometry_loss.backward(); connectivity={}
    for key,params in groups.items(): connectivity[key]=any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0.0 for p in params)
    if not all(connectivity.values()): raise RuntimeError("H_GEO_P3_GRADIENT_CONNECTIVITY_FAIL:"+json.dumps(connectivity,sort_keys=True))
    if any(p.grad is not None for p in v8.parameters()): raise RuntimeError("H_GEO_P3_SHARED_V8_RECEIVED_GRADIENT")
    carrier.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True); shared_after=shared_v8_hashes_v1(v8)
    if shared_after!=shared_before: raise RuntimeError("H_GEO_P3_SHARED_V8_HASH_DRIFT")
    result={"schema":SCHEMA,"status":"PASS_H_GEO_P3_A100_PREFLIGHT__EXPLICIT_CARRIER_TRAINING_AUTHORIZED","execution_source_sha":source_sha,"binding_sha256":bsha,"preregistration_sha256":psha,"dataset_receipt_sha256":sha256_file(dr),"v8_checkpoint_sha256":binding["v8_initialization"]["checkpoint"]["sha256"],"gpu":gpu,"dataset_subject_count":len(receipt["subjects"]),"trainable_parameter_names":list(names),"trainable_parameter_counts":{k:_count(v) for k,v in groups.items()},"training_aux_parameter_count":sum(p.numel() for p in aux.parameters()),"gradient_connectivity":connectivity,"initial_highres_zero_residual_max_abs_error":init_refine_error,"initial_position_projection_absmax":init_position_absmax,"high_resolution_geometry_plane_resolution":int(hi.shape[-1]),"shared_v8_hashes_before":shared_before,"shared_v8_hashes_after":shared_after,"p2_checkpoint_used_for_initialization":False,"teacher_truth_in_inference_forward":False,"tessa_invoked":False,"optimizer_steps_executed":0,"scientific_pass_claimed":False}
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print("H_GEO_P3_PREFLIGHT_PASS",json.dumps({"params":result["trainable_parameter_counts"],"gpu":gpu},sort_keys=True),flush=True); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--expected-source-sha",required=True); p.add_argument("--binding",default="canonical/H_GEO_P3_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P3_PREREGISTRATION_V1.json"); p.add_argument("--dataset-root",required=True); p.add_argument("--dataset-receipt",required=True); p.add_argument("--prepared-root",required=True); p.add_argument("--prepared-manifest",required=True); p.add_argument("--v8-checkpoint",required=True); p.add_argument("--output",required=True); preflight(p.parse_args()); return 0
if __name__=="__main__": raise SystemExit(main())
