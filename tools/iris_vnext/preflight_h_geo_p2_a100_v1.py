#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from models.iris.vnext.metric_geometry_v1 import metric_geometry_objective_v1
from tools.iris_vnext.h_geo_p2_runtime_v1 import build_subject_runtime_v1,create_aux_head_v1,frozen_module_hashes_v1,load_binding_v1,load_prereg_v1,set_h_geo_p2_trainable_v1,verify_p1_dataset_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_exact_v8_model,module_state_sha256_v1,require_a100_runtime_v1,require_exact_clean_source_v1,sha256_file

SCHEMA="RealSaS.HGeoP2A100Preflight.v1"

def preflight(args):
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower(); require_exact_clean_source_v1(repo,source_sha)
    bp=Path(args.binding).resolve(); pp=Path(args.preregistration).resolve(); binding=load_binding_v1(bp); prereg=load_prereg_v1(pp); bsha=sha256_file(bp); psha=sha256_file(pp)
    if prereg["architecture"]["p1_checkpoint_initialization_allowed"] is not False: raise RuntimeError("H_GEO_P2_P1_INIT_PREREG_DRIFT")
    gpu=require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"])); device=torch.device("cuda")
    dr=Path(args.dataset_receipt).resolve(); receipt,_=verify_p1_dataset_v1(dataset_root=Path(args.dataset_root),dataset_receipt=dr,binding=binding)
    model=load_exact_v8_model(Path(args.v8_checkpoint),device); v8_hash=module_state_sha256_v1(model.scene_core); frozen_before=frozen_module_hashes_v1(model)
    scene_params,field_params,names=set_h_geo_p2_trainable_v1(model); aux=create_aux_head_v1(device)
    runtime=build_subject_runtime_v1(model=model,prepared_root=Path(args.prepared_root),prepared_manifest=Path(args.prepared_manifest),dataset_root=Path(args.dataset_root),dataset_receipt=dr,binding=binding,split="train",device=device,feature_chunk=256,cache_v8_baseline=False)
    item=runtime["Snake"]; a=item["arrays"]; idx=torch.arange(0,min(64,len(a["xyz_iris"]))); xyz=a["xyz_iris"][idx].to(device); src=item["source"]
    model.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
    planes=model.encode_scene(src["image_tokens"],src["token_xy"],src["camera_features"]); local=model.scene_core.query(planes,xyz[None]); pred=aux(local["H"].squeeze(0))
    losses=metric_geometry_objective_v1(pred,distance_iris=a["distance_iris"][idx].to(device),offset_to_surface_iris=a["offset_to_surface_iris"][idx].to(device),normal_unit=a["normal_unit"][idx].to(device),normal_valid=a["normal_valid"][idx].to(device).bool(),near_label=a["near_label"][idx].to(device).bool(),offset_valid_distance_iris=float(prereg["truth"]["offset_valid_distance_iris"]))
    weights=prereg["training"]["loss_weights"]
    geometry_loss=float(weights["distance"])*losses["distance"]+float(weights["offset"])*losses["offset"]+float(weights["normal"])*losses["normal"]+float(weights["near"])*losses["near"]
    if not bool(torch.isfinite(geometry_loss).item()): raise RuntimeError("H_GEO_P2_PREFLIGHT_NONFINITE_GEOMETRY_LOSS")
    geometry_loss.backward()
    scene_grad=any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0 for p in scene_params); field_grad=any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0 for p in field_params)
    if not scene_grad or not field_grad: raise RuntimeError("H_GEO_P2_GRADIENT_CONNECTIVITY_FAIL")
    model.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
    result={"schema":SCHEMA,"status":"PASS_H_GEO_P2_A100_PREFLIGHT__SCENE_FIELD_TRAINING_AUTHORIZED","execution_source_sha":source_sha,"binding_sha256":bsha,"preregistration_sha256":psha,"dataset_receipt_sha256":sha256_file(dr),"v8_checkpoint_sha256":binding["v8_initialization"]["checkpoint"]["sha256"],"v8_scene_core_sha256":v8_hash,"gpu":gpu,"dataset_subject_count":len(receipt["subjects"]),"scene_trainable_parameter_count":sum(p.numel() for p in scene_params),"field_trainable_parameter_count":sum(p.numel() for p in field_params),"training_aux_parameter_count":sum(p.numel() for p in aux.parameters()),"trainable_existing_parameter_names":list(names),"gradient_connectivity":{"scene":scene_grad,"field":field_grad},"frozen_module_hashes":frozen_before,"p1_checkpoint_used_for_initialization":False,"teacher_truth_in_inference_forward":False,"tessa_invoked":False,"optimizer_steps_executed":0,"scientific_pass_claimed":False}
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print("H_GEO_P2_PREFLIGHT_PASS",json.dumps({"scene":result["scene_trainable_parameter_count"],"field":result["field_trainable_parameter_count"],"aux":result["training_aux_parameter_count"],"gpu":gpu},sort_keys=True),flush=True); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--expected-source-sha",required=True); p.add_argument("--binding",default="canonical/H_GEO_P2_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P2_PREREGISTRATION_V1.json"); p.add_argument("--dataset-root",required=True); p.add_argument("--dataset-receipt",required=True); p.add_argument("--prepared-root",required=True); p.add_argument("--prepared-manifest",required=True); p.add_argument("--v8-checkpoint",required=True); p.add_argument("--output",required=True); preflight(p.parse_args()); return 0
if __name__=="__main__": raise SystemExit(main())
