#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed A100 admission for H-GEO-P1. Executes zero optimizer steps."""

import argparse, json
from pathlib import Path
import torch
from tools.iris_vnext.h_geo_p1_runtime_v1 import create_aux_head_v1, load_binding_v1, load_prereg_v1, set_h_geo_p1_trainable_v1, verify_dataset_receipt_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_exact_v8_model, module_state_sha256_v1, require_a100_runtime_v1, require_exact_clean_source_v1, sha256_file

SCHEMA="RealSaS.HGeoP1A100Preflight.v1"

def preflight(args):
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower(); require_exact_clean_source_v1(repo,source_sha); binding_path=Path(args.binding).resolve(); prereg_path=Path(args.preregistration).resolve(); binding=load_binding_v1(binding_path); prereg=load_prereg_v1(prereg_path); runtime=prereg["runtime_admission"]
    if runtime["accelerator"]!="NVIDIA_A100_CLASS" or float(runtime["minimum_cuda_memory_gib"])!=39.0: raise ValueError("H_GEO_P1_RUNTIME_PREREG_DRIFT")
    gpu=require_a100_runtime_v1(39.0); binding_sha=sha256_file(binding_path); prereg_sha=sha256_file(prereg_path); dataset_receipt=Path(args.dataset_receipt).resolve(); rows=verify_dataset_receipt_v1(dataset_root=Path(args.dataset_root),dataset_receipt=dataset_receipt,binding=binding,binding_sha256=binding_sha,preregistration_sha256=prereg_sha); receipt=json.loads(dataset_receipt.read_text()); max_replay=float(receipt["maximum_normalization_replay_error_iris"])
    if max_replay>float(prereg["truth"]["normalization_recovery_max_error_iris"]): raise RuntimeError("H_GEO_P1_NORMALIZATION_REPLAY_GATE_FAIL")
    device=torch.device("cuda"); model=load_exact_v8_model(Path(args.v8_checkpoint),device); before_hash=module_state_sha256_v1(model.scene_core); trainable,names=set_h_geo_p1_trainable_v1(model); aux=create_aux_head_v1(device); trainable_count=sum(p.numel() for p in trainable); aux_count=sum(p.numel() for p in aux.parameters() if p.requires_grad)
    if trainable_count<=0 or aux_count<=0: raise RuntimeError("H_GEO_P1_TRAINABLE_COUNT_INVALID")
    result={"schema":SCHEMA,"status":"PASS_H_GEO_P1_A100_PREFLIGHT__CARRIER_TRAINING_AUTHORIZED","execution_source_sha":source_sha,"binding_sha256":binding_sha,"preregistration_sha256":prereg_sha,"dataset_receipt_sha256":sha256_file(dataset_receipt),"v8_checkpoint_sha256":binding["v8"]["checkpoint"]["sha256"],"v8_scene_core_sha256":before_hash,"gpu":gpu,"dataset_subject_count":len(rows),"maximum_normalization_replay_error_iris":max_replay,"existing_trainable_parameter_count":trainable_count,"training_aux_parameter_count":aux_count,"trainable_existing_parameter_names":list(names),"teacher_truth_in_inference_forward":False,"tessa_invoked":False,"optimizer_steps_executed":0,"scientific_pass_claimed":False}
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print("H_GEO_P1_PREFLIGHT_PASS",json.dumps({"gpu":gpu,"existing_trainable":trainable_count,"aux":aux_count,"max_norm_error":max_replay},sort_keys=True),flush=True); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--expected-source-sha",required=True); p.add_argument("--binding",default="canonical/H_GEO_P1_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P1_PREREGISTRATION_V1.json"); p.add_argument("--dataset-root",required=True); p.add_argument("--dataset-receipt",required=True); p.add_argument("--v8-checkpoint",required=True); p.add_argument("--output",required=True); preflight(p.parse_args()); return 0
if __name__=="__main__": raise SystemExit(main())
