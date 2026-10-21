#!/usr/bin/env python3
from __future__ import annotations

"""Independent H-GEO-P1 court: fresh equal-capacity probes on P1 H vs frozen V8 H."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.iris.vnext.h_rep_probe_v1 import HRepresentationProbeV1, build_probe_input_v1, h_rep_probe_metrics_v1, h_rep_probe_objective_v1
from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS
from tools.iris_vnext.h_geo_p1_runtime_v1 import load_binding_v1, load_dataset_split_v1, load_prereg_v1, verify_dataset_receipt_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_exact_v8_model, load_json, load_subject_source, module_state_sha256_v1, require_a100_runtime_v1, require_exact_clean_source_v1, sha256_file

SCHEMA="RealSaS.HGeoP1IndependentCourt.v1"
ARMS=("P1_H_XYZ","V8_H_XYZ","XYZ_ONLY","P1_SHUFFLED_H_XYZ")

def mean(rows,key): return float(np.mean([r[key] for r in rows]))
def vmax(rows,key): return float(max(r[key] for r in rows))
def vmin(rows,key): return float(min(r[key] for r in rows))
def reduction(value,base): return 1.0-float(value)/max(float(base),1e-12)

def cache_features(v8,p1,*,prepared_root,prepared_manifest,dataset_root,dataset_receipt,binding,binding_sha,prereg_sha,split,device,chunk):
    manifest=load_json(Path(prepared_manifest)); mr={str(r["subject_id"]):r for r in manifest["subjects"]}; ex={str(r["subject_id"]):r for r in binding["prepared"]["subjects"]}; ds=verify_dataset_receipt_v1(dataset_root=Path(dataset_root),dataset_receipt=Path(dataset_receipt),binding=binding,binding_sha256=binding_sha,preregistration_sha256=prereg_sha); out={}
    with torch.inference_mode():
        for si,subject in enumerate(SUBJECT_IDS):
            if mr[subject]["shard_sha256"]!=ex[subject]["sha256"]: raise RuntimeError("H_GEO_P1_COURT_PREPARED_DRIFT:"+subject)
            shard=(Path(prepared_root)/ex[subject]["file"]).resolve(); src=load_subject_source(shard,ex[subject]["sha256"],device); planes=v8.encode_scene(src["image_tokens"],src["token_xy"],src["camera_features"]).float().detach(); a=load_dataset_split_v1((Path(dataset_root)/ds[subject]["payload_file"]).resolve(),split); p1h=[]; v8h=[]
            for start in range(0,len(a["xyz_iris"]),int(chunk)):
                q=a["xyz_iris"][start:start+int(chunk)].to(device=device,dtype=torch.float32)[None]; p1h.append(p1.scene_core.query(planes,q)["H"].squeeze(0).cpu()); v8h.append(v8.scene_core.query(planes,q)["H"].squeeze(0).cpu())
            a["P1_H"]=torch.cat(p1h).float(); a["V8_H"]=torch.cat(v8h).float(); g=torch.Generator(device="cpu"); g.manual_seed(27040000+si+(0 if split=="train" else 1000)); a["shuffle_index"]=torch.randperm(len(a["xyz_iris"]),generator=g); out[subject]={"arrays":a,"planes":planes,"source":src,"shard":shard}; print(f"H_GEO_P1_COURT_CACHE split={split} subject={subject} rows={len(a['xyz_iris'])}",flush=True)
    return out

def probe_input(arm,item,index,device):
    a=item["arrays"]; xyz=a["xyz_iris"][index].to(device); zero=torch.zeros(len(index),device=device); p1=a["P1_H"][index].to(device); v8=a["V8_H"][index].to(device)
    if arm=="P1_H_XYZ": return build_probe_input_v1("H_XYZ",h=p1,xyz_iris=xyz,raw_field=zero)
    if arm=="V8_H_XYZ": return build_probe_input_v1("H_XYZ",h=v8,xyz_iris=xyz,raw_field=zero)
    if arm=="XYZ_ONLY": return build_probe_input_v1("XYZ_ONLY",h=p1,xyz_iris=xyz,raw_field=zero)
    sh=a["P1_H"][a["shuffle_index"][index]].to(device); return build_probe_input_v1("SHUFFLED_H_XYZ",h=p1,xyz_iris=xyz,raw_field=zero,shuffled_h=sh)

def evaluate(model,arm,item,device,chunk):
    a=item["arrays"]; chunks=[]
    with torch.no_grad():
        for start in range(0,len(a["xyz_iris"]),int(chunk)):
            idx=torch.arange(start,min(len(a["xyz_iris"]),start+int(chunk))); chunks.append(model(probe_input(arm,item,idx,device)))
    pred={k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    return h_rep_probe_metrics_v1(pred,xyz_iris=a["xyz_iris"].to(device),nearest_reference_iris=a["closest_surface_iris"].to(device),distance_iris=a["distance_iris"].to(device),normal_unit=a["normal_unit"].to(device),normal_valid=a["normal_valid"].to(device),near_label=a["near_label"].to(device))

def audit(args):
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower(); require_exact_clean_source_v1(repo,source_sha); bp=Path(args.binding).resolve(); pp=Path(args.preregistration).resolve(); binding=load_binding_v1(bp); prereg=load_prereg_v1(pp); bsha=sha256_file(bp); psha=sha256_file(pp); gpu=require_a100_runtime_v1(float(prereg["runtime_admission"]["minimum_cuda_memory_gib"])); device=torch.device("cuda")
    trp=Path(args.training_result).resolve(); tr=json.loads(trp.read_text()); ckpt=Path(args.checkpoint).resolve(); dr=Path(args.dataset_receipt).resolve(); dsha=sha256_file(dr)
    if tr.get("status")!="COMPLETED_H_GEO_P1_CARRIER_3600_STEPS__AWAITING_INDEPENDENT_COURT" or sha256_file(ckpt)!=tr.get("checkpoint_sha256"): raise RuntimeError("H_GEO_P1_COURT_TRAINING_INVALID")
    for k,v in {"execution_source_sha":source_sha,"binding_sha256":bsha,"preregistration_sha256":psha,"dataset_receipt_sha256":dsha}.items():
        if tr.get(k)!=v: raise RuntimeError("H_GEO_P1_COURT_AUTHORITY_DRIFT:"+k)
    payload=torch.load(ckpt,map_location="cpu",weights_only=True)
    if payload.get("schema")!="RealSaS.HGeoP1Checkpoint.v1" or int(payload.get("completed_steps",-1))!=int(prereg["training"]["carrier_steps"]): raise RuntimeError("H_GEO_P1_COURT_CHECKPOINT_INVALID")
    v8=load_exact_v8_model(Path(args.v8_checkpoint),device); p1=load_exact_v8_model(Path(args.v8_checkpoint),device); p1.scene_core.field.load_state_dict(payload["field_state"],strict=True); v8.eval(); p1.eval()
    if module_state_sha256_v1(v8.scene_core)!=tr["v8_scene_core_sha256"]: raise RuntimeError("H_GEO_P1_COURT_V8_HASH_DRIFT")
    train=cache_features(v8,p1,prepared_root=args.prepared_root,prepared_manifest=args.prepared_manifest,dataset_root=args.dataset_root,dataset_receipt=dr,binding=binding,binding_sha=bsha,prereg_sha=psha,split="train",device=device,chunk=args.feature_chunk); hold=cache_features(v8,p1,prepared_root=args.prepared_root,prepared_manifest=args.prepared_manifest,dataset_root=args.dataset_root,dataset_receipt=dr,binding=binding,binding_sha=bsha,prereg_sha=psha,split="holdout",device=device,chunk=args.feature_chunk)
    kinds={"P1_H_XYZ":"H_XYZ","V8_H_XYZ":"H_XYZ","XYZ_ONLY":"XYZ_ONLY","P1_SHUFFLED_H_XYZ":"SHUFFLED_H_XYZ"}; models={a:HRepresentationProbeV1(kinds[a]).to(device) for a in ARMS}; counts={a:m.trainable_parameter_count() for a,m in models.items()}
    if len(set(counts.values()))!=1: raise RuntimeError("H_GEO_P1_COURT_PROBE_CAPACITY_DRIFT")
    opts={a:torch.optim.AdamW(models[a].parameters(),lr=2e-4,betas=(0.9,0.95),weight_decay=0.01) for a in ARMS}; steps=int(prereg["independent_court"]["probe_steps_per_arm"]); batch=int(prereg["independent_court"]["probe_batch_size"])
    for step in range(steps):
        subject=SUBJECT_IDS[step%8]; item=train[subject]; a=item["arrays"]; g=torch.Generator(device="cpu"); g.manual_seed(27050000+step); idx=torch.randint(0,len(a["xyz_iris"]),(min(batch,len(a["xyz_iris"])),),generator=g); target={k:v[idx].to(device) for k,v in a.items() if k not in ("P1_H","V8_H","shuffle_index")}
        for arm in ARMS:
            opts[arm].zero_grad(set_to_none=True); pred=models[arm](probe_input(arm,item,idx,device)); loss=h_rep_probe_objective_v1(pred,distance_iris=target["distance_iris"],offset_to_reference_iris=target["offset_to_surface_iris"],normal_unit=target["normal_unit"],normal_valid=target["normal_valid"].bool(),near_label=target["near_label"].bool(),offset_valid_distance_iris=float(prereg["truth"]["offset_valid_distance_iris"])); loss["total"].backward(); torch.nn.utils.clip_grad_norm_(models[arm].parameters(),1.0); opts[arm].step()
        if step==0 or (step+1)%100==0 or step+1==steps: print("H_GEO_P1_FRESH_PROBE_STEP",step+1,"/",steps,subject,flush=True)
    subjects=[]
    for subject in SUBJECT_IDS:
        metrics={arm:evaluate(models[arm],arm,hold[subject],device,args.eval_chunk) for arm in ARMS}; subjects.append({"subject_id":subject,"arms":metrics}); print("H_GEO_P1_COURT_SUBJECT",subject,json.dumps(metrics,sort_keys=True),flush=True)
    aggregate={}
    for arm in ARMS:
        rows=[s["arms"][arm] for s in subjects]; aggregate[arm]={"mean_distance_mae_iris":mean(rows,"distance_mae_iris"),"maximum_subject_endpoint_error_p95_iris":vmax(rows,"endpoint_error_p95_iris"),"minimum_subject_normal_absolute_dot_mean":vmin(rows,"normal_absolute_dot_mean"),"minimum_subject_near_balanced_accuracy":vmin(rows,"near_balanced_accuracy"),"mean_near_balanced_accuracy":mean(rows,"near_balanced_accuracy")}
    retention=[]
    with torch.inference_mode():
        for subject in SUBJECT_IDS:
            item=hold[subject]; shard=torch.load(item["shard"],map_location="cpu",weights_only=True); anchor=torch.as_tensor(shard["geometry_points"]).float()[:4096].to(device); planes=item["planes"]; lv8=v8.scene_core.query(planes,anchor[None]); lp1=p1.scene_core.query(planes,anchor[None]); cos=float(F.cosine_similarity(lv8["H"],lp1["H"],dim=-1).mean()); fmae=float((lv8["field"]-lp1["field"]).abs().mean()); retention.append({"subject_id":subject,"mean_h_cosine_similarity":cos,"raw_field_mae_iris":fmae})
    min_cos=min(x["mean_h_cosine_similarity"] for x in retention); max_field=max(x["raw_field_mae_iris"] for x in retention); c=prereg["independent_court"]; p=aggregate["P1_H_XYZ"]; b=aggregate["V8_H_XYZ"]; x=aggregate["XYZ_ONLY"]; s=aggregate["P1_SHUFFLED_H_XYZ"]
    gates={"absolute":{"mean_distance_mae":p["mean_distance_mae_iris"]<=c["maximum_p1_mean_distance_mae_iris"],"max_endpoint_p95":p["maximum_subject_endpoint_error_p95_iris"]<=c["maximum_p1_subject_endpoint_p95_iris"],"min_normal":p["minimum_subject_normal_absolute_dot_mean"]>=c["minimum_p1_subject_normal_alignment"],"min_near_ba":p["minimum_subject_near_balanced_accuracy"]>=c["minimum_p1_subject_near_balanced_accuracy"]},"vs_v8":{"distance":reduction(p["mean_distance_mae_iris"],b["mean_distance_mae_iris"])>=c["minimum_distance_mae_reduction_vs_v8"],"endpoint":reduction(p["maximum_subject_endpoint_error_p95_iris"],b["maximum_subject_endpoint_error_p95_iris"])>=c["minimum_endpoint_p95_reduction_vs_v8"],"normal":p["minimum_subject_normal_absolute_dot_mean"]-b["minimum_subject_normal_absolute_dot_mean"]>=c["minimum_worst_subject_normal_gain_vs_v8"],"near":p["mean_near_balanced_accuracy"]-b["mean_near_balanced_accuracy"]>=c["minimum_mean_near_ba_gain_vs_v8"]},"local_alignment":{"distance_xyz":reduction(p["mean_distance_mae_iris"],x["mean_distance_mae_iris"])>=c["minimum_p1_h_distance_mae_reduction_vs_xyz"],"distance_shuffled":reduction(p["mean_distance_mae_iris"],s["mean_distance_mae_iris"])>=c["minimum_p1_h_distance_mae_reduction_vs_shuffled"],"near_xyz":p["mean_near_balanced_accuracy"]-x["mean_near_balanced_accuracy"]>=c["minimum_p1_h_near_ba_gain_vs_xyz"],"near_shuffled":p["mean_near_balanced_accuracy"]-s["mean_near_balanced_accuracy"]>=c["minimum_p1_h_near_ba_gain_vs_shuffled"]},"retention":{"h_cosine":min_cos>=c["minimum_anchor_h_cosine_similarity_vs_v8"],"raw_field":max_field<=c["maximum_anchor_raw_field_mae_vs_v8"]}}
    passed=all(all(v.values()) for v in gates.values()); status="SCIENTIFIC_PASS_H_GEO_P1_METRIC_CARRIER_RESCUE" if passed else "SCIENTIFIC_FAIL_H_GEO_P1_METRIC_CARRIER_RESCUE"; next_action=prereg["interpretation"]["pass_next_action" if passed else "fail_next_action"]
    result={"schema":SCHEMA,"status":status,"execution_source_sha":source_sha,"binding_sha256":bsha,"preregistration_sha256":psha,"dataset_receipt_sha256":dsha,"training_result_sha256":sha256_file(trp),"training_checkpoint_sha256":sha256_file(ckpt),"gpu":gpu,"probe_optimizer_steps_per_arm":steps,"probe_parameter_counts":counts,"aggregate":aggregate,"subjects":subjects,"retention_subjects":retention,"retention_global":{"minimum_subject_mean_h_cosine_similarity":min_cos,"maximum_subject_raw_field_mae_iris":max_field},"gates":gates,"teacher_truth_in_inference_forward":False,"tessa_invoked":False,"carrier_optimizer_steps_executed_by_court":0,"scientific_pass_claimed":passed,"surf_a2_authorized":False,"tessa_authorized":False,"next_action":next_action}; out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print("H_GEO_P1_COURT",status,json.dumps({"aggregate":aggregate,"gates":gates,"retention":result["retention_global"]},sort_keys=True),flush=True); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--expected-source-sha",required=True); p.add_argument("--binding",default="canonical/H_GEO_P1_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P1_PREREGISTRATION_V1.json"); p.add_argument("--training-result",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--dataset-root",required=True); p.add_argument("--dataset-receipt",required=True); p.add_argument("--prepared-root",required=True); p.add_argument("--prepared-manifest",required=True); p.add_argument("--v8-checkpoint",required=True); p.add_argument("--output",required=True); p.add_argument("--feature-chunk",type=int,default=4096); p.add_argument("--eval-chunk",type=int,default=4096); audit(p.parse_args()); return 0
if __name__=="__main__": raise SystemExit(main())
