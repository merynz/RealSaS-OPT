#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS, SUBJECT_SOURCE_IDS, make_query_split_v1, recover_prepared_normalization_v1

SCHEMA="RealSaS.HGeoP1ExactTriangleDataset.v1"

def sha256_file(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def J(path):
    v=json.loads(Path(path).read_text());
    if not isinstance(v,dict): raise ValueError("H_GEO_P1_JSON_OBJECT_REQUIRED")
    return v

def build(args):
    binding_path=Path(args.binding).resolve(); prereg_path=Path(args.preregistration).resolve(); binding=J(binding_path); prereg=J(prereg_path)
    if binding.get("schema")!="RealSaS.HGeoP1InputBinding.v1" or prereg.get("schema")!="RealSaS.HGeoP1Preregistration.v1": raise ValueError("H_GEO_P1_AUTHORITY_SCHEMA_INVALID")
    prepared_root=Path(args.prepared_root).resolve(); teacher_root=Path(args.teacher_root).resolve(); manifest_path=Path(args.prepared_manifest).resolve(); teacher_batch_path=Path(args.teacher_batch_receipt).resolve(); manifest=J(manifest_path); teacher_batch=J(teacher_batch_path)
    if sha256_file(manifest_path)!=binding["prepared"]["manifest"]["sha256"] or sha256_file(teacher_batch_path)!=binding["teachers"]["batch_receipt"]["sha256"]: raise ValueError("H_GEO_P1_INPUT_RECEIPT_SHA_DRIFT")
    mr={str(r["subject_id"]):r for r in manifest["subjects"]}; tr={str(r["subject_id"]):r for r in teacher_batch["subjects"]}; ep={str(r["subject_id"]):r for r in binding["prepared"]["subjects"]}; et={str(r["subject_id"]):r for r in binding["teachers"]["subjects"]}
    if tuple(mr)!=SUBJECT_IDS or tuple(tr)!=SUBJECT_IDS: raise ValueError("H_GEO_P1_SUBJECT_ORDER_DRIFT")
    output=Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise RuntimeError("H_GEO_P1_DATASET_OUTPUT_MUST_BE_EMPTY")
    output.mkdir(parents=True,exist_ok=True); rows=[]; maxerr=0.0; truth=prereg["truth"]; sampling=prereg["sampling"]
    for si,subject in enumerate(SUBJECT_IDS):
        if mr[subject]["shard_file"]!=ep[subject]["file"] or mr[subject]["shard_sha256"]!=ep[subject]["sha256"] or tr[subject]["teacher_file"]!=et[subject]["file"] or tr[subject]["teacher_sha256"]!=et[subject]["sha256"]: raise ValueError("H_GEO_P1_SUBJECT_BINDING_DRIFT:"+subject)
        pp=(prepared_root/ep[subject]["file"]).resolve(); tp=(teacher_root/et[subject]["file"]).resolve()
        if sha256_file(pp)!=ep[subject]["sha256"] or sha256_file(tp)!=et[subject]["sha256"]: raise ValueError("H_GEO_P1_SUBJECT_SHA_DRIFT:"+subject)
        shard=torch.load(pp,map_location="cpu",weights_only=True)
        with np.load(tp,allow_pickle=False) as data: rest=np.asarray(data["rest_vertices_source_world"],dtype=np.float64); faces=np.asarray(data["faces"],dtype=np.int64)
        norm=recover_prepared_normalization_v1(subject_id=subject,rest_vertices_source_world=rest,faces=faces,stored_surface_points_iris=torch.as_tensor(shard["teacher_surface_points_iris"]).float().numpy(),maximum_error_iris=float(truth["normalization_recovery_max_error_iris"])); maxerr=max(maxerr,norm.maximum_replay_error_iris); verts=norm.apply(rest)
        common=dict(vertices_iris=verts,faces=faces,offsets_iris=sampling["surface_normal_offsets_iris"],near_threshold_iris=float(truth["near_threshold_iris"]),normal_valid_distance_iris=float(truth["normal_valid_distance_iris"]))
        train=make_query_split_v1(seed=27010000+si,near_count=int(sampling["train_near_per_subject"]),uniform_count=int(sampling["train_uniform_per_subject"]),**common); hold=make_query_split_v1(seed=27020000+si,near_count=int(sampling["holdout_near_per_subject"]),uniform_count=int(sampling["holdout_uniform_per_subject"]),**common)
        out=output/f"{subject.lower().replace('-','').replace(' ','')}.h_geo_p1.npz"; np.savez_compressed(out,**{f"train_{k}":v for k,v in train.items()},**{f"holdout_{k}":v for k,v in hold.items()})
        row={"subject_id":subject,"source_subject_id":SUBJECT_SOURCE_IDS[subject],"prepared_file":ep[subject]["file"],"prepared_sha256":ep[subject]["sha256"],"teacher_file":et[subject]["file"],"teacher_sha256":et[subject]["sha256"],"teacher_vertex_count":len(rest),"teacher_face_count":len(faces),"normalization":{"scale":norm.scale,"translation":list(norm.translation),"center_source_world":list(norm.center_source_world),"half_extent_source_world":norm.half_extent_source_world,"maximum_replay_error_iris":norm.maximum_replay_error_iris},"payload_file":out.name,"payload_sha256":sha256_file(out),"train_count":len(train["xyz_iris"]),"holdout_count":len(hold["xyz_iris"]),"train_near_positive_fraction":float(train["near_label"].mean()),"holdout_near_positive_fraction":float(hold["near_label"].mean())}; rows.append(row); print("H_GEO_P1_DATASET_SUBJECT",subject,f"vertices={len(rest)}",f"faces={len(faces)}",f"norm_error={norm.maximum_replay_error_iris:.3e}",flush=True)
    receipt={"schema":SCHEMA,"status":"MATERIALIZED_H_GEO_P1_EXACT_TRIANGLE_DATASET","input_binding_sha256":sha256_file(binding_path),"preregistration_sha256":sha256_file(prereg_path),"prepared_manifest_sha256":sha256_file(manifest_path),"teacher_batch_receipt_sha256":sha256_file(teacher_batch_path),"subject_order":list(SUBJECT_IDS),"subjects":rows,"maximum_normalization_replay_error_iris":maxerr,"truth_semantics":"EXACT_CLOSEST_POINT_TO_AUTHORED_REST_TRIANGLES__UNSIGNED_DISTANCE__NO_SDF_SIGN_CLAIM","teacher_truth_in_inference_forward":False,"tessa_invoked":False,"optimizer_steps_executed":0,"scientific_pass_claimed":False,"next_action":"RUN_H_GEO_P1_A100_PREFLIGHT"}; (output/"H_GEO_P1_DATASET_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); print("H_GEO_P1_DATASET_PASS",f"subjects={len(rows)}",f"max_norm_error={maxerr:.3e}",flush=True); return receipt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--binding",default="canonical/H_GEO_P1_INPUT_BINDING_V1.json"); p.add_argument("--preregistration",default="canonical/H_GEO_P1_PREREGISTRATION_V1.json"); p.add_argument("--prepared-root",required=True); p.add_argument("--prepared-manifest",required=True); p.add_argument("--teacher-root",required=True); p.add_argument("--teacher-batch-receipt",required=True); p.add_argument("--output-dir",required=True); build(p.parse_args()); return 0
if __name__=="__main__": raise SystemExit(main())
