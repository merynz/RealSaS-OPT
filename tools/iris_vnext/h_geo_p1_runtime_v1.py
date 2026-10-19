from __future__ import annotations

"""Runtime helpers shared by H-GEO-P1 trainer, preflight and court."""

from pathlib import Path
from typing import Any
import numpy as np
import torch
from models.iris.vnext.metric_geometry_v1 import IRISMetricGeometryAuxHeadV1
from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS
from tools.iris_vnext.h_rep_probe_authority_v1 import load_json, load_subject_source, sha256_file

DATASET_SCHEMA = "RealSaS.HGeoP1ExactTriangleDataset.v1"


def load_binding_v1(path: Path) -> dict[str, Any]:
    value=load_json(path)
    if value.get("schema")!="RealSaS.HGeoP1InputBinding.v1": raise ValueError("H_GEO_P1_BINDING_SCHEMA_INVALID")
    return value


def load_prereg_v1(path: Path) -> dict[str, Any]:
    value=load_json(path)
    if value.get("schema")!="RealSaS.HGeoP1Preregistration.v1": raise ValueError("H_GEO_P1_PREREG_SCHEMA_INVALID")
    return value


def load_dataset_split_v1(path:Path,split:str)->dict[str,torch.Tensor]:
    if split not in ("train","holdout"): raise ValueError("H_GEO_P1_SPLIT_INVALID")
    keys=("xyz_iris","closest_surface_iris","distance_iris","offset_to_surface_iris","normal_unit","normal_valid","near_label")
    with np.load(path,allow_pickle=False) as data:
        out={}
        for key in keys:
            name=f"{split}_{key}"
            if name not in data: raise ValueError("H_GEO_P1_DATASET_FIELD_MISSING:"+name)
            value=np.asarray(data[name]); out[key]=torch.from_numpy(value.astype(np.bool_) if key in ("normal_valid","near_label") else value.astype(np.float32))
    n=len(out["xyz_iris"])
    if out["xyz_iris"].shape!=(n,3) or out["closest_surface_iris"].shape!=(n,3) or out["offset_to_surface_iris"].shape!=(n,3) or out["normal_unit"].shape!=(n,3): raise ValueError("H_GEO_P1_DATASET_VECTOR_SHAPE_INVALID")
    if out["distance_iris"].shape!=(n,) or out["normal_valid"].shape!=(n,) or out["near_label"].shape!=(n,): raise ValueError("H_GEO_P1_DATASET_SCALAR_SHAPE_INVALID")
    for key,value in out.items():
        if value.dtype!=torch.bool and not bool(torch.isfinite(value).all().item()): raise ValueError("H_GEO_P1_DATASET_NONFINITE:"+key)
    return out


def verify_dataset_receipt_v1(*,dataset_root:Path,dataset_receipt:Path,binding,binding_sha256:str,preregistration_sha256:str):
    dataset_root=dataset_root.resolve(); receipt=load_json(dataset_receipt.resolve())
    if receipt.get("schema")!=DATASET_SCHEMA or receipt.get("status")!="MATERIALIZED_H_GEO_P1_EXACT_TRIANGLE_DATASET": raise ValueError("H_GEO_P1_DATASET_RECEIPT_INVALID")
    if receipt.get("input_binding_sha256")!=binding_sha256 or receipt.get("preregistration_sha256")!=preregistration_sha256: raise ValueError("H_GEO_P1_DATASET_AUTHORITY_DRIFT")
    if tuple(receipt.get("subject_order") or ())!=SUBJECT_IDS or receipt.get("teacher_truth_in_inference_forward") is not False or receipt.get("tessa_invoked") is not False: raise ValueError("H_GEO_P1_DATASET_SCOPE_DRIFT")
    rows={str(r["subject_id"]):r for r in receipt.get("subjects") or ()}
    if tuple(rows)!=SUBJECT_IDS: raise ValueError("H_GEO_P1_DATASET_ROW_ORDER_DRIFT")
    for subject in SUBJECT_IDS:
        path=(dataset_root/rows[subject]["payload_file"]).resolve()
        if not path.is_relative_to(dataset_root) or sha256_file(path)!=rows[subject]["payload_sha256"]: raise ValueError("H_GEO_P1_DATASET_PAYLOAD_DRIFT:"+subject)
    return rows


def set_h_geo_p1_trainable_v1(model):
    for p in model.parameters(): p.requires_grad_(False)
    field=model.scene_core.field; modules=(field.input_norm,field.input_projection,field.blocks,field.feature_projection,field.final_norm)
    for module in modules:
        for p in module.parameters(): p.requires_grad_(True)
    names=tuple(name for name,p in model.named_parameters() if p.requires_grad); allowed=("scene_core.field.input_norm.","scene_core.field.input_projection.","scene_core.field.blocks.","scene_core.field.feature_projection.","scene_core.field.final_norm.")
    if not names or any(not name.startswith(allowed) for name in names): raise RuntimeError("H_GEO_P1_TRAINABLE_CENSUS_DRIFT")
    return [p for p in model.parameters() if p.requires_grad],names


def build_subject_runtime_v1(*,model,prepared_root:Path,prepared_manifest:Path,dataset_root:Path,dataset_receipt:Path,binding,binding_sha256:str,preregistration_sha256:str,split:str,device,feature_chunk:int=4096,cache_baseline:bool=False):
    manifest=load_json(prepared_manifest.resolve()); rows={str(r["subject_id"]):r for r in manifest.get("subjects") or ()}; expected={str(r["subject_id"]):r for r in binding["prepared"]["subjects"]}; dsrows=verify_dataset_receipt_v1(dataset_root=dataset_root,dataset_receipt=dataset_receipt,binding=binding,binding_sha256=binding_sha256,preregistration_sha256=preregistration_sha256)
    if tuple(rows)!=SUBJECT_IDS or sha256_file(prepared_manifest.resolve())!=binding["prepared"]["manifest"]["sha256"]: raise ValueError("H_GEO_P1_RUNTIME_MANIFEST_DRIFT")
    runtime={}; model.eval()
    with torch.inference_mode():
        for subject in SUBJECT_IDS:
            row=rows[subject]; ex=expected[subject]
            if row["shard_file"]!=ex["file"] or row["shard_sha256"]!=ex["sha256"]: raise ValueError("H_GEO_P1_RUNTIME_PREPARED_BINDING_DRIFT:"+subject)
            shard=(prepared_root/ex["file"]).resolve(); source=load_subject_source(shard,ex["sha256"],device); planes=model.encode_scene(source["image_tokens"],source["token_xy"],source["camera_features"]).float().detach(); arrays=load_dataset_split_v1((dataset_root/dsrows[subject]["payload_file"]).resolve(),split)
            if cache_baseline:
                hs=[]; fs=[]; xyz=arrays["xyz_iris"]
                for start in range(0,len(xyz),int(feature_chunk)):
                    q=xyz[start:start+int(feature_chunk)].to(device=device,dtype=torch.float32)[None]; local=model.scene_core.query(planes,q); hs.append(local["H"].squeeze(0).cpu().float()); fs.append(local["field"].squeeze(0).cpu().float())
                arrays["baseline_H"]=torch.cat(hs,0); arrays["baseline_field"]=torch.cat(fs,0)
            runtime[subject]={"scene_planes":planes,"arrays":arrays,"source":source,"shard":shard}; print(f"H_GEO_P1_RUNTIME_CACHE split={split} subject={subject} rows={len(arrays['xyz_iris'])}",flush=True)
    return runtime


def create_aux_head_v1(device): return IRISMetricGeometryAuxHeadV1().to(device=device,dtype=torch.float32)
