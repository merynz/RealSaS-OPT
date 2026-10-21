from __future__ import annotations

"""Runtime helpers for H-GEO-P2 existing-architecture scene+field control."""

from pathlib import Path
from typing import Any
import torch

from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS
from tools.iris_vnext.h_geo_p1_runtime_v1 import load_dataset_split_v1, create_aux_head_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_json, load_subject_source, sha256_file

P1_DATASET_SCHEMA="RealSaS.HGeoP1ExactTriangleDataset.v1"


def load_binding_v1(path:Path)->dict[str,Any]:
    value=load_json(path)
    if value.get("schema")!="RealSaS.HGeoP2InputBinding.v1": raise ValueError("H_GEO_P2_BINDING_SCHEMA_INVALID")
    return value


def load_prereg_v1(path:Path)->dict[str,Any]:
    value=load_json(path)
    if value.get("schema")!="RealSaS.HGeoP2Preregistration.v1": raise ValueError("H_GEO_P2_PREREG_SCHEMA_INVALID")
    return value


def verify_p1_dataset_v1(*,dataset_root:Path,dataset_receipt:Path,binding:dict[str,Any]):
    dataset_root=dataset_root.resolve(); dataset_receipt=dataset_receipt.resolve(); authority=binding["exact_triangle_dataset"]["receipt"]
    if sha256_file(dataset_receipt)!=authority["sha256"]: raise ValueError("H_GEO_P2_P1_DATASET_RECEIPT_SHA_DRIFT")
    receipt=load_json(dataset_receipt)
    if receipt.get("schema")!=P1_DATASET_SCHEMA or receipt.get("status")!="MATERIALIZED_H_GEO_P1_EXACT_TRIANGLE_DATASET": raise ValueError("H_GEO_P2_P1_DATASET_RECEIPT_INVALID")
    if tuple(receipt.get("subject_order") or ())!=SUBJECT_IDS: raise ValueError("H_GEO_P2_P1_DATASET_SUBJECT_ORDER_DRIFT")
    if receipt.get("teacher_truth_in_inference_forward") is not False or receipt.get("tessa_invoked") is not False: raise ValueError("H_GEO_P2_P1_DATASET_SCOPE_DRIFT")
    rows={str(r["subject_id"]):r for r in receipt.get("subjects") or ()}
    if tuple(rows)!=SUBJECT_IDS: raise ValueError("H_GEO_P2_P1_DATASET_ROWS_DRIFT")
    for subject in SUBJECT_IDS:
        path=(dataset_root/rows[subject]["payload_file"]).resolve()
        if not path.is_relative_to(dataset_root) or sha256_file(path)!=str(rows[subject]["payload_sha256"]): raise ValueError("H_GEO_P2_P1_DATASET_PAYLOAD_DRIFT:"+subject)
    return receipt,rows


def set_h_geo_p2_trainable_v1(model):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.scene_core.scene.parameters(): p.requires_grad_(True)
    field=model.scene_core.field
    for module in (field.input_norm,field.input_projection,field.blocks,field.feature_projection,field.final_norm):
        for p in module.parameters(): p.requires_grad_(True)
    names=tuple(name for name,p in model.named_parameters() if p.requires_grad)
    allowed=("scene_core.scene.","scene_core.field.input_norm.","scene_core.field.input_projection.","scene_core.field.blocks.","scene_core.field.feature_projection.","scene_core.field.final_norm.")
    if not names or any(not n.startswith(allowed) for n in names): raise RuntimeError("H_GEO_P2_TRAINABLE_CENSUS_DRIFT")
    scene=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("scene_core.scene.")]
    fieldp=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("scene_core.field.")]
    if not scene or not fieldp: raise RuntimeError("H_GEO_P2_TRAINABLE_GROUP_EMPTY")
    return scene,fieldp,names


def frozen_module_hashes_v1(model):
    from tools.iris_vnext.h_rep_probe_authority_v1 import module_state_sha256_v1
    return {
        "scene_core.view_bank":module_state_sha256_v1(model.scene_core.view_bank),
        "scene_core.cube_extraction":module_state_sha256_v1(model.scene_core.cube_extraction),
        "mechanical_demand":module_state_sha256_v1(model.mechanical_demand),
        "appearance":module_state_sha256_v1(model.appearance),
        "relations":module_state_sha256_v1(model.relations),
        "structure":module_state_sha256_v1(model.structure),
    }


def build_subject_runtime_v1(*,model,prepared_root:Path,prepared_manifest:Path,dataset_root:Path,dataset_receipt:Path,binding:dict[str,Any],split:str,device,feature_chunk:int=4096,cache_v8_baseline:bool=False):
    prepared_root=prepared_root.resolve(); manifest_path=prepared_manifest.resolve(); manifest=load_json(manifest_path); rows={str(r["subject_id"]):r for r in manifest.get("subjects") or ()}; expected={str(r["subject_id"]):r for r in binding["prepared"]["subjects"]}
    if tuple(rows)!=SUBJECT_IDS or sha256_file(manifest_path)!=binding["prepared"]["manifest"]["sha256"]: raise ValueError("H_GEO_P2_MANIFEST_DRIFT")
    _,dsrows=verify_p1_dataset_v1(dataset_root=dataset_root,dataset_receipt=dataset_receipt,binding=binding)
    runtime={}; model.eval()
    for subject in SUBJECT_IDS:
        row=rows[subject]; ex=expected[subject]
        if row["shard_file"]!=ex["file"] or row["shard_sha256"]!=ex["sha256"]: raise ValueError("H_GEO_P2_PREPARED_BINDING_DRIFT:"+subject)
        shard=(prepared_root/ex["file"]).resolve()
        # Keep live P2 source tensors ordinary (not inference tensors): scene-core
        # parameter gradients may need to save these values for backward.
        source=load_subject_source(shard,ex["sha256"],device)
        arrays=load_dataset_split_v1((dataset_root/dsrows[subject]["payload_file"]).resolve(),split)
        if cache_v8_baseline:
            hs=[]; fs=[]; xyz=arrays["xyz_iris"]
            with torch.inference_mode():
                planes=model.encode_scene(source["image_tokens"],source["token_xy"],source["camera_features"]).float()
                for start in range(0,len(xyz),int(feature_chunk)):
                    q=xyz[start:start+int(feature_chunk)].to(device=device,dtype=torch.float32)[None]
                    local=model.scene_core.query(planes,q)
                    hs.append(local["H"].squeeze(0).detach().cpu().float())
                    fs.append(local["field"].squeeze(0).detach().cpu().float())
            # clone outside inference_mode so retention targets are ordinary tensors.
            arrays["baseline_H"]=torch.cat(hs).clone()
            arrays["baseline_field"]=torch.cat(fs).clone()
        runtime[subject]={"source":source,"arrays":arrays,"shard":shard}
        print(f"H_GEO_P2_RUNTIME_CACHE split={split} subject={subject} rows={len(arrays['xyz_iris'])}",flush=True)
    return runtime
