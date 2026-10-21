from __future__ import annotations

"""Runtime helpers for H-GEO-P3 explicit high-resolution geometry carrier."""

from pathlib import Path
from typing import Any
import torch

from models.iris.vnext.explicit_geometry_carrier_v1 import ExplicitGeometryCarrierV1
from tools.iris_vnext.h_geo_p1_geometry_v1 import SUBJECT_IDS
from tools.iris_vnext.h_geo_p1_runtime_v1 import load_dataset_split_v1, create_aux_head_v1
from tools.iris_vnext.h_geo_p2_runtime_v1 import verify_p1_dataset_v1
from tools.iris_vnext.h_rep_probe_authority_v1 import load_json, load_subject_source, module_state_sha256_v1, sha256_file


def load_binding_v1(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HGeoP3InputBinding.v1": raise ValueError("H_GEO_P3_BINDING_SCHEMA_INVALID")
    return value


def load_prereg_v1(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "RealSaS.HGeoP3Preregistration.v1": raise ValueError("H_GEO_P3_PREREG_SCHEMA_INVALID")
    return value


def shared_v8_hashes_v1(model) -> dict[str, str]:
    return {
        "scene_core": module_state_sha256_v1(model.scene_core),
        "mechanical_demand": module_state_sha256_v1(model.mechanical_demand),
        "appearance": module_state_sha256_v1(model.appearance),
        "relations": module_state_sha256_v1(model.relations),
        "structure": module_state_sha256_v1(model.structure),
    }


def create_explicit_geometry_carrier_v1(v8_model, device) -> ExplicitGeometryCarrierV1:
    carrier = ExplicitGeometryCarrierV1(v8_model.config).to(device)
    carrier.initialize_from_v8(v8_model)
    return carrier


def set_h_geo_p3_trainable_v1(carrier: ExplicitGeometryCarrierV1):
    for p in carrier.parameters(): p.requires_grad_(False)
    for p in carrier.geometry_scene.parameters(): p.requires_grad_(True)
    for p in carrier.highres_refiner.parameters(): p.requires_grad_(True)
    body = carrier.geometry_field
    for module in (body.input_norm, body.input_projection, body.blocks, body.feature_projection, body.final_norm):
        for p in module.parameters(): p.requires_grad_(True)
    for p in carrier.position_projection.parameters(): p.requires_grad_(True)
    names = tuple(name for name, p in carrier.named_parameters() if p.requires_grad)
    allowed = ("geometry_scene.","highres_refiner.","geometry_field.input_norm.","geometry_field.input_projection.","geometry_field.blocks.","geometry_field.feature_projection.","geometry_field.final_norm.","position_projection.")
    if not names or any(not n.startswith(allowed) for n in names): raise RuntimeError("H_GEO_P3_TRAINABLE_CENSUS_DRIFT")
    groups = {
        "scene": [p for n,p in carrier.named_parameters() if p.requires_grad and n.startswith("geometry_scene.")],
        "refiner": [p for n,p in carrier.named_parameters() if p.requires_grad and n.startswith("highres_refiner.")],
        "field": [p for n,p in carrier.named_parameters() if p.requires_grad and n.startswith("geometry_field.")],
        "position": [p for n,p in carrier.named_parameters() if p.requires_grad and n.startswith("position_projection.")],
    }
    if any(not v for v in groups.values()): raise RuntimeError("H_GEO_P3_TRAINABLE_GROUP_EMPTY")
    return groups, names


def build_subject_runtime_v1(*,v8_model,prepared_root:Path,prepared_manifest:Path,dataset_root:Path,dataset_receipt:Path,binding:dict[str,Any],split:str,device):
    prepared_root=prepared_root.resolve(); manifest_path=prepared_manifest.resolve(); manifest=load_json(manifest_path)
    rows={str(r["subject_id"]):r for r in manifest.get("subjects") or ()}; expected={str(r["subject_id"]):r for r in binding["prepared"]["subjects"]}
    if tuple(rows)!=SUBJECT_IDS or sha256_file(manifest_path)!=binding["prepared"]["manifest"]["sha256"]: raise ValueError("H_GEO_P3_MANIFEST_DRIFT")
    _, dsrows = verify_p1_dataset_v1(dataset_root=dataset_root,dataset_receipt=dataset_receipt,binding=binding)
    runtime={}; v8_model.eval()
    for subject in SUBJECT_IDS:
        row=rows[subject]; ex=expected[subject]
        if row["shard_file"]!=ex["file"] or row["shard_sha256"]!=ex["sha256"]: raise ValueError("H_GEO_P3_PREPARED_BINDING_DRIFT:"+subject)
        shard=(prepared_root/ex["file"]).resolve(); source=load_subject_source(shard,ex["sha256"],device)
        arrays=load_dataset_split_v1((dataset_root/dsrows[subject]["payload_file"]).resolve(),split)
        with torch.no_grad():
            memory=v8_model.scene_core.view_bank(source["image_tokens"],source["token_xy"],source["camera_features"]).detach().clone()
        runtime[subject]={"source":source,"frozen_view_memory":memory,"arrays":arrays,"shard":shard}
        print(f"H_GEO_P3_RUNTIME_CACHE split={split} subject={subject} rows={len(arrays['xyz_iris'])}",flush=True)
    return runtime


__all__=["build_subject_runtime_v1","create_aux_head_v1","create_explicit_geometry_carrier_v1","load_binding_v1","load_prereg_v1","set_h_geo_p3_trainable_v1","shared_v8_hashes_v1","verify_p1_dataset_v1"]
