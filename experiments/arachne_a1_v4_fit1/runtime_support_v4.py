from __future__ import annotations

"""Execution support for Arachne A1 v4.

This module contains only frozen-authority loading, evaluation and proposal helpers.
It deliberately has no dependency on the superseded A1-v3 predictor or objective.
"""

from hashlib import sha256
import json
from pathlib import Path
import math
import numpy as np
import torch

from compiler.realsas_compiler_core.substrate.scene_first_signed import rigging_surface_from_scene_first_zero_mesh_v1
from compiler.realsas_compiler_core.types import QualifiedJoint
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from models.skin_field_codec.v7.skin_field_codec_v7 import ArachneSkinFieldConfigV7, SkinFieldCodecV7
from models.arachne.v4.loss_v4 import normalize_joint_fields

EXPECTED_A0_MODEL_SHA = "8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7"
EXPECTED_BANK_SHA = "b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a"
EXPECTED_A0_CONFIG_HASH = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
EXPECTED_ZERO_SHA = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
EXPECTED_SKELETON_FILE_SHA = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"
EXPECTED_SURFACE_LINEAGE = "67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"
IRIS_CHECKPOINT_SHA = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
SOURCE_RUN_ID = "20260904T220929Z"
CAMERA_SHA = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)


def sha_file(path: Path, chunk: int = 8 << 20) -> str:
    h=sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=float)+"\n",encoding="utf-8"); tmp.replace(path)


def atomic_torch_save(payload, path: Path) -> str:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,tmp); tmp.replace(path); return sha_file(path)


def load_cameras(data_dir: Path) -> tuple[dict,...]:
    cams=[]
    for i,expected in enumerate(CAMERA_SHA):
        p=data_dir/f"V{i}.camera.json"
        if not p.exists() or sha_file(p)!=expected: raise RuntimeError(f"CAMERA_SHA_DRIFT_V{i}")
        cams.append(json.loads(p.read_text(encoding="utf-8")))
    cams.sort(key=lambda x:int(x["view_index"]))
    if [int(c["view_index"]) for c in cams]!=list(range(8)): raise RuntimeError("CAMERA_VIEW_INDEX_DRIFT")
    if [int(c["yaw_deg"]) for c in cams]!=[0,45,90,135,180,225,270,315]: raise RuntimeError("CAMERA_YAW_CONTRACT_DRIFT")
    return tuple(cams)


def build_surface(zero_path: Path, cameras: tuple[dict,...]):
    if sha_file(zero_path)!=EXPECTED_ZERO_SHA: raise RuntimeError("ZERO_SURFACE_SHA_DRIFT")
    with np.load(zero_path,allow_pickle=False) as z:
        world=np.asarray(z["vertices"],np.float64); faces=np.asarray(z["faces"],np.int64); hints=np.asarray(z["normals"],np.float64)
    center=np.asarray(cameras[0]["center"],np.float64); half=float(cameras[0]["half_extent"])
    s=rigging_surface_from_scene_first_zero_mesh_v1((world-center[None])/half,faces,hints,cameras,normalization_center=center,normalization_half_extent=half,authority_label="IRIS_SCENE_FIRST_SIGNED_V3_PROMOTED_MAGE_FIT",source_run_id=SOURCE_RUN_ID,source_checkpoint_sha256=IRIS_CHECKPOINT_SHA,source_zero_surface_sha256=EXPECTED_ZERO_SHA,target_nodes=1024,normal_k=64,visibility_depth_tolerance_norm=.02,metadata={"camera_contract":"CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG","teacher_truth_used":False,"geppetto_reference_strength_fit1":True})
    if s.geometry_lineage_hash!=EXPECTED_SURFACE_LINEAGE: raise RuntimeError(f"SURFACE_LINEAGE_DRIFT::{s.geometry_lineage_hash}")
    if len(s.surface_nodes)!=950 or len(s.local_relations)!=2813: raise RuntimeError("SURFACE_WITNESS_CARDINALITY_DRIFT")
    return s


def load_skeleton(path: Path) -> QualifiedSkeletonIRV2:
    if sha_file(path)!=EXPECTED_SKELETON_FILE_SHA: raise RuntimeError("QUALIFIED_SKELETON_FILE_SHA_DRIFT")
    d=json.loads(path.read_text(encoding="utf-8"))
    if d.get("skeleton_lineage_hash")!=EXPECTED_SKELETON_LINEAGE: raise RuntimeError("SKELETON_LINEAGE_DRIFT")
    joints=tuple(QualifiedJoint(canonical_joint_id=str(x["canonical_joint_id"]),position=tuple(map(float,x["position"])),parent_canonical_id=None if x.get("parent_canonical_id") is None else str(x["parent_canonical_id"]),support_surface_ids=tuple(map(str,x.get("support_surface_ids",()))),source_proposal_id=str(x.get("source_proposal_id",""))) for x in d["joints"])
    if len(joints)!=22: raise RuntimeError("QUALIFIED_JOINT_COUNT_DRIFT")
    deform=tuple(map(str,d.get("deform_root_ids",()))) or tuple(j.canonical_joint_id for j in joints if j.parent_canonical_id is None)
    return QualifiedSkeletonIRV2(joints=joints,deform_root_ids=deform,assembly_root_binding=dict(d.get("assembly_root_binding",{})),qualification_report=dict(d.get("qualification_report",{})),skeleton_lineage_hash=str(d["skeleton_lineage_hash"]))


def load_bank(path: Path) -> dict[str,np.ndarray]:
    if sha_file(path)!=EXPECTED_BANK_SHA: raise RuntimeError("A1_SUPERVISION_BANK_SHA_DRIFT")
    with np.load(path,allow_pickle=False) as z: d={k:np.asarray(z[k]) for k in z.files}
    req={"surface_ids","joint_ids","gsa_geometry7","gsa_world","gsa_teacher_weights","gsa_supervision_mask","dense8k_geometry7","dense8k_world","dense8k_teacher_weights","holdout_geometry7","holdout_world","holdout_teacher_weights","condition_query_indices"}
    if not req.issubset(d): raise RuntimeError(f"A1_BANK_SCHEMA_MISSING::{sorted(req-set(d))}")
    if d["gsa_geometry7"].shape!=(950,7) or d["gsa_teacher_weights"].shape!=(950,22): raise RuntimeError("A1_BANK_GSA_SHAPE_DRIFT")
    if d["dense8k_geometry7"].shape[1:]!=(7,) or d["dense8k_teacher_weights"].shape[1:]!=(22,): raise RuntimeError("A1_BANK_DENSE_SHAPE_DRIFT")
    if d["holdout_geometry7"].shape[1:]!=(7,) or d["holdout_teacher_weights"].shape[1:]!=(22,): raise RuntimeError("A1_BANK_HOLDOUT_SHAPE_DRIFT")
    if d["condition_query_indices"].shape!=(384,): raise RuntimeError("A1_BANK_CONDITION_QUERY_DRIFT")
    return d


def load_frozen_codec(path: Path, device: torch.device):
    if sha_file(path)!=EXPECTED_A0_MODEL_SHA: raise RuntimeError("A0_K4_MODEL_SHA_DRIFT")
    try: payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    except Exception: payload=torch.load(path,map_location="cpu",weights_only=False)
    if int(payload.get("token_count",-1))!=4 or payload.get("config_hash")!=EXPECTED_A0_CONFIG_HASH: raise RuntimeError("A0_K4_PAYLOAD_CONTRACT_DRIFT")
    cfg=ArachneSkinFieldConfigV7()
    if cfg.config_hash!=EXPECTED_A0_CONFIG_HASH: raise RuntimeError("LOCAL_V7_CONFIG_HASH_DRIFT")
    codec=SkinFieldCodecV7(cfg); codec.load_state_dict(payload["model"],strict=True); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    codec=codec.to(device=device,dtype=torch.float32)
    if any(p.requires_grad for p in codec.parameters()): raise RuntimeError("FROZEN_CODEC_HAS_TRAINABLE_PARAMETER")
    return codec,payload


def conditioning_to_torch(c,device):
    names=("surface_positions_normalized","surface_normals","surface_normal_valid","surface_support_views","surface_raster_xy","surface_raster_valid","surface_observed","surface_completed","surface_mask","edge_index","edge_features","edge_mask","joint_positions_normalized","joint_mask","parent_indices","root_mask","deform_root_mask","joint_depth_normalized","support_anchor_matrix","pair_geometry","pair_mask")
    bools={"surface_normal_valid","surface_support_views","surface_raster_valid","surface_observed","surface_completed","surface_mask","edge_mask","joint_mask","root_mask","deform_root_mask","support_anchor_matrix","pair_mask"}; ints={"edge_index","parent_indices"}
    return {n:torch.as_tensor(getattr(c,n),device=device,dtype=torch.bool if n in bools else torch.long if n in ints else torch.float32) for n in names}


def decode_logits(codec, prepared_condition, field_tokens, query_geometry, *, chunk=1024):
    if field_tokens.ndim!=4 or field_tokens.shape[0]!=1: raise ValueError("field token shape drift")
    q=torch.as_tensor(query_geometry,device=field_tokens.device,dtype=torch.float32); pieces=[]
    for start in range(0,len(q),int(chunk)):
        qg=q[start:start+chunk][None]
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True): qemb=codec.geometry_embedding(qg).detach()
        pj=[]
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
            for j in range(field_tokens.shape[1]): pj.append(codec.decoder.query_logits(prepared_condition,field_tokens[:,j],qemb))
        pieces.append(torch.stack(pj,dim=-1).float())
    return torch.cat(pieces,dim=1)


def _legacy_probe_transforms(j:int,device):
    poses=4; t=torch.eye(4,dtype=torch.float32,device=device)[None,None].repeat(1,poses,j,1,1)
    for ji in range(j):
        u=float(ji+1)/j; t[0,1,ji,0,3]=.10*u; t[0,1,ji,1,3]=.035*(-1 if ji%2 else 1); t[0,2,ji,1,3]=.085*u; t[0,2,ji,2,3]=.030*(ji-(j-1)/2); t[0,3,ji,0,3]=-.055*(ji-(j-1)/2); t[0,3,ji,2,3]=.070*u
    return t


def _lbs(rest,weights,transforms):
    ones=torch.ones((*rest.shape[:2],1),dtype=rest.dtype,device=rest.device); hom=torch.cat([rest,ones],-1); moved=torch.einsum("bpjac,bnc->bpjna",transforms,hom)[...,:3]; return torch.einsum("bnj,bpjna->bpna",weights,moved)


def _deform_ratio(rest,wt,wp,T,mask):
    td=_lbs(rest,wt,T); pd=_lbs(rest,wp,T); rp=rest[:,None].expand_as(td); m=mask[:,None,:,None].to(td.dtype); den=(m.sum()*td.shape[1]*td.shape[-1]).clamp_min(1.0); motion=torch.sqrt((((td-rp)*m).square().sum()/den).clamp_min(1e-12)); err=torch.sqrt((((pd-td)*m).square().sum()/den).clamp_min(1e-12)); return float((err/motion.clamp_min(1e-6)).cpu()),float(motion.cpu()),float(err.cpu())


def metrics(pred,truth,world,device,mask=None):
    pred=np.asarray(pred,np.float64); truth=np.asarray(truth,np.float64); mask=np.ones(len(pred),bool) if mask is None else np.asarray(mask,bool); row=np.abs(pred-truth).sum(1)
    dom=float((pred[mask].argmax(1)==truth[mask].argmax(1)).mean()); top3=np.argpartition(-pred[mask],kth=2,axis=1)[:,:3]; td=truth[mask].argmax(1); top3inc=float(np.mean([td[i] in top3[i] for i in range(len(td))]))
    rest=torch.as_tensor(np.asarray(world,np.float32)[None],device=device); wt=torch.as_tensor(truth.astype(np.float32)[None],device=device); wp=torch.as_tensor(pred.astype(np.float32)[None],device=device); mt=torch.as_tensor(mask[None],device=device,dtype=torch.bool); deform,motion,err=_deform_ratio(rest,wt,wp,_legacy_probe_transforms(pred.shape[1],device),mt)
    return {"rows_total":int(len(pred)),"rows_evaluated":int(mask.sum()),"row_l1_mean":float(row[mask].mean()),"row_l1_p50":float(np.quantile(row[mask],.50)),"row_l1_p90":float(np.quantile(row[mask],.90)),"row_l1_p95":float(np.quantile(row[mask],.95)),"row_l1_p99":float(np.quantile(row[mask],.99)),"cvar10":float(row[mask][row[mask]>=np.quantile(row[mask],.90)].mean()),"deformation_error_ratio":deform,"deformation_error_rms":err,"teacher_motion_rms":motion,"dominant_accuracy":dom,"teacher_dominant_top3_inclusion":top3inc,"simplex_max_abs_residual":float(np.max(np.abs(pred.sum(1)-1.0)))}


__all__=["EXPECTED_A0_MODEL_SHA","EXPECTED_BANK_SHA","EXPECTED_A0_CONFIG_HASH","EXPECTED_SURFACE_LINEAGE","EXPECTED_SKELETON_LINEAGE","sha_file","write_json","atomic_torch_save","load_cameras","build_surface","load_skeleton","load_bank","load_frozen_codec","conditioning_to_torch","decode_logits","metrics"]
