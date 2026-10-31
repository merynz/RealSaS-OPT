#!/usr/bin/env python3
from __future__ import annotations

"""Build source-import and camera-observation authority candidate from one packet.

This producer binds immutable raw source bytes to the admitted bind-rest scene,
normalization, exact 8 observation rasters, masks and camera bundle. It does not
infer geometry, appearance sampler semantics, compositing, rig truth or training
admission.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

SCHEMA="RealSaS.SourceObservationAuthority.v1"
STATUS="PASS_SOURCE_OBSERVATION_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def digest(value)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load(path:Path)->dict:
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict): raise ValueError("SOURCE_OBSERVATION_JSON_OBJECT")
    return value


def matrix(value,code):
    m=np.asarray(value,np.float64)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-10,rtol=0): raise ValueError(code)
    return m


def build(subject_id:str,source_path:Path,packet_root:Path,output_dir:Path)->dict:
    source_path=source_path.resolve(); packet_root=packet_root.resolve(); output_dir=output_dir.resolve()
    if not source_path.is_file(): raise FileNotFoundError(str(source_path))
    source_sha=sha256(source_path)
    paths={name:packet_root/name for name in ("SOURCE_REST_RENDER_CANDIDATE.json","SOURCE_SCENE_SNAPSHOT.json","IRIS_OBSERVATION_ADMISSION.json","camera_bundle.json")}
    if not all(p.is_file() for p in paths.values()): raise FileNotFoundError("SOURCE_OBSERVATION_PACKET_CONTROL_FILES")
    candidate=load(paths["SOURCE_REST_RENDER_CANDIDATE.json"]); snapshot=load(paths["SOURCE_SCENE_SNAPSHOT.json"]); proof=load(paths["IRIS_OBSERVATION_ADMISSION.json"]); camera=load(paths["camera_bundle.json"])
    if candidate.get("schema")!="RealSaS.Controlled8ViewSourceRestCandidate.v2" or snapshot.get("schema")!="RealSaS.ImportedSourceSceneSnapshot.v2" or proof.get("schema")!="RealSaS.IrisBindObservationAdmission.v1" or camera.get("schema")!="RealSaS.CameraProjectionBundle.v1": raise ValueError("SOURCE_OBSERVATION_SCHEMA")
    if any(str(x.get("subject_id") or x.get("subject_key") or subject_id)!=subject_id for x in (candidate,proof)): raise ValueError("SOURCE_OBSERVATION_SUBJECT")
    if str(candidate.get("source_sha256"))!=source_sha or str(snapshot.get("source_sha256"))!=source_sha or str(proof.get("source_sha256"))!=source_sha: raise ValueError("SOURCE_OBSERVATION_SOURCE_SHA")
    candidate_sha=sha256(paths["SOURCE_REST_RENDER_CANDIDATE.json"]); snapshot_sha=sha256(paths["SOURCE_SCENE_SNAPSHOT.json"]); proof_sha=sha256(paths["IRIS_OBSERVATION_ADMISSION.json"]); camera_sha=sha256(paths["camera_bundle.json"])
    if str(proof.get("candidate_sha256"))!=candidate_sha or str(proof.get("camera_sha256"))!=camera_sha or str((candidate.get("source_scene_snapshot") or {}).get("sha256"))!=snapshot_sha or str((candidate.get("camera_bundle") or {}).get("sha256"))!=camera_sha: raise ValueError("SOURCE_OBSERVATION_CONTROL_BINDING")
    if candidate.get("raw_source_mutated",(candidate.get("import") or {}).get("raw_source_mutated")) is not False or candidate.get("pose_address")!="SOURCE_BIND_REST" or candidate.get("native_frame0_is_rest_assumed") is not False: raise ValueError("SOURCE_OBSERVATION_REST_CONTRACT")
    if (candidate.get("internal_consistency") or {}).get("animation_free_rest") is not True or (candidate.get("internal_consistency") or {}).get("normalization_uniform") is not True: raise ValueError("SOURCE_OBSERVATION_INTERNAL_CONSISTENCY")
    if proof.get("status")!="PASS_IRIS_BIND_OBSERVATIONS" or proof.get("scope")!="IMAGE_DERIVED_FSTAR_BIND_OBSERVATIONS__NOT_FULL_MECHANICAL_TRUTH": raise ValueError("SOURCE_OBSERVATION_PROOF_STATUS")
    norm=matrix(snapshot.get("source_world_to_canonical"),"SOURCE_OBSERVATION_NORMALIZATION"); inv=matrix(snapshot.get("canonical_to_source_world"),"SOURCE_OBSERVATION_NORMALIZATION_INVERSE")
    if not np.allclose(norm@inv,np.eye(4),atol=1e-6,rtol=0): raise ValueError("SOURCE_OBSERVATION_NORMALIZATION_INVERSE")
    nodes=list(snapshot.get("nodes") or [])
    if not nodes: raise ValueError("SOURCE_OBSERVATION_EMPTY_SCENE")
    node_index={str(row.get("node_id")):row for row in nodes}
    if len(node_index)!=len(nodes) or "" in node_index: raise ValueError("SOURCE_OBSERVATION_NODE_IDENTITY")
    for row in nodes:
        world=matrix(row.get("node_to_source_world"),"SOURCE_OBSERVATION_NODE_WORLD"); local=matrix(row.get("effective_node_to_parent"),"SOURCE_OBSERVATION_NODE_LOCAL"); parent=row.get("parent_node_id"); parent_world=np.eye(4) if parent is None else matrix(node_index[parent].get("node_to_source_world"),"SOURCE_OBSERVATION_PARENT_WORLD")
        if not np.allclose(world,parent_world@local,atol=1e-6,rtol=1e-6): raise ValueError("SOURCE_OBSERVATION_PARENT_WORLD_DRIFT")
        if any(int(row.get(k,0))!=0 for k in ("active_action_count","active_nla_track_count","driver_count")): raise ValueError("SOURCE_OBSERVATION_LIVE_REST_DRIVER")
    if camera.get("per_view_recenter") is not False or camera.get("per_view_pose_change") is not False or len(camera.get("cameras") or [])!=8: raise ValueError("SOURCE_OBSERVATION_CAMERA_POLICY")
    half=float(camera.get("common_half_extent",0)); center=np.asarray(camera.get("common_center_xyz"),np.float64)
    if not np.isfinite(half) or half<=0 or center.shape!=(3,) or not np.isfinite(center).all(): raise ValueError("SOURCE_OBSERVATION_CAMERA_NORMALIZATION")
    expected=np.eye(4); expected[:3,:3]/=half; expected[:3,3]=-center/half
    if not np.allclose(norm,expected,atol=1e-6,rtol=1e-6): raise ValueError("SOURCE_OBSERVATION_CAMERA_SCENE_NORMALIZATION_DRIFT")
    cviews={int(row.get("view_index",-1)):row for row in candidate.get("views") or []}; pviews={int(row.get("view_index",-1)):row for row in proof.get("views") or []}
    if set(cviews)!=set(range(8)) or set(pviews)!=set(range(8)): raise ValueError("SOURCE_OBSERVATION_VIEW_SET")
    rows=[]
    for i in range(8):
        raster=packet_root/f"V{i}.png"; mask=packet_root/f"V{i}.mask.bin"
        if not raster.is_file() or not mask.is_file(): raise FileNotFoundError("SOURCE_OBSERVATION_VIEW_FILE")
        rsha,msha=sha256(raster),sha256(mask)
        if str((cviews[i].get("raster") or {}).get("sha256"))!=rsha or str((cviews[i].get("foreground_mask") or {}).get("sha256"))!=msha or str(pviews[i].get("raster_sha256"))!=rsha or str(pviews[i].get("mask_sha256"))!=msha: raise ValueError("SOURCE_OBSERVATION_VIEW_HASH_BINDING")
        with Image.open(raster) as image:
            rgba=np.asarray(image.convert("RGBA"),np.uint8)
        raw_mask=np.frombuffer(mask.read_bytes(),np.uint8)
        if rgba.shape!=(1024,1024,4) or raw_mask.size!=1024*1024 or np.any((raw_mask!=0)&(raw_mask!=1)): raise ValueError("SOURCE_OBSERVATION_VIEW_LAYOUT")
        rows.append({"view_index":i,"raster":{"path":str(raster),"sha256":rsha,"decoded_rgba_sha256":hashlib.sha256(rgba.tobytes()).hexdigest()},"mask":{"path":str(mask),"sha256":msha,"foreground_pixel_count":int(np.count_nonzero(raw_mask))},"maximum_mask_gap_px":float(pviews[i].get("maximum_mask_gap_px",float("inf"))),"border_margin_px":int(pviews[i].get("border_margin_px",-1))})
    policy=dict(proof.get("policy") or {})
    if int(policy.get("resolution",0))!=1024 or int(policy.get("view_count",0))!=8 or float(policy.get("maximum_mask_gap_px",float("inf")))!=2.0 or int(policy.get("minimum_border_margin_px",-1))!=8: raise ValueError("SOURCE_OBSERVATION_PROOF_POLICY")
    if float(proof.get("relative_vertex_error",float("inf")))>1e-5 or max(row["maximum_mask_gap_px"] for row in rows)>2.0 or min(row["border_margin_px"] for row in rows)<8: raise ValueError("SOURCE_OBSERVATION_PROOF_TOLERANCE")
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"source":{"path":str(source_path),"sha256":source_sha},"controls":{"candidate":{"path":str(paths["SOURCE_REST_RENDER_CANDIDATE.json"]),"sha256":candidate_sha},"snapshot":{"path":str(paths["SOURCE_SCENE_SNAPSHOT.json"]),"sha256":snapshot_sha},"proof":{"path":str(paths["IRIS_OBSERVATION_ADMISSION.json"]),"sha256":proof_sha},"camera":{"path":str(paths["camera_bundle.json"]),"sha256":camera_sha}},"views":rows,"counts":{"node_count":len(nodes),"bone_count":len(snapshot.get("bones") or []),"view_count":8},"family_candidate_outcomes":{"SOURCE_IMPORT":"QUALIFIED_CORE_CANDIDATE","CAMERA_OBSERVATION":"QUALIFIED_CORE_CANDIDATE"},"scope":{"appearance_sampler_qualified":False,"native_lit_material_parity_claimed":False,"full_mechanical_truth_claimed":False,"training_authorized":False},"authority":{"independent_court_required":True,"final_family_authority_minted":False,"training_authorized":False}}
    manifest["authority_identity_sha256"]=digest(manifest); output_dir.mkdir(parents=True,exist_ok=True); path=output_dir/"SOURCE_OBSERVATION_AUTHORITY_MANIFEST_V1.json"; path.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n"); return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subject-id",required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--packet-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(); r=build(a.subject_id,a.source,a.packet_root,a.output_dir); print(json.dumps({"status":r["status"],"subject_id":r["subject_id"],"source_sha256":r["source"]["sha256"]},sort_keys=True))


if __name__=="__main__": main()
