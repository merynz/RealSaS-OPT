from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    robust_zero_surface_normals_v1,
    mesh_connected_component_labels_v1,
    _adaptive_voxel_compact,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _skin,
    _tracks_for_clip,
)
from tools.demo.run_knight_baseline_mechanical_causality_v1 import (
    _face_vertex_indices,
    _skin_discontinuity,
    _triangle_3d_metrics,
)

RUN_ROOT=Path("/home/monster/realsas_authority/runs/SUBJECT2_KNIGHT_DEMO_V2_20260924")
OUT=Path("canonical/knight_invented_face_smear_causality_v1")
EXPECTED={
    "candidate_lineage_hash":"c1db3416db84ca49f06dec3a22f8864fb54a7898a098f99975937e764d732909",
    "skeleton_lineage_hash":"e54548b5dbd39e399fbd901edaef57efee167b665c9628f1f38820fa62c34850",
    "skin_lineage_hash":"b30417b9375a52c646618f0fb94ea8ddd7285fea959e460522dd5f8b6b70cb2a",
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pct(x,p):
    return float(np.percentile(np.asarray(x,dtype=np.float64),p))

def describe(values, mask):
    a=np.asarray(values,dtype=np.float64)[mask]
    return {
      "count":int(len(a)),
      "mean":float(np.mean(a)) if len(a) else None,
      "p50":pct(a,50) if len(a) else None,
      "p95":pct(a,95) if len(a) else None,
      "p99":pct(a,99) if len(a) else None,
      "max":float(np.max(a)) if len(a) else None,
    }

def threshold_counts(values,mask,thresholds):
    a=np.asarray(values,dtype=np.float64)
    n=max(1,int(np.count_nonzero(mask)))
    return {
      str(t):{
        "count":int(np.count_nonzero(mask & (a>float(t)))),
        "fraction":float(np.count_nonzero(mask & (a>float(t)))/n),
      }
      for t in thresholds
    }

def relative_risk(values, invented, threshold):
    a=np.asarray(values,dtype=np.float64)
    inv_rate=np.count_nonzero(invented & (a>threshold))/max(1,np.count_nonzero(invented))
    sup=~invented
    sup_rate=np.count_nonzero(sup & (a>threshold))/max(1,np.count_nonzero(sup))
    return {
      "threshold":float(threshold),
      "invented_rate":float(inv_rate),
      "supported_rate":float(sup_rate),
      "risk_ratio":float(inv_rate/max(sup_rate,1e-12)),
    }

# Exact artifacts.
p12=RUN_ROOT/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"
p15=RUN_ROOT/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"
p18=RUN_ROOT/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"
p28=RUN_ROOT/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"
p32=RUN_ROOT/"artifacts/32_SKIN_QUALIFIED/qualified_skin.json"
p05=RUN_ROOT/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"

seal=json.loads(p12.read_text())
surface=json.loads(p15.read_text())
candidate=canonical_mesh_candidate_from_dict(json.loads(p18.read_text()))
skeleton=qualified_skeleton_from_dict(json.loads(p28.read_text()))
skin=qualified_skin_from_dict(json.loads(p32.read_text()))
cameras=qualified_camera_set_from_dict(json.loads(p05.read_text()))

if candidate.candidate_lineage_hash!=EXPECTED["candidate_lineage_hash"]:
    raise RuntimeError("CANDIDATE_LINEAGE_DRIFT")
if skeleton.skeleton_lineage_hash!=EXPECTED["skeleton_lineage_hash"]:
    raise RuntimeError("SKELETON_LINEAGE_DRIFT")
if skin.skin_lineage_hash!=EXPECTED["skin_lineage_hash"]:
    raise RuntimeError("SKIN_LINEAGE_DRIFT")

npz=Path(str(seal["npz_path"]))
with np.load(npz,allow_pickle=False) as data:
    dense_v=np.asarray(data["vertices_normalized"],np.float64)
    dense_f=np.asarray(data["faces"],np.int64)
    hints=np.asarray(data["implicit_normals"],np.float64)

meta=dict(surface.get("metadata") or {})
target=int(meta["compact_target_nodes"])
component_aware=bool(meta["component_aware_compaction"])
dense_n=robust_zero_surface_normals_v1(dense_v,hints,k=64)
labels=mesh_connected_component_labels_v1(len(dense_v),dense_f) if component_aware else None
cp,cn,edges,divisions,inverse=_adaptive_voxel_compact(
    dense_v,dense_f,dense_n,
    target_nodes=target,
    preserve_connected_components=component_aware,
    precomputed_component_labels=labels,
)

nodes=list(surface.get("surface_nodes") or [])
if len(nodes)!=len(cp): raise RuntimeError("COMPACT_COUNT_DRIFT")
sid_to_idx={}
for i,node in enumerate(nodes):
    pg=str(node.get("persistence_group_id") or "")
    if pg!=f"SCENE_FIRST_SIGNED_ZERO:{i:05d}":
        raise RuntimeError(f"SURFACE_INDEX_DRIFT:{i}:{pg}")
    sid_to_idx[str(node["surface_id"])]=i

mapped=np.asarray(inverse[dense_f],np.int64)
dense_face_incidence=set()
for tri in mapped:
    vals=tuple(sorted(map(int,tri.tolist())))
    if len(set(vals))==3:
        dense_face_incidence.add(vals)

cid_to_sid={}
for v in candidate.vertices:
    md=dict(v.metadata or {})
    sid=str(md.get("source_surface_id") or "")
    if sid not in sid_to_idx:
        raise RuntimeError(f"CANDIDATE_SURFACE_MISSING:{v.candidate_vertex_id}:{sid}")
    cid_to_sid[str(v.candidate_vertex_id)]=sid

invented=np.zeros((len(candidate.faces),),dtype=bool)
for fi,face in enumerate(candidate.faces):
    tri=tuple(sorted(sid_to_idx[cid_to_sid[str(cid)]] for cid in face))
    invented[fi]=tri not in dense_face_incidence

if int(np.count_nonzero(invented))!=515:
    raise RuntimeError(f"INVENTED_COUNT_DRIFT:{np.count_nonzero(invented)}")

joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
face_index=_face_vertex_indices(candidate)
disc=_skin_discontinuity(W,face_index)
rest=np.asarray([v.P for v in candidate.vertices],np.float64)
rest_face=rest[face_index]

# Exact baseline RUN motion.
motion_path=RUN_ROOT/"inputs/motion/quaternius_knight_v1/demo_run_v1.motion.json"
payload=json.loads(motion_path.read_text())
source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
camera_tuple=tuple(sorted(cameras.cameras,key=lambda c:int(c.view_index)))
tracks,mapping_details=_tracks_for_clip(payload,skeleton,camera_tuple,source_report)

duration=float(payload["duration_seconds"])
sample_times=np.linspace(0.0,duration,4,endpoint=not bool(payload.get("loop")))
max_edge=np.zeros((len(candidate.faces),),dtype=np.float64)
max_area=np.zeros((len(candidate.faces),),dtype=np.float64)
for t in sample_times:
    mats,_joint_pos,_frame_hash=_joint_pose_v2(
        skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=camera_tuple
    )
    posed=_skin(rest,W,joint_ids,mats)
    metric=_triangle_3d_metrics(rest_face,posed[face_index])
    max_edge=np.maximum(max_edge,np.nan_to_num(metric["max_edge_ratio"],nan=0.0,posinf=np.inf))
    max_area=np.maximum(max_area,np.nan_to_num(metric["area_ratio"],nan=0.0,posinf=np.inf))

supported=~invented
baseline_fraction=float(np.mean(invented))
rank=np.argsort(max_edge)[::-1]

def top_enrichment(k):
    ids=rank[:min(k,len(rank))]
    c=int(np.count_nonzero(invented[ids]))
    frac=c/max(1,len(ids))
    return {
      "k":int(k),
      "invented_count":c,
      "invented_fraction":float(frac),
      "baseline_invented_fraction":baseline_fraction,
      "enrichment":float(frac/max(baseline_fraction,1e-12)),
    }

report={
 "schema":"RealSaS.KnightInventedFaceSmearCausality.v1",
 "status":"AUDIT_ONLY__EXACT_BASELINE_CORRELATION",
 "authority":{
   "candidate_lineage_hash":candidate.candidate_lineage_hash,
   "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
   "skin_lineage_hash":skin.skin_lineage_hash,
   "motion_sha256":sha(motion_path),
   "stage12_npz_sha256":sha(npz),
   "stage15_sha256":sha(p15),
   "stage18_sha256":sha(p18),
   "stage32_sha256":sha(p32),
 },
 "population":{
   "face_count":len(candidate.faces),
   "invented_face_count":int(np.count_nonzero(invented)),
   "invented_fraction":baseline_fraction,
   "dense_supported_face_count":int(np.count_nonzero(supported)),
 },
 "skin_discontinuity":{
   "invented":describe(disc,invented),
   "supported":describe(disc,supported),
   "thresholds_invented":threshold_counts(disc,invented,[0.1,0.5,1.0,1.5,1.9]),
   "thresholds_supported":threshold_counts(disc,supported,[0.1,0.5,1.0,1.5,1.9]),
   "risk_ratio_gt_0_5":relative_risk(disc,invented,0.5),
   "risk_ratio_gt_1_9":relative_risk(disc,invented,1.9),
 },
 "actual_run_3d":{
   "sample_times_seconds":[float(x) for x in sample_times],
   "max_edge_ratio_invented":describe(max_edge,invented),
   "max_edge_ratio_supported":describe(max_edge,supported),
   "max_area_ratio_invented":describe(max_area,invented),
   "max_area_ratio_supported":describe(max_area,supported),
   "edge_thresholds_invented":threshold_counts(max_edge,invented,[2,4,10,20,50,100]),
   "edge_thresholds_supported":threshold_counts(max_edge,supported,[2,4,10,20,50,100]),
   "area_thresholds_invented":threshold_counts(max_area,invented,[2,4,10,20,50,100]),
   "area_thresholds_supported":threshold_counts(max_area,supported,[2,4,10,20,50,100]),
   "edge_risk_ratios":[relative_risk(max_edge,invented,t) for t in [4,10,20,50,100]],
   "top_edge_pathology_enrichment":[top_enrichment(k) for k in [50,100,250,500,1000,2000]],
 },
 "finding":{
   "id":"KNIGHT_INVENTED_FACE_SMEAR_CAUSAL_CORRELATION",
   "claim_boundary":"Invented-vs-dense-supported comparison holds mesh, skin, skeleton and exact baseline RUN motion fixed. Correlation/enrichment can identify whether incidence invention is a major owner, but does not by itself prove that preserving dense incidence is the only acceptable product discretization.",
 }
}

OUT.mkdir(parents=True,exist_ok=True)
(OUT/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
