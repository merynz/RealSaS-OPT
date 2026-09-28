from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    robust_zero_surface_normals_v1, mesh_connected_component_labels_v1, _adaptive_voxel_compact,
)

RUN=Path("/home/monster/realsas_authority/runs/SUBJECT2_KNIGHT_DEMO_V2_20260924")
OUT=Path("canonical/knight_invented_face_skin_causality_audit_v1")
OUT.mkdir(parents=True,exist_ok=True)

p12=RUN/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"
p15=RUN/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"
p18=RUN/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"
p32=RUN/"artifacts/32_SKIN_QUALIFIED/qualified_skin.json"
for p in (p12,p15,p18,p32):
    if not p.is_file(): raise RuntimeError(f"MISSING:{p}")

z=json.loads(p12.read_text()); surf=json.loads(p15.read_text())
mesh=json.loads(p18.read_text()); skin=json.loads(p32.read_text())
npz=Path(str(z["npz_path"]))
with np.load(npz,allow_pickle=False) as d:
    v=np.asarray(d["vertices_normalized"],np.float64)
    f=np.asarray(d["faces"],np.int64)
    h=np.asarray(d["implicit_normals"],np.float64)

meta=dict(surf.get("metadata") or {})
target=int(meta["compact_target_nodes"])
component_aware=bool(meta["component_aware_compaction"])
dense_n=robust_zero_surface_normals_v1(v,h,k=64)
labels=mesh_connected_component_labels_v1(len(v),f) if component_aware else None
_,_,_,_,inverse=_adaptive_voxel_compact(
    v,f,dense_n,target_nodes=target,
    preserve_connected_components=component_aware,
    precomputed_component_labels=labels,
)
mapped=np.asarray(inverse[f],np.int64)
dense_face_incidence={
    tuple(sorted(map(int,tri.tolist())))
    for tri in mapped if len(set(map(int,tri.tolist())))==3
}

nodes=list(surf.get("surface_nodes") or [])
sid_to_idx={}
for i,node in enumerate(nodes):
    sid=str(node["surface_id"])
    pg=str(node.get("persistence_group_id") or "")
    if pg!=f"SCENE_FIRST_SIGNED_ZERO:{i:05d}":
        raise RuntimeError(f"SURFACE_INDEX_IDENTITY_DRIFT:{i}:{pg}")
    sid_to_idx[sid]=i

rows=list(skin.get("rows") or [])
joints=sorted({str(inf["joint_id"]) for row in rows for inf in (row.get("influences") or [])})
jix={j:i for i,j in enumerate(joints)}
W={}
for row in rows:
    sid=str(row["surface_id"])
    w=np.zeros((len(joints),),np.float64)
    for inf in row.get("influences") or []:
        w[jix[str(inf["joint_id"])]]=float(inf["weight"])
    if abs(float(w.sum())-1.0)>1e-6:
        raise RuntimeError(f"SKIN_SIMPLEX_DRIFT:{sid}:{w.sum()}")
    W[sid]=w

cid_to_sid={}
for row in mesh.get("vertices") or []:
    cid=str(row["candidate_vertex_id"])
    sid=str((row.get("metadata") or {}).get("source_surface_id") or "")
    if sid not in sid_to_idx or sid not in W:
        raise RuntimeError(f"MISSING_SURFACE_OR_SKIN:{cid}:{sid}")
    cid_to_sid[cid]=sid

def max_l1(sids):
    w=[W[x] for x in sids]
    return max(float(np.abs(w[0]-w[1]).sum()),
               float(np.abs(w[1]-w[2]).sum()),
               float(np.abs(w[2]-w[0]).sum()))

records=[]
for fi,face in enumerate(mesh.get("faces") or []):
    sids=[cid_to_sid[str(cid)] for cid in face]
    tri=tuple(sorted(sid_to_idx[x] for x in sids))
    records.append((fi, tri not in dense_face_incidence, max_l1(sids), sids, tri))

invented=np.asarray([r[1] for r in records],bool)
disc=np.asarray([r[2] for r in records],np.float64)
supported=~invented

def stats(mask):
    vals=disc[mask]
    return {
        "face_count":int(mask.sum()),
        "mean_l1":float(vals.mean()),
        "p95_l1":float(np.percentile(vals,95)),
        "p99_l1":float(np.percentile(vals,99)),
        "max_l1":float(vals.max()),
    }

thresholds=(0.1,0.5,1.0,1.5,1.9)
enrichment={}
for t in thresholds:
    p_inv=float(np.mean(disc[invented]>t))
    p_sup=float(np.mean(disc[supported]>t))
    enrichment[str(t)]={
        "invented_high_count":int(np.count_nonzero(invented & (disc>t))),
        "supported_high_count":int(np.count_nonzero(supported & (disc>t))),
        "P_high_given_invented":p_inv,
        "P_high_given_supported":p_sup,
        "relative_risk":(p_inv/p_sup) if p_sup>0 else None,
        "fraction_of_all_high_faces_that_are_invented":
            float(np.count_nonzero(invented & (disc>t))/max(1,np.count_nonzero(disc>t))),
    }

top=sorted(records,key=lambda r:r[2],reverse=True)[:100]
top_summary={
    "top100_invented_count":sum(1 for r in top if r[1]),
    "top100_invented_fraction":sum(1 for r in top if r[1])/100.0,
    "rows":[
        {"face_index":r[0],"invented":r[1],"max_skin_l1":r[2],
         "surface_ids":r[3],"compact_indices":r[4]}
        for r in top[:30]
    ],
}

report={
    "schema":"RealSaS.KnightInventedFaceSkinCausalityAudit.v1",
    "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
    "artifact_sha256":{
        "stage12":hashlib.sha256(p12.read_bytes()).hexdigest(),
        "stage15":hashlib.sha256(p15.read_bytes()).hexdigest(),
        "stage18":hashlib.sha256(p18.read_bytes()).hexdigest(),
        "stage32":hashlib.sha256(p32.read_bytes()).hexdigest(),
    },
    "counts":{"total":len(records),"invented":int(invented.sum()),"supported":int(supported.sum())},
    "invented_stats":stats(invented),
    "supported_stats":stats(supported),
    "enrichment":enrichment,
    "top_skin_discontinuity_faces":top_summary,
    "finding":{
        "id":"KNIGHT_CLIQUE_INVENTION_X_SKIN_DISCONTINUITY",
        "claim_boundary":"Measures association between missing dense face incidence and Stage32 skin discontinuity on the exact Knight baseline. Association alone is not yet the final deformation-causality proof."
    }
}
(OUT/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "counts":report["counts"],
    "invented_stats":report["invented_stats"],
    "supported_stats":report["supported_stats"],
    "enrichment":report["enrichment"],
    "top100":report["top_skin_discontinuity_faces"]["top100_invented_count"]
},indent=2,sort_keys=True))
