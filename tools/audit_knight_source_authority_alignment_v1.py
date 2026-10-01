from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

def q(v):
    if isinstance(v,(str,int,float,bool)) or v is None: return v
    if isinstance(v,list): return [q(x) for x in v[:8]]
    if isinstance(v,dict): return {k:q(v[k]) for k in list(v)[:20]}
    return str(type(v).__name__)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--motion",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    motion=json.loads(a.motion.read_text())
    srows=tuple(motion.get("source_skeleton") or ())
    tracks=tuple(motion.get("tracks") or ())
    with np.load(a.source,allow_pickle=False) as z:
        heads=np.asarray(z["bone_heads_source"],dtype=np.float64)
        restw=np.asarray(z["rest_world_source"],dtype=np.float64)
        restl=np.asarray(z["rest_local_source"],dtype=np.float64)
        skin=np.asarray(z["skin"],dtype=np.float64)
    rowpos=np.asarray([r.get("rest_position") for r in srows],dtype=np.float64) if srows else np.zeros((0,3))
    candidates={}
    if len(srows)==len(heads):
        for name,arr in {
            "bone_heads_source":heads,
            "rest_world_translation":restw[:,:3,3],
            "rest_local_translation":restl[:,:3,3],
        }.items():
            d=np.linalg.norm(rowpos-arr,axis=1)
            candidates[name]={
                "max_error_same_order":float(np.max(d)),
                "p95_error_same_order":float(np.quantile(d,.95)),
                "mean_error_same_order":float(np.mean(d)),
            }
    payload={
        "schema":"RealSaS.KnightSourceAuthorityAlignmentAudit.v1",
        "motion_top_keys":list(motion.keys()),
        "source_skeleton_count":len(srows),
        "track_count":len(tracks),
        "source_npz_bone_count":int(len(heads)),
        "source_npz_skin_shape":list(skin.shape),
        "source_skeleton_field_keys":sorted(set().union(*(set(r.keys()) for r in srows))) if srows else [],
        "track_field_keys":sorted(set().union(*(set(r.keys()) for r in tracks))) if tracks else [],
        "first_source_rows":[q(r) for r in srows[:8]],
        "first_tracks":[q(r) for r in tracks[:4]],
        "same_order_alignment_candidates":candidates,
        "row_order_joint_ids":[str(r.get("source_joint_id")) for r in srows],
        "track_joint_ids":[str(r.get("source_joint_id")) for r in tracks],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__": main()
