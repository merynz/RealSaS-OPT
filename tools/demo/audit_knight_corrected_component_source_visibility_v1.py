from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from tools.demo.render_knight_motion_preview_v1 import _ctx


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()

    ctx=_ctx(a.authority_root.resolve(),a.run_id)
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    cameras=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    obs=qualified_observation_set_from_dict(stage_output_payload(
        ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1"))
    _rgba,masks=_load_source_inputs(ctx,obs)

    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    comp_by_vid={str(v.candidate_vertex_id):str(v.component_id) for v in candidate.vertices}
    face_comp=[]
    for face in candidate.faces:
        cs={comp_by_vid[str(x)] for x in face}
        if len(cs)!=1: raise RuntimeError("CROSS_COMPONENT_FACE")
        face_comp.append(next(iter(cs)))
    comp_ids=sorted(set(face_comp))
    stats={cid:{"face_count":0,"visible_face_ids":set(),"visible_pixels_by_view":[0]*8,"visible_faces_by_view":[0]*8} for cid in comp_ids}
    for cid in face_comp: stats[cid]["face_count"]+=1

    cams={int(c.view_index):c for c in cameras.cameras}
    for view in range(8):
        mask=np.asarray(masks[view],dtype=bool)
        vis=rasterize_visible_owner(candidate,cams[view],width=mask.shape[1],height=mask.shape[0])
        owner=np.asarray(vis.owner_face_index,dtype=np.int64)
        take=mask & (owner>=0)
        if not np.any(take): continue
        ids=owner[take]
        uniq,counts=np.unique(ids,return_counts=True)
        by_comp={}
        for fi,count in zip(uniq.tolist(),counts.tolist()):
            cid=face_comp[int(fi)]
            stats[cid]["visible_face_ids"].add(int(fi))
            stats[cid]["visible_pixels_by_view"][view]+=int(count)
            by_comp.setdefault(cid,0)
            by_comp[cid]+=1
        for cid,n in by_comp.items():
            stats[cid]["visible_faces_by_view"][view]=int(n)

    rows=[]
    for cid in comp_ids:
        s=stats[cid]
        rows.append({
            "component_id":cid,
            "face_count":s["face_count"],
            "source_visible_face_count_union":len(s["visible_face_ids"]),
            "source_visible_face_fraction":len(s["visible_face_ids"])/max(1,s["face_count"]),
            "visible_pixels_by_view":s["visible_pixels_by_view"],
            "visible_faces_by_view":s["visible_faces_by_view"],
            "source_visible_any_view":len(s["visible_face_ids"])>0,
        })
    unseen=[r for r in rows if not r["source_visible_any_view"]]
    print("COMPONENT_VISIBILITY_AUDIT="+json.dumps({
        "component_count":len(rows),
        "unseen_component_count":len(unseen),
        "unseen_components":unseen,
        "smallest_visible_components":sorted(
            [r for r in rows if r["source_visible_any_view"]],
            key=lambda r:(sum(r["visible_pixels_by_view"]),r["component_id"])
        )[:12],
    },sort_keys=True))

if __name__=="__main__":
    main()
