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
    provenance=stage_output_payload(
        ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.CompactedDenseFaceProvenance.v1")
    cameras=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    obs=qualified_observation_set_from_dict(stage_output_payload(
        ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1"))
    _rgba,masks=_load_source_inputs(ctx,obs)

    compact_faces=[tuple(map(str,x)) for x in provenance["compact_faces"]]
    source_ids=sorted({sid for f in compact_faces for sid in f})
    parent={sid:sid for sid in source_ids}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if ra<rb: parent[rb]=ra
        else: parent[ra]=rb
    for f in compact_faces:
        union(f[0],f[1]); union(f[1],f[2]); union(f[2],f[0])
    roots=sorted({find(x) for x in source_ids})
    raw_index={r:i for i,r in enumerate(roots)}
    raw_by_sid={sid:raw_index[find(sid)] for sid in source_ids}

    def support_surface_ids(vertex):
        return [str(sid) for sid,_ in vertex.support_binding.coefficients]

    vertices={str(v.candidate_vertex_id):v for v in candidate.vertices}
    face_raw=[]
    face_mech=[]
    for fi,face in enumerate(candidate.faces):
        raw=set()
        mech=set()
        for vid in map(str,face):
            v=vertices[vid]
            mech.add(str(v.component_id))
            for sid in support_surface_ids(v):
                if sid not in raw_by_sid:
                    raise RuntimeError(f"SOURCE_SUPPORT_OUTSIDE_COMPACT_FACE_DOMAIN:{sid}")
                raw.add(raw_by_sid[sid])
        if len(mech)!=1:
            raise RuntimeError(f"CROSS_MECH_FACE:{fi}")
        if len(raw)!=1:
            raise RuntimeError(f"CROSS_RAW_SOURCE_FACE:{fi}:{sorted(raw)}")
        face_raw.append(next(iter(raw)))
        face_mech.append(next(iter(mech)))

    stats={rid:{
        "candidate_face_count":0,
        "mechanical_components":set(),
        "visible_face_ids":set(),
        "visible_pixels_by_view":[0]*8,
        "visible_faces_by_view":[0]*8,
    } for rid in sorted(set(face_raw))}
    for fi,rid in enumerate(face_raw):
        stats[rid]["candidate_face_count"]+=1
        stats[rid]["mechanical_components"].add(face_mech[fi])

    cams={int(c.view_index):c for c in cameras.cameras}
    for view in range(8):
        mask=np.asarray(masks[view],dtype=bool)
        vis=rasterize_visible_owner(candidate,cams[view],width=mask.shape[1],height=mask.shape[0])
        owner=np.asarray(vis.owner_face_index,dtype=np.int64)
        take=mask&(owner>=0)
        if not np.any(take):continue
        uniq,counts=np.unique(owner[take],return_counts=True)
        seen={}
        for fi,count in zip(uniq.tolist(),counts.tolist()):
            rid=face_raw[int(fi)]
            stats[rid]["visible_face_ids"].add(int(fi))
            stats[rid]["visible_pixels_by_view"][view]+=int(count)
            seen[rid]=seen.get(rid,0)+1
        for rid,n in seen.items():
            stats[rid]["visible_faces_by_view"][view]=int(n)

    rows=[]
    for rid in sorted(stats):
        s=stats[rid]
        rows.append({
            "source_component_index":int(rid),
            "candidate_face_count":int(s["candidate_face_count"]),
            "mechanical_component_count":len(s["mechanical_components"]),
            "mechanical_components":sorted(s["mechanical_components"]),
            "source_visible_face_count_union":len(s["visible_face_ids"]),
            "source_visible_any_view":bool(s["visible_face_ids"]),
            "visible_pixels_by_view":s["visible_pixels_by_view"],
            "visible_faces_by_view":s["visible_faces_by_view"],
        })
    unseen=[r for r in rows if not r["source_visible_any_view"]]
    print("SOURCE_TOPOLOGY_APPEARANCE_AUDIT="+json.dumps({
        "source_component_count":len(rows),
        "source_component_without_anchor_count":len(unseen),
        "source_components_without_anchor":unseen,
        "components":rows,
    },sort_keys=True))

if __name__=="__main__":
    main()
