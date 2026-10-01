from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import (
    exact, load, faces_index, teacher_weights, stress, source_cat,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def edge_face_map(faces):
    out=defaultdict(list)
    for fi,(a,b,c) in enumerate(faces.tolist()):
        for x,y in ((a,b),(b,c),(c,a)):
            out[tuple(sorted((int(x),int(y))))].append(int(fi))
    return out


def unsafe_face_components(faces,unsafe,efm):
    adj=[set() for _ in range(len(faces))]
    for edge,inc in efm.items():
        u=[fi for fi in inc if unsafe[fi]]
        if len(u)>1:
            for i in range(len(u)):
                for j in range(i+1,len(u)):
                    adj[u[i]].add(u[j]);adj[u[j]].add(u[i])
    seen=set(); comps=[]
    for s in np.where(unsafe)[0]:
        s=int(s)
        if s in seen:continue
        q=[s];seen.add(s);comp=[]
        while q:
            u=q.pop();comp.append(u)
            for v in adj[u]:
                if v not in seen:seen.add(v);q.append(v)
        comps.append(sorted(comp))
    return sorted(comps,key=len,reverse=True)


def graph_components(edges):
    nbr=defaultdict(set)
    for a,b in edges:
        nbr[a].add(b);nbr[b].add(a)
    seen=set();rows=[]
    for s in sorted(nbr):
        if s in seen:continue
        q=[s];seen.add(s);verts=[];subedges=set()
        while q:
            u=q.pop();verts.append(u)
            for v in nbr[u]:
                subedges.add(tuple(sorted((u,v))))
                if v not in seen:seen.add(v);q.append(v)
        deg=[len(nbr[v]) for v in verts]
        rows.append({
            "vertex_count":len(verts),
            "edge_count":len(subedges),
            "degree_histogram":dict(Counter(map(str,deg))),
            "is_simple_cycle":bool(len(verts)>=3 and len(subedges)==len(verts) and all(d==2 for d in deg)),
            "is_simple_open_chain":bool(len(verts)>=2 and deg.count(1)==2 and all(d in (1,2) for d in deg)),
            "branch_vertex_count":sum(d>2 for d in deg),
            "endpoint_count":sum(d==1 for d in deg),
            "vertices":verts[:256],
        })
    return sorted(rows,key=lambda x:x["edge_count"],reverse=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

    faces=faces_index(cand)
    P=np.asarray([v.P for v in cand.vertices],dtype=np.float64)
    W,jids,tri,sf=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    unsafe,stress_summary=stress(P,W,faces,sk,cams,env,policy)
    efm=edge_face_map(faces)
    comps=unsafe_face_components(faces,unsafe,efm)
    source_sets=[set(map(int,x)) for x in sf.tolist()]
    dom=np.argmax(W,axis=1)

    rows=[]
    total_frontier=set()
    for ci,comp in enumerate(comps):
        cset=set(comp)
        frontier=set()
        internal=set()
        original_boundary=set()
        verts=set()
        noninc=0
        for fi in comp:
            face=faces[fi]
            verts.update(map(int,face))
            ts=[int(tri[int(v)]) for v in face]
            cats=(source_cat(ts[0],ts[1],source_sets),source_cat(ts[1],ts[2],source_sets),source_cat(ts[2],ts[0],source_sets))
            noninc+=int("NONINCIDENT" in cats)
            for x,y in ((int(face[0]),int(face[1])),(int(face[1]),int(face[2])),(int(face[2]),int(face[0]))):
                e=tuple(sorted((x,y)))
                inc=efm[e]
                unsafe_inc=sum(bool(unsafe[j]) for j in inc)
                safe_inc=sum(not bool(unsafe[j]) for j in inc)
                if unsafe_inc>0 and safe_inc>0:
                    frontier.add(e)
                elif len(inc)==1:
                    original_boundary.add(e)
                elif unsafe_inc>=2:
                    internal.add(e)
        total_frontier.update(frontier)
        fg=graph_components(sorted(frontier))
        bverts=sorted({v for e in frontier for v in e})
        joint_counts=Counter(jids[int(dom[v])] for v in bverts)
        extent=(P[np.asarray(sorted(verts),dtype=np.int64)].max(axis=0)-P[np.asarray(sorted(verts),dtype=np.int64)].min(axis=0)) if verts else np.zeros(3)
        rows.append({
            "component_index":ci,
            "unsafe_face_count":len(comp),
            "vertex_count":len(verts),
            "source_nonincident_face_count":noninc,
            "source_nonincident_fraction":float(noninc/max(1,len(comp))),
            "frontier_edge_count":len(frontier),
            "frontier_vertex_count":len(bverts),
            "frontier_graph_component_count":len(fg),
            "frontier_simple_cycle_count":sum(bool(x["is_simple_cycle"]) for x in fg),
            "frontier_open_chain_count":sum(bool(x["is_simple_open_chain"]) for x in fg),
            "frontier_branch_component_count":sum(int(x["branch_vertex_count"]>0) for x in fg),
            "frontier_graph_components":fg[:32],
            "original_mesh_boundary_edge_count":len(original_boundary),
            "bbox_extent":extent.tolist(),
            "boundary_dominant_joint_counts":dict(joint_counts.most_common()),
            "top_boundary_joint_fraction":float(joint_counts.most_common(1)[0][1]/max(1,len(bverts))) if bverts else None,
            "face_indices":comp[:512],
        })

    agg={
        "unsafe_face_count":int(unsafe.sum()),
        "unsafe_component_count":len(comps),
        "component_sizes":[len(x) for x in comps[:128]],
        "frontier_edge_count_unique":len(total_frontier),
        "components_with_only_closed_cycle_frontiers":sum(
            bool(r["frontier_graph_component_count"]>0 and
                 r["frontier_simple_cycle_count"]==r["frontier_graph_component_count"])
            for r in rows
        ),
        "components_with_any_open_chain":sum(r["frontier_open_chain_count"]>0 for r in rows),
        "components_with_branching_frontier":sum(r["frontier_branch_component_count"]>0 for r in rows),
        "largest_component_faces":len(comps[0]) if comps else 0,
        "largest_component_fraction":float(len(comps[0])/max(1,int(unsafe.sum()))) if comps else 0.0,
    }
    report={
        "schema":"RealSaS.KnightTeacherWeightSeamMorphologyCourt.v1",
        "status":"ORACLE_DIAGNOSTIC_ONLY__NO_REPAIR",
        "stress":stress_summary,
        "aggregate":agg,
        "components":rows,
        "claim_boundary":"Teacher weights isolate topology for morphology measurement only. No teacher topology is used to build frontier components.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TEACHER_WEIGHT_SEAM_MORPHOLOGY_COURT_PASS",json.dumps({
        "stress":stress_summary,"aggregate":agg,
        "top_components":[{k:v for k,v in r.items() if k not in ("face_indices","frontier_graph_components","boundary_dominant_joint_counts")} for r in rows[:12]],
    },sort_keys=True))

if __name__=="__main__":main()
