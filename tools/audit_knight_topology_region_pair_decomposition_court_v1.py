from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
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
    exact, load, faces_index, teacher_weights, stress,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def source_components(vertex_count, faces):
    parent=np.arange(vertex_count,dtype=np.int64)
    rank=np.zeros(vertex_count,dtype=np.int8)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(int(a)),find(int(b))
        if ra==rb:return
        if rank[ra]<rank[rb]:ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:rank[ra]+=1
    for a,b,c in faces.tolist():
        union(a,b);union(b,c);union(c,a)
    roots=[find(i) for i in range(vertex_count)]
    uniq={r:i for i,r in enumerate(sorted(set(roots)))}
    return np.asarray([uniq[r] for r in roots],dtype=np.int64)


def face_connected_components(face_ids,faces):
    ids=set(map(int,face_ids))
    edge_faces=defaultdict(list)
    for fi in ids:
        a,b,c=map(int,faces[fi])
        for x,y in ((a,b),(b,c),(c,a)):
            edge_faces[tuple(sorted((x,y)))].append(fi)
    adj=defaultdict(set)
    for inc in edge_faces.values():
        if len(inc)>1:
            for i in range(len(inc)):
                for j in range(i+1,len(inc)):
                    adj[inc[i]].add(inc[j]);adj[inc[j]].add(inc[i])
    seen=set();out=[]
    for s in sorted(ids):
        if s in seen:continue
        q=[s];seen.add(s);comp=[]
        while q:
            u=q.pop();comp.append(u)
            for v in adj[u]:
                if v not in seen:seen.add(v);q.append(v)
        out.append(comp)
    return sorted(out,key=len,reverse=True)


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

    src_comp_by_vertex=source_components(int(sf.max())+1,sf)
    tri_comp=np.asarray([src_comp_by_vertex[int(row[0])] for row in sf],dtype=np.int64)
    # Source faces are manifold within a component; verify all 3 vertices agree.
    if any(len({int(src_comp_by_vertex[int(v)]) for v in row})!=1 for row in sf.tolist()):
        raise RuntimeError("SOURCE_FACE_CROSSES_COMPONENT")

    groups=defaultdict(list)
    cross=0;same=0
    for fi in np.where(unsafe)[0]:
        fi=int(fi)
        labels=tuple(sorted({int(tri_comp[int(tri[int(v)])]) for v in faces[fi]}))
        if len(labels)>1:
            key="CROSS:"+",".join(map(str,labels));cross+=1
        else:
            key="SAME:"+str(labels[0]);same+=1
        groups[key].append(fi)

    rows=[]
    all_subsizes=[]
    for key,ids in groups.items():
        comps=face_connected_components(ids,faces)
        sizes=[len(x) for x in comps]
        all_subsizes.extend(sizes)
        verts=np.unique(faces[np.asarray(ids,dtype=np.int64)].reshape(-1))
        extent=P[verts].max(axis=0)-P[verts].min(axis=0)
        rows.append({
            "region_pair_key":key,
            "face_count":len(ids),
            "connected_patch_count":len(comps),
            "connected_patch_sizes":sizes[:128],
            "largest_patch_size":sizes[0] if sizes else 0,
            "vertex_count":int(len(verts)),
            "bbox_extent":extent.tolist(),
        })
    rows.sort(key=lambda r:r["face_count"],reverse=True)
    all_subsizes.sort(reverse=True)

    report={
        "schema":"RealSaS.KnightTeacherTopologyRegionPairDecompositionCourt.v1",
        "status":"ORACLE_DIAGNOSTIC_ONLY__NO_PRODUCT_REPAIR",
        "stress":stress_summary,
        "source_component_count":int(len(set(src_comp_by_vertex.tolist()))),
        "unsafe":{
            "total":int(unsafe.sum()),
            "cross_source_component":cross,
            "same_source_component":same,
            "cross_fraction":float(cross/max(1,int(unsafe.sum()))),
        },
        "region_pair_decomposition":{
            "group_count":len(rows),
            "connected_patch_count_total":int(sum(r["connected_patch_count"] for r in rows)),
            "largest_patch_sizes":all_subsizes[:128],
            "patches_le_32_faces":int(sum(s<=32 for s in all_subsizes)),
            "patches_gt_128_faces":int(sum(s>128 for s in all_subsizes)),
            "groups":rows,
        },
        "finding":{
            "unsafe_is_mostly_cross_component":bool(cross>0.8*max(1,int(unsafe.sum()))),
            "region_pair_decomposition_breaks_giant_blob":bool(all_subsizes and all_subsizes[0]<0.5*int(unsafe.sum())),
        },
        "claim_boundary":"Source components are teacher-only labels used to test whether a mechanically correct region decomposition makes the residual topology repair local. Product inference may not use these labels.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TOPOLOGY_REGION_PAIR_DECOMPOSITION_COURT_PASS",json.dumps({
        "unsafe":report["unsafe"],
        "decomposition":{k:v for k,v in report["region_pair_decomposition"].items() if k!="groups"},
        "top_groups":rows[:20],
        "finding":report["finding"],
    },sort_keys=True))

if __name__=="__main__":main()
