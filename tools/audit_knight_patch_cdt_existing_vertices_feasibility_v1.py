from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_metric,_violates,_local_scale,
    _sampled_symmetric_local_deviation,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.mesh._historical_v05 import triangulate_production_cdt

def face_adjacency(faces):
    edge_faces=defaultdict(list)
    for fi,face in enumerate(faces):
        a,b,c=map(str,face)
        for e in (_edge(a,b),_edge(b,c),_edge(c,a)):
            edge_faces[e].append(fi)
    adj={i:set() for i in range(len(faces))}
    for rows in edge_faces.values():
        for a in rows:
            for b in rows:
                if a!=b:adj[a].add(b)
    return adj

def expand(seed,adj,hops):
    seen={int(seed)};front={int(seed)}
    for _ in range(int(hops)):
        nxt=set()
        for x in front:nxt.update(adj.get(x,()))
        nxt-=seen;seen|=nxt;front=nxt
    return seen

def boundary_cycle(faces,patch):
    counts=defaultdict(int)
    for fi in patch:
        a,b,c=map(str,faces[fi])
        for e in (_edge(a,b),_edge(b,c),_edge(c,a)):
            counts[e]+=1
    edges=[e for e,n in counts.items() if n==1]
    graph=defaultdict(set)
    for a,b in edges:graph[a].add(b);graph[b].add(a)
    if len(graph)<3 or any(len(nbs)!=2 for nbs in graph.values()):return None
    start=min(graph);cycle=[start];prev=None;cur=start
    while True:
        nbs=sorted(graph[cur]);nxt=nbs[0] if nbs[0]!=prev else nbs[1]
        if nxt==start:break
        if nxt in cycle:return None
        cycle.append(nxt);prev,cur=cur,nxt
        if len(cycle)>len(graph):return None
    return tuple(cycle) if len(cycle)==len(graph) else None

def pca_chart(vertex_ids,positions):
    ids=tuple(sorted(set(map(str,vertex_ids))))
    P=np.asarray([positions[v] for v in ids],dtype=np.float64)
    center=np.mean(P,axis=0)
    U,S,Vt=np.linalg.svd(P-center,full_matrices=False)
    ex=np.asarray(Vt[0],dtype=np.float64);ey=np.asarray(Vt[1],dtype=np.float64)
    normal=np.cross(ex,ey)
    normal/=max(float(np.linalg.norm(normal)),1e-18)
    xy={vid:(float(np.dot(positions[vid]-center,ex)),float(np.dot(positions[vid]-center,ey))) for vid in ids}
    return center,ex,ey,normal,xy

def signed_area2(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

def orient_boundary(cycle,xy):
    area=0.0
    for i,v in enumerate(cycle):
        a=xy[v];b=xy[cycle[(i+1)%len(cycle)]]
        area+=a[0]*b[1]-b[0]*a[1]
    return tuple(cycle) if area>=0 else (cycle[0],)+tuple(reversed(cycle[1:]))

def chart_injective(old_faces,xy):
    signs=[]
    for face in old_faces:
        a,b,c=[xy[str(v)] for v in face]
        s=float(signed_area2(a,b,c))
        if abs(s)<=1e-12:return False
        signs.append(1 if s>0 else -1)
    return len(set(signs))==1

def patch_quality(faces,positions,policy):
    ms=[_metric(f,positions) for f in faces]
    return {
      "violations":sum(_violates(m,policy) for m in ms),
      "min_angle":min(float(m["min_angle_deg"]) for m in ms),
      "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in ms),
      "degenerate":sum(bool(m["degenerate"]) for m in ms),
    }

def match_points(result_vertices,known_xy,eps):
    out=[]
    used=set()
    for point in result_vertices:
        p=np.asarray(point,dtype=np.float64)
        rows=[(float(np.linalg.norm(p-np.asarray(xy,dtype=np.float64))),i) for i,xy in enumerate(known_xy)]
        dist,idx=min(rows)
        if dist>eps:return None
        out.append(idx);used.add(idx)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--static-v6-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    cand=canonical_mesh_candidate_from_dict(loadj(a.static_v6_dir/"final_candidate.json"))
    positions={str(v.candidate_vertex_id):np.asarray(v.P,dtype=np.float64) for v in cand.vertices}
    faces=[tuple(map(str,f)) for f in cand.faces]
    metrics=[_metric(f,positions) for f in faces]
    bad={i for i,m in enumerate(metrics) if _violates(m,policy)}
    adj=face_adjacency(faces)
    accepted=[];reject=Counter()

    for seed in sorted(bad):
        best=None
        for hops in (1,2,3):
            patch=expand(seed,adj,hops)
            cycle=boundary_cycle(faces,patch)
            if cycle is None:
                reject[f"H{hops}_NONDISK"]+=1;continue
            old_faces=tuple(faces[i] for i in sorted(patch))
            patch_vertices={str(v) for f in old_faces for v in f}
            boundary=set(cycle)
            interior=sorted(patch_vertices-boundary)
            if not interior:
                reject[f"H{hops}_NO_INTERIOR"]+=1;continue
            _center,_ex,_ey,_normal,xy=pca_chart(patch_vertices,positions)
            if not chart_injective(old_faces,xy):
                reject[f"H{hops}_NONINJECTIVE_CHART"]+=1;continue
            cycle=orient_boundary(cycle,xy)
            outer=[xy[v] for v in cycle]
            support=[xy[v] for v in interior]
            scale=max(
                max(abs(x) for p in outer+support for x in p),
                1.0,
            )
            result=triangulate_production_cdt(
                outer,
                support_points=support,
                target_min_angle_deg=0.0,
                max_boundary_vertices=max(512,len(outer)+64),
                max_support_vertices=max(512,len(support)+64),
                max_constraint_recovery_iterations=128,
                max_quality_iterations=0,
                min_feature_spacing=1e-10*scale,
            )
            if not bool(result.success) or int(result.missing_constraint_count)!=0:
                reject[f"H{hops}_CDT_FAIL"]+=1;continue
            if int(result.constraint_split_count)!=0 or int(result.quality_insert_count)!=0:
                reject[f"H{hops}_NEW_POINT"]+=1;continue
            known_ids=list(cycle)+interior
            known_xy=[xy[v] for v in known_ids]
            mapped=match_points(result.vertices,known_xy,eps=1e-7*scale)
            if mapped is None:
                reject[f"H{hops}_UNBOUND_POINT"]+=1;continue
            new_faces=[]
            for tri in result.triangles:
                ids=tuple(known_ids[mapped[int(i)]] for i in tri)
                if len(set(ids))==3:new_faces.append(ids)
            if not new_faces:
                reject[f"H{hops}_NO_FACE"]+=1;continue

            oldq=patch_quality(old_faces,positions,policy)
            newq=patch_quality(new_faces,positions,policy)
            if newq["degenerate"]:
                reject[f"H{hops}_DEGENERATE"]+=1;continue
            monotone=(
              newq["violations"]<=oldq["violations"]
              and newq["min_angle"]+1e-9>=oldq["min_angle"]
              and newq["max_aspect"]<=oldq["max_aspect"]+1e-9
            )
            strict=(
              newq["violations"]<oldq["violations"]
              or newq["min_angle"]>oldq["min_angle"]+1e-7
              or newq["max_aspect"]+1e-7<oldq["max_aspect"]
            )
            if not (monotone and strict):
                reject[f"H{hops}_QUALITY"]+=1;continue

            local_scale=_local_scale(patch_vertices,positions)
            allowed=float(policy.g1_max_normal_refinement_ratio)*local_scale
            dev=_sampled_symmetric_local_deviation(old_faces,new_faces,positions)
            if dev>allowed+1e-12:
                reject[f"H{hops}_G1"]+=1;continue

            trial=[f for i,f in enumerate(faces) if i not in patch]+new_faces
            if len(set(trial))!=len(trial):
                reject[f"H{hops}_DUPLICATE"]+=1;continue
            topo=_manifold_report(trial)
            if not topo["passed"]:
                reject[f"H{hops}_TOPOLOGY"]+=1;continue
            covered=sum(fi in bad for fi in patch)
            row={
              "seed_face":seed,"hops":hops,"patch_face_count":len(patch),
              "boundary_vertex_count":len(cycle),"interior_vertex_count":len(interior),
              "covered_bad_faces":covered,
              "old_quality":oldq,"new_quality":newq,
              "deviation":float(dev),"allowed":float(allowed),
              "backend_name":str(result.backend_name),
            }
            key=(-covered,newq["violations"],-newq["min_angle"],newq["max_aspect"],dev,hops)
            if best is None or key<best[0]:best=(key,row)
        if best is not None:accepted.append(best[1])

    covered=set()
    for row in accepted:
        patch=expand(row["seed_face"],adj,row["hops"])
        covered.update(fi for fi in bad if fi in patch)
    report={
      "schema":"RealSaS.KnightPatchCDTExistingVerticesFeasibility.v1",
      "status":"MEASURED_AUDIT_ONLY",
      "bad_face_count":len(bad),
      "accepted_patch_count":len(accepted),
      "covered_bad_face_count":len(covered),
      "rejections":dict(sorted(reject.items())),
      "accepted":accepted,
      "success":bool(accepted),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_PATCH_CDT_EXISTING_VERTICES_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"accepted_patches":len(accepted),
      "covered_bad_faces":len(covered),"rejections":report["rejections"],
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
