from __future__ import annotations
import argparse,json,functools,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_incident_faces_by_vertex,_metric,_violates,
    _local_scale,_sampled_symmetric_local_deviation,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report

def qtuple(faces,positions,policy):
    ms=[_metric(f,positions) for f in faces]
    if not ms:
        return None
    return {
      "violations":sum(_violates(m,policy) for m in ms),
      "min_angle":min(float(m["min_angle_deg"]) for m in ms),
      "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in ms),
      "degenerate":sum(bool(m["degenerate"]) for m in ms),
    }

def vertex_link_cycle(v,incident_faces):
    g=defaultdict(set)
    for f in incident_faces:
        row=[str(x) for x in f]
        if v not in row: continue
        others=[x for x in row if x!=v]
        if len(others)!=2: return None
        a,b=others;g[a].add(b);g[b].add(a)
    if len(g)<3 or any(len(nbs)!=2 for nbs in g.values()):
        return None
    start=min(g)
    cycle=[start];prev=None;cur=start
    while True:
        nbs=sorted(g[cur])
        nxt=nbs[0] if nbs[0]!=prev else nbs[1]
        if nxt==start:
            break
        if nxt in cycle:
            return None
        cycle.append(nxt);prev,cur=cur,nxt
        if len(cycle)>len(g):
            return None
    return cycle if len(cycle)==len(g) else None

def avg_old_normal(faces,positions):
    n=np.zeros(3,float)
    for f in faces:
        a,b,c=[np.asarray(positions[str(x)],float) for x in f]
        n+=np.cross(b-a,c-a)
    norm=float(np.linalg.norm(n))
    return None if norm<=1e-12 else n/norm

def orient_triangles(tris,positions,target_normal):
    if target_normal is None:return tuple(tuple(t) for t in tris)
    out=[]
    for tri in tris:
        a,b,c=[np.asarray(positions[str(x)],float) for x in tri]
        n=np.cross(b-a,c-a)
        if float(np.dot(n,target_normal))<0:
            out.append((tri[0],tri[2],tri[1]))
        else: out.append(tuple(tri))
    return tuple(out)

def best_polygon_triangulation(cycle,positions,policy,forbidden_chords):
    n=len(cycle)
    if n<3:return None
    @functools.lru_cache(None)
    def solve(i,j):
        if j-i<2:
            return (0,float("inf"),0.0,())
        best=None
        for k in range(i+1,j):
            # Internal chords required by this split. Polygon boundary edges are free.
            chords=[]
            if k>i+1: chords.append(_edge(cycle[i],cycle[k]))
            if j>k+1: chords.append(_edge(cycle[k],cycle[j]))
            if any(ch in forbidden_chords for ch in chords):
                continue
            left=solve(i,k);right=solve(k,j)
            if left is None or right is None:continue
            tri=(cycle[i],cycle[k],cycle[j])
            m=_metric(tri,positions)
            if bool(m["degenerate"]):continue
            viol=int(_violates(m,policy))
            total_viol=int(left[0]+right[0]+viol)
            minang=min(float(left[1]),float(right[1]),float(m["min_angle_deg"]))
            maxasp=max(float(left[2]),float(right[2]),float(m["aspect_longest_over_min_altitude"]))
            tris=left[3]+right[3]+(tri,)
            key=(total_viol,-minang,maxasp,tuple(sorted(tris)))
            if best is None or key<best[0]:
                best=(key,(total_viol,minang,maxasp,tris))
        return None if best is None else best[1]
    return solve(0,n-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--static-v4-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    cand=canonical_mesh_candidate_from_dict(loadj(a.static_v4_dir/"final_candidate.json"))
    vertices={str(v.candidate_vertex_id):v for v in cand.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    faces=[tuple(map(str,f)) for f in cand.faces]
    inc=_edge_incidence(faces);incident=_incident_faces_by_vertex(faces)
    metrics=[_metric(f,positions) for f in faces]
    bad={i for i,m in enumerate(metrics) if _violates(m,policy)}
    bad_vertices=sorted({str(v) for i in bad for v in faces[i]})
    boundary={v for e,rows in inc.items() if len(rows)==1 for v in e}
    external_edge_inc={e:set(rows) for e,rows in inc.items()}
    accepted=[];reject=Counter()

    for v in bad_vertices:
        if v in boundary:
            reject["BOUNDARY_VERTEX"]+=1;continue
        idx=set(incident.get(v,set()))
        old_faces=tuple(faces[i] for i in sorted(idx))
        cycle=vertex_link_cycle(v,old_faces)
        if cycle is None:
            reject["NONDISK_LINK"]+=1;continue
        oldq=qtuple(old_faces,positions,policy)
        if oldq is None or oldq["violations"]<=0:
            reject["NO_LOCAL_VIOLATION"]+=1;continue

        # Any chord already used by a face outside this cavity would create
        # non-manifold incidence if re-used as an internal retriangulation edge.
        boundary_edges={_edge(cycle[i],cycle[(i+1)%len(cycle)]) for i in range(len(cycle))}
        forbidden=set()
        for i in range(len(cycle)):
            for j in range(i+1,len(cycle)):
                e=_edge(cycle[i],cycle[j])
                if e in boundary_edges:continue
                rows=external_edge_inc.get(e,set())
                if any(fi not in idx for fi in rows):
                    forbidden.add(e)

        best=best_polygon_triangulation(cycle,positions,policy,frozenset(forbidden))
        if best is None:
            reject["NO_TRIANGULATION"]+=1;continue
        _,_,_,raw_new=best
        new_faces=orient_triangles(raw_new,positions,avg_old_normal(old_faces,positions))
        newq=qtuple(new_faces,positions,policy)
        if newq is None or newq["degenerate"]:
            reject["DEGENERATE"]+=1;continue
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
            reject["QUALITY"]+=1;continue

        local_vertices={x for f in old_faces for x in f}
        scale=_local_scale(local_vertices,positions)
        allowed=float(policy.g1_max_normal_refinement_ratio)*scale
        dev=_sampled_symmetric_local_deviation(old_faces,new_faces,positions)
        if dev>allowed+1e-12:
            reject["G1_SHAPE"]+=1;continue

        remaining=[f for i,f in enumerate(faces) if i not in idx]
        trial=remaining+list(new_faces)
        if len(set(trial))!=len(trial):
            reject["DUPLICATE_FACE"]+=1;continue
        topo=_manifold_report(trial)
        if not topo["passed"]:
            reject["TOPOLOGY"]+=1;continue

        fixed_bad=sum(1 for fi in bad if fi in idx)
        accepted.append({
          "removed_vertex":v,
          "valence":len(cycle),
          "old_incident_face_count":len(old_faces),
          "new_face_count":len(new_faces),
          "bad_faces_in_cavity":fixed_bad,
          "old_quality":oldq,"new_quality":newq,
          "deviation":dev,"allowed":allowed,
          "cycle":cycle,"new_faces":new_faces,
        })

    covered=set()
    for row in accepted:
        v=row["removed_vertex"]
        covered.update(fi for fi in bad if v in faces[fi])

    report={
      "schema":"RealSaS.KnightCavityRetriangulationFeasibility.v1",
      "status":"MEASURED_AUDIT_ONLY",
      "bad_face_count":len(bad),
      "bad_vertex_count":len(bad_vertices),
      "accepted_cavity_count":len(accepted),
      "covered_bad_face_count":len(covered),
      "rejections":dict(sorted(reject.items())),
      "accepted":accepted,
      "success":bool(accepted),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_CAVITY_RETRIANGULATION_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"bad_vertices":len(bad_vertices),
      "accepted_cavities":len(accepted),"covered_bad_faces":len(covered),
      "rejections":report["rejections"],"success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
