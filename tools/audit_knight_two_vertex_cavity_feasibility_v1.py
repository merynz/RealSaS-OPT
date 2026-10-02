from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import loadj
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_incident_faces_by_vertex,_metric,_violates,
    _local_scale,_sampled_symmetric_local_deviation,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.canonical_mesh_quality_cavity_remesh_v1 import (
    _best_polygon_triangulation,_patch_quality,_average_patch_normal,_orient_triangles,
)

def boundary_cycle_for_removed_vertices(faces,cavity_indices,removed):
    counts=Counter()
    for fi in cavity_indices:
        a,b,c=map(str,faces[fi])
        for e in (_edge(a,b),_edge(b,c),_edge(c,a)):
            if e[0] in removed or e[1] in removed:
                continue
            counts[e]+=1
    boundary=[e for e,n in counts.items() if n==1]
    if len(boundary)<3:
        return None
    g=defaultdict(set)
    for a,b in boundary:
        g[a].add(b);g[b].add(a)
    if any(len(nbs)!=2 for nbs in g.values()):
        return None
    start=min(g);cycle=[start];prev=None;cur=start
    while True:
        nbs=sorted(g[cur])
        nxt=nbs[0] if nbs[0]!=prev else nbs[1]
        if nxt==start:break
        if nxt in cycle:return None
        cycle.append(nxt);prev,cur=cur,nxt
        if len(cycle)>len(g):return None
    if len(cycle)!=len(g):return None
    return tuple(cycle)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--static-v5-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    cand=canonical_mesh_candidate_from_dict(loadj(a.static_v5_dir/"final_candidate.json"))
    vertices={str(v.candidate_vertex_id):v for v in cand.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    faces=[tuple(map(str,f)) for f in cand.faces]
    incidence=_edge_incidence(faces)
    incident=_incident_faces_by_vertex(faces)
    metrics=[_metric(f,positions) for f in faces]
    bad={i for i,m in enumerate(metrics) if _violates(m,policy)}
    boundary_vertices={v for e,rows in incidence.items() if len(rows)==1 for v in e}

    # The shortest edge of each bad face is the micro-edge witness.
    candidate_edges=set()
    for fi in bad:
        face=faces[fi]
        lengths=[]
        for i,j in ((0,1),(1,2),(2,0)):
            e=_edge(face[i],face[j])
            p=positions[e[0]];q=positions[e[1]]
            d=sum((float(p[k])-float(q[k]))**2 for k in range(3))**0.5
            lengths.append((d,e))
        candidate_edges.add(min(lengths)[1])

    accepted=[];rejected=Counter()
    for edge in sorted(candidate_edges):
        u,v=edge
        removed={u,v}
        if u in boundary_vertices or v in boundary_vertices:
            rejected["BOUNDARY"]+=1;continue
        if vertices[u].component_id!=vertices[v].component_id:
            rejected["COMPONENT"]+=1;continue
        cavity_indices=set(incident.get(u,set()))|set(incident.get(v,set()))
        old_faces=tuple(faces[i] for i in sorted(cavity_indices))
        cycle=boundary_cycle_for_removed_vertices(faces,cavity_indices,removed)
        if cycle is None:
            rejected["NONDISK_CAVITY"]+=1;continue
        oldq=_patch_quality(old_faces,positions,policy)
        if oldq is None or int(oldq["violation_count"])<=0:
            rejected["NO_LOCAL_VIOLATION"]+=1;continue

        boundary_edges={_edge(cycle[i],cycle[(i+1)%len(cycle)]) for i in range(len(cycle))}
        forbidden=set()
        for i in range(len(cycle)):
            for j in range(i+1,len(cycle)):
                e=_edge(cycle[i],cycle[j])
                if e in boundary_edges:continue
                rows=incidence.get(e,())
                if any(fi not in cavity_indices for fi in rows):
                    forbidden.add(e)

        best=_best_polygon_triangulation(cycle,positions,policy,frozenset(forbidden))
        if best is None:
            rejected["NO_TRIANGULATION"]+=1;continue
        _,_,_,raw=best
        new_faces=_orient_triangles(raw,positions,_average_patch_normal(old_faces,positions))
        newq=_patch_quality(new_faces,positions,policy)
        if newq is None or int(newq["degenerate_count"])>0:
            rejected["DEGENERATE"]+=1;continue
        monotone=(
            int(newq["violation_count"])<=int(oldq["violation_count"])
            and float(newq["min_angle_deg"])+1e-9>=float(oldq["min_angle_deg"])
            and float(newq["max_aspect"])<=float(oldq["max_aspect"])+1e-9
        )
        strict=(
            int(newq["violation_count"])<int(oldq["violation_count"])
            or float(newq["min_angle_deg"])>float(oldq["min_angle_deg"])+1e-7
            or float(newq["max_aspect"])+1e-7<float(oldq["max_aspect"])
        )
        if not (monotone and strict):
            rejected["QUALITY"]+=1;continue

        local_vertices={str(x) for face in old_faces for x in face}
        scale=_local_scale(local_vertices,positions)
        allowed=float(policy.g1_max_normal_refinement_ratio)*scale
        deviation=_sampled_symmetric_local_deviation(old_faces,new_faces,positions)
        if deviation>allowed+1e-12:
            rejected["G1_SHAPE"]+=1;continue

        trial=[face for i,face in enumerate(faces) if i not in cavity_indices]
        trial.extend(new_faces)
        if len(set(trial))!=len(trial):
            rejected["DUPLICATE_FACE"]+=1;continue
        topo=_manifold_report(trial)
        if not topo["passed"]:
            rejected["TOPOLOGY"]+=1;continue

        covered=sum(fi in bad for fi in cavity_indices)
        accepted.append({
            "removed_edge":edge,
            "removed_vertices":[u,v],
            "boundary_cycle":cycle,
            "boundary_size":len(cycle),
            "old_face_count":len(old_faces),
            "new_face_count":len(new_faces),
            "bad_faces_in_cavity":covered,
            "old_quality":oldq,
            "new_quality":newq,
            "deviation":float(deviation),
            "allowed":float(allowed),
        })

    covered_bad=set()
    for row in accepted:
        rem=set(row["removed_vertices"])
        for fi in bad:
            if rem.intersection(faces[fi]):
                covered_bad.add(fi)

    report={
      "schema":"RealSaS.KnightTwoVertexCavityFeasibility.v1",
      "status":"MEASURED_AUDIT_ONLY",
      "bad_face_count":len(bad),
      "candidate_micro_edge_count":len(candidate_edges),
      "accepted_cavity_count":len(accepted),
      "covered_bad_face_count":len(covered_bad),
      "rejections":dict(sorted(rejected.items())),
      "accepted":accepted,
      "success":bool(accepted),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TWO_VERTEX_CAVITY_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"candidate_micro_edges":len(candidate_edges),
      "accepted_cavities":len(accepted),"covered_bad_faces":len(covered_bad),
      "rejections":report["rejections"],"success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
