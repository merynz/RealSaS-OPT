from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj
from tools.audit_knight_patch_cdt_existing_vertices_feasibility_v1 import (
    face_adjacency,expand,boundary_cycle,orient_boundary,patch_quality,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mechanical_partition_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _metric,_violates,_local_scale,_sampled_symmetric_local_deviation,
    mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.canonical_cdt_adapter_v1 import _barycentric_2d
from compiler.realsas_compiler_core.mesh._historical_v05 import triangulate_production_cdt

def pca_chart(patch_vertices,positions):
    ids=sorted(patch_vertices)
    P=np.asarray([positions[v] for v in ids],dtype=np.float64)
    center=P.mean(axis=0)
    _,s,vt=np.linalg.svd(P-center[None,:],full_matrices=False)
    if len(s)<2 or float(s[1])<=1e-12:
        raise RuntimeError("PCA_CHART_DEGENERATE")
    e0=vt[0];e1=vt[1]
    return {
        v:(float(np.dot(np.asarray(positions[v])-center,e0)),
           float(np.dot(np.asarray(positions[v])-center,e1)))
        for v in ids
    }

def orient2(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

def on_segment(a,b,p,eps=1e-12):
    return (
        min(a[0],b[0])-eps<=p[0]<=max(a[0],b[0])+eps
        and min(a[1],b[1])-eps<=p[1]<=max(a[1],b[1])+eps
        and abs(orient2(a,b,p))<=eps
    )

def seg_intersects(a,b,c,d,eps=1e-12):
    o1=orient2(a,b,c);o2=orient2(a,b,d);o3=orient2(c,d,a);o4=orient2(c,d,b)
    if ((o1>eps and o2<-eps) or (o1<-eps and o2>eps)) and ((o3>eps and o4<-eps) or (o3<-eps and o4>eps)):
        return True
    return (
        on_segment(a,b,c,eps) or on_segment(a,b,d,eps)
        or on_segment(c,d,a,eps) or on_segment(c,d,b,eps)
    )

def simple_boundary(cycle,xy):
    n=len(cycle)
    for i in range(n):
        a=xy[cycle[i]];b=xy[cycle[(i+1)%n]]
        for j in range(i+1,n):
            if j==i or (j+1)%n==i or (i+1)%n==j:
                continue
            if i==0 and j==n-1:
                continue
            c=xy[cycle[j]];d=xy[cycle[(j+1)%n]]
            if seg_intersects(a,b,c,d):
                return False
    return True

def locate_metric_point(point,old_faces,xy,positions,tol=1e-8):
    candidates=[]
    for fi,face in enumerate(old_faces):
        tri=(xy[str(face[0])],xy[str(face[1])],xy[str(face[2])])
        try:
            bary=_barycentric_2d(point,tri)
        except Exception:
            continue
        b=np.asarray(bary,dtype=np.float64)
        if float(b.min())<-tol or float(b.max())>1.0+tol:
            continue
        p=sum(float(w)*np.asarray(positions[str(v)],dtype=np.float64) for v,w in zip(face,bary))
        candidates.append((fi,tuple(map(float,bary)),p))
    if not candidates:
        return None
    base=candidates[0][2]
    if any(float(np.linalg.norm(row[2]-base))>1e-6 for row in candidates[1:]):
        return "AMBIGUOUS"
    candidates.sort(key=lambda row:(-min(row[1]),row[0]))
    fi,bary,p=candidates[0]
    return fi,bary,tuple(map(float,p))

def nearest_boundary(point,cycle,xy,tol):
    p=np.asarray(point,dtype=np.float64)
    rows=[(float(np.linalg.norm(p-np.asarray(xy[v],dtype=np.float64))),v) for v in cycle]
    d,v=min(rows)
    return v if d<=tol else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--static-v7-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    cand=canonical_mesh_candidate_from_dict(loadj(a.static_v7_dir/"final_candidate.json"))
    partition=mechanical_partition_from_dict(loadj(a.static_v7_dir/"partition.json"))
    protected=mechanical_quality_protected_surface_ids_v1(partition)
    vertex_rows={str(v.candidate_vertex_id):v for v in cand.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertex_rows.items()}
    faces=[tuple(map(str,f)) for f in cand.faces]
    metrics=[_metric(f,positions) for f in faces]
    bad={i for i,m in enumerate(metrics) if _violates(m,policy)}
    adj=face_adjacency(faces)
    accepted=[];reject=Counter()

    for seed in sorted(bad):
        best=None
        for hops in (1,2,3,4,5,6,7,8):
            patch=expand(seed,adj,hops)
            if len(patch)>320:
                reject[f"H{hops}_TOO_LARGE"]+=1;continue
            cycle=boundary_cycle(faces,patch)
            if cycle is None:
                reject[f"H{hops}_NONDISK"]+=1;continue
            old_faces=tuple(faces[i] for i in sorted(patch))
            patch_vertices={str(v) for f in old_faces for v in f}
            boundary=set(cycle)
            interior=sorted(patch_vertices-boundary)
            if not interior:
                reject[f"H{hops}_NO_INTERIOR"]+=1;continue
            if any(
                protected.intersection(
                    str(sid) for sid,_ in vertex_rows[v].support_binding.coefficients
                )
                for v in interior
            ):
                reject[f"H{hops}_PROTECTED"]+=1;continue
            try:
                xy=pca_chart(patch_vertices,positions)
            except RuntimeError:
                reject[f"H{hops}_PCA_FAIL"]+=1;continue
            if any(abs(orient2(xy[f[0]],xy[f[1]],xy[f[2]]))<=1e-12 for f in old_faces):
                reject[f"H{hops}_PCA_DEGENERATE_FACE"]+=1;continue
            if not simple_boundary(cycle,xy):
                reject[f"H{hops}_PCA_BOUNDARY_SELF_INTERSECT"]+=1;continue
            cycle=orient_boundary(cycle,xy)
            outer=[xy[v] for v in cycle]
            cscale=max(max(abs(x) for p in outer for x in p),1.0)
            result=triangulate_production_cdt(
                outer,
                support_points=None,
                target_min_angle_deg=float(policy.g3_min_angle_deg),
                max_boundary_vertices=max(1024,len(outer)+256),
                max_support_vertices=4096,
                max_constraint_recovery_iterations=256,
                max_quality_iterations=192,
                min_feature_spacing=1e-10*cscale,
            )
            if not bool(result.success) or int(result.missing_constraint_count)!=0:
                reject[f"H{hops}_CDT_FAIL"]+=1;continue
            if int(result.constraint_split_count)!=0:
                reject[f"H{hops}_BOUNDARY_SPLIT"]+=1;continue

            local_ids=[];local_positions={};inserted=0;point_fail=False
            tol=1e-7*cscale
            for pi,point in enumerate(result.vertices):
                known=nearest_boundary(point,cycle,xy,tol)
                if known is not None:
                    local_ids.append(known);local_positions[known]=positions[known];continue
                loc=locate_metric_point(point,old_faces,xy,positions)
                if loc is None:
                    point_fail=True;reject[f"H{hops}_UNBOUND_POINT"]+=1;break
                if loc=="AMBIGUOUS":
                    point_fail=True;reject[f"H{hops}_AMBIGUOUS_LIFT"]+=1;break
                fi,bary,p3=loc
                nid=f"PCAM:{seed}:{hops}:{pi}"
                local_ids.append(nid);local_positions[nid]=p3;inserted+=1
            if point_fail:continue

            new_faces=[]
            for tri in result.triangles:
                ids=tuple(local_ids[int(i)] for i in tri)
                if len(set(ids))==3:new_faces.append(ids)
            if not new_faces:
                reject[f"H{hops}_NO_FACE"]+=1;continue
            oldq=patch_quality(old_faces,positions,policy)
            newq=patch_quality(new_faces,local_positions,policy)
            if int(newq["degenerate"])>0:
                reject[f"H{hops}_DEGENERATE"]+=1;continue
            if int(newq["violations"])!=0:
                reject[f"H{hops}_RESIDUAL_QUALITY"]+=1;continue

            merged=dict(positions);merged.update(local_positions)
            scale=_local_scale(patch_vertices,positions)
            allowed=float(policy.g1_max_normal_refinement_ratio)*scale
            dev=_sampled_symmetric_local_deviation(old_faces,new_faces,merged)
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
              "boundary_vertex_count":len(cycle),"removed_interior_vertex_count":len(interior),
              "inserted_vertex_count":inserted,"covered_bad_faces":covered,
              "old_quality":oldq,"new_quality":newq,
              "deviation":float(dev),"allowed":float(allowed),
              "kernel_quality_insert_count":int(result.quality_insert_count),
            }
            key=(-covered,inserted,len(interior),-newq["min_angle"],newq["max_aspect"],dev,hops)
            if best is None or key<best[0]:best=(key,row)
        if best is not None:accepted.append(best[1])

    covered=set()
    for row in accepted:
        patch=expand(row["seed_face"],adj,row["hops"])
        covered.update(fi for fi in bad if fi in patch)
    report={
      "schema":"RealSaS.KnightPatchPCAMetricRemeshFeasibility.v1",
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
    print("KNIGHT_PATCH_PCA_METRIC_REMESH_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"accepted_patches":len(accepted),
      "covered_bad_faces":len(covered),
      "inserted_total":sum(r["inserted_vertex_count"] for r in accepted),
      "removed_total":sum(r["removed_interior_vertex_count"] for r in accepted),
      "rejections":report["rejections"],"success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
