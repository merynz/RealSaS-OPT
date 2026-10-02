from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj
from tools.audit_knight_patch_cdt_existing_vertices_feasibility_v1 import (
    face_adjacency,expand,boundary_cycle,tutte_chart,chart_injective,
    orient_boundary,patch_quality,
)
from tools.audit_knight_patch_cdt_steiner_feasibility_v1 import (
    locate_in_old_chart,nearest_known,bary_position,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mechanical_partition_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _metric,_violates,_local_scale,_sampled_symmetric_local_deviation,
    _combine_support_bindings,mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.mesh._historical_v05 import triangulate_production_cdt

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
        for hops in (1,2,3,4,5,6,7):
            patch=expand(seed,adj,hops)
            if len(patch)>256:
                reject[f"H{hops}_TOO_LARGE"]+=1;continue
            cycle=boundary_cycle(faces,patch)
            if cycle is None:
                reject[f"H{hops}_NONDISK"]+=1;continue
            old_faces=tuple(faces[i] for i in sorted(patch))
            patch_vertices=sorted({str(v) for f in old_faces for v in f})
            boundary=set(cycle)
            interior=sorted(set(patch_vertices)-boundary)
            if not interior:
                reject[f"H{hops}_NO_INTERIOR"]+=1;continue
            if any(
                protected.intersection(
                    str(sid) for sid,_ in vertex_rows[v].support_binding.coefficients
                )
                for v in interior
            ):
                reject[f"H{hops}_PROTECTED"]+=1;continue

            # Old patch chart is only source point-location authority.
            try:
                xy=tutte_chart(old_faces,cycle,positions)
            except RuntimeError:
                reject[f"H{hops}_TUTTE_FAIL"]+=1;continue
            if not chart_injective(old_faces,xy):
                reject[f"H{hops}_NONINJECTIVE"]+=1;continue
            cycle=orient_boundary(cycle,xy)
            outer=[xy[v] for v in cycle]
            scale=max(max(abs(x) for p in outer for x in p),1.0)

            # Interior vertices are deliberately NOT supplied. The CDT is free
            # to recreate only the source-bound Steiner support it needs.
            result=triangulate_production_cdt(
                outer,
                support_points=None,
                target_min_angle_deg=float(policy.g3_min_angle_deg),
                max_boundary_vertices=max(1024,len(outer)+256),
                max_support_vertices=4096,
                max_constraint_recovery_iterations=256,
                max_quality_iterations=192,
                min_feature_spacing=1e-10*scale,
            )
            if not bool(result.success) or int(result.missing_constraint_count)!=0:
                reject[f"H{hops}_CDT_FAIL"]+=1;continue
            if int(result.constraint_split_count)!=0:
                reject[f"H{hops}_BOUNDARY_SPLIT"]+=1;continue

            known_ids=list(cycle)
            tol=1e-7*scale
            local_ids=[];local_positions={};local_supports={}
            inserted=0;point_fail=False
            for pi,point in enumerate(result.vertices):
                known=nearest_known(point,known_ids,xy,tol)
                if known is not None:
                    local_ids.append(known)
                    local_positions[known]=positions[known]
                    local_supports[known]=vertex_rows[known].support_binding
                    continue
                loc=locate_in_old_chart(point,old_faces,xy)
                if loc is None:
                    point_fail=True;break
                fi,bary=loc
                source_face=old_faces[fi]
                support=_combine_support_bindings(
                    [vertex_rows[str(v)] for v in source_face],bary
                )
                new_id=f"PURGEST:{seed}:{hops}:{pi}"
                local_ids.append(new_id)
                local_positions[new_id]=bary_position(source_face,bary,positions)
                local_supports[new_id]=support
                inserted+=1
            if point_fail:
                reject[f"H{hops}_UNBOUND_STEINER"]+=1;continue

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
            pscale=_local_scale(set(patch_vertices),positions)
            allowed=float(policy.g1_max_normal_refinement_ratio)*pscale
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
              "boundary_vertex_count":len(cycle),
              "removed_interior_vertex_count":len(interior),
              "inserted_steiner_count":inserted,
              "kernel_quality_insert_count":int(result.quality_insert_count),
              "covered_bad_faces":covered,
              "old_quality":oldq,"new_quality":newq,
              "deviation":float(dev),"allowed":float(allowed),
              "backend_name":str(result.backend_name),
            }
            key=(-covered,inserted,len(interior),-newq["min_angle"],newq["max_aspect"],dev,hops)
            if best is None or key<best[0]:best=(key,row)
        if best is not None:accepted.append(best[1])

    covered=set()
    for row in accepted:
        patch=expand(row["seed_face"],adj,row["hops"])
        covered.update(fi for fi in bad if fi in patch)
    report={
      "schema":"RealSaS.KnightPatchPurgeSteinerFeasibility.v1",
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
    print("KNIGHT_PATCH_PURGE_STEINER_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"accepted_patches":len(accepted),
      "covered_bad_faces":len(covered),
      "inserted_total":sum(r["inserted_steiner_count"] for r in accepted),
      "removed_total":sum(r["removed_interior_vertex_count"] for r in accepted),
      "rejections":report["rejections"],"success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
