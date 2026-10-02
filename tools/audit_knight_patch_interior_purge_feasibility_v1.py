from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import loadj
from tools.audit_knight_patch_cdt_existing_vertices_feasibility_v1 import (
    face_adjacency,expand,boundary_cycle,patch_quality,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mechanical_partition_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _metric,_violates,_local_scale,_sampled_symmetric_local_deviation,
    mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.canonical_mesh_quality_cavity_remesh_v1 import (
    _best_polygon_triangulation,_average_patch_normal,_orient_triangles,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--static-v6-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    cand=canonical_mesh_candidate_from_dict(loadj(a.static_v6_dir/"final_candidate.json"))
    partition=mechanical_partition_from_dict(loadj(a.static_v6_dir/"partition.json"))
    protected=mechanical_quality_protected_surface_ids_v1(partition)
    vertices={str(v.candidate_vertex_id):v for v in cand.vertices}
    positions={str(v.candidate_vertex_id):tuple(map(float,v.P)) for v in cand.vertices}
    faces=[tuple(map(str,f)) for f in cand.faces]
    metrics=[_metric(f,positions) for f in faces]
    bad={i for i,m in enumerate(metrics) if _violates(m,policy)}
    adj=face_adjacency(faces)
    accepted=[];reject=Counter()

    for seed in sorted(bad):
        best=None
        for hops in (1,2,3,4,5):
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
            if any(
                protected.intersection(
                    str(sid) for sid,_ in vertices[v].support_binding.coefficients
                )
                for v in interior
            ):
                reject[f"H{hops}_PROTECTED_INTERIOR"]+=1;continue

            # Avoid chords already owned by outside faces.
            boundary_edges={
                tuple(sorted((cycle[i],cycle[(i+1)%len(cycle)])))
                for i in range(len(cycle))
            }
            external_edges={}
            for i,face in enumerate(faces):
                if i in patch: continue
                a0,b0,c0=face
                for e in (
                    tuple(sorted((a0,b0))),
                    tuple(sorted((b0,c0))),
                    tuple(sorted((c0,a0))),
                ):
                    external_edges[e]=external_edges.get(e,0)+1
            forbidden=frozenset(
                e for e in external_edges
                if e not in boundary_edges and e[0] in boundary and e[1] in boundary
            )
            best_tri=_best_polygon_triangulation(
                tuple(cycle),positions,policy,forbidden
            )
            if best_tri is None:
                reject[f"H{hops}_NO_TRIANGULATION"]+=1;continue
            _,_,_,raw=best_tri
            new_faces=_orient_triangles(
                raw,positions,_average_patch_normal(old_faces,positions)
            )
            oldq=patch_quality(old_faces,positions,policy)
            newq=patch_quality(new_faces,positions,policy)
            if int(newq["degenerate"])>0:
                reject[f"H{hops}_DEGENERATE"]+=1;continue
            if int(newq["violations"])!=0:
                reject[f"H{hops}_RESIDUAL_QUALITY"]+=1;continue
            if (
                float(newq["min_angle"])+1e-9<float(oldq["min_angle"])
                or float(newq["max_aspect"])>float(oldq["max_aspect"])+1e-9
            ):
                reject[f"H{hops}_COMPANION_REGRESSION"]+=1;continue

            scale=_local_scale(patch_vertices,positions)
            allowed=float(policy.g1_max_normal_refinement_ratio)*scale
            dev=_sampled_symmetric_local_deviation(old_faces,new_faces,positions)
            if dev>allowed+1e-12:
                reject[f"H{hops}_G1"]+=1;continue

            trial=[f for i,f in enumerate(faces) if i not in patch]+list(new_faces)
            if len(set(trial))!=len(trial):
                reject[f"H{hops}_DUPLICATE"]+=1;continue
            topo=_manifold_report(trial)
            if not topo["passed"]:
                reject[f"H{hops}_TOPOLOGY"]+=1;continue
            covered=sum(fi in bad for fi in patch)
            row={
              "seed_face":seed,"hops":hops,"patch_face_count":len(patch),
              "boundary_vertex_count":len(cycle),"removed_interior_vertex_count":len(interior),
              "covered_bad_faces":covered,"old_quality":oldq,"new_quality":newq,
              "deviation":float(dev),"allowed":float(allowed),
            }
            key=(-covered,len(interior),-newq["min_angle"],newq["max_aspect"],dev,hops)
            if best is None or key<best[0]:best=(key,row)
        if best is not None:accepted.append(best[1])

    covered=set()
    for row in accepted:
        patch=expand(row["seed_face"],adj,row["hops"])
        covered.update(fi for fi in bad if fi in patch)

    report={
      "schema":"RealSaS.KnightPatchInteriorPurgeFeasibility.v1",
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
    print("KNIGHT_PATCH_INTERIOR_PURGE_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"accepted_patches":len(accepted),
      "covered_bad_faces":len(covered),"rejections":report["rejections"],
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
