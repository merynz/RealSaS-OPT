from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from tools.audit_knight_repaired_quality_collapse_v1 import loadj,build_repaired_surface
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_vertex_neighbors,_incident_faces_by_vertex,
    _link_condition_holds,_collapse_local_faces,_metric,_violates,
    _local_scale,_closest_point_barycentric,_face_normal,
)
from compiler.realsas_compiler_core.hashing import content_sha256

def closest_point_triangle_distance(point, tri):
    p=np.asarray(point,float); a,b,c=[np.asarray(x,float) for x in tri]
    # reuse barycentric closest point helper for deterministic distance
    q,bary=_closest_point_barycentric(p,(a,b,c))
    return float(np.linalg.norm(p-np.asarray(q,float)))

def sample_face(face,positions):
    p=[np.asarray(positions[str(v)],float) for v in face]
    return (
      (p[0]+p[1]+p[2])/3.0,
      (p[0]+p[1])/2.0,
      (p[1]+p[2])/2.0,
      (p[2]+p[0])/2.0,
    )

def surface_distance(point,faces,positions):
    if not faces: return float("inf")
    return min(
      closest_point_triangle_distance(point,tuple(positions[str(v)] for v in face))
      for face in faces
    )

def separate_surface_deviation(old_faces,new_faces,old_positions,new_positions):
    vals=[]
    for face in old_faces:
        vals.extend(surface_distance(p,new_faces,new_positions) for p in sample_face(face,old_positions))
    for face in new_faces:
        vals.extend(surface_distance(p,old_faces,old_positions) for p in sample_face(face,new_positions))
    return float(max(vals,default=0.0))

def quality(faces,positions,policy):
    ms=[_metric(face,positions) for face in faces]
    if not ms:
        return None
    return {
      "violation_count":sum(_violates(m,policy) for m in ms),
      "min_angle_deg":min(float(m["min_angle_deg"]) for m in ms),
      "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in ms),
      "degenerate_count":sum(bool(m["degenerate"]) for m in ms),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--static-v4-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    current=canonical_mesh_candidate_from_dict(loadj(a.static_v4_dir/"final_candidate.json"))

    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_ALTERNATING_V3",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    reference=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({
        "court":"REPAIRED_STATIC_QUALITY_ALTERNATING_V3",
        "policy":policy.qualification_policy_lineage_hash,
      }),
      explicit_face_provenance=explicit,
    )

    vertices={str(v.candidate_vertex_id):v for v in current.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    ref_vertices={str(v.candidate_vertex_id):v for v in reference.vertices}
    ref_positions={vid:tuple(map(float,v.P)) for vid,v in ref_vertices.items()}
    faces=[tuple(map(str,f)) for f in current.faces]
    inc=_edge_incidence(faces)
    neighbors=_vertex_neighbors(faces)
    incident=_incident_faces_by_vertex(faces)
    ref_incident=_incident_faces_by_vertex(reference.faces)
    metrics=[_metric(f,positions) for f in faces]
    bad={i for i,m in enumerate(metrics) if _violates(m,policy)}
    boundary={v for e,rows in inc.items() if len(rows)==1 for v in e}

    candidate_edges=sorted({_edge(faces[i][j],faces[i][(j+1)%3]) for i in bad for j in range(3)})
    rejections=Counter()
    accepted=[]
    fractions=(0.25,0.5,0.75)

    for edge in candidate_edges:
        u,v=edge
        if u in boundary or v in boundary:
            rejections["BOUNDARY"]+=1;continue
        if vertices[u].component_id!=vertices[v].component_id:
            rejections["COMPONENT"]+=1;continue
        if not _link_condition_holds(faces,edge=edge,incidence=inc,neighbors=neighbors):
            rejections["LINK"]+=1;continue

        best=None
        for keep,remove in ((u,v),(v,u)):
            affected=set(incident[keep])|set(incident[remove])
            old_faces=tuple(faces[i] for i in sorted(affected))
            new_faces=_collapse_local_faces(faces,keep=keep,remove=remove,affected=affected)
            if new_faces is None:
                rejections["DUPLICATE"]+=1;continue
            if not new_faces:
                rejections["EMPTY"]+=1;continue
            oldq=quality(old_faces,positions,policy)
            if oldq is None:continue
            ref_rows=set(ref_incident.get(u,set()))|set(ref_incident.get(v,set()))
            ref_faces=[
                tuple(map(str,reference.faces[i]))
                for i in sorted(ref_rows)
                if all(str(x) in ref_positions for x in reference.faces[i])
            ]
            if not ref_faces:
                rejections["NO_REFERENCE"]+=1;continue

            pu=np.asarray(positions[u],float);pv=np.asarray(positions[v],float)
            for frac in fractions:
                target=(1.0-frac)*pu+frac*pv
                proj_best=None
                for rf in ref_faces:
                    tri=tuple(ref_positions[str(x)] for x in rf)
                    q,bary=_closest_point_barycentric(target,tri)
                    dist=float(np.linalg.norm(np.asarray(q,float)-target))
                    key=(dist,tuple(rf))
                    if proj_best is None or key<proj_best[0]:
                        proj_best=(key,rf,tuple(map(float,q)),tuple(map(float,bary)))
                if proj_best is None:continue
                _,rf,projected,bary=proj_best
                new_positions=dict(positions);new_positions[keep]=projected
                newq=quality(new_faces,new_positions,policy)
                if newq is None or newq["degenerate_count"]:
                    rejections["DEGENERATE"]+=1;continue

                monotone=(
                  newq["violation_count"]<=oldq["violation_count"]
                  and newq["min_angle_deg"]+1e-9>=oldq["min_angle_deg"]
                  and newq["max_aspect"]<=oldq["max_aspect"]+1e-9
                )
                strict=(
                  newq["violation_count"]<oldq["violation_count"]
                  or newq["min_angle_deg"]>oldq["min_angle_deg"]+1e-7
                  or newq["max_aspect"]+1e-7<oldq["max_aspect"]
                )
                if not (monotone and strict):
                    rejections["QUALITY"]+=1;continue

                # Match each surviving old face to its rewritten new face and
                # require orientation to remain in the same hemisphere.
                orient_ok=True
                for of in old_faces:
                    rew=tuple(keep if str(x)==remove else str(x) for x in of)
                    if len(set(rew))<3: continue
                    oldn=_face_normal(of,positions)
                    newn=_face_normal(rew,new_positions)
                    if oldn is None or newn is None or float(np.dot(oldn,newn))<=0.0:
                        orient_ok=False;break
                if not orient_ok:
                    rejections["ORIENTATION"]+=1;continue

                local_vertices={str(x) for f in old_faces for x in f}
                scale=_local_scale(local_vertices,positions)
                if scale<=1e-10:
                    rejections["SCALE"]+=1;continue
                allowed=float(policy.g1_max_normal_refinement_ratio)*scale
                dev=separate_surface_deviation(old_faces,new_faces,positions,new_positions)
                move=max(
                    float(np.linalg.norm(np.asarray(projected)-pu)),
                    float(np.linalg.norm(np.asarray(projected)-pv)),
                )
                dev=max(dev,move)
                if dev>allowed+1e-12:
                    rejections["G1_SHAPE"]+=1;continue

                row={
                  "edge":edge,"keep":keep,"remove":remove,"fraction":frac,
                  "reference_face":rf,"barycentric":bary,
                  "projected":projected,"oldq":oldq,"newq":newq,
                  "deviation":dev,"allowed":allowed,
                  "edge_length":float(np.linalg.norm(pu-pv)),
                }
                key=(
                  -int(oldq["violation_count"]-newq["violation_count"]),
                  -float(newq["min_angle_deg"]-oldq["min_angle_deg"]),
                  float(newq["max_aspect"]),float(dev),keep,remove,frac,
                )
                if best is None or key<best[0]:
                    best=(key,row)
        if best is not None:
            accepted.append(best[1])
        else:
            rejections["EDGE_NO_ACCEPTED_PLACEMENT"]+=1

    covered_bad=set()
    for row in accepted:
        e=set(row["edge"])
        for fi in bad:
            if e.issubset(set(faces[fi])):
                covered_bad.add(fi)

    report={
      "schema":"RealSaS.KnightProjectedEdgeContractionFeasibility.v1",
      "status":"MEASURED_AUDIT_ONLY",
      "bad_face_count":len(bad),
      "candidate_edge_count":len(candidate_edges),
      "accepted_edge_count":len(accepted),
      "bad_faces_with_accepted_own_edge":len(covered_bad),
      "rejections":dict(sorted(rejections.items())),
      "accepted":accepted,
      "best_min_angle_after":max((r["newq"]["min_angle_deg"] for r in accepted),default=None),
      "success":bool(accepted),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_PROJECTED_EDGE_CONTRACTION_FEASIBILITY="+json.dumps({
      "bad_faces":len(bad),"candidate_edges":len(candidate_edges),
      "accepted_edges":len(accepted),"covered_bad_faces":len(covered_bad),
      "rejections":report["rejections"],"success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
