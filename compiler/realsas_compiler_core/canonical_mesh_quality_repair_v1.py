from __future__ import annotations

"""Deterministic fixed-vertex triangle-quality repair.

This pass changes connectivity only. It never moves, inserts, or deletes vertices.
Each accepted interior-edge flip must:
- stay inside one mechanical component,
- keep the mesh manifold at the flipped edge,
- strictly improve the local frozen G3 triangle-quality tuple,
- not regress the companion G3 metric,
- remain inside the frozen G1 local geometric-deviation budget,
- preserve every vertex support binding exactly.

The pass is intentionally conservative. Residual violations are evidence for a
later projected adaptive-remeshing phase; they are never hidden.
"""

from dataclasses import replace
import math
from collections import defaultdict

import numpy as np

from .hashing import content_sha256
from .mesh.conditioning_v1 import triangle_rest_metric
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


_EPS = 1e-10


def _edge(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _metric(face, positions):
    return triangle_rest_metric(tuple(positions[str(v)] for v in face))


def _violates(metric: dict, policy: MeshQualificationPolicyIR) -> bool:
    return (
        bool(metric["degenerate"])
        or float(metric["min_angle_deg"]) + 1e-9 < float(policy.g3_min_angle_deg)
        or float(metric["aspect_longest_over_min_altitude"]) - 1e-9
        > float(policy.g3_max_aspect_longest_over_min_altitude)
    )


def _pair_quality(faces, positions, policy):
    metrics = tuple(_metric(face, positions) for face in faces)
    return {
        "metrics": metrics,
        "violation_count": sum(_violates(m, policy) for m in metrics),
        "min_angle_deg": min(float(m["min_angle_deg"]) for m in metrics),
        "max_aspect": max(float(m["aspect_longest_over_min_altitude"]) for m in metrics),
        "degenerate_count": sum(bool(m["degenerate"]) for m in metrics),
    }


def _point_triangle_distance(point, tri) -> float:
    # Ericson-style closest-point region tests; deterministic float64.
    p=np.asarray(point,dtype=np.float64)
    a,b,c=(np.asarray(x,dtype=np.float64) for x in tri)
    ab=b-a; ac=c-a; ap=p-a
    d1=float(np.dot(ab,ap)); d2=float(np.dot(ac,ap))
    if d1<=0.0 and d2<=0.0: return float(np.linalg.norm(ap))
    bp=p-b; d3=float(np.dot(ab,bp)); d4=float(np.dot(ac,bp))
    if d3>=0.0 and d4<=d3: return float(np.linalg.norm(bp))
    vc=d1*d4-d3*d2
    if vc<=0.0 and d1>=0.0 and d3<=0.0:
        v=d1/(d1-d3); q=a+v*ab; return float(np.linalg.norm(p-q))
    cp=p-c; d5=float(np.dot(ab,cp)); d6=float(np.dot(ac,cp))
    if d6>=0.0 and d5<=d6: return float(np.linalg.norm(cp))
    vb=d5*d2-d1*d6
    if vb<=0.0 and d2>=0.0 and d6<=0.0:
        w=d2/(d2-d6); q=a+w*ac; return float(np.linalg.norm(p-q))
    va=d3*d6-d5*d4
    if va<=0.0 and (d4-d3)>=0.0 and (d5-d6)>=0.0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); q=b+w*(c-b); return float(np.linalg.norm(p-q))
    denom=1.0/(va+vb+vc)
    v=vb*denom; w=vc*denom
    q=a+ab*v+ac*w
    return float(np.linalg.norm(p-q))


def _surface_distance(point, faces, positions) -> float:
    return min(
        _point_triangle_distance(point, tuple(positions[str(v)] for v in face))
        for face in faces
    )


def _sampled_symmetric_local_deviation(old_faces, new_faces, positions) -> float:
    def samples(face):
        p=[np.asarray(positions[str(v)],dtype=np.float64) for v in face]
        return (
            (p[0]+p[1]+p[2])/3.0,
            (p[0]+p[1])/2.0,
            (p[1]+p[2])/2.0,
            (p[2]+p[0])/2.0,
        )
    values=[]
    for face in old_faces:
        values.extend(_surface_distance(p,new_faces,positions) for p in samples(face))
    for face in new_faces:
        values.extend(_surface_distance(p,old_faces,positions) for p in samples(face))
    return float(max(values,default=0.0))


def _local_scale(vertex_ids, positions) -> float:
    points=[np.asarray(positions[str(v)],dtype=np.float64) for v in sorted(set(vertex_ids))]
    lengths=[]
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            d=float(np.linalg.norm(points[i]-points[j]))
            if math.isfinite(d) and d>_EPS: lengths.append(d)
    if not lengths:
        return 0.0
    return float(np.median(np.asarray(lengths,dtype=np.float64)))


def _edge_incidence(faces):
    inc=defaultdict(list)
    for fi,face in enumerate(faces):
        a,b,c=map(str,face)
        for e in (_edge(a,b),_edge(b,c),_edge(c,a)):
            inc[e].append(fi)
    return inc


def _opposite(face, edge):
    rows=[str(v) for v in face if str(v) not in edge]
    if len(rows)!=1:
        raise QualificationError("QUALITY_FLIP_EDGE_FACE_INCIDENCE_INVALID")
    return rows[0]


def _report(faces, positions, policy):
    metrics=[_metric(face,positions) for face in faces]
    finite=[float(m["aspect_longest_over_min_altitude"]) for m in metrics if not bool(m["degenerate"])]
    return {
        "schema":"RealSaS.FixedVertexMeshQualityReport.v1",
        "face_count":len(faces),
        "degenerate_face_count":sum(bool(m["degenerate"]) for m in metrics),
        "below_min_angle_face_count":sum(
            bool(m["degenerate"]) or float(m["min_angle_deg"])+1e-9<float(policy.g3_min_angle_deg)
            for m in metrics
        ),
        "above_max_aspect_face_count":sum(
            bool(m["degenerate"]) or float(m["aspect_longest_over_min_altitude"])-1e-9>float(policy.g3_max_aspect_longest_over_min_altitude)
            for m in metrics
        ),
        "policy_violating_face_count":sum(_violates(m,policy) for m in metrics),
        "min_angle_deg":min(float(m["min_angle_deg"]) for m in metrics),
        "max_aspect_longest_over_min_altitude":max(finite,default=float("inf")),
    }


def repair_candidate_fixed_vertex_flips_v1(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    max_passes: int = 12,
) -> tuple[CanonicalMeshCandidateIR, dict]:
    if int(max_passes)<1:
        raise QualificationError("QUALITY_FLIP_MAX_PASSES_INVALID")
    vertices={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_FLIP_REQUIRES_FACES")

    before=_report(faces,positions,policy)
    accepted=[]
    rejected_shape=0
    rejected_topology=0

    for pass_index in range(int(max_passes)):
        inc=_edge_incidence(faces)
        proposals=[]
        existing_edges=set(inc)
        for edge,adj in sorted(inc.items()):
            if len(adj)!=2:
                continue
            i,j=adj
            f0=faces[i]; f1=faces[j]
            c=_opposite(f0,edge); d=_opposite(f1,edge)
            a,b=edge
            if len({a,b,c,d})!=4:
                continue
            if _edge(c,d) in existing_edges:
                rejected_topology+=1
                continue
            components={vertices[v].component_id for v in (a,b,c,d)}
            if len(components)!=1:
                continue

            old_faces=(f0,f1)
            oldq=_pair_quality(old_faces,positions,policy)
            if oldq["violation_count"]<=0:
                continue

            # Both windings have identical intrinsic quality. Use deterministic
            # candidate-id ordering; later raster/fidelity gates remain authority.
            new_faces=((c,d,a),(d,c,b))
            newq=_pair_quality(new_faces,positions,policy)
            if newq["degenerate_count"]:
                continue
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
                continue

            scale=_local_scale((a,b,c,d),positions)
            if scale<=_EPS:
                continue
            deviation=_sampled_symmetric_local_deviation(old_faces,new_faces,positions)
            allowed=float(policy.g1_max_normal_refinement_ratio)*scale
            if deviation>allowed+1e-12:
                rejected_shape+=1
                continue

            improvement=(
                oldq["violation_count"]-newq["violation_count"],
                newq["min_angle_deg"]-oldq["min_angle_deg"],
                oldq["max_aspect"]-newq["max_aspect"],
            )
            proposals.append({
                "edge":edge,"faces":(i,j),"new_faces":new_faces,
                "old_quality":oldq,"new_quality":newq,
                "deviation":deviation,"allowed":allowed,
                "improvement":improvement,
            })

        if not proposals:
            break
        proposals.sort(
            key=lambda row:(
                -int(row["improvement"][0]),
                -float(row["improvement"][1]),
                -float(row["improvement"][2]),
                row["edge"],
            )
        )
        used_faces=set()
        applied=0
        for row in proposals:
            i,j=row["faces"]
            if i in used_faces or j in used_faces:
                continue
            faces[i],faces[j]=row["new_faces"]
            used_faces.update((i,j))
            accepted.append({
                "pass_index":pass_index,
                "old_edge":row["edge"],
                "new_edge":_edge(
                    _opposite(candidate.faces[i],row["edge"]),
                    _opposite(candidate.faces[j],row["edge"]),
                ) if pass_index==0 else _edge(row["new_faces"][0][0],row["new_faces"][0][1]),
                "local_sampled_surface_deviation":row["deviation"],
                "allowed_local_deviation":row["allowed"],
                "old_violation_count":row["old_quality"]["violation_count"],
                "new_violation_count":row["new_quality"]["violation_count"],
                "old_min_angle_deg":row["old_quality"]["min_angle_deg"],
                "new_min_angle_deg":row["new_quality"]["min_angle_deg"],
                "old_max_aspect":row["old_quality"]["max_aspect"],
                "new_max_aspect":row["new_quality"]["max_aspect"],
            })
            applied+=1
        if applied==0:
            break

    faces_tuple=tuple(sorted(tuple(face) for face in faces))
    edges=tuple(sorted({
        _edge(face[i],face[j])
        for face in faces_tuple
        for i,j in ((0,1),(1,2),(2,0))
    }))
    after=_report(faces_tuple,positions,policy)
    producer_policy_hash=content_sha256({
        "schema":"RealSaS.FixedVertexDelaunayFlipRepairPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_passes":int(max_passes),
        "g1_local_deviation_ratio":float(policy.g1_max_normal_refinement_ratio),
        "selection":"MONOTONE_G3_VIOLATION_MIN_ANGLE_ASPECT_WITH_G1_LOCAL_DEVIATION",
        "vertex_motion":"FORBIDDEN",
        "vertex_insertion":"FORBIDDEN",
        "vertex_deletion":"FORBIDDEN",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=candidate.vertices,
        faces=faces_tuple,
        edges=edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.FixedVertexDelaunayFlipRepair.v1",
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "quality_repair":{
                "algorithm":"FIXED_VERTEX_MONOTONE_EDGE_FLIP_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "vertex_motion":False,
                "vertex_insertion":False,
                "vertex_deletion":False,
                "support_binding_changed":False,
                "accepted_flip_count":len(accepted),
                "rejected_shape_deviation_count":int(rejected_shape),
                "rejected_topology_count":int(rejected_topology),
                "before":before,
                "after":after,
                "accepted_flips":accepted,
            },
        },
    )
    result=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return result,dict(result.metadata["quality_repair"])
