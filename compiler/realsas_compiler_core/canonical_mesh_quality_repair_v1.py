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
from .types import QualificationError, SurfaceSupportBinding


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


def _vertex_neighbors(faces):
    out=defaultdict(set)
    for face in faces:
        a,b,c=map(str,face)
        out[a].update((b,c))
        out[b].update((a,c))
        out[c].update((a,b))
    return out


def _incident_faces_by_vertex(faces):
    out=defaultdict(set)
    for fi,face in enumerate(faces):
        for vid in face:
            out[str(vid)].add(fi)
    return out


def _collapse_local_faces(faces, *, keep: str, remove: str, affected: set[int]):
    rows=[]
    for fi in sorted(affected):
        face=tuple(keep if str(v)==remove else str(v) for v in faces[fi])
        if len(set(face))<3:
            continue
        rows.append(face)
    if len(set(rows))!=len(rows):
        return None
    return tuple(rows)


def _link_condition_holds(
    faces,
    *,
    edge: tuple[str,str],
    incidence,
    neighbors,
) -> bool:
    adjacent=incidence.get(edge,())
    if len(adjacent)!=2:
        return False
    u,v=edge
    opposite={_opposite(faces[fi],edge) for fi in adjacent}
    shared=set(neighbors[u]).intersection(neighbors[v])
    return shared==opposite


def repair_candidate_endpoint_collapses_v1(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    max_collapses: int = 256,
    seed_only_violating_faces: bool = True,
) -> tuple[CanonicalMeshCandidateIR, dict]:
    """Greedy manifold-preserving halfedge collapse for residual quality defects.

    This pass never invents a vertex position. One endpoint survives exactly,
    including its support binding and component identity. Candidate collapses are
    restricted to interior manifold edges satisfying the link condition and are
    accepted only if local G3 quality improves monotonically and sampled local
    surface deviation remains inside frozen G1.
    """
    if int(max_collapses)<1:
        raise QualificationError("QUALITY_COLLAPSE_MAX_COLLAPSES_INVALID")
    vertex_rows={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertex_rows.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_COLLAPSE_REQUIRES_FACES")

    before=_report(faces,positions,policy)
    accepted=[]
    rejected_link=0
    rejected_shape=0
    rejected_quality=0
    rejected_duplicate=0

    for collapse_index in range(int(max_collapses)):
        incidence=_edge_incidence(faces)
        neighbors=_vertex_neighbors(faces)
        incident_by_vertex=_incident_faces_by_vertex(faces)
        metrics=[_metric(face,positions) for face in faces]
        violating={i for i,m in enumerate(metrics) if _violates(m,policy)}
        if not violating:
            break

        candidate_edges=set()
        for fi in sorted(violating):
            a,b,c=faces[fi]
            candidate_edges.update((_edge(a,b),_edge(b,c),_edge(c,a)))

        proposals=[]
        for edge in sorted(candidate_edges):
            u,v=edge
            if u not in vertex_rows or v not in vertex_rows:
                continue
            if vertex_rows[u].component_id!=vertex_rows[v].component_id:
                continue
            if not _link_condition_holds(
                faces,edge=edge,incidence=incidence,neighbors=neighbors
            ):
                rejected_link+=1
                continue

            for keep,remove in ((u,v),(v,u)):
                affected=set(incident_by_vertex[keep])|set(incident_by_vertex[remove])
                old_faces=tuple(faces[i] for i in sorted(affected))
                new_faces=_collapse_local_faces(
                    faces,keep=keep,remove=remove,affected=affected
                )
                if new_faces is None:
                    rejected_duplicate+=1
                    continue
                if not new_faces:
                    continue

                oldq={
                    "metrics":tuple(_metric(face,positions) for face in old_faces),
                }
                oldq["violation_count"]=sum(_violates(m,policy) for m in oldq["metrics"])
                oldq["min_angle_deg"]=min(float(m["min_angle_deg"]) for m in oldq["metrics"])
                oldq["max_aspect"]=max(float(m["aspect_longest_over_min_altitude"]) for m in oldq["metrics"])

                new_metrics=tuple(_metric(face,positions) for face in new_faces)
                if any(bool(m["degenerate"]) for m in new_metrics):
                    rejected_quality+=1
                    continue
                newq={
                    "metrics":new_metrics,
                    "violation_count":sum(_violates(m,policy) for m in new_metrics),
                    "min_angle_deg":min(float(m["min_angle_deg"]) for m in new_metrics),
                    "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in new_metrics),
                }
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
                    rejected_quality+=1
                    continue

                local_vertices={str(x) for face in old_faces for x in face}
                scale=_local_scale(local_vertices,positions)
                if scale<=_EPS:
                    continue
                deviation=_sampled_symmetric_local_deviation(
                    old_faces,new_faces,positions
                )
                allowed=float(policy.g1_max_normal_refinement_ratio)*scale
                if deviation>allowed+1e-12:
                    rejected_shape+=1
                    continue

                edge_len=float(np.linalg.norm(
                    np.asarray(positions[u],dtype=np.float64)
                    -np.asarray(positions[v],dtype=np.float64)
                ))
                proposals.append({
                    "edge":edge,
                    "keep":keep,
                    "remove":remove,
                    "affected":affected,
                    "new_faces":new_faces,
                    "old_quality":oldq,
                    "new_quality":newq,
                    "deviation":deviation,
                    "allowed":allowed,
                    "edge_length":edge_len,
                })

        if not proposals:
            break
        proposals.sort(key=lambda row:(
            -int(row["old_quality"]["violation_count"]-row["new_quality"]["violation_count"]),
            -float(row["new_quality"]["min_angle_deg"]-row["old_quality"]["min_angle_deg"]),
            float(row["new_quality"]["max_aspect"]),
            float(row["edge_length"]),
            row["keep"],
            row["remove"],
        ))
        row=proposals[0]
        keep=str(row["keep"]); remove=str(row["remove"])
        affected=set(row["affected"])
        untouched=[face for i,face in enumerate(faces) if i not in affected]
        faces=untouched+list(row["new_faces"])
        vertex_rows.pop(remove,None)
        positions.pop(remove,None)
        accepted.append({
            "collapse_index":collapse_index,
            "edge":row["edge"],
            "kept_vertex_id":keep,
            "removed_vertex_id":remove,
            "edge_length":row["edge_length"],
            "local_sampled_surface_deviation":row["deviation"],
            "allowed_local_deviation":row["allowed"],
            "old_violation_count":row["old_quality"]["violation_count"],
            "new_violation_count":row["new_quality"]["violation_count"],
            "old_min_angle_deg":row["old_quality"]["min_angle_deg"],
            "new_min_angle_deg":row["new_quality"]["min_angle_deg"],
            "old_max_aspect":row["old_quality"]["max_aspect"],
            "new_max_aspect":row["new_quality"]["max_aspect"],
        })

    faces_tuple=tuple(sorted(tuple(face) for face in faces))
    used_ids={str(v) for face in faces_tuple for v in face}
    vertices=tuple(
        row for vid,row in sorted(vertex_rows.items())
        if vid in used_ids
    )
    edges=tuple(sorted({
        _edge(face[i],face[j])
        for face in faces_tuple
        for i,j in ((0,1),(1,2),(2,0))
    }))
    after=_report(faces_tuple,positions,policy)
    producer_policy_hash=content_sha256({
        "schema":"RealSaS.EndpointHalfedgeCollapseRepairPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_collapses":int(max_collapses),
        "seed_only_violating_faces":bool(seed_only_violating_faces),
        "topology_guard":"INTERIOR_MANIFOLD_LINK_CONDITION",
        "placement":"KEEP_EXISTING_ENDPOINT_ONLY",
        "g1_local_deviation_ratio":float(policy.g1_max_normal_refinement_ratio),
        "selection":"MONOTONE_G3_THEN_MIN_EDGE",
        "new_vertex_creation":"FORBIDDEN",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices,
        faces=faces_tuple,
        edges=edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.EndpointHalfedgeCollapseRepair.v1",
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "endpoint_collapse_quality_repair":{
                "algorithm":"ENDPOINT_HALFEDGE_COLLAPSE_LINK_CONDITION_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "new_vertex_creation":False,
                "surviving_vertex_motion":False,
                "surviving_support_binding_changed":False,
                "accepted_collapse_count":len(accepted),
                "rejected_link_condition_count":int(rejected_link),
                "rejected_shape_deviation_count":int(rejected_shape),
                "rejected_quality_count":int(rejected_quality),
                "rejected_duplicate_face_count":int(rejected_duplicate),
                "before":before,
                "after":after,
                "accepted_collapses":accepted,
            },
        },
    )
    result=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return result,dict(result.metadata["endpoint_collapse_quality_repair"])


def _closest_point_barycentric(point, tri):
    """Return closest point and barycentric weights on one 3D triangle."""
    p=np.asarray(point,dtype=np.float64)
    a,b,c=(np.asarray(x,dtype=np.float64) for x in tri)
    ab=b-a; ac=c-a; ap=p-a
    d1=float(np.dot(ab,ap)); d2=float(np.dot(ac,ap))
    if d1<=0.0 and d2<=0.0:
        return a,np.asarray([1.0,0.0,0.0],dtype=np.float64)
    bp=p-b; d3=float(np.dot(ab,bp)); d4=float(np.dot(ac,bp))
    if d3>=0.0 and d4<=d3:
        return b,np.asarray([0.0,1.0,0.0],dtype=np.float64)
    vc=d1*d4-d3*d2
    if vc<=0.0 and d1>=0.0 and d3<=0.0:
        v=d1/(d1-d3)
        return a+v*ab,np.asarray([1.0-v,v,0.0],dtype=np.float64)
    cp=p-c; d5=float(np.dot(ab,cp)); d6=float(np.dot(ac,cp))
    if d6>=0.0 and d5<=d6:
        return c,np.asarray([0.0,0.0,1.0],dtype=np.float64)
    vb=d5*d2-d1*d6
    if vb<=0.0 and d2>=0.0 and d6<=0.0:
        w=d2/(d2-d6)
        return a+w*ac,np.asarray([1.0-w,0.0,w],dtype=np.float64)
    va=d3*d6-d5*d4
    if va<=0.0 and (d4-d3)>=0.0 and (d5-d6)>=0.0:
        w=(d4-d3)/((d4-d3)+(d5-d6))
        return b+w*(c-b),np.asarray([0.0,1.0-w,w],dtype=np.float64)
    denom=1.0/(va+vb+vc)
    v=vb*denom; w=vc*denom; u=1.0-v-w
    return a+ab*v+ac*w,np.asarray([u,v,w],dtype=np.float64)


def _combine_support_bindings(vertices, weights):
    rows=defaultdict(float)
    for vertex,weight in zip(vertices,weights):
        w=float(weight)
        if w<=1e-14:
            continue
        for sid,coeff in vertex.support_binding.coefficients:
            rows[str(sid)]+=w*float(coeff)
    cleaned=[(sid,value) for sid,value in sorted(rows.items()) if value>1e-12]
    total=sum(value for _,value in cleaned)
    if total<=0.0:
        raise QualificationError("QUALITY_RELAX_SUPPORT_EMPTY")
    coeffs=tuple((sid,float(value/total)) for sid,value in cleaned)
    mode=(
        "IDENTITY_SURFACE_NODE"
        if len(coeffs)==1 and abs(coeffs[0][1]-1.0)<=1e-12
        else "LOCAL_CONVEX_INTERPOLATION"
    )
    return SurfaceSupportBinding(mode,coeffs)


def _face_normal(face, positions):
    p=[np.asarray(positions[str(v)],dtype=np.float64) for v in face]
    n=np.cross(p[1]-p[0],p[2]-p[0])
    norm=float(np.linalg.norm(n))
    if not math.isfinite(norm) or norm<=_EPS:
        return None
    return n/norm


def mechanical_quality_protected_surface_ids_v1(partition) -> frozenset[str]:
    """Surface IDs that must remain immobile during local quality relaxation.

    PRESERVE_CONTINUITY is a connectivity invariant, not a vertex-motion ban.
    Only explicit cuts and unresolved boundaries hard-protect their endpoint
    supports. Topological boundary vertices are protected independently by the
    relaxation operator itself.
    """
    rows = tuple(getattr(partition, "boundary_constraints", ()) or ())
    protected: set[str] = set()
    for row in rows:
        decision = str(getattr(row, "decision", ""))
        if decision not in {"SEPARATE", "PRESERVE_CONTINUITY", "UNKNOWN"}:
            raise QualificationError("QUALITY_PROTECTION_BOUNDARY_DECISION_INVALID")
        if decision in {"SEPARATE", "UNKNOWN"}:
            protected.add(str(row.a_surface_id))
            protected.add(str(row.b_surface_id))
    return frozenset(sorted(protected))


def repair_candidate_projected_relaxation_v1(
    candidate: CanonicalMeshCandidateIR,
    reference_candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    protected_surface_ids: set[str] | frozenset[str] = frozenset(),
    max_moves: int = 256,
    relaxation_fractions: tuple[float,...] = (0.125,0.25,0.5,0.75,1.0),
) -> tuple[CanonicalMeshCandidateIR, dict]:
    """Quality-targeted tangential relaxation projected to immutable source mesh.

    Only interior vertices in currently violating faces are considered. The
    Laplacian target is projected onto the vertex's immutable reference 1-ring
    triangles; the projected barycentric coordinates are composed with existing
    SurfaceSupportBindings, yielding an exact LOCAL_CONVEX_INTERPOLATION rather
    than a free geometric point.
    """
    if int(max_moves)<1:
        raise QualificationError("QUALITY_RELAX_MAX_MOVES_INVALID")
    fractions=tuple(float(x) for x in relaxation_fractions)
    if not fractions or any((not math.isfinite(x) or x<=0.0 or x>1.0) for x in fractions):
        raise QualificationError("QUALITY_RELAX_FRACTIONS_INVALID")

    reference_vertices={str(v.candidate_vertex_id):v for v in reference_candidate.vertices}
    reference_positions={vid:tuple(map(float,v.P)) for vid,v in reference_vertices.items()}
    reference_incident=_incident_faces_by_vertex(reference_candidate.faces)

    vertices={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    before=_report(faces,positions,policy)
    protected_surface_ids={str(x) for x in protected_surface_ids}

    accepted=[]
    rejected_boundary=0
    rejected_protected=0
    rejected_projection=0
    rejected_quality=0
    rejected_shape=0
    rejected_orientation=0

    for move_index in range(int(max_moves)):
        incidence=_edge_incidence(faces)
        neighbors=_vertex_neighbors(faces)
        incident=_incident_faces_by_vertex(faces)
        metrics=[_metric(face,positions) for face in faces]
        violating={i for i,m in enumerate(metrics) if _violates(m,policy)}
        if not violating:
            break

        boundary_vertices={
            vid for edge,adj in incidence.items() if len(adj)==1 for vid in edge
        }
        seed_vertices=sorted({
            str(v) for fi in violating for v in faces[fi]
        })
        proposals=[]

        for vid in seed_vertices:
            if vid not in vertices or vid not in reference_vertices:
                continue
            if vid in boundary_vertices:
                rejected_boundary+=1
                continue
            if protected_surface_ids.intersection(
                sid for sid,_ in vertices[vid].support_binding.coefficients
            ):
                rejected_protected+=1
                continue
            nbs=sorted(neighbors.get(vid,()))
            if len(nbs)<3 or any(nb not in positions for nb in nbs):
                continue
            component=vertices[vid].component_id
            if any(vertices[nb].component_id!=component for nb in nbs):
                continue

            patch_indices=set(incident[vid])
            old_faces=tuple(faces[i] for i in sorted(patch_indices))
            old_metrics=tuple(_metric(face,positions) for face in old_faces)
            oldq={
                "violation_count":sum(_violates(m,policy) for m in old_metrics),
                "min_angle_deg":min(float(m["min_angle_deg"]) for m in old_metrics),
                "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in old_metrics),
            }
            if oldq["violation_count"]<=0:
                continue

            current=np.asarray(positions[vid],dtype=np.float64)
            centroid=np.mean(
                np.asarray([positions[nb] for nb in nbs],dtype=np.float64),
                axis=0,
            )
            local_scale=_local_scale([vid,*nbs],positions)
            if local_scale<=_EPS:
                continue

            ref_face_indices=reference_incident.get(vid,set())
            ref_faces=[
                tuple(map(str,reference_candidate.faces[i]))
                for i in sorted(ref_face_indices)
            ]
            ref_faces=[
                face for face in ref_faces
                if all(x in reference_positions for x in face)
                and len({reference_vertices[x].component_id for x in face})==1
                and reference_vertices[face[0]].component_id==component
            ]
            if not ref_faces:
                rejected_projection+=1
                continue

            for fraction in fractions:
                target=current+fraction*(centroid-current)
                best=None
                for ref_face in ref_faces:
                    tri=tuple(reference_positions[x] for x in ref_face)
                    projected,bary=_closest_point_barycentric(target,tri)
                    distance=float(np.linalg.norm(target-projected))
                    key=(distance,tuple(ref_face))
                    if best is None or key<best[0]:
                        best=(key,ref_face,projected,bary)
                if best is None:
                    continue
                _,ref_face,projected,bary=best
                displacement=float(np.linalg.norm(projected-current))
                allowed_displacement=float(policy.g1_max_normal_refinement_ratio)*local_scale
                if displacement>allowed_displacement+1e-12:
                    rejected_shape+=1
                    continue

                new_positions=dict(positions)
                new_positions[vid]=tuple(map(float,projected))
                new_metrics=tuple(_metric(face,new_positions) for face in old_faces)
                if any(bool(m["degenerate"]) for m in new_metrics):
                    rejected_quality+=1
                    continue
                newq={
                    "violation_count":sum(_violates(m,policy) for m in new_metrics),
                    "min_angle_deg":min(float(m["min_angle_deg"]) for m in new_metrics),
                    "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in new_metrics),
                }
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
                    rejected_quality+=1
                    continue

                orientation_ok=True
                for face in old_faces:
                    oldn=_face_normal(face,positions)
                    newn=_face_normal(face,new_positions)
                    if oldn is None or newn is None or float(np.dot(oldn,newn))<=0.0:
                        orientation_ok=False
                        break
                if not orientation_ok:
                    rejected_orientation+=1
                    continue

                new_faces=old_faces
                deviation=_sampled_symmetric_local_deviation(
                    old_faces,new_faces,new_positions
                )
                # The topology is unchanged; supplement the symmetric face test
                # with direct vertex displacement because the old/new helper sees
                # one shared face list.
                deviation=max(deviation,displacement)
                if deviation>allowed_displacement+1e-12:
                    rejected_shape+=1
                    continue

                ref_vertex_rows=[reference_vertices[x] for x in ref_face]
                support=_combine_support_bindings(ref_vertex_rows,bary)
                proposals.append({
                    "vertex_id":vid,
                    "fraction":fraction,
                    "projected":tuple(map(float,projected)),
                    "support":support,
                    "reference_face":ref_face,
                    "barycentric":tuple(map(float,bary)),
                    "displacement":displacement,
                    "allowed":allowed_displacement,
                    "old_quality":oldq,
                    "new_quality":newq,
                })

        if not proposals:
            break
        proposals.sort(key=lambda row:(
            -int(row["old_quality"]["violation_count"]-row["new_quality"]["violation_count"]),
            -float(row["new_quality"]["min_angle_deg"]-row["old_quality"]["min_angle_deg"]),
            float(row["new_quality"]["max_aspect"]),
            float(row["displacement"]),
            str(row["vertex_id"]),
            float(row["fraction"]),
        ))
        row=proposals[0]
        vid=str(row["vertex_id"])
        old_vertex=vertices[vid]
        vertices[vid]=replace(
            old_vertex,
            support_binding=row["support"],
            P=row["projected"],
            metadata={
                **dict(old_vertex.metadata or {}),
                "projected_quality_relaxation":{
                    "algorithm":"PROJECTED_LOCAL_RELAXATION_V1",
                    "reference_face":list(row["reference_face"]),
                    "barycentric":list(row["barycentric"]),
                    "fraction":row["fraction"],
                    "displacement":row["displacement"],
                    "allowed_displacement":row["allowed"],
                },
            },
        )
        positions[vid]=row["projected"]
        accepted.append({
            "move_index":move_index,
            "vertex_id":vid,
            "fraction":row["fraction"],
            "reference_face":row["reference_face"],
            "barycentric":row["barycentric"],
            "displacement":row["displacement"],
            "allowed_displacement":row["allowed"],
            "old_violation_count":row["old_quality"]["violation_count"],
            "new_violation_count":row["new_quality"]["violation_count"],
            "old_min_angle_deg":row["old_quality"]["min_angle_deg"],
            "new_min_angle_deg":row["new_quality"]["min_angle_deg"],
            "old_max_aspect":row["old_quality"]["max_aspect"],
            "new_max_aspect":row["new_quality"]["max_aspect"],
        })

    vertices_tuple=tuple(vertices[vid] for vid in sorted(vertices))
    after=_report(faces,positions,policy)
    producer_policy_hash=content_sha256({
        "schema":"RealSaS.ProjectedLocalRelaxationRepairPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "reference_candidate_lineage_hash":reference_candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_moves":int(max_moves),
        "relaxation_fractions":list(fractions),
        "projection":"IMMUTABLE_REFERENCE_ONE_RING_TRIANGLES",
        "support":"COMPOSED_LOCAL_CONVEX_INTERPOLATION",
        "boundary_vertices":"PROTECTED",
        "protected_surface_ids":sorted(protected_surface_ids),
        "g1_displacement_ratio":float(policy.g1_max_normal_refinement_ratio),
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices_tuple,
        faces=tuple(faces),
        edges=candidate.edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.ProjectedLocalRelaxation.v1",
        producer_policy_hash=producer_policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "projected_local_relaxation":{
                "algorithm":"PROJECTED_LOCAL_RELAXATION_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "reference_candidate_lineage_hash":reference_candidate.candidate_lineage_hash,
                "accepted_move_count":len(accepted),
                "rejected_boundary_count":int(rejected_boundary),
                "rejected_protected_count":int(rejected_protected),
                "rejected_projection_count":int(rejected_projection),
                "rejected_quality_count":int(rejected_quality),
                "rejected_shape_count":int(rejected_shape),
                "rejected_orientation_count":int(rejected_orientation),
                "before":before,
                "after":after,
                "accepted_moves":accepted,
            },
        },
    )
    result=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return result,dict(result.metadata["projected_local_relaxation"])
