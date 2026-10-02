from __future__ import annotations

"""Generic batched endpoint-collapse quality repair.

This is an audit candidate for replacing the O(K*F) one-collapse-per-global-rebuild
loop in canonical_mesh_quality_repair_v1. It preserves the same local admission
rules and batches only collapses whose closed one-ring vertex neighborhoods are
pairwise disjoint. Disjoint stars make the local rewrites commute; a global
manifold-incidence check is still applied after every batch fail-closed.
"""

from collections import defaultdict
from dataclasses import replace
import numpy as np

from .canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_vertex_neighbors,_incident_faces_by_vertex,
    _link_condition_holds,_collapse_local_faces,_metric,_violates,
    _local_scale,_sampled_symmetric_local_deviation,_report,_EPS,
)
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report


def _global_nonmanifold_edge_count(faces) -> int:
    inc=_edge_incidence(faces)
    return sum(len(rows)>2 for rows in inc.values())


def repair_candidate_endpoint_collapses_batched_v2(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    max_batches: int = 64,
    max_collapses: int = 2048,
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_batches)<1 or int(max_collapses)<1:
        raise QualificationError("QUALITY_BATCHED_COLLAPSE_LIMIT_INVALID")

    vertex_rows={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertex_rows.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_BATCHED_COLLAPSE_REQUIRES_FACES")

    initial_topology=_manifold_report(faces)
    if not initial_topology["passed"]:
        raise QualificationError(
            f"QUALITY_BATCHED_COLLAPSE_INPUT_NONMANIFOLD:"
            f"{initial_topology['nonmanifold_edge_count']}:"
            f"{initial_topology['illegal_vertex_link_count']}"
        )
    before=_report(faces,positions,policy)
    accepted=[]
    batch_rows=[]
    rejected_link=rejected_shape=rejected_quality=rejected_duplicate=0

    for batch_index in range(int(max_batches)):
        if len(accepted)>=int(max_collapses):
            break
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
            if not _link_condition_holds(faces,edge=edge,incidence=incidence,neighbors=neighbors):
                rejected_link+=1
                continue
            for keep,remove in ((u,v),(v,u)):
                affected=set(incident_by_vertex[keep])|set(incident_by_vertex[remove])
                old_faces=tuple(faces[i] for i in sorted(affected))
                new_faces=_collapse_local_faces(faces,keep=keep,remove=remove,affected=affected)
                if new_faces is None:
                    rejected_duplicate+=1
                    continue
                if not new_faces:
                    continue
                old_metrics=tuple(_metric(face,positions) for face in old_faces)
                new_metrics=tuple(_metric(face,positions) for face in new_faces)
                if any(bool(m["degenerate"]) for m in new_metrics):
                    rejected_quality+=1
                    continue
                oldq={
                    "violation_count":sum(_violates(m,policy) for m in old_metrics),
                    "min_angle_deg":min(float(m["min_angle_deg"]) for m in old_metrics),
                    "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in old_metrics),
                }
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
                local_vertices={str(x) for face in old_faces for x in face}
                scale=_local_scale(local_vertices,positions)
                if scale<=_EPS:
                    continue
                deviation=_sampled_symmetric_local_deviation(old_faces,new_faces,positions)
                allowed=float(policy.g1_max_normal_refinement_ratio)*scale
                if deviation>allowed+1e-12:
                    rejected_shape+=1
                    continue
                edge_len=float(np.linalg.norm(
                    np.asarray(positions[u],dtype=np.float64)
                    -np.asarray(positions[v],dtype=np.float64)
                ))
                proposals.append({
                    "edge":edge,"keep":keep,"remove":remove,
                    "affected":affected,"local_vertices":frozenset(local_vertices),
                    "oldq":oldq,"newq":newq,"deviation":deviation,
                    "allowed":allowed,"edge_length":edge_len,
                })

        if not proposals:
            break
        proposals.sort(key=lambda row:(
            -int(row["oldq"]["violation_count"]-row["newq"]["violation_count"]),
            -float(row["newq"]["min_angle_deg"]-row["oldq"]["min_angle_deg"]),
            float(row["newq"]["max_aspect"]),
            float(row["edge_length"]),
            row["keep"],row["remove"],
        ))

        selected=[]
        occupied=set()
        remaining=int(max_collapses)-len(accepted)
        for row in proposals:
            if len(selected)>=remaining:
                break
            if occupied.intersection(row["local_vertices"]):
                continue
            selected.append(row)
            occupied.update(row["local_vertices"])
        if not selected:
            break

        mapping={str(row["remove"]):str(row["keep"]) for row in selected}
        new_faces=[]
        for face in faces:
            rewritten=tuple(mapping.get(str(v),str(v)) for v in face)
            if len(set(rewritten))<3:
                continue
            new_faces.append(rewritten)
        if len(set(new_faces))!=len(new_faces):
            raise QualificationError("QUALITY_BATCHED_COLLAPSE_CREATED_DUPLICATE_FACE")
        faces=new_faces
        for row in selected:
            rem=str(row["remove"])
            vertex_rows.pop(rem,None)
            positions.pop(rem,None)

        topo=_manifold_report(faces)
        if not topo["passed"]:
            raise QualificationError(
                f"QUALITY_BATCHED_COLLAPSE_POSTBATCH_NONMANIFOLD:"
                f"{batch_index}:{topo['nonmanifold_edge_count']}:"
                f"{topo['illegal_vertex_link_count']}"
            )

        for row in selected:
            accepted.append({
                "batch_index":batch_index,
                "edge":row["edge"],"kept_vertex_id":row["keep"],"removed_vertex_id":row["remove"],
                "edge_length":row["edge_length"],
                "local_sampled_surface_deviation":row["deviation"],
                "allowed_local_deviation":row["allowed"],
                "old_violation_count":row["oldq"]["violation_count"],
                "new_violation_count":row["newq"]["violation_count"],
                "old_min_angle_deg":row["oldq"]["min_angle_deg"],
                "new_min_angle_deg":row["newq"]["min_angle_deg"],
                "old_max_aspect":row["oldq"]["max_aspect"],
                "new_max_aspect":row["newq"]["max_aspect"],
            })
        batch_rows.append({
            "batch_index":batch_index,
            "accepted_count":len(selected),
            "occupied_closed_one_ring_vertex_count":len(occupied),
            "face_count_after":len(faces),
            "topology":topo,
        })

    faces_tuple=tuple(sorted(tuple(face) for face in faces))
    used_ids={str(v) for face in faces_tuple for v in face}
    vertices=tuple(row for vid,row in sorted(vertex_rows.items()) if vid in used_ids)
    edges=tuple(sorted({_edge(face[i],face[j]) for face in faces_tuple for i,j in ((0,1),(1,2),(2,0))}))
    after=_report(faces_tuple,positions,policy)
    final_topology=_manifold_report(faces_tuple)
    policy_hash=content_sha256({
        "schema":"RealSaS.EndpointHalfedgeBatchedCollapseRepairPolicy.v2",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_batches":int(max_batches),"max_collapses":int(max_collapses),
        "independence_guard":"PAIRWISE_DISJOINT_CLOSED_ONE_RING_VERTEX_SETS",
        "topology_guard":"EDGE_AND_VERTEX_LINK_MANIFOLD_PRE_AND_POST_BATCH",
        "placement":"KEEP_EXISTING_ENDPOINT_ONLY",
        "selection":"MONOTONE_G3_WITH_FROZEN_G1_LOCAL_DEVIATION",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices,faces=faces_tuple,edges=edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.EndpointHalfedgeBatchedCollapseRepair.v2",
        producer_policy_hash=policy_hash,candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "endpoint_batched_collapse_quality_repair":{
                "algorithm":"ENDPOINT_HALFEDGE_BATCHED_DISJOINT_STAR_COLLAPSE_V2",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "new_vertex_creation":False,"surviving_vertex_motion":False,
                "surviving_support_binding_changed":False,
                "accepted_collapse_count":len(accepted),
                "batch_count":len(batch_rows),"batches":batch_rows,
                "rejected_link_condition_count":int(rejected_link),
                "rejected_shape_deviation_count":int(rejected_shape),
                "rejected_quality_count":int(rejected_quality),
                "rejected_duplicate_face_count":int(rejected_duplicate),
                "before":before,"after":after,
                "initial_topology":initial_topology,"final_topology":final_topology,
                "accepted_collapses":accepted,
            },
        },
    )
    result=replace(provisional,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional))
    return result,dict(result.metadata["endpoint_batched_collapse_quality_repair"])