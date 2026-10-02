from __future__ import annotations

"""Generic batched projected relaxation for residual static mesh quality defects.

The operator preserves connectivity. Candidate moves are identical in spirit to
repair_candidate_projected_relaxation_v1, but pairwise-disjoint closed one-ring
stars are applied in the same batch. Every moved vertex is projected onto the
immutable reference candidate surface and receives a convex-composed support
binding. Local G3 quality must improve monotonically, orientation must be
preserved, and displacement must remain within frozen G1.
"""

from collections import defaultdict
from dataclasses import replace
import math
import numpy as np

from .canonical_mesh_quality_repair_v1 import (
    _EPS,
    _closest_point_barycentric,
    _combine_support_bindings,
    _edge_incidence,
    _face_normal,
    _incident_faces_by_vertex,
    _local_scale,
    _metric,
    _report,
    _sampled_symmetric_local_deviation,
    _vertex_neighbors,
    _violates,
)
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def repair_candidate_projected_relaxation_batched_v2(
    candidate: CanonicalMeshCandidateIR,
    reference_candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    protected_surface_ids: set[str] | frozenset[str] = frozenset(),
    max_batches: int = 64,
    max_moves: int = 2048,
    relaxation_fractions: tuple[float,...] = (0.125,0.25,0.5,0.75,1.0),
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_batches)<1 or int(max_moves)<1:
        raise QualificationError("QUALITY_BATCHED_RELAX_LIMIT_INVALID")
    fractions=tuple(float(x) for x in relaxation_fractions)
    if not fractions or any((not math.isfinite(x) or x<=0.0 or x>1.0) for x in fractions):
        raise QualificationError("QUALITY_BATCHED_RELAX_FRACTIONS_INVALID")

    reference_vertices={str(v.candidate_vertex_id):v for v in reference_candidate.vertices}
    reference_positions={vid:tuple(map(float,v.P)) for vid,v in reference_vertices.items()}
    reference_incident=_incident_faces_by_vertex(reference_candidate.faces)

    vertices={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_BATCHED_RELAX_REQUIRES_FACES")
    topo0=_manifold_report(faces)
    if not topo0["passed"]:
        raise QualificationError(
            f"QUALITY_BATCHED_RELAX_INPUT_NONMANIFOLD:"
            f"{topo0['nonmanifold_edge_count']}:{topo0['illegal_vertex_link_count']}"
        )

    protected_surface_ids={str(x) for x in protected_surface_ids}
    before=_report(faces,positions,policy)
    accepted=[]
    batch_rows=[]
    rejected_boundary=0
    rejected_protected=0
    rejected_projection=0
    rejected_quality=0
    rejected_shape=0
    rejected_orientation=0
    rejected_batch_conflict=0

    for batch_index in range(int(max_batches)):
        if len(accepted)>=int(max_moves):
            break
        incidence=_edge_incidence(faces)
        neighbors=_vertex_neighbors(faces)
        incident=_incident_faces_by_vertex(faces)
        metrics=[_metric(face,positions) for face in faces]
        violating={i for i,m in enumerate(metrics) if _violates(m,policy)}
        if not violating:
            break

        boundary_vertices={vid for edge,adj in incidence.items() if len(adj)==1 for vid in edge}
        seed_vertices=sorted({str(v) for fi in violating for v in faces[fi]})
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
            centroid=np.mean(np.asarray([positions[nb] for nb in nbs],dtype=np.float64),axis=0)
            local_vertices=frozenset({vid,*nbs})
            local_scale=_local_scale(local_vertices,positions)
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

            best_for_vertex=None
            for fraction in fractions:
                target=current+float(fraction)*(centroid-current)
                best_projection=None
                for ref_face in ref_faces:
                    tri=tuple(reference_positions[x] for x in ref_face)
                    projected,bary=_closest_point_barycentric(target,tri)
                    distance=float(np.linalg.norm(target-projected))
                    key=(distance,tuple(ref_face))
                    if best_projection is None or key<best_projection[0]:
                        best_projection=(key,ref_face,projected,bary)
                if best_projection is None:
                    rejected_projection+=1
                    continue
                _,ref_face,projected,bary=best_projection
                displacement=float(np.linalg.norm(projected-current))
                allowed=float(policy.g1_max_normal_refinement_ratio)*local_scale
                if displacement>allowed+1e-12:
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

                deviation=_sampled_symmetric_local_deviation(old_faces,old_faces,new_positions)
                deviation=max(float(deviation),displacement)
                if deviation>allowed+1e-12:
                    rejected_shape+=1
                    continue

                support=_combine_support_bindings(
                    [reference_vertices[x] for x in ref_face],
                    bary,
                )
                improvement=(
                    int(oldq["violation_count"]-newq["violation_count"]),
                    float(newq["min_angle_deg"]-oldq["min_angle_deg"]),
                    float(oldq["max_aspect"]-newq["max_aspect"]),
                    -displacement,
                )
                row={
                    "vertex_id":vid,
                    "fraction":float(fraction),
                    "projected":tuple(map(float,projected)),
                    "support":support,
                    "reference_face":tuple(ref_face),
                    "barycentric":tuple(map(float,bary)),
                    "displacement":displacement,
                    "allowed":allowed,
                    "oldq":oldq,
                    "newq":newq,
                    "local_vertices":local_vertices,
                    "improvement":improvement,
                }
                key=(
                    -improvement[0],-improvement[1],-improvement[2],
                    displacement,vid,float(fraction),
                )
                if best_for_vertex is None or key<best_for_vertex[0]:
                    best_for_vertex=(key,row)
            if best_for_vertex is not None:
                proposals.append(best_for_vertex[1])

        if not proposals:
            break
        proposals.sort(key=lambda row:(
            -int(row["improvement"][0]),
            -float(row["improvement"][1]),
            -float(row["improvement"][2]),
            float(row["displacement"]),
            str(row["vertex_id"]),
        ))

        selected=[]
        occupied=set()
        remaining=int(max_moves)-len(accepted)
        for row in proposals:
            if len(selected)>=remaining:
                break
            if occupied.intersection(row["local_vertices"]):
                rejected_batch_conflict+=1
                continue
            selected.append(row)
            occupied.update(row["local_vertices"])
        if not selected:
            break

        before_batch=_report(faces,positions,policy)
        for row in selected:
            vid=str(row["vertex_id"])
            old=vertices[vid]
            positions[vid]=tuple(row["projected"])
            vertices[vid]=replace(
                old,
                support_binding=row["support"],
                P=tuple(row["projected"]),
                metadata={
                    **dict(old.metadata or {}),
                    "projected_quality_relaxation":{
                        "operator":"BATCHED_DISJOINT_STAR_PROJECTED_RELAXATION_V2",
                        "reference_face":list(row["reference_face"]),
                        "barycentric":list(row["barycentric"]),
                        "fraction":float(row["fraction"]),
                    },
                },
            )

        after_batch=_report(faces,positions,policy)
        if int(after_batch["policy_violating_face_count"])>int(before_batch["policy_violating_face_count"]):
            raise QualificationError("QUALITY_BATCHED_RELAX_GLOBAL_QUALITY_REGRESSION")
        topo=_manifold_report(faces)
        if not topo["passed"]:
            raise QualificationError(
                f"QUALITY_BATCHED_RELAX_POSTBATCH_NONMANIFOLD:"
                f"{batch_index}:{topo['nonmanifold_edge_count']}:{topo['illegal_vertex_link_count']}"
            )

        for row in selected:
            accepted.append({
                "batch_index":batch_index,
                "vertex_id":row["vertex_id"],
                "fraction":row["fraction"],
                "reference_face":row["reference_face"],
                "barycentric":row["barycentric"],
                "displacement":row["displacement"],
                "allowed_displacement":row["allowed"],
                "old_violation_count":row["oldq"]["violation_count"],
                "new_violation_count":row["newq"]["violation_count"],
                "old_min_angle_deg":row["oldq"]["min_angle_deg"],
                "new_min_angle_deg":row["newq"]["min_angle_deg"],
                "old_max_aspect":row["oldq"]["max_aspect"],
                "new_max_aspect":row["newq"]["max_aspect"],
            })
        batch_rows.append({
            "batch_index":batch_index,
            "accepted_move_count":len(selected),
            "occupied_closed_one_ring_vertex_count":len(occupied),
            "before_violations":int(before_batch["policy_violating_face_count"]),
            "after_violations":int(after_batch["policy_violating_face_count"]),
            "topology":topo,
        })

    vertices_tuple=tuple(vertices[vid] for vid in sorted(vertices))
    after=_report(faces,positions,policy)
    final_topology=_manifold_report(faces)
    policy_hash=content_sha256({
        "schema":"RealSaS.ProjectedBatchedRelaxationRepairPolicy.v2",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "reference_candidate_lineage_hash":reference_candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_batches":int(max_batches),"max_moves":int(max_moves),
        "fractions":fractions,
        "batch_independence":"PAIRWISE_DISJOINT_CLOSED_ONE_RING_VERTEX_SETS",
        "topology_guard":"CONNECTIVITY_IMMUTABLE__EDGE_AND_VERTEX_LINK_MANIFOLD_PRE_POST",
        "projection":"IMMUTABLE_REFERENCE_TRIANGLE_CLOSEST_POINT",
        "support_binding":"REFERENCE_FACE_BARYCENTRIC_CONVEX_COMPOSITION",
        "selection":"MONOTONE_G3_WITH_FROZEN_G1_DISPLACEMENT_AND_ORIENTATION",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices_tuple,
        faces=tuple(faces),
        edges=candidate.edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.ProjectedBatchedRelaxationRepair.v2",
        producer_policy_hash=policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "projected_batched_relaxation_quality_repair":{
                "algorithm":"PROJECTED_BATCHED_DISJOINT_STAR_RELAXATION_V2",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "reference_candidate_lineage_hash":reference_candidate.candidate_lineage_hash,
                "accepted_move_count":len(accepted),
                "batch_count":len(batch_rows),
                "batches":batch_rows,
                "rejected_boundary_count":int(rejected_boundary),
                "rejected_protected_count":int(rejected_protected),
                "rejected_projection_count":int(rejected_projection),
                "rejected_quality_count":int(rejected_quality),
                "rejected_shape_count":int(rejected_shape),
                "rejected_orientation_count":int(rejected_orientation),
                "rejected_batch_conflict_count":int(rejected_batch_conflict),
                "before":before,"after":after,
                "initial_topology":topo0,"final_topology":final_topology,
                "accepted_moves":accepted,
            },
        },
    )
    result=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return result,dict(result.metadata["projected_batched_relaxation_quality_repair"])
