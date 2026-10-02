from __future__ import annotations

"""Generic source-surface-constrained vertex quality optimization.

Residual static-quality defects may remain after topology-safe flip/collapse,
projected centroid relaxation, and local cavity remeshing.  This operator keeps
topology fixed and searches admitted immutable reference triangles for a better
position of an existing candidate vertex.

Admission is fail-closed:
- input and output edge/vertex-link manifoldness must pass,
- boundary and protected vertices are not moved,
- only vertices present in the immutable reference candidate are eligible,
- candidate positions are barycentric points on reference triangles incident to
  the same reference vertex,
- moved support is the convex composition of that exact reference triangle,
- all incident current faces must become policy-clean,
- face orientation must remain in the same hemisphere,
- frozen G1 displacement/deviation budget must pass,
- pairwise-disjoint closed one-rings may be moved in one batch.

This is audit authority only until Stage14 explicitly owns source-surface
mechanical remeshing/optimization.
"""

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .canonical_mesh_quality_repair_v1 import (
    _edge_incidence,
    _face_normal,
    _incident_faces_by_vertex,
    _local_scale,
    _metric,
    _report,
    _sampled_symmetric_local_deviation,
    _violates,
    _combine_support_bindings,
)
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def _patch_quality(faces, positions, policy):
    metrics=tuple(_metric(face,positions) for face in faces)
    if not metrics:
        return None
    return {
        "violation_count":sum(_violates(m,policy) for m in metrics),
        "min_angle_deg":min(float(m["min_angle_deg"]) for m in metrics),
        "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in metrics),
        "degenerate_count":sum(bool(m["degenerate"]) for m in metrics),
    }


def _bary_grid(resolution: int):
    n=int(resolution)
    if n<2:
        raise QualificationError("QUALITY_SOURCE_OPT_GRID_INVALID")
    for i in range(n+1):
        for j in range(n+1-i):
            a=float(i)/float(n)
            b=float(j)/float(n)
            c=1.0-a-b
            yield (a,b,c)


def repair_candidate_source_surface_quality_v1(
    candidate: CanonicalMeshCandidateIR,
    reference_candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    protected_surface_ids: set[str] | frozenset[str] = frozenset(),
    grid_schedule: tuple[int,...] = (8,16,32,64),
    max_batches: int = 32,
    max_moves: int = 256,
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_batches)<1 or int(max_moves)<1:
        raise QualificationError("QUALITY_SOURCE_OPT_LIMIT_INVALID")
    if not grid_schedule:
        raise QualificationError("QUALITY_SOURCE_OPT_GRID_MISSING")

    vertex_rows={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,row.P)) for vid,row in vertex_rows.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_SOURCE_OPT_REQUIRES_FACES")
    topo0=_manifold_report(faces)
    if not topo0["passed"]:
        raise QualificationError(
            f"QUALITY_SOURCE_OPT_INPUT_NONMANIFOLD:"
            f"{topo0['nonmanifold_edge_count']}:"
            f"{topo0['illegal_vertex_link_count']}"
        )

    ref_rows={str(v.candidate_vertex_id):v for v in reference_candidate.vertices}
    ref_pos={vid:tuple(map(float,row.P)) for vid,row in ref_rows.items()}
    ref_faces=[tuple(map(str,face)) for face in reference_candidate.faces]
    ref_incident=_incident_faces_by_vertex(ref_faces)

    protected={str(x) for x in protected_surface_ids}
    before=_report(faces,positions,policy)
    accepted=[]
    batch_rows=[]
    rejected=defaultdict(int)

    for batch_index in range(int(max_batches)):
        if len(accepted)>=int(max_moves):
            break

        incidence=_edge_incidence(faces)
        incident=_incident_faces_by_vertex(faces)
        metrics=[_metric(face,positions) for face in faces]
        violating={i for i,m in enumerate(metrics) if _violates(m,policy)}
        if not violating:
            break
        boundary_vertices={
            vid for edge,rows in incidence.items() if len(rows)==1 for vid in edge
        }

        seed_vertices=sorted({str(v) for fi in violating for v in faces[fi]})
        proposals=[]

        for vid in seed_vertices:
            if vid not in vertex_rows or vid not in ref_rows:
                rejected["missing_reference_vertex"]+=1
                continue
            if vid in boundary_vertices:
                rejected["boundary_vertex"]+=1
                continue
            if protected.intersection(
                str(sid) for sid,_ in vertex_rows[vid].support_binding.coefficients
            ):
                rejected["protected_support"]+=1
                continue

            patch_indices=set(incident.get(vid,set()))
            old_faces=tuple(faces[i] for i in sorted(patch_indices))
            oldq=_patch_quality(old_faces,positions,policy)
            if oldq is None or int(oldq["violation_count"])<=0:
                rejected["no_local_violation"]+=1
                continue

            local_vertices={str(x) for face in old_faces for x in face}
            scale=_local_scale(local_vertices,positions)
            if scale<=1e-12:
                rejected["scale"]+=1
                continue
            allowed=float(policy.g1_max_normal_refinement_ratio)*float(scale)
            current=np.asarray(positions[vid],dtype=np.float64)

            old_normals={}
            for fi in sorted(patch_indices):
                old_normals[fi]=_face_normal(faces[fi],positions)

            best=None
            ref_face_indices=sorted(ref_incident.get(vid,set()))
            if not ref_face_indices:
                rejected["no_reference_face"]+=1
                continue

            found_resolution=None
            for resolution in tuple(int(x) for x in grid_schedule):
                found_this_resolution=False
                for rfi in ref_face_indices:
                    ref_face=ref_faces[rfi]
                    if any(x not in ref_pos for x in ref_face):
                        continue
                    if len({ref_rows[x].component_id for x in ref_face})!=1:
                        continue
                    if ref_rows[vid].component_id!=vertex_rows[vid].component_id:
                        continue

                    tri=np.asarray([ref_pos[x] for x in ref_face],dtype=np.float64)
                    for bary in _bary_grid(resolution):
                        b=np.asarray(bary,dtype=np.float64)
                        point=tuple(map(float,b@tri))
                        displacement=float(np.linalg.norm(np.asarray(point)-current))
                        if displacement>allowed+1e-12:
                            continue

                        new_positions=dict(positions)
                        new_positions[vid]=point
                        newq=_patch_quality(old_faces,new_positions,policy)
                        if newq is None or int(newq["degenerate_count"])>0:
                            continue
                        if int(newq["violation_count"])!=0:
                            continue

                        orientation_ok=True
                        for fi in sorted(patch_indices):
                            oldn=old_normals[fi]
                            newn=_face_normal(faces[fi],new_positions)
                            if oldn is None or newn is None or float(np.dot(oldn,newn))<=0.0:
                                orientation_ok=False
                                break
                        if not orientation_ok:
                            rejected["orientation"]+=1
                            continue

                        deviation=_sampled_symmetric_local_deviation(
                            old_faces,old_faces,new_positions
                        )
                        deviation=max(float(deviation),displacement)
                        if deviation>allowed+1e-12:
                            rejected["g1_shape"]+=1
                            continue

                        support=_combine_support_bindings(
                            [ref_rows[x] for x in ref_face],
                            bary,
                        )
                        key=(
                            displacement,
                            -float(newq["min_angle_deg"]),
                            float(newq["max_aspect"]),
                            tuple(ref_face),
                            tuple(float(x) for x in bary),
                        )
                        row={
                            "vertex_id":vid,
                            "reference_face":tuple(ref_face),
                            "barycentric":tuple(float(x) for x in bary),
                            "point":point,
                            "support":support,
                            "displacement":displacement,
                            "allowed":allowed,
                            "oldq":oldq,
                            "newq":newq,
                            "local_vertices":frozenset(local_vertices),
                            "grid_resolution":resolution,
                            "deviation":deviation,
                        }
                        if best is None or key<best[0]:
                            best=(key,row)
                        found_this_resolution=True
                if found_this_resolution:
                    found_resolution=resolution
                    break

            if best is None:
                rejected["no_feasible_source_point"]+=1
                continue
            proposals.append(best[1])

        if not proposals:
            break

        proposals.sort(key=lambda row:(
            float(row["displacement"]),
            -float(row["newq"]["min_angle_deg"]),
            float(row["newq"]["max_aspect"]),
            str(row["vertex_id"]),
        ))
        selected=[]
        occupied=set()
        remaining=int(max_moves)-len(accepted)
        for row in proposals:
            if len(selected)>=remaining:
                break
            if occupied.intersection(row["local_vertices"]):
                rejected["batch_conflict"]+=1
                continue
            selected.append(row)
            occupied.update(row["local_vertices"])
        if not selected:
            break

        before_batch=_report(faces,positions,policy)
        for row in selected:
            vid=str(row["vertex_id"])
            old=vertex_rows[vid]
            positions[vid]=tuple(row["point"])
            vertex_rows[vid]=replace(
                old,
                P=tuple(row["point"]),
                support_binding=row["support"],
                metadata={
                    **dict(old.metadata or {}),
                    "source_surface_quality_optimization":{
                        "operator":"SOURCE_SURFACE_BARYCENTRIC_QUALITY_SEARCH_V1",
                        "reference_face":list(row["reference_face"]),
                        "barycentric":list(row["barycentric"]),
                        "grid_resolution":int(row["grid_resolution"]),
                    },
                },
            )

        after_batch=_report(faces,positions,policy)
        if int(after_batch["policy_violating_face_count"])>int(before_batch["policy_violating_face_count"]):
            raise QualificationError("QUALITY_SOURCE_OPT_GLOBAL_QUALITY_REGRESSION")
        topo=_manifold_report(faces)
        if not topo["passed"]:
            raise QualificationError(
                f"QUALITY_SOURCE_OPT_POSTBATCH_NONMANIFOLD:"
                f"{batch_index}:{topo['nonmanifold_edge_count']}:"
                f"{topo['illegal_vertex_link_count']}"
            )

        for row in selected:
            accepted.append({
                "batch_index":batch_index,
                "vertex_id":row["vertex_id"],
                "reference_face":row["reference_face"],
                "barycentric":row["barycentric"],
                "grid_resolution":row["grid_resolution"],
                "displacement":row["displacement"],
                "allowed_displacement":row["allowed"],
                "local_sampled_surface_deviation":row["deviation"],
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
            "before_violations":int(before_batch["policy_violating_face_count"]),
            "after_violations":int(after_batch["policy_violating_face_count"]),
            "topology":topo,
        })

    vertices_tuple=tuple(vertex_rows[vid] for vid in sorted(vertex_rows))
    after=_report(faces,positions,policy)
    final_topology=_manifold_report(faces)
    policy_hash=content_sha256({
        "schema":"RealSaS.SourceSurfaceQualityOptimizationPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "reference_candidate_lineage_hash":reference_candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "grid_schedule":tuple(int(x) for x in grid_schedule),
        "max_batches":int(max_batches),
        "max_moves":int(max_moves),
        "topology":"IMMUTABLE",
        "search_domain":"REFERENCE_INCIDENT_TRIANGLES_BARYCENTRIC_GRID",
        "selection":"MIN_DISPLACEMENT__MAX_MIN_ANGLE__MIN_MAX_ASPECT",
        "g1":"FROZEN_LOCAL_SCALE_DISPLACEMENT_AND_SAMPLED_DEVIATION",
        "product_authority":"AUDIT_ONLY__STAGE14_SOURCE_SURFACE_OPTIMIZATION_REQUIRED",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices_tuple,
        faces=tuple(faces),
        edges=candidate.edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.SourceSurfaceQualityOptimization.audit.v1",
        producer_policy_hash=policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "source_surface_quality_optimization":{
                "algorithm":"SOURCE_SURFACE_BARYCENTRIC_QUALITY_SEARCH_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "reference_candidate_lineage_hash":reference_candidate.candidate_lineage_hash,
                "audit_only":True,
                "product_authority_minted":False,
                "stage14_source_surface_optimization_required":True,
                "accepted_move_count":len(accepted),
                "batch_count":len(batch_rows),
                "batches":batch_rows,
                "rejected_counts":dict(sorted(rejected.items())),
                "before":before,
                "after":after,
                "initial_topology":topo0,
                "final_topology":final_topology,
                "accepted_moves":accepted,
            },
        },
    )
    result=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return result,dict(result.metadata["source_surface_quality_optimization"])
