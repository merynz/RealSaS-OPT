from __future__ import annotations

"""Generic interior-edge cavity retriangulation for residual mesh-quality defects.

This audit operator removes both endpoints of a problematic interior edge and
retriangulates the fixed boundary of the union of their incident one-rings.
It is stronger than endpoint collapse because it does not force either endpoint
to survive. It is still conservative:
- input and post-batch edge/vertex-link manifoldness are required,
- both removed endpoints must be interior, same mechanical component, and
  unprotected,
- the cavity boundary must be one simple cycle,
- surviving vertex positions/supports are immutable,
- external chord reuse is forbidden,
- G3 must improve monotonically and strictly,
- sampled symmetric G1 deviation must stay within the frozen budget,
- pairwise-disjoint cavities may be batched,
- no product triangle authority is minted in this audit operator.
"""

from collections import defaultdict
from dataclasses import replace

from .canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_incident_faces_by_vertex,_metric,_violates,
    _local_scale,_sampled_symmetric_local_deviation,_report,
)
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from .canonical_mesh_quality_cavity_remesh_v1 import (
    _patch_quality,_average_patch_normal,_orient_triangles,
    _best_polygon_triangulation,
)
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def _boundary_cycle_for_patch(faces, patch_indices):
    counts=defaultdict(int)
    for fi in patch_indices:
        a,b,c=map(str,faces[fi])
        for edge in (_edge(a,b),_edge(b,c),_edge(c,a)):
            counts[edge]+=1
    boundary=[edge for edge,n in counts.items() if int(n)==1]
    if len(boundary)<3:
        return None
    graph=defaultdict(set)
    for a,b in boundary:
        graph[a].add(b);graph[b].add(a)
    if any(len(rows)!=2 for rows in graph.values()):
        return None
    start=min(graph)
    cycle=[start]
    prev=None
    cur=start
    while True:
        nbs=sorted(graph[cur])
        nxt=nbs[0] if nbs[0]!=prev else nbs[1]
        if nxt==start:
            break
        if nxt in cycle:
            return None
        cycle.append(nxt)
        prev,cur=cur,nxt
        if len(cycle)>len(graph):
            return None
    if len(cycle)!=len(graph):
        return None
    return tuple(cycle)


def repair_candidate_edge_cavity_retriangulation_v1(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    protected_surface_ids: set[str] | frozenset[str] = frozenset(),
    max_batches: int = 64,
    max_removed_edges: int = 2048,
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_batches)<1 or int(max_removed_edges)<1:
        raise QualificationError("QUALITY_EDGE_CAVITY_LIMIT_INVALID")

    vertex_rows={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,row.P)) for vid,row in vertex_rows.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_EDGE_CAVITY_REQUIRES_FACES")

    initial_topology=_manifold_report(faces)
    if not initial_topology["passed"]:
        raise QualificationError(
            f"QUALITY_EDGE_CAVITY_INPUT_NONMANIFOLD:"
            f"{initial_topology['nonmanifold_edge_count']}:"
            f"{initial_topology['illegal_vertex_link_count']}"
        )

    protected={str(x) for x in protected_surface_ids}
    before=_report(faces,positions,policy)
    accepted=[]
    batch_rows=[]
    rejected=defaultdict(int)

    for batch_index in range(int(max_batches)):
        if len(accepted)>=int(max_removed_edges):
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
        candidate_edges=sorted({
            _edge(faces[fi][i],faces[fi][j])
            for fi in violating
            for i,j in ((0,1),(1,2),(2,0))
        })
        proposals=[]

        for edge in candidate_edges:
            u,v=edge
            if u not in vertex_rows or v not in vertex_rows:
                continue
            if u in boundary_vertices or v in boundary_vertices:
                rejected["boundary_endpoint"]+=1
                continue
            if vertex_rows[u].component_id!=vertex_rows[v].component_id:
                rejected["component_mismatch"]+=1
                continue
            if protected.intersection(
                str(sid)
                for endpoint in (u,v)
                for sid,_ in vertex_rows[endpoint].support_binding.coefficients
            ):
                rejected["protected_support"]+=1
                continue
            if len(incidence.get(edge,()))!=2:
                rejected["not_interior_edge"]+=1
                continue

            cavity_indices=set(incident.get(u,set()))|set(incident.get(v,set()))
            if not cavity_indices:
                rejected["empty_cavity"]+=1
                continue
            cycle=_boundary_cycle_for_patch(faces,cavity_indices)
            if cycle is None:
                rejected["nondisk_boundary"]+=1
                continue
            if u in cycle or v in cycle:
                rejected["endpoint_on_cavity_boundary"]+=1
                continue

            old_faces=tuple(faces[i] for i in sorted(cavity_indices))
            oldq=_patch_quality(old_faces,positions,policy)
            if oldq is None or int(oldq["violation_count"])<=0:
                rejected["no_local_violation"]+=1
                continue

            boundary_edges={
                _edge(cycle[i],cycle[(i+1)%len(cycle)])
                for i in range(len(cycle))
            }
            forbidden=set()
            for i in range(len(cycle)):
                for j in range(i+1,len(cycle)):
                    chord=_edge(cycle[i],cycle[j])
                    if chord in boundary_edges:
                        continue
                    rows=incidence.get(chord,())
                    if any(fi not in cavity_indices for fi in rows):
                        forbidden.add(chord)

            best=_best_polygon_triangulation(
                cycle,positions,policy,frozenset(forbidden)
            )
            if best is None:
                rejected["no_triangulation"]+=1
                continue
            _,_,_,raw_faces=best
            new_faces=_orient_triangles(
                raw_faces,positions,_average_patch_normal(old_faces,positions)
            )
            newq=_patch_quality(new_faces,positions,policy)
            if newq is None or int(newq["degenerate_count"])>0:
                rejected["degenerate"]+=1
                continue

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
                rejected["quality"]+=1
                continue

            local_vertices={str(x) for face in old_faces for x in face}
            scale=_local_scale(local_vertices,positions)
            if scale<=1e-10:
                rejected["scale"]+=1
                continue
            deviation=_sampled_symmetric_local_deviation(
                old_faces,new_faces,positions
            )
            allowed=float(policy.g1_max_normal_refinement_ratio)*scale
            if deviation>allowed+1e-12:
                rejected["g1_shape"]+=1
                continue

            trial=[
                face for i,face in enumerate(faces)
                if i not in cavity_indices
            ]
            trial.extend(new_faces)
            if len(set(trial))!=len(trial):
                rejected["duplicate_face"]+=1
                continue
            topology=_manifold_report(trial)
            if not topology["passed"]:
                rejected["topology"]+=1
                continue

            edge_length=sum(
                (float(positions[u][k])-float(positions[v][k]))**2
                for k in range(3)
            )**0.5
            proposals.append({
                "edge":edge,
                "cavity_indices":frozenset(cavity_indices),
                "cavity_vertices":frozenset(local_vertices),
                "old_faces":old_faces,
                "new_faces":new_faces,
                "old_quality":oldq,
                "new_quality":newq,
                "deviation":float(deviation),
                "allowed":float(allowed),
                "boundary_size":len(cycle),
                "edge_length":float(edge_length),
                "bad_faces_in_cavity":sum(fi in violating for fi in cavity_indices),
            })

        if not proposals:
            break

        proposals.sort(key=lambda row:(
            -int(row["old_quality"]["violation_count"]-row["new_quality"]["violation_count"]),
            -float(row["new_quality"]["min_angle_deg"]-row["old_quality"]["min_angle_deg"]),
            float(row["new_quality"]["max_aspect"]),
            float(row["deviation"]),
            float(row["edge_length"]),
            row["edge"],
        ))

        selected=[]
        occupied=set()
        remaining=int(max_removed_edges)-len(accepted)
        for row in proposals:
            if len(selected)>=remaining:
                break
            if occupied.intersection(row["cavity_vertices"]):
                rejected["batch_conflict"]+=1
                continue
            selected.append(row)
            occupied.update(row["cavity_vertices"])
        if not selected:
            break

        remove_face_indices=set()
        replacement_faces=[]
        removed_vertices=set()
        for row in selected:
            remove_face_indices.update(row["cavity_indices"])
            replacement_faces.extend(row["new_faces"])
            removed_vertices.update(row["edge"])

        next_faces=[
            face for i,face in enumerate(faces)
            if i not in remove_face_indices
        ]
        next_faces.extend(replacement_faces)
        if len(set(next_faces))!=len(next_faces):
            raise QualificationError("QUALITY_EDGE_CAVITY_BATCH_DUPLICATE_FACE")

        before_batch=_report(faces,positions,policy)
        faces=next_faces
        for vid in sorted(removed_vertices):
            vertex_rows.pop(str(vid),None)
            positions.pop(str(vid),None)

        topology=_manifold_report(faces)
        if not topology["passed"]:
            raise QualificationError(
                f"QUALITY_EDGE_CAVITY_POSTBATCH_NONMANIFOLD:"
                f"{batch_index}:{topology['nonmanifold_edge_count']}:"
                f"{topology['illegal_vertex_link_count']}"
            )
        after_batch=_report(faces,positions,policy)
        if int(after_batch["policy_violating_face_count"])>int(before_batch["policy_violating_face_count"]):
            raise QualificationError("QUALITY_EDGE_CAVITY_GLOBAL_QUALITY_REGRESSION")

        for row in selected:
            accepted.append({
                "batch_index":batch_index,
                "removed_edge":row["edge"],
                "boundary_size":row["boundary_size"],
                "bad_faces_in_cavity":row["bad_faces_in_cavity"],
                "old_face_count":len(row["old_faces"]),
                "new_face_count":len(row["new_faces"]),
                "old_violation_count":row["old_quality"]["violation_count"],
                "new_violation_count":row["new_quality"]["violation_count"],
                "old_min_angle_deg":row["old_quality"]["min_angle_deg"],
                "new_min_angle_deg":row["new_quality"]["min_angle_deg"],
                "old_max_aspect":row["old_quality"]["max_aspect"],
                "new_max_aspect":row["new_quality"]["max_aspect"],
                "local_sampled_surface_deviation":row["deviation"],
                "allowed_local_deviation":row["allowed"],
                "edge_length":row["edge_length"],
            })
        batch_rows.append({
            "batch_index":batch_index,
            "accepted_edge_cavity_count":len(selected),
            "removed_vertex_count":len(removed_vertices),
            "removed_face_count":len(remove_face_indices),
            "added_face_count":len(replacement_faces),
            "before_violations":int(before_batch["policy_violating_face_count"]),
            "after_violations":int(after_batch["policy_violating_face_count"]),
            "topology":topology,
        })

    faces_tuple=tuple(sorted(tuple(map(str,face)) for face in faces))
    used_ids={str(v) for face in faces_tuple for v in face}
    vertices_tuple=tuple(
        row for vid,row in sorted(vertex_rows.items()) if vid in used_ids
    )
    edges=tuple(sorted({
        _edge(face[i],face[j])
        for face in faces_tuple for i,j in ((0,1),(1,2),(2,0))
    }))
    after=_report(faces_tuple,positions,policy)
    final_topology=_manifold_report(faces_tuple)
    policy_hash=content_sha256({
        "schema":"RealSaS.EdgeCavityRetriangulationQualityRepairPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_batches":int(max_batches),
        "max_removed_edges":int(max_removed_edges),
        "boundary_rule":"FIXED_COMBINED_ONE_RING_SIMPLE_CYCLE__BOUNDARY_ENDPOINT_REMOVAL_FORBIDDEN",
        "topology_rule":"INTERIOR_EDGE_ONLY__NO_EXTERNAL_CHORD_REUSE__FULL_MANIFOLD_POST_BATCH",
        "geometry_rule":"SURVIVING_VERTEX_POSITIONS_IMMUTABLE__FROZEN_G1_SAMPLED_DEVIATION",
        "selection":"MIN_VIOLATIONS__MAX_MIN_ANGLE__MIN_MAX_ASPECT",
        "provenance_claim":"AUDIT_ONLY__STAGE14_SOURCE_SURFACE_REMESH_AUTHORITY_REQUIRED_FOR_PRODUCT_PROMOTION",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices_tuple,
        faces=faces_tuple,
        edges=edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.EdgeCavityRetriangulationQualityRepair.v1",
        producer_policy_hash=policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "edge_cavity_retriangulation_quality_repair":{
                "algorithm":"INTERIOR_EDGE_COMBINED_STAR_CAVITY_RETRIANGULATION_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "audit_only":True,
                "product_triangle_authority_minted":False,
                "stage14_surface_remesh_authority_required_for_product_promotion":True,
                "accepted_edge_cavity_count":len(accepted),
                "batch_count":len(batch_rows),
                "batches":batch_rows,
                "rejected_counts":dict(sorted(rejected.items())),
                "before":before,
                "after":after,
                "initial_topology":initial_topology,
                "final_topology":final_topology,
                "accepted_edge_cavities":accepted,
                "surviving_vertex_motion":False,
                "surviving_support_binding_changed":False,
                "new_vertex_creation":False,
            },
        },
    )
    result=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return result,dict(result.metadata["edge_cavity_retriangulation_quality_repair"])
