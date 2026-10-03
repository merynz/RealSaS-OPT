from __future__ import annotations

"""Generic local cavity retriangulation for residual static mesh-quality defects.

The operator removes an interior vertex whose one-ring is a topological disk and
retriangulates the fixed one-ring boundary using only surviving vertices.  It:
- requires edge + vertex-link manifold input,
- preserves the cavity boundary and all surviving vertex positions/supports,
- forbids internal chords already used outside the cavity,
- chooses the lexicographically best triangulation under frozen G3,
- requires strict local quality improvement with no companion-metric regression,
- requires sampled symmetric G1 surface deviation within the frozen budget,
- batches only pairwise-disjoint cavities,
- rechecks full manifoldness and global quality after every batch.

No product provenance claim is minted here: a Stage14 mechanical-remesh authority
must explicitly bind any retriangulated faces to source-surface support before
promotion.
"""

import functools
from collections import defaultdict
from dataclasses import replace

import numpy as np

from .canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_incident_faces_by_vertex,_metric,_violates,
    _local_scale,_sampled_symmetric_local_deviation,_report,
)
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def _vertex_link_cycle(vertex_id: str, incident_faces):
    v=str(vertex_id)
    graph=defaultdict(set)
    for face in incident_faces:
        row=[str(x) for x in face]
        if v not in row:
            continue
        others=[x for x in row if x!=v]
        if len(others)!=2:
            return None
        a,b=others
        graph[a].add(b);graph[b].add(a)
    if len(graph)<3 or any(len(nbs)!=2 for nbs in graph.values()):
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
    return tuple(cycle) if len(cycle)==len(graph) else None


def _patch_quality(faces,positions,policy):
    metrics=tuple(_metric(face,positions) for face in faces)
    if not metrics:
        return None
    return {
        "violation_count":sum(_violates(m,policy) for m in metrics),
        "min_angle_deg":min(float(m["min_angle_deg"]) for m in metrics),
        "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in metrics),
        "degenerate_count":sum(bool(m["degenerate"]) for m in metrics),
    }


def _average_patch_normal(faces,positions):
    total=np.zeros(3,dtype=np.float64)
    for face in faces:
        a,b,c=(np.asarray(positions[str(v)],dtype=np.float64) for v in face)
        total+=np.cross(b-a,c-a)
    norm=float(np.linalg.norm(total))
    if norm<=1e-12:
        return None
    return total/norm


def _orient_triangles(triangles,positions,target_normal):
    if target_normal is None:
        return tuple(tuple(map(str,t)) for t in triangles)
    out=[]
    for tri in triangles:
        a,b,c=(np.asarray(positions[str(v)],dtype=np.float64) for v in tri)
        normal=np.cross(b-a,c-a)
        if float(np.dot(normal,target_normal))<0.0:
            out.append((str(tri[0]),str(tri[2]),str(tri[1])))
        else:
            out.append(tuple(map(str,tri)))
    return tuple(out)


def _best_polygon_triangulation(cycle,positions,policy,forbidden_chords):
    cycle=tuple(map(str,cycle))
    n=len(cycle)
    if n<3:
        return None

    @functools.lru_cache(None)
    def solve(i,j):
        if j-i<2:
            return (0,float("inf"),0.0,())
        best=None
        for k in range(i+1,j):
            chords=[]
            if k>i+1:
                chords.append(_edge(cycle[i],cycle[k]))
            if j>k+1:
                chords.append(_edge(cycle[k],cycle[j]))
            if any(chord in forbidden_chords for chord in chords):
                continue
            left=solve(i,k)
            right=solve(k,j)
            if left is None or right is None:
                continue
            tri=(cycle[i],cycle[k],cycle[j])
            metric=_metric(tri,positions)
            if bool(metric["degenerate"]):
                continue
            violation=int(_violates(metric,policy))
            total_violation=int(left[0]+right[0]+violation)
            min_angle=min(float(left[1]),float(right[1]),float(metric["min_angle_deg"]))
            max_aspect=max(
                float(left[2]),float(right[2]),
                float(metric["aspect_longest_over_min_altitude"]),
            )
            triangles=left[3]+right[3]+(tri,)
            key=(total_violation,-min_angle,max_aspect,tuple(sorted(triangles)))
            if best is None or key<best[0]:
                best=(key,(total_violation,min_angle,max_aspect,triangles))
        return None if best is None else best[1]

    return solve(0,n-1)


def repair_candidate_cavity_retriangulation_v1(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    protected_surface_ids: set[str] | frozenset[str] = frozenset(),
    max_batches: int = 64,
    max_removed_vertices: int = 2048,
    proposal_admissibility=None,
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_batches)<1 or int(max_removed_vertices)<1:
        raise QualificationError("QUALITY_CAVITY_REMESH_LIMIT_INVALID")

    vertex_rows={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,row.P)) for vid,row in vertex_rows.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_CAVITY_REMESH_REQUIRES_FACES")

    initial_topology=_manifold_report(faces)
    if not initial_topology["passed"]:
        raise QualificationError(
            f"QUALITY_CAVITY_REMESH_INPUT_NONMANIFOLD:"
            f"{initial_topology['nonmanifold_edge_count']}:"
            f"{initial_topology['illegal_vertex_link_count']}"
        )

    protected={str(x) for x in protected_surface_ids}
    before=_report(faces,positions,policy)
    accepted=[]
    batch_rows=[]
    rejected=defaultdict(int)

    for batch_index in range(int(max_batches)):
        if len(accepted)>=int(max_removed_vertices):
            break

        incidence=_edge_incidence(faces)
        incident=_incident_faces_by_vertex(faces)
        metrics=[_metric(face,positions) for face in faces]
        violating={i for i,m in enumerate(metrics) if _violates(m,policy)}
        if not violating:
            break
        boundary_vertices={vid for edge,rows in incidence.items() if len(rows)==1 for vid in edge}
        seed_vertices=sorted({str(v) for fi in violating for v in faces[fi]})
        proposals=[]

        for vid in seed_vertices:
            if vid not in vertex_rows:
                continue
            if vid in boundary_vertices:
                rejected["boundary_vertex"]+=1
                continue
            if protected.intersection(
                str(sid) for sid,_ in vertex_rows[vid].support_binding.coefficients
            ):
                rejected["protected_support"]+=1
                continue

            cavity_indices=set(incident.get(vid,set()))
            old_faces=tuple(faces[i] for i in sorted(cavity_indices))
            cycle=_vertex_link_cycle(vid,old_faces)
            if cycle is None:
                rejected["nondisk_link"]+=1
                continue
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
                    edge=_edge(cycle[i],cycle[j])
                    if edge in boundary_edges:
                        continue
                    rows=incidence.get(edge,())
                    if any(fi not in cavity_indices for fi in rows):
                        forbidden.add(edge)

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

            trial=[face for i,face in enumerate(faces) if i not in cavity_indices]
            trial.extend(new_faces)
            if len(set(trial))!=len(trial):
                rejected["duplicate_face"]+=1
                continue
            topology=_manifold_report(trial)
            if not topology["passed"]:
                rejected["topology"]+=1
                continue

            proposal={
                "operator":"vertex_cavity",
                "removed_vertex_id":vid,
                "cavity_indices":frozenset(cavity_indices),
                "cavity_vertices":frozenset(local_vertices),
                "old_faces":old_faces,
                "new_faces":new_faces,
                "old_quality":oldq,
                "new_quality":newq,
                "deviation":float(deviation),
                "allowed":float(allowed),
                "valence":len(cycle),
                "bad_faces_in_cavity":sum(fi in violating for fi in cavity_indices),
            }
            if proposal_admissibility is not None and not bool(proposal_admissibility(proposal)):
                rejected["mechanical_admissibility"]+=1
                continue
            proposals.append(proposal)

        if not proposals:
            break

        proposals.sort(key=lambda row:(
            -int(row["old_quality"]["violation_count"]-row["new_quality"]["violation_count"]),
            -float(row["new_quality"]["min_angle_deg"]-row["old_quality"]["min_angle_deg"]),
            float(row["new_quality"]["max_aspect"]),
            float(row["deviation"]),
            str(row["removed_vertex_id"]),
        ))

        selected=[]
        occupied_vertices=set()
        remaining=int(max_removed_vertices)-len(accepted)
        for row in proposals:
            if len(selected)>=remaining:
                break
            if occupied_vertices.intersection(row["cavity_vertices"]):
                rejected["batch_conflict"]+=1
                continue
            selected.append(row)
            occupied_vertices.update(row["cavity_vertices"])
        if not selected:
            break

        remove_face_indices=set()
        replacement_faces=[]
        for row in selected:
            remove_face_indices.update(row["cavity_indices"])
            replacement_faces.extend(row["new_faces"])
        next_faces=[
            face for i,face in enumerate(faces)
            if i not in remove_face_indices
        ]
        next_faces.extend(replacement_faces)
        if len(set(next_faces))!=len(next_faces):
            raise QualificationError("QUALITY_CAVITY_REMESH_BATCH_DUPLICATE_FACE")

        before_batch=_report(faces,positions,policy)
        faces=next_faces
        for row in selected:
            vid=str(row["removed_vertex_id"])
            vertex_rows.pop(vid,None)
            positions.pop(vid,None)

        topology=_manifold_report(faces)
        if not topology["passed"]:
            raise QualificationError(
                f"QUALITY_CAVITY_REMESH_POSTBATCH_NONMANIFOLD:"
                f"{batch_index}:{topology['nonmanifold_edge_count']}:"
                f"{topology['illegal_vertex_link_count']}"
            )
        after_batch=_report(faces,positions,policy)
        if int(after_batch["policy_violating_face_count"])>int(before_batch["policy_violating_face_count"]):
            raise QualificationError("QUALITY_CAVITY_REMESH_GLOBAL_QUALITY_REGRESSION")

        for row in selected:
            accepted.append({
                "batch_index":batch_index,
                "removed_vertex_id":row["removed_vertex_id"],
                "valence":row["valence"],
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
            })
        batch_rows.append({
            "batch_index":batch_index,
            "accepted_cavity_count":len(selected),
            "removed_face_count":len(remove_face_indices),
            "added_face_count":len(replacement_faces),
            "before_violations":int(before_batch["policy_violating_face_count"]),
            "after_violations":int(after_batch["policy_violating_face_count"]),
            "topology":topology,
        })

    faces_tuple=tuple(sorted(tuple(map(str,face)) for face in faces))
    used_ids={str(v) for face in faces_tuple for v in face}
    vertices_tuple=tuple(
        row for vid,row in sorted(vertex_rows.items())
        if vid in used_ids
    )
    edges=tuple(sorted({
        _edge(face[i],face[j])
        for face in faces_tuple for i,j in ((0,1),(1,2),(2,0))
    }))
    after=_report(faces_tuple,positions,policy)
    final_topology=_manifold_report(faces_tuple)

    policy_hash=content_sha256({
        "schema":"RealSaS.CavityRetriangulationQualityRepairPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_batches":int(max_batches),
        "max_removed_vertices":int(max_removed_vertices),
        "boundary_rule":"FIXED_ONE_RING_CYCLE__BOUNDARY_VERTEX_REMOVAL_FORBIDDEN",
        "topology_rule":"DISK_LINK_ONLY__NO_EXTERNAL_INTERNAL_CHORD_REUSE__FULL_MANIFOLD_POST_BATCH",
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
        producer_id="RealSaS.CanonicalMesh.CavityRetriangulationQualityRepair.v1",
        producer_policy_hash=policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "cavity_retriangulation_quality_repair":{
                "algorithm":"FIXED_BOUNDARY_VERTEX_REMOVAL_CAVITY_RETRIANGULATION_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "audit_only":True,
                "product_triangle_authority_minted":False,
                "stage14_surface_remesh_authority_required_for_product_promotion":True,
                "accepted_cavity_count":len(accepted),
                "batch_count":len(batch_rows),
                "batches":batch_rows,
                "rejected_counts":dict(sorted(rejected.items())),
                "before":before,
                "after":after,
                "initial_topology":initial_topology,
                "final_topology":final_topology,
                "accepted_cavities":accepted,
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
    return result,dict(result.metadata["cavity_retriangulation_quality_repair"])
