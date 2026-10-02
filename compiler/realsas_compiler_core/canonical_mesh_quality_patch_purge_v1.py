from __future__ import annotations

"""Generic fixed-boundary multi-vertex patch remeshing quality repair.

Audit-only operator:
- starts from a violating face,
- expands a bounded face-adjacency patch,
- requires one simple boundary cycle,
- removes every unprotected interior vertex,
- retriangulates only the fixed surviving boundary vertices,
- forbids reuse of chords already owned outside the patch,
- requires strict/monotone G3 improvement and frozen G1 deviation,
- requires complete edge + vertex-link manifoldness after every batch,
- batches only pairwise-disjoint patches.

This intentionally does NOT mint product triangle authority. Product promotion
belongs in the Stage14 source-surface mechanical remesher, where remeshed faces
receive explicit source-surface support/provenance.
"""

from collections import defaultdict
from dataclasses import replace

from .canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_metric,_violates,_local_scale,
    _sampled_symmetric_local_deviation,_report,
)
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from .canonical_mesh_quality_cavity_remesh_v1 import (
    _best_polygon_triangulation,_average_patch_normal,_orient_triangles,
    _patch_quality,
)
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def _face_adjacency(faces):
    edge_faces=defaultdict(list)
    for fi,face in enumerate(faces):
        a,b,c=map(str,face)
        for edge in (_edge(a,b),_edge(b,c),_edge(c,a)):
            edge_faces[edge].append(fi)
    adj={i:set() for i in range(len(faces))}
    for rows in edge_faces.values():
        for a in rows:
            adj[a].update(b for b in rows if b!=a)
    return adj


def _expand(seed,adj,hops):
    seen={int(seed)}
    front={int(seed)}
    for _ in range(int(hops)):
        nxt=set()
        for fi in front:
            nxt.update(adj.get(fi,()))
        nxt-=seen
        seen|=nxt
        front=nxt
    return frozenset(seen)


def _boundary_cycle(faces,patch):
    counts=defaultdict(int)
    for fi in patch:
        a,b,c=map(str,faces[fi])
        for edge in (_edge(a,b),_edge(b,c),_edge(c,a)):
            counts[edge]+=1
    boundary=[edge for edge,n in counts.items() if int(n)==1]
    graph=defaultdict(set)
    for a,b in boundary:
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


def repair_candidate_patch_interior_purge_v1(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    protected_surface_ids: set[str] | frozenset[str] = frozenset(),
    max_hops: int = 5,
    max_patch_faces: int = 128,
    max_batches: int = 64,
    max_patches: int = 2048,
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_hops)<1 or int(max_patch_faces)<1 or int(max_batches)<1 or int(max_patches)<1:
        raise QualificationError("QUALITY_PATCH_PURGE_LIMIT_INVALID")

    vertex_rows={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,row.P)) for vid,row in vertex_rows.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_PATCH_PURGE_REQUIRES_FACES")

    initial_topology=_manifold_report(faces)
    if not initial_topology["passed"]:
        raise QualificationError(
            f"QUALITY_PATCH_PURGE_INPUT_NONMANIFOLD:"
            f"{initial_topology['nonmanifold_edge_count']}:"
            f"{initial_topology['illegal_vertex_link_count']}"
        )

    protected={str(x) for x in protected_surface_ids}
    before=_report(faces,positions,policy)
    accepted=[]
    batches=[]
    rejected=defaultdict(int)

    for batch_index in range(int(max_batches)):
        if len(accepted)>=int(max_patches):
            break

        metrics=[_metric(face,positions) for face in faces]
        violating={i for i,m in enumerate(metrics) if _violates(m,policy)}
        if not violating:
            break
        adj=_face_adjacency(faces)
        incidence=_edge_incidence(faces)
        proposals=[]

        for seed in sorted(violating):
            best=None
            for hops in range(1,int(max_hops)+1):
                patch=_expand(seed,adj,hops)
                if len(patch)>int(max_patch_faces):
                    rejected["patch_too_large"]+=1
                    continue
                cycle=_boundary_cycle(faces,patch)
                if cycle is None:
                    rejected["nondisk_patch"]+=1
                    continue
                old_faces=tuple(faces[i] for i in sorted(patch))
                patch_vertices={str(v) for face in old_faces for v in face}
                boundary=set(cycle)
                interior=sorted(patch_vertices-boundary)
                if not interior:
                    rejected["no_interior_vertex"]+=1
                    continue
                if any(
                    protected.intersection(
                        str(sid)
                        for sid,_ in vertex_rows[v].support_binding.coefficients
                    )
                    for v in interior
                ):
                    rejected["protected_interior"]+=1
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
                        if any(fi not in patch for fi in rows):
                            forbidden.add(chord)

                tri=_best_polygon_triangulation(
                    cycle,positions,policy,frozenset(forbidden)
                )
                if tri is None:
                    rejected["no_triangulation"]+=1
                    continue
                _,_,_,raw_faces=tri
                new_faces=_orient_triangles(
                    raw_faces,positions,_average_patch_normal(old_faces,positions)
                )
                oldq=_patch_quality(old_faces,positions,policy)
                newq=_patch_quality(new_faces,positions,policy)
                if newq is None or int(newq["degenerate_count"])>0:
                    rejected["degenerate"]+=1
                    continue
                # This operator is reserved for closure patches. It must remove
                # every quality violation in the selected cavity.
                if int(newq["violation_count"])!=0:
                    rejected["residual_quality"]+=1
                    continue
                monotone=(
                    float(newq["min_angle_deg"])+1e-9>=float(oldq["min_angle_deg"])
                    and float(newq["max_aspect"])<=float(oldq["max_aspect"])+1e-9
                )
                if not monotone:
                    rejected["companion_regression"]+=1
                    continue

                scale=_local_scale(patch_vertices,positions)
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
                    if i not in patch
                ]
                trial.extend(new_faces)
                if len(set(trial))!=len(trial):
                    rejected["duplicate_face"]+=1
                    continue
                topology=_manifold_report(trial)
                if not topology["passed"]:
                    rejected["topology"]+=1
                    continue

                covered=sum(fi in violating for fi in patch)
                row={
                    "seed_face":int(seed),
                    "hops":int(hops),
                    "patch_indices":patch,
                    "patch_vertices":frozenset(patch_vertices),
                    "interior_vertices":tuple(interior),
                    "old_faces":old_faces,
                    "new_faces":tuple(new_faces),
                    "old_quality":oldq,
                    "new_quality":newq,
                    "deviation":float(deviation),
                    "allowed":float(allowed),
                    "covered_bad_faces":int(covered),
                    "boundary_vertex_count":len(cycle),
                }
                key=(
                    -int(covered),
                    len(interior),
                    -float(newq["min_angle_deg"]),
                    float(newq["max_aspect"]),
                    float(deviation),
                    int(hops),
                    int(seed),
                )
                if best is None or key<best[0]:
                    best=(key,row)
            if best is not None:
                proposals.append(best[1])

        if not proposals:
            break

        proposals.sort(key=lambda row:(
            -int(row["covered_bad_faces"]),
            len(row["interior_vertices"]),
            -float(row["new_quality"]["min_angle_deg"]),
            float(row["new_quality"]["max_aspect"]),
            float(row["deviation"]),
            int(row["hops"]),
            int(row["seed_face"]),
        ))
        selected=[]
        occupied=set()
        remaining=int(max_patches)-len(accepted)
        for row in proposals:
            if len(selected)>=remaining:
                break
            if occupied.intersection(row["patch_vertices"]):
                rejected["batch_conflict"]+=1
                continue
            selected.append(row)
            occupied.update(row["patch_vertices"])
        if not selected:
            break

        remove_indices=set()
        replacement=[]
        removed_vertices=set()
        for row in selected:
            remove_indices.update(row["patch_indices"])
            replacement.extend(row["new_faces"])
            removed_vertices.update(row["interior_vertices"])

        next_faces=[
            face for i,face in enumerate(faces)
            if i not in remove_indices
        ]
        next_faces.extend(replacement)
        if len(set(next_faces))!=len(next_faces):
            raise QualificationError("QUALITY_PATCH_PURGE_BATCH_DUPLICATE_FACE")

        before_batch=_report(faces,positions,policy)
        faces=next_faces
        used_after={str(v) for face in faces for v in face}
        for vid in sorted(removed_vertices):
            if vid not in used_after:
                vertex_rows.pop(vid,None)
                positions.pop(vid,None)

        topology=_manifold_report(faces)
        if not topology["passed"]:
            raise QualificationError(
                f"QUALITY_PATCH_PURGE_POSTBATCH_NONMANIFOLD:"
                f"{batch_index}:{topology['nonmanifold_edge_count']}:"
                f"{topology['illegal_vertex_link_count']}"
            )
        after_batch=_report(faces,positions,policy)
        if int(after_batch["policy_violating_face_count"])>int(before_batch["policy_violating_face_count"]):
            raise QualificationError("QUALITY_PATCH_PURGE_GLOBAL_QUALITY_REGRESSION")

        for row in selected:
            accepted.append({
                "batch_index":int(batch_index),
                "seed_face":row["seed_face"],
                "hops":row["hops"],
                "covered_bad_faces":row["covered_bad_faces"],
                "old_face_count":len(row["old_faces"]),
                "new_face_count":len(row["new_faces"]),
                "removed_interior_vertex_count":len(row["interior_vertices"]),
                "boundary_vertex_count":row["boundary_vertex_count"],
                "old_violation_count":row["old_quality"]["violation_count"],
                "new_violation_count":row["new_quality"]["violation_count"],
                "old_min_angle_deg":row["old_quality"]["min_angle_deg"],
                "new_min_angle_deg":row["new_quality"]["min_angle_deg"],
                "old_max_aspect":row["old_quality"]["max_aspect"],
                "new_max_aspect":row["new_quality"]["max_aspect"],
                "local_sampled_surface_deviation":row["deviation"],
                "allowed_local_deviation":row["allowed"],
            })
        batches.append({
            "batch_index":int(batch_index),
            "accepted_patch_count":len(selected),
            "removed_face_count":len(remove_indices),
            "added_face_count":len(replacement),
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
        for face in faces_tuple
        for i,j in ((0,1),(1,2),(2,0))
    }))
    after=_report(faces_tuple,positions,policy)
    final_topology=_manifold_report(faces_tuple)

    policy_hash=content_sha256({
        "schema":"RealSaS.PatchInteriorPurgeQualityRepairPolicy.v1",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_hops":int(max_hops),
        "max_patch_faces":int(max_patch_faces),
        "max_batches":int(max_batches),
        "max_patches":int(max_patches),
        "boundary_rule":"FIXED_SIMPLE_CYCLE",
        "interior_rule":"REMOVE_UNPROTECTED_INTERIOR_VERTICES",
        "topology_rule":"NO_EXTERNAL_CHORD_REUSE__FULL_MANIFOLD_POST_BATCH",
        "geometry_rule":"SURVIVING_VERTEX_POSITIONS_IMMUTABLE__FROZEN_G1_SAMPLED_DEVIATION",
        "quality_rule":"PATCH_MUST_CLOSE_ALL_G3_VIOLATIONS",
        "provenance_claim":"AUDIT_ONLY__STAGE14_SOURCE_SURFACE_REMESH_AUTHORITY_REQUIRED_FOR_PRODUCT_PROMOTION",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=vertices_tuple,
        faces=faces_tuple,
        edges=edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.PatchInteriorPurgeQualityRepair.v1",
        producer_policy_hash=policy_hash,
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "patch_interior_purge_quality_repair":{
                "algorithm":"FIXED_BOUNDARY_MULTI_VERTEX_INTERIOR_PURGE_V1",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "audit_only":True,
                "product_triangle_authority_minted":False,
                "stage14_surface_remesh_authority_required_for_product_promotion":True,
                "accepted_patch_count":len(accepted),
                "batch_count":len(batches),
                "batches":batches,
                "rejected_counts":dict(sorted(rejected.items())),
                "before":before,
                "after":after,
                "initial_topology":initial_topology,
                "final_topology":final_topology,
                "accepted_patches":accepted,
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
    return result,dict(result.metadata["patch_interior_purge_quality_repair"])
