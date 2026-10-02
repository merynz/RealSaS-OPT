from __future__ import annotations

"""Generic topology-safe fixed-vertex edge-flip quality repair.

Compared with v1, flips in one batch must have pairwise-disjoint four-vertex
patches, and full edge + vertex-link manifoldness is checked before repair and
after every pass. This keeps the fixed-vertex/G1/G3 semantics while making the
batching topology-safe.
"""

from collections import defaultdict
from dataclasses import replace

from .canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_opposite,_pair_quality,_local_scale,
    _sampled_symmetric_local_deviation,_report,_EPS,
)
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def _vertex_link_legal(faces) -> tuple[int,list[str]]:
    links=defaultdict(lambda:defaultdict(set))
    incident=defaultdict(int)
    for face in faces:
        a,b,c=map(str,face)
        for v,x,y in ((a,b,c),(b,a,c),(c,a,b)):
            incident[v]+=1
            links[v][x].add(y)
            links[v][y].add(x)
    bad=[]
    for v,g in links.items():
        nodes=sorted(g)
        deg={n:len(g[n]) for n in nodes}
        seen=set(); comps=0
        for n in nodes:
            if n in seen: continue
            comps+=1
            st=[n];seen.add(n)
            while st:
                x=st.pop()
                for y in g[x]:
                    if y not in seen:
                        seen.add(y);st.append(y)
        d1=sum(d==1 for d in deg.values())
        cycle=(comps==1 and all(d==2 for d in deg.values()))
        path=(comps==1 and d1==2 and all(d in (1,2) for d in deg.values()))
        if not (cycle or path):
            bad.append(v)
    return len(bad),sorted(bad)


def _manifold_report(faces) -> dict:
    inc=_edge_incidence(faces)
    bad_edges=[edge for edge,rows in inc.items() if len(rows)>2]
    bad_vertex_count,bad_vertices=_vertex_link_legal(faces)
    return {
        "nonmanifold_edge_count":len(bad_edges),
        "illegal_vertex_link_count":bad_vertex_count,
        "bad_edge_examples":bad_edges[:16],
        "bad_vertex_examples":bad_vertices[:16],
        "passed":bool(not bad_edges and bad_vertex_count==0),
    }


def repair_candidate_fixed_vertex_flips_topology_safe_v2(
    candidate: CanonicalMeshCandidateIR,
    policy: MeshQualificationPolicyIR,
    *,
    max_passes: int = 12,
) -> tuple[CanonicalMeshCandidateIR,dict]:
    if int(max_passes)<1:
        raise QualificationError("QUALITY_TOPO_SAFE_FLIP_MAX_PASSES_INVALID")
    vertices={str(v.candidate_vertex_id):v for v in candidate.vertices}
    positions={vid:tuple(map(float,v.P)) for vid,v in vertices.items()}
    faces=[tuple(map(str,face)) for face in candidate.faces]
    if not faces:
        raise QualificationError("QUALITY_TOPO_SAFE_FLIP_REQUIRES_FACES")

    initial_topology=_manifold_report(faces)
    if not initial_topology["passed"]:
        raise QualificationError(
            f"QUALITY_TOPO_SAFE_FLIP_INPUT_NONMANIFOLD:"
            f"{initial_topology['nonmanifold_edge_count']}:"
            f"{initial_topology['illegal_vertex_link_count']}"
        )

    before=_report(faces,positions,policy)
    accepted=[]
    pass_rows=[]
    rejected_shape=0
    rejected_topology=0
    rejected_batch_conflict=0

    for pass_index in range(int(max_passes)):
        inc=_edge_incidence(faces)
        proposals=[]
        existing_edges=set(inc)

        for edge,adj in sorted(inc.items()):
            if len(adj)!=2:
                continue
            i,j=adj
            f0=faces[i];f1=faces[j]
            c=_opposite(f0,edge);d=_opposite(f1,edge)
            a,b=edge
            quad=frozenset((a,b,c,d))
            if len(quad)!=4:
                continue
            if _edge(c,d) in existing_edges:
                rejected_topology+=1
                continue
            if len({vertices[v].component_id for v in quad})!=1:
                continue

            old_faces=(f0,f1)
            oldq=_pair_quality(old_faces,positions,policy)
            if oldq["violation_count"]<=0:
                continue
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
            scale=_local_scale(tuple(quad),positions)
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
                "edge":edge,"faces":(i,j),"new_faces":new_faces,"quad":quad,
                "oldq":oldq,"newq":newq,"deviation":deviation,"allowed":allowed,
                "improvement":improvement,
            })

        if not proposals:
            break
        proposals.sort(key=lambda row:(
            -int(row["improvement"][0]),
            -float(row["improvement"][1]),
            -float(row["improvement"][2]),
            row["edge"],
        ))
        occupied=set()
        selected=[]
        for row in proposals:
            if occupied.intersection(row["quad"]):
                rejected_batch_conflict+=1
                continue
            selected.append(row)
            occupied.update(row["quad"])
        if not selected:
            break

        for row in selected:
            i,j=row["faces"]
            faces[i],faces[j]=row["new_faces"]

        if len(set(faces))!=len(faces):
            raise QualificationError("QUALITY_TOPO_SAFE_FLIP_DUPLICATE_FACE")
        topo=_manifold_report(faces)
        if not topo["passed"]:
            raise QualificationError(
                f"QUALITY_TOPO_SAFE_FLIP_POSTPASS_NONMANIFOLD:"
                f"{pass_index}:{topo['nonmanifold_edge_count']}:{topo['illegal_vertex_link_count']}"
            )

        for row in selected:
            accepted.append({
                "pass_index":pass_index,
                "old_edge":row["edge"],
                "new_edge":_edge(row["new_faces"][0][0],row["new_faces"][0][1]),
                "local_sampled_surface_deviation":row["deviation"],
                "allowed_local_deviation":row["allowed"],
                "old_violation_count":row["oldq"]["violation_count"],
                "new_violation_count":row["newq"]["violation_count"],
                "old_min_angle_deg":row["oldq"]["min_angle_deg"],
                "new_min_angle_deg":row["newq"]["min_angle_deg"],
                "old_max_aspect":row["oldq"]["max_aspect"],
                "new_max_aspect":row["newq"]["max_aspect"],
            })
        pass_rows.append({
            "pass_index":pass_index,
            "accepted_flip_count":len(selected),
            "occupied_vertex_count":len(occupied),
            "topology":topo,
        })

    faces_tuple=tuple(sorted(tuple(face) for face in faces))
    edges=tuple(sorted({
        _edge(face[i],face[j])
        for face in faces_tuple for i,j in ((0,1),(1,2),(2,0))
    }))
    after=_report(faces_tuple,positions,policy)
    final_topology=_manifold_report(faces_tuple)
    policy_hash=content_sha256({
        "schema":"RealSaS.FixedVertexTopologySafeFlipRepairPolicy.v2",
        "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "max_passes":int(max_passes),
        "batch_independence":"PAIRWISE_DISJOINT_FOUR_VERTEX_PATCHES",
        "topology_guard":"EDGE_AND_VERTEX_LINK_MANIFOLD_PRE_AND_POST_PASS",
        "selection":"MONOTONE_G3_WITH_FROZEN_G1_LOCAL_DEVIATION",
        "vertex_motion":"FORBIDDEN","vertex_insertion":"FORBIDDEN","vertex_deletion":"FORBIDDEN",
    })
    provisional=CanonicalMeshCandidateIR(
        vertices=candidate.vertices,faces=faces_tuple,edges=edges,
        surface_binding_hash=candidate.surface_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        carrier_policy_binding_hash=candidate.carrier_policy_binding_hash,
        producer_id="RealSaS.CanonicalMesh.FixedVertexTopologySafeFlipRepair.v2",
        producer_policy_hash=policy_hash,candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "topology_safe_flip_quality_repair":{
                "algorithm":"FIXED_VERTEX_TOPOLOGY_SAFE_EDGE_FLIP_V2",
                "input_candidate_lineage_hash":candidate.candidate_lineage_hash,
                "accepted_flip_count":len(accepted),
                "rejected_shape_deviation_count":int(rejected_shape),
                "rejected_topology_count":int(rejected_topology),
                "rejected_batch_conflict_count":int(rejected_batch_conflict),
                "before":before,"after":after,
                "initial_topology":initial_topology,"final_topology":final_topology,
                "passes":pass_rows,"accepted_flips":accepted,
                "vertex_motion":False,"vertex_insertion":False,"vertex_deletion":False,
                "support_binding_changed":False,
            },
        },
    )
    result=replace(provisional,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional))
    return result,dict(result.metadata["topology_safe_flip_quality_repair"])
