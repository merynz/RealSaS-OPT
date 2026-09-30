from __future__ import annotations

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
)
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    build_compacted_dense_face_provenance_v1,
    validate_compacted_dense_face_provenance_v1,
)
from compiler.realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _fixture():
    # Three source triangles supply AB, BC and CA separately.  ABC is therefore
    # a 3-clique in the pairwise relation graph but is NOT a source face.
    pos={
        "a":(0.0,0.0,0.0),
        "b":(2.0,0.0,0.0),
        "c":(1.0,2.0,0.0),
        "d":(1.0,-1.0,0.0),
        "e":(3.0,1.0,0.0),
        "f":(-1.0,1.0,0.0),
    }
    ids=tuple(pos)
    dense_faces=(("a","b","d"),("b","c","e"),("a","c","f"))
    index={sid:i for i,sid in enumerate(ids)}
    face_idx=tuple(tuple(index[sid] for sid in face) for face in dense_faces)

    edge_set=set()
    for face in dense_faces:
        for i,j in ((0,1),(1,2),(2,0)):
            edge_set.add(tuple(sorted((face[i],face[j]))))
    nodes=tuple(
        SurfaceNode(sid,pos[sid],(0,),("p",),(f"obs-{sid}",))
        for sid in ids
    )
    relations=tuple(
        SurfaceRelation(f"r-{a}-{b}",a,b,"LOCAL_NEIGHBOR",1.0)
        for a,b in sorted(edge_set)
    )
    surface=RiggingSurfaceIR(
        nodes,relations,"surface-face-provenance",
        metadata={
            "compact_voxel_divisions":0,
            "component_aware_compaction":False,
            "source_zero_surface_sha256":"b"*64,
        },
    )
    partition=build_structural_partition(surface)
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(c.component_id,"MESH",("auto",))
            for c in partition.components
        ),
    )
    vertices=[pos[sid] for sid in ids]
    return surface,partition,carrier,vertices,face_idx


def _source_face_set(candidate):
    by_id={v.candidate_vertex_id:v for v in candidate.vertices}
    out=set()
    for face in candidate.faces:
        sids=[]
        for vid in face:
            coeffs=tuple(by_id[vid].support_binding.coefficients)
            assert len(coeffs)==1 and abs(float(coeffs[0][1])-1.0)<=1e-12
            sids.append(str(coeffs[0][0]))
        out.add(tuple(sorted(sids)))
    return out


def test_exact_face_provenance_blocks_three_edge_clique_face_minting():
    surface,partition,carrier,vertices,face_idx=_fixture()
    provenance=build_compacted_dense_face_provenance_v1(
        vertices,face_idx,surface,source_zero_surface_sha256="b"*64
    )
    validate_compacted_dense_face_provenance_v1(provenance,surface=surface)

    assert tuple(provenance["compact_faces"])==(
        ("a","b","d"),("a","c","f"),("b","c","e"),
    )
    assert ("a","b","c") not in set(provenance["compact_faces"])
    assert provenance["three_clique_face_minting_allowed"] is False

    loose=build_canonical_relation_candidate(
        surface,partition,carrier,producer_policy_hash="loose"
    )
    assert ("a","b","c") in _source_face_set(loose)

    strict=build_canonical_relation_candidate(
        surface,partition,carrier,
        producer_policy_hash="strict",
        explicit_face_provenance=tuple(provenance["compact_faces"]),
    )
    strict_faces=_source_face_set(strict)
    assert ("a","b","c") not in strict_faces
    assert strict_faces==set(provenance["compact_faces"])
    assert strict.metadata["face_provenance_mode"]=="EXPLICIT_SOURCE_FACE_PROVENANCE"
    assert strict.metadata["three_clique_face_minting_allowed"] is False
