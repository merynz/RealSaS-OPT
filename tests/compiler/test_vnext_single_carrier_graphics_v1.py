from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.graphics_runtime_v1 import build_runtime_graphics_v1
from compiler.realsas_compiler_core.graphics_v1 import (
    GraphicsCompilePolicyV1,
    GraphicsStaticRelationEvidenceV1,
    GraphicsViewRelationEvidenceV1,
    build_graphics_evidence_v1,
    compile_graphics_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _graphics_evidence(mesh_binding_hash: str):
    static = (
        GraphicsStaticRelationEvidenceV1(0, 1, 0.0, 1.0, 0.9, 0.8, 0.8, 0.9, "s01"),
        GraphicsStaticRelationEvidenceV1(1, 2, 0.0, 1.0, 0.0, 0.8, 0.8, 0.9, "s12"),
    )
    views = []
    for view in range(8):
        views.extend(
            (
                GraphicsViewRelationEvidenceV1(view, 0, 1, 0.95, 0.95, 1.0, 1.0, 0.95, f"v{view}-01"),
                GraphicsViewRelationEvidenceV1(view, 1, 2, 0.90, 0.95, 1.0, 1.0, 0.90, f"v{view}-12"),
                GraphicsViewRelationEvidenceV1(view, 2, 0, 0.70, 0.80, 1.0, 1.0, 0.70, f"v{view}-20"),
            )
        )
    return build_graphics_evidence_v1(
        mesh_binding_hash=mesh_binding_hash,
        static_relations=static,
        view_relations=tuple(views),
    )


def test_graphics_compiles_role_free_order_and_runtime_consumes_materialized_content():
    faces = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    evidence = _graphics_evidence("mesh")
    compiled = compile_graphics_v1(
        canonical_mesh_binding_hash="mesh",
        canonical_mesh_skin_binding_hash="mesh-skin",
        source_graphics_binding_hash="source-graphics",
        faces=faces,
        evidence=evidence,
        policy=GraphicsCompilePolicyV1(allow_unqualified_order_arbitration=True),
        mechanical_domain_ids=("joint:hand",),
        mechanical_attachment_evidence={
            (0, 1): {
                "persistent_contact": 0.95,
                "parent_mechanical_domain_id": "joint:hand",
                "anchor_face_index": 0,
                "child_face_index": 1,
                "barycentric_anchor": (0.2, 0.3, 0.5),
                "evidence_ref": "mechanical-contact",
            }
        },
    )

    assert compiled.metadata["categorical_semantics_used"] is False
    assert compiled.metadata["single_canonical_mesh_authority"] is True
    assert compiled.metadata["independent_visual_mesh_allowed"] is False
    assert len(compiled.fragments) == 3
    assert len(compiled.attachments) == 1
    # Face 1 is the moving child, face 0 is merely the parent anchor.
    assert compiled.attachments[0].fragment_id == compiled.fragments[1].fragment_id
    assert compiled.attachments[0].anchor_face_index == 0
    assert all(
        view.metadata["triangle_array_order_is_presentation_authority"] is False
        for view in compiled.views
    )

    runtime = build_runtime_graphics_v1(compiled, face_count=3)
    assert runtime.face_fragment_index == (0, 1, 2)
    assert len(runtime.views) == 8
    assert runtime.metadata["hash_only_binding_forbidden"] is True
    assert runtime.metadata["behaviorally_consumable"] is True
    # A contradictory source relation was resolved for deterministic display,
    # not independently certified as a true source occlusion order.
    assert runtime.metadata["representation_policy_arbitration_used"] is True
    assert runtime.metadata["unresolved_source_order_conflict"] is True
    assert runtime.metadata["pose_conditioned_order_execution"] is False
    assert all(len(view.topological_fragment_order) == 3 for view in runtime.views)


def test_graphics_cannot_bind_a_second_independent_visual_mesh():
    evidence = build_graphics_evidence_v1(
        mesh_binding_hash="proposal-A",
        static_relations=(
            GraphicsStaticRelationEvidenceV1(0, 1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
        view_relations=(),
    )
    with pytest.raises(QualificationError, match="CANONICAL_MESH_BINDING_DRIFT"):
        compile_graphics_v1(
            canonical_mesh_binding_hash="canonical-B",
            canonical_mesh_skin_binding_hash="skin",
            source_graphics_binding_hash="source",
            faces=((0, 1, 2), (3, 4, 5)),
            evidence=evidence,
        )


def test_graphics_acyclic_source_order_does_not_claim_pose_conditioned_coverage():
    evidence = build_graphics_evidence_v1(
        mesh_binding_hash="mesh",
        static_relations=(
            GraphicsStaticRelationEvidenceV1(0, 1, 0., 1., 0., .8, .8, 1., "static"),
        ),
        view_relations=(
            GraphicsViewRelationEvidenceV1(0, 0, 1, .95, .9, 1., 1., .95, "view"),
        ),
    )
    compiled = compile_graphics_v1(
        canonical_mesh_binding_hash="mesh",
        canonical_mesh_skin_binding_hash="mesh-skin",
        source_graphics_binding_hash="source-graphics",
        faces=((0, 1, 2), (3, 4, 5)),
        evidence=evidence,
    )
    runtime = build_runtime_graphics_v1(compiled, face_count=2)
    assert runtime.metadata["unresolved_source_order_conflict"] is False
    assert runtime.metadata["representation_policy_arbitration_used"] is False
    assert runtime.metadata["pose_conditioned_order_execution"] is False


def test_conflicting_source_front_back_relations_never_silently_become_canonical_truth():
    faces = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    evidence = _graphics_evidence("mesh")
    with pytest.raises(QualificationError, match="GRAPHICS_SOURCE_ORDER_CONFLICT_UNQUALIFIED"):
        compile_graphics_v1(
            canonical_mesh_binding_hash="mesh",
            canonical_mesh_skin_binding_hash="mesh-skin",
            source_graphics_binding_hash="source-graphics",
            faces=faces, evidence=evidence,
        )
    preview = compile_graphics_v1(
        canonical_mesh_binding_hash="mesh",
        canonical_mesh_skin_binding_hash="mesh-skin",
        source_graphics_binding_hash="source-graphics",
        faces=faces, evidence=evidence,
        policy=GraphicsCompilePolicyV1(
            allow_unqualified_order_arbitration=True,
        ),
    )
    assert preview.metadata["order_arbitration_not_source_truth"] is True
    assert any(v.metadata["cycle_edges_removed"] for v in preview.views)
    runtime = build_runtime_graphics_v1(preview, face_count=3)
    assert runtime.metadata["unresolved_source_order_conflict"] is True
    assert runtime.metadata["pose_conditioned_order_execution"] is False


@pytest.mark.parametrize("mutated,code", [
    ({"child_face_index": None}, "GRAPHICS_ATTACHMENT_PARENT_CHILD_FACE_BINDING_INVALID"),
    ({"child_face_index": 0}, "GRAPHICS_ATTACHMENT_PARENT_CHILD_FACE_BINDING_INVALID"),
    ({"anchor_face_index": 2}, "GRAPHICS_ATTACHMENT_PARENT_CHILD_FACE_BINDING_INVALID"),
    ({"parent_mechanical_domain_id": "nonexistent"}, "GRAPHICS_ATTACHMENT_MECHANICAL_DOMAIN_UNQUALIFIED"),
])
def test_graphics_rejects_unbound_or_ambiguous_attachment(mutated, code):
    evidence = _graphics_evidence("mesh")
    mechanical = {
        "persistent_contact": .95,
        "parent_mechanical_domain_id": "joint:hand",
        "anchor_face_index": 0,
        "child_face_index": 1,
        "barycentric_anchor": (.2, .3, .5),
        "evidence_ref": "mechanical-contact",
    }
    mechanical.update(mutated)
    with pytest.raises(QualificationError, match=code):
        compile_graphics_v1(
            canonical_mesh_binding_hash="mesh",
            canonical_mesh_skin_binding_hash="mesh-skin",
            source_graphics_binding_hash="source-graphics",
            faces=((0,1,2),(3,4,5),(6,7,8)),
            evidence=evidence,
            mechanical_domain_ids=("joint:hand",),
            mechanical_attachment_evidence={(0,1):mechanical},
            policy=GraphicsCompilePolicyV1(
                allow_unqualified_order_arbitration=True,
            ),
        )


def test_graphics_rejects_unqualified_parent_even_with_valid_face_addresses():
    evidence = _graphics_evidence("mesh")
    with pytest.raises(QualificationError, match="MECHANICAL_DOMAIN_UNQUALIFIED"):
        compile_graphics_v1(
            canonical_mesh_binding_hash="mesh",
            canonical_mesh_skin_binding_hash="mesh-skin",
            source_graphics_binding_hash="source-graphics",
            faces=((0,1,2),(3,4,5),(6,7,8)),
            evidence=evidence,
            mechanical_attachment_evidence={(0,1):{
                "persistent_contact": .95,
                "parent_mechanical_domain_id": "joint:hand",
                "anchor_face_index": 0,
                "child_face_index": 1,
                "barycentric_anchor": (.2, .3, .5),
            }},
            policy=GraphicsCompilePolicyV1(
                allow_unqualified_order_arbitration=True,
            ),
        )


@pytest.mark.parametrize("invalid,expected", [
    ({"persistent_contact": 1.5}, "GRAPHICS_PROBABILITY_INVALID:mechanical_contact"),
    ({"barycentric_anchor": (float("nan"), .5, .5)}, "GRAPHICS_ATTACHMENT_BARYCENTRIC_INVALID"),
    ({"barycentric_anchor": (-.1, .6, .5)}, "GRAPHICS_ATTACHMENT_BARYCENTRIC_INVALID"),
])
def test_graphics_attachment_rejects_invalid_numeric_source_evidence(invalid, expected):
    m = {
        "persistent_contact": .95,
        "parent_mechanical_domain_id": "joint:hand",
        "anchor_face_index": 0,
        "child_face_index": 1,
        "barycentric_anchor": (.2, .3, .5),
        "evidence_ref": "source-contact",
    }
    m.update(invalid)
    with pytest.raises(QualificationError, match=expected):
        compile_graphics_v1(
            canonical_mesh_binding_hash="mesh",
            canonical_mesh_skin_binding_hash="mesh-skin",
            source_graphics_binding_hash="source",
            faces=((0,1,2),(3,4,5),(6,7,8)),
            evidence=_graphics_evidence("mesh"),
            mechanical_domain_ids=("joint:hand",),
            mechanical_attachment_evidence={(0,1):m},
            policy=GraphicsCompilePolicyV1(allow_unqualified_order_arbitration=True),
        )


def test_graphics_invalid_unused_relation_probability_must_not_escape_court():
    evidence = build_graphics_evidence_v1(
        mesh_binding_hash="mesh",
        static_relations=(
            GraphicsStaticRelationEvidenceV1(
                0,1,1.5,1.,0.,0.,0.,1.,"unused-confidence",
            ),
        ),
        view_relations=(),
    )
    with pytest.raises(QualificationError, match="GRAPHICS_PROBABILITY_INVALID:same"):
        compile_graphics_v1(
            canonical_mesh_binding_hash="mesh",
            canonical_mesh_skin_binding_hash="skin",
            source_graphics_binding_hash="source",
            faces=((0,1,2),(3,4,5)), evidence=evidence,
        )


def test_transitive_region_affinity_never_erases_an_explicit_nonadjacent_cut():
    faces=((0,1,2),(1,2,3),(2,3,4))
    # Locally 0~1 and 1~2; source evidence 0!=2. A normal union-find would
    # incorrectly collapse all three into one drawable via transitivity.
    a=GraphicsStaticRelationEvidenceV1(0,1,.99,.01,0.,0.,0.,.98,"same-01")
    b=GraphicsStaticRelationEvidenceV1(1,2,.98,.01,0.,0.,0.,.97,"same-12")
    conflict=GraphicsStaticRelationEvidenceV1(0,2,.01,.99,0.,0.,0.,.99,"cut-02")
    results=[]
    for relations in ((a,b,conflict),(conflict,b,a)):
        evidence=build_graphics_evidence_v1(
            mesh_binding_hash="mesh",static_relations=relations,view_relations=(),
        )
        compiled=compile_graphics_v1(
            canonical_mesh_binding_hash="mesh",
            canonical_mesh_skin_binding_hash="skin",
            source_graphics_binding_hash="source",
            faces=faces,evidence=evidence,
        )
        results.append(compiled)
    first,second=results
    assert len(first.fragments)==2
    assert tuple(f.face_indices for f in first.fragments)==((0,1),(2,))
    assert tuple(f.fragment_id for f in first.fragments)==tuple(
        f.fragment_id for f in second.fragments
    )
    assert first.metadata["transitive_cut_constraints_enforced"] is True
    assert first.metadata["candidate_merge_blocked_by_cut"] == ((1,2),)
    assert first.metadata["pose_conditioned_visibility_proven"] is False


def test_explicit_cut_court_does_not_turn_weak_ambiguous_relation_into_forced_split():
    faces=((0,1,2),(1,2,3),(2,3,4))
    # 0/2 cut is below the qualified candidate-policy confidence threshold.
    evidence=build_graphics_evidence_v1(
        mesh_binding_hash="mesh",
        static_relations=(
            GraphicsStaticRelationEvidenceV1(0,1,.99,.01,0.,0.,0.,.99),
            GraphicsStaticRelationEvidenceV1(1,2,.99,.01,0.,0.,0.,.99),
            GraphicsStaticRelationEvidenceV1(0,2,.01,.99,0.,0.,0.,.2),
        ),
        view_relations=(),
    )
    compiled=compile_graphics_v1(
        canonical_mesh_binding_hash="mesh",
        canonical_mesh_skin_binding_hash="skin",
        source_graphics_binding_hash="source",
        faces=faces,evidence=evidence,
    )
    assert len(compiled.fragments)==1
    assert compiled.metadata["candidate_merge_blocked_by_cut"]==()


def test_unobserved_fragment_visibility_remains_unknown_and_uses_depth_raster_policy():
    evidence=build_graphics_evidence_v1(
        mesh_binding_hash="mesh",
        static_relations=(
            GraphicsStaticRelationEvidenceV1(0,1,0.,1.,0.,0.,0.,1.),
        ),
        view_relations=(
            GraphicsViewRelationEvidenceV1(0,0,1,.95,.95,1.,1.,1.),
        ),
    )
    compiled=compile_graphics_v1(
        canonical_mesh_binding_hash="mesh",
        canonical_mesh_skin_binding_hash="skin",
        source_graphics_binding_hash="source",
        faces=((0,1,2),(3,4,5),(6,7,8)),
        evidence=evidence,
    )
    assert len(compiled.fragments)==3
    view=compiled.views[0]
    unknown=(compiled.fragments[2].fragment_id,)
    assert view.metadata["visibility_unknown_fragment_ids"] == unknown
    assert unknown[0] in view.visible_fragment_ids
    assert view.metadata["missing_source_view_visibility_is_not_positive_label"] is True
    assert compiled.metadata["source_visibility_coverage_complete"] is False
    runtime=build_runtime_graphics_v1(compiled,face_count=3)
    assert runtime.metadata["source_visibility_coverage_complete"] is False
    assert runtime.metadata["unknown_visibility_fragment_ids_by_view"]["0"] == unknown
    assert runtime.metadata["unknown_visibility_raster_policy"] == (
        "DRAW_GEOMETRY_AND_DEFER_OCCLUSION_TO_DEPTH"
    )


def test_no_view_evidence_never_mints_complete_source_visibility():
    evidence=build_graphics_evidence_v1(
        mesh_binding_hash="mesh",
        static_relations=(
            GraphicsStaticRelationEvidenceV1(0,1,0.,1.,0.,0.,0.,1.),
        ),
        view_relations=(),
    )
    compiled=compile_graphics_v1(
        canonical_mesh_binding_hash="mesh",
        canonical_mesh_skin_binding_hash="skin",
        source_graphics_binding_hash="source",
        faces=((0,1,2),(3,4,5)),
        evidence=evidence,
    )
    assert compiled.metadata["source_visibility_coverage_complete"] is False
    assert compiled.metadata["view_relation_eight_direction_address_coverage"] is False
    runtime=build_runtime_graphics_v1(compiled,face_count=2)
    assert runtime.metadata["source_visibility_coverage_complete"] is False
    assert runtime.metadata["view_relation_eight_direction_address_coverage"] is False
