from __future__ import annotations

import pytest
from dataclasses import replace

from compiler.realsas_compiler_core.graphics_runtime_v1 import build_runtime_graphics_v1
from compiler.realsas_compiler_core.graphics_v1 import (
    GraphicsCompilePolicyV1,
    compile_graphics_on_canonical_puppet_v1,
    compile_neural_region_graphics_on_canonical_puppet_v1,
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


def _canonical_graphics_parents():
    from tests.compiler.test_v6_typed_rig_skin_main_integration_v1 import (
        _real_rig_fixture, _qualified_synthetic_carrier_skin,
    )
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
    mesh,rig=_real_rig_fixture()
    skin=mint_canonical_skin_v2(
        mesh,rig,
        qualified_skin=_qualified_synthetic_carrier_skin(mesh,rig),
        skin_code_proposal_hash="a"*64,
        skin_codec_architecture_id="fixture-codec-not-model-release",
        skin_codec_checkpoint_hash="b"*64,
    )
    return mesh,rig,skin


def test_graphics_compiles_only_from_actual_same_canonical_mesh_rig_skin():
    mesh,rig,skin=_canonical_graphics_parents()
    evidence=build_graphics_evidence_v1(
        mesh_binding_hash=mesh.mesh_lineage_hash,
        static_relations=(),view_relations=(),
    )
    compiled=compile_graphics_on_canonical_puppet_v1(
        mesh=mesh,rig=rig,skin=skin,source_graphics_binding_hash="source-bound",
        evidence=evidence,
    )
    assert compiled.canonical_mesh_binding_hash==mesh.mesh_lineage_hash
    assert compiled.canonical_mesh_skin_binding_hash==skin.skin_lineage_hash
    assert len(compiled.fragments)==1
    assert compiled.fragments[0].face_indices==(0,)
    assert compiled.metadata["source_visibility_coverage_complete"] is False
    runtime=build_runtime_graphics_v1(compiled,face_count=len(mesh.faces))
    assert runtime.face_fragment_index==(0,)
    assert runtime.metadata["source_visibility_coverage_complete"] is False


def test_canonical_graphics_wrapper_rejects_self_rehashed_cross_puppet_skin():
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
        canonical_skin_lineage_hash_v2,
    )
    mesh,rig,skin=_canonical_graphics_parents()
    wrong=replace(skin,mesh_lineage_hash="f"*64,skin_lineage_hash="")
    wrong=replace(wrong,skin_lineage_hash=canonical_skin_lineage_hash_v2(wrong))
    evidence=build_graphics_evidence_v1(
        mesh_binding_hash=mesh.mesh_lineage_hash,
        static_relations=(),view_relations=(),
    )
    with pytest.raises(QualificationError,match="GRAPHICS_CANONICAL_PUPPET_PARENT_DRIFT"):
        compile_graphics_on_canonical_puppet_v1(
            mesh=mesh,rig=rig,skin=wrong,
            source_graphics_binding_hash="source-bound",
            evidence=evidence,
        )


def test_canonical_graphics_wrapper_rejects_wrong_mesh_binding_despite_valid_parents():
    mesh,rig,skin=_canonical_graphics_parents()
    evidence=build_graphics_evidence_v1(
        mesh_binding_hash="unrelated-proposal",
        static_relations=(),view_relations=(),
    )
    with pytest.raises(QualificationError,match="GRAPHICS_CANONICAL_MESH_BINDING_DRIFT"):
        compile_graphics_on_canonical_puppet_v1(
            mesh=mesh,rig=rig,skin=skin,
            source_graphics_binding_hash="source-bound",
            evidence=evidence,
        )


def _qualified_neural_graphics_fixture(*, mutate_candidate=None):
    """Pure Compiler fixture, no torch/ML dependency in V6 typed courts."""
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
        canonical_mesh_lineage_hash_v1,
    )
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import mint_canonical_rig_v2
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
    from compiler.realsas_compiler_core.rig_program_ir_v1 import qualify_rig_program_v1
    from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
        RSSvNextInferenceEvidenceIRV1, seal_inference_evidence_v1,
    )
    from tests.compiler.test_control_ready_rig_v2 import _mesh, _proposal
    from tests.compiler.test_v6_typed_rig_skin_main_integration_v1 import (
        _qualified_synthetic_carrier_skin,
    )

    bare = _mesh()
    n = len(bare.vertices)
    points = [list(v.P) for v in bare.vertices]
    indices = {
        v.canonical_mesh_vertex_id: i for i, v in enumerate(bare.vertices)
    }
    network_mesh = {
        "topology_tokens": [128, 129, 4],
        "rest_xyz": points,
        "base_xyz": points,
        "local_features": [[.1, .2, .3]] * n,
        "faces": [[indices[vid] for vid in face] for face in bare.faces],
        "vertex_components": [0] * n,
        "face_components": [0] * len(bare.faces),
        "normals_xyz": [[0., 0., 1.]] * n,
        "semantic_logits": [[4., .2, -3.]] * n,
        "presentation_role_logits": [[.1, .2, .3, .4]] * n,
        "presentation_role_schema_id": "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1",
        "affordance_logits": [[.5, .5]] * n,
        "graphics_logits": [[.1, .2, .3, .4, .5]] * n,
        "static_quality_logits": [[0., 0., 0., 0.]] * n,
        "unsigned_distance": [.1] * n,
        "sign_logits": [.7] * n,
        "source_rays": {
            "rgba": [[[.1, .2, .3, .8], [.2, .3, .4, .4]]] * n,
            "valid": [[True, True]] * n,
            "alpha": [[.8, .4]] * n,
            "appearance_authority": "ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE",
            "depth_authority": "UNKNOWN_NO_SOURCE_DEPTH",
        },
        "source_authority": "LEARNED_PREDICTED_TOPOLOGY",
        "signed_solid_authority": "UNQUALIFIED",
        "canonical_mesh_authority": False,
    }
    scene = {
        "source_semantic_patch_logits": [[[0., 0., 0., 0.]], [[0., 0., 0., 0.]]],
        "source_semantic_log_uncertainty": [[[0.]], [[0.]]],
        "region_relation_logits": [
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
        ],
        "region_relation_valid_mask": [[False, True], [True, False]],
        "presentation_role_probabilities": [[.25, .25, .25, .25]] * n,
        "constructed_camera_features": None,
        "constructed_camera_angles": None,
        "authority": "PREDICTED_FROM_TWO_ORIGINAL_VIEWS",
        "region_query_xyz": points,
        "region_membership_logits": [[4., .2, -3.]] * n,
        "region_graphics_logits": [[0., 0., 0., 0., 0.]] * n,
    }
    if mutate_candidate:
        mutate_candidate(network_mesh)
    evidence = seal_inference_evidence_v1(RSSvNextInferenceEvidenceIRV1(
        checkpoint_sha256="a" * 64,
        observed_front_sha256="b" * 64,
        observed_back_sha256="c" * 64,
        camera_binding_sha256="d" * 64,
        mesh_status="UNQUALIFIED_MESH_CANDIDATE",
        mesh=network_mesh,
        scene=scene,
    ))
    mesh = replace(
        bare, proposal_evidence_refs=(evidence.evidence_hash,),
        mesh_lineage_hash="",
    )
    mesh = replace(mesh, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    rig = mint_canonical_rig_v2(
        mesh, qualified_program=qualify_rig_program_v1(_proposal(mesh)),
    )
    skin = mint_canonical_skin_v2(
        mesh, rig,
        qualified_skin=_qualified_synthetic_carrier_skin(mesh, rig),
        skin_code_proposal_hash="a" * 64,
        skin_codec_architecture_id="fixture-codec-not-modelrelease",
        skin_codec_checkpoint_hash="b" * 64,
    )
    mapping = tuple(v.source_candidate_vertex_id for v in mesh.vertices)
    return evidence, mapping, mesh, rig, skin


def test_neural_anonymous_regions_lower_into_real_canonical_graphics_without_source_authority():
    evidence, mapping, mesh, rig, skin = _qualified_neural_graphics_fixture()
    compiled = compile_neural_region_graphics_on_canonical_puppet_v1(
        evidence=evidence, expected_checkpoint_sha256="a" * 64,
        candidate_vertex_ids_by_index=mapping,
        mesh=mesh, rig=rig, skin=skin,
    )
    assert compiled.metadata["origin"] == "NEURAL_ANONYMOUS_REGION_CANDIDATE"
    assert compiled.metadata["qualified_drawable_source_truth"] is False
    assert compiled.metadata["qualified_source_occlusion"] is False
    assert compiled.metadata["qualified_source_attachments"] is False
    assert compiled.metadata["independent_editability_proven"] is False
    assert compiled.metadata["exact_candidate_to_canonical_face_map"] == (0,)
    assert compiled.canonical_mesh_binding_hash == mesh.mesh_lineage_hash
    assert compiled.canonical_mesh_skin_binding_hash == skin.skin_lineage_hash
    assert len(compiled.fragments) == len(mesh.faces)
    assert not compiled.views
    assert not compiled.attachments
    runtime = build_runtime_graphics_v1(compiled, face_count=len(mesh.faces))
    assert runtime.face_fragment_index == (0,)
    assert runtime.metadata["source_visibility_coverage_complete"] is False


def test_neural_graphics_cannot_skip_static_source_qualification_or_relabel_vertex_addresses():
    evidence, mapping, mesh, rig, skin = _qualified_neural_graphics_fixture()
    with pytest.raises(QualificationError, match="GRAPHICS_NEURAL_VERTEX_POSITION_DRIFT"):
        compile_neural_region_graphics_on_canonical_puppet_v1(
            evidence=evidence, expected_checkpoint_sha256="a" * 64,
            candidate_vertex_ids_by_index=tuple(reversed(mapping)),
            mesh=mesh, rig=rig, skin=skin,
        )
    without_parent = replace(
        mesh, proposal_evidence_refs=(), mesh_lineage_hash="",
    )
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_lineage_hash_v1
    without_parent = replace(
        without_parent, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(without_parent),
    )
    with pytest.raises(QualificationError, match="GRAPHICS_NEURAL_STATIC_QUALIFICATION_PARENT_MISSING"):
        compile_neural_region_graphics_on_canonical_puppet_v1(
            evidence=evidence, expected_checkpoint_sha256="a" * 64,
            candidate_vertex_ids_by_index=mapping,
            mesh=without_parent, rig=rig, skin=skin,
        )


def test_neural_graphics_rejects_orientation_flipped_candidate_even_when_parent_is_resealed():
    def invert_winding(mesh):
        mesh["faces"] = [list(reversed(face)) for face in mesh["faces"]]
    evidence, mapping, mesh, rig, skin = _qualified_neural_graphics_fixture(
        mutate_candidate=invert_winding,
    )
    with pytest.raises(QualificationError, match="GRAPHICS_NEURAL_FACE_TOPOLOGY_OR_WINDING_DRIFT"):
        compile_neural_region_graphics_on_canonical_puppet_v1(
            evidence=evidence, expected_checkpoint_sha256="a" * 64,
            candidate_vertex_ids_by_index=mapping,
            mesh=mesh, rig=rig, skin=skin,
        )
