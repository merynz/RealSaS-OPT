import pytest

from compiler.realsas_compiler_core.graphics_relation_evidence_v1 import (
    GraphicsFaceVisibilityEvidenceV1,
    IRISGraphicsStaticPairEvidenceV1,
    IRISGraphicsViewPairEvidenceV1,
    build_iris_graphics_evidence_v1,
    compile_iris_graphics_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


MESH = "a" * 64
SKIN = "b" * 64
SOURCE = "c" * 64
H = "d" * 64
MODEL = "model-release:iris-v3"


def _evidence(*, front_ab=0.9, front_ba=0.1, visibility=True):
    return build_iris_graphics_evidence_v1(
        mesh_binding_hash=MESH,
        shared_h_transport_hash=H,
        model_release_binding=MODEL,
        static_pairs=(
            IRISGraphicsStaticPairEvidenceV1(
                face_a=0,
                face_b=1,
                same_drawable_ab=0.1,
                same_drawable_ba=0.1,
                cut_ab=0.9,
                cut_ba=0.9,
                attachment_ab=0.0,
                attachment_ba=0.0,
                evidence_ref="static:0:1",
            ),
        ),
        view_pairs=(
            IRISGraphicsViewPairEvidenceV1(
                view_index=0,
                face_a=0,
                face_b=1,
                front_ab=front_ab,
                front_ba=front_ba,
                occlusion_ab=0.9,
                occlusion_ba=0.9,
                projected_alpha_boundary_ab=0.8,
                projected_alpha_boundary_ba=0.8,
                projected_appearance_boundary_ab=0.8,
                projected_appearance_boundary_ba=0.8,
                support_ref="view0:overlap:0:1",
                evidence_ref="view:0:0:1",
            ),
        ),
        visibility=(
            GraphicsFaceVisibilityEvidenceV1(
                view_index=0,
                face_index=0,
                visible=visibility,
                support_ref="view0:raster:face0",
                authority="DETERMINISTIC_ZBUFFER",
            ),
            GraphicsFaceVisibilityEvidenceV1(
                view_index=0,
                face_index=1,
                visible=True,
                support_ref="view0:raster:face1",
                authority="DETERMINISTIC_ZBUFFER",
            ),
        ),
    )


def test_bidirectional_front_support_compiles_without_neural_confidence():
    evidence = _evidence()
    compiled = compile_iris_graphics_v1(
        canonical_mesh_binding_hash=MESH,
        canonical_mesh_skin_binding_hash=SKIN,
        source_graphics_binding_hash=SOURCE,
        faces=((0, 1, 2), (1, 3, 2)),
        evidence=evidence,
    )

    assert compiled.metadata["neural_confidence_used"] is False
    assert compiled.metadata["pairwise_visibility_used"] is False
    assert compiled.metadata["local_interleaving_without_executable_support"] == "FAIL_CLOSED"
    assert len(compiled.fragments) == 2
    assert len(compiled.views) == 1
    assert len(compiled.views[0].order_edges) == 1
    edge = compiled.views[0].order_edges[0]
    assert edge.front_fragment_id != edge.back_fragment_id
    assert 0.0 < edge.confidence <= 1.0


def test_same_drawable_requires_both_ordered_directions_to_support_union():
    evidence = build_iris_graphics_evidence_v1(
        mesh_binding_hash=MESH,
        shared_h_transport_hash=H,
        model_release_binding=MODEL,
        static_pairs=(
            IRISGraphicsStaticPairEvidenceV1(
                face_a=0,
                face_b=1,
                same_drawable_ab=0.95,
                same_drawable_ba=0.10,
                cut_ab=0.05,
                cut_ba=0.05,
                attachment_ab=0.0,
                attachment_ba=0.0,
            ),
        ),
        view_pairs=(),
        visibility=(),
    )
    compiled = compile_iris_graphics_v1(
        canonical_mesh_binding_hash=MESH,
        canonical_mesh_skin_binding_hash=SKIN,
        source_graphics_binding_hash=SOURCE,
        faces=((0, 1, 2), (1, 3, 2)),
        evidence=evidence,
    )
    assert len(compiled.fragments) == 2


def test_opposite_local_precedence_fails_closed_in_global_v1_runtime():
    evidence = build_iris_graphics_evidence_v1(
        mesh_binding_hash=MESH,
        shared_h_transport_hash=H,
        model_release_binding=MODEL,
        static_pairs=(
            IRISGraphicsStaticPairEvidenceV1(0, 1, 0.1, 0.1, 0.9, 0.9, 0.0, 0.0),
        ),
        view_pairs=(
            IRISGraphicsViewPairEvidenceV1(
                0, 0, 1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1,
                "support:left",
            ),
            IRISGraphicsViewPairEvidenceV1(
                0, 0, 1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1,
                "support:right",
            ),
        ),
        visibility=(),
    )
    with pytest.raises(QualificationError, match="LOCAL_INTERLEAVING_RUNTIME_SUPPORT_REQUIRED"):
        compile_iris_graphics_v1(
            canonical_mesh_binding_hash=MESH,
            canonical_mesh_skin_binding_hash=SKIN,
            source_graphics_binding_hash=SOURCE,
            faces=((0, 1, 2), (1, 3, 2)),
            evidence=evidence,
        )


def test_visibility_is_independent_and_missing_does_not_mean_visible():
    evidence = _evidence(visibility=False)
    compiled = compile_iris_graphics_v1(
        canonical_mesh_binding_hash=MESH,
        canonical_mesh_skin_binding_hash=SKIN,
        source_graphics_binding_hash=SOURCE,
        faces=((0, 1, 2), (1, 3, 2)),
        evidence=evidence,
    )
    visible = compiled.views[0].visible_fragment_ids
    assert len(visible) == 1
