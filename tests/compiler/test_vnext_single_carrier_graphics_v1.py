from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.graphics_runtime_v1 import build_runtime_graphics_v1
from compiler.realsas_compiler_core.graphics_v1 import (
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
        mechanical_attachment_evidence={
            (0, 1): {
                "persistent_contact": 0.95,
                "parent_mechanical_domain_id": "joint:hand",
                "anchor_face_index": 0,
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
    assert all(
        view.metadata["triangle_array_order_is_presentation_authority"] is False
        for view in compiled.views
    )

    runtime = build_runtime_graphics_v1(compiled, face_count=3)
    assert runtime.face_fragment_index == (0, 1, 2)
    assert len(runtime.views) == 8
    assert runtime.metadata["hash_only_binding_forbidden"] is True
    assert runtime.metadata["behaviorally_consumable"] is True
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
