from __future__ import annotations

"""RSS model-driven regional relation -> sparse Compiler face-pair proposals.

All scores remain neural candidates. The source renders, mechanical contact,
2D presentation order and live attachment solver remain independently
qualified courts. Test fixture topology is synthetic but all head outputs
come from the real unified RSS model.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_face_relation_schedule_v1 import (
    compile_rss_vnext_sparse_face_relations_v1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    compile_rss_vnext_surface_candidates_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.models.test_rss_vnext_surface_decisions_v1 import (
    _real_surface_fixture, _fanout,
)


def _two_face_compiled():
    cfg, state, rgba, camera, evidence, carrier = _real_surface_fixture(
        two_faces=True,
    )
    reader = _fanout(cfg, evidence, carrier)
    surface = compile_rss_vnext_surface_candidates_v1(reader)
    assert len(surface.faces) == 2
    return cfg, state, reader, surface


def _expected_relation(state, surface, fa, fb):
    graph = state.region_relation_logits[0].detach().float().numpy()
    predictions = 1. / (1. + np.exp(-graph))
    a = np.asarray(surface.faces[fa].region_probabilities[:-1])
    b = np.asarray(surface.faces[fb].region_probabilities[:-1])
    w = np.outer(a, b)
    np.fill_diagonal(w, 0.)
    score = (predictions * w[..., None]).sum((0, 1)) / w.sum()
    return tuple(float(x) for x in score)


def test_real_rss_face_relation_compiler_preserves_direction_and_unknown_mass():
    cfg, state, reader, surface = _two_face_compiled()
    before = reader.query(0)
    matrix_before = reader.region_relation_matrix_v1()
    schedule = compile_rss_vnext_sparse_face_relations_v1(
        reader, surface, requested_face_pairs=[(0, 1), (1, 0)],
    )
    assert schedule.inference_evidence_hash == reader.evidence.evidence_hash
    assert schedule.unified_model_config_sha256 == reader.model_config_sha256
    assert schedule.neural_scene_carrier_sha256 == reader.neural_carrier_sha256
    assert schedule.candidate_mesh_sha256 == surface.candidate_mesh_sha256
    assert len(schedule.face_relations) == 2
    forward, reverse = schedule.face_relations
    assert forward.state == reverse.state == "UNQUALIFIED_SPARSE_LEARNED_PAIR"
    for row in schedule.face_relations:
        expected = _expected_relation(
            state, surface, row.from_face_index, row.to_face_index,
        )
        assert row.same_drawable_affinity == pytest.approx(expected[0], abs=1e-6)
        assert row.cut_affinity == pytest.approx(expected[1], abs=1e-6)
        assert row.directed_attachment_affinity == pytest.approx(expected[2], abs=1e-6)
        assert row.known_cross_slot_support > 0
        assert row.from_unknown_probability > 0
        assert row.to_unknown_probability > 0
        assert not row.source_drawable_qualified
        assert not row.source_separability_qualified
        assert not row.source_contact_qualified
        assert not row.mechanical_parent_anchor_qualified
        assert not row.source_visibility_qualified
    assert forward.same_drawable_affinity == pytest.approx(
        reverse.same_drawable_affinity, abs=1e-7,
    )
    assert forward.cut_affinity == pytest.approx(
        reverse.cut_affinity, abs=1e-7,
    )
    assert not schedule.independent_drawable_authority
    assert not schedule.attachment_authority
    assert not schedule.pose_conditioned_presentation_authority
    # The consumer of R×R scores must not mutate the full neural carrier.
    matrix_before[0, 1, 2] = 1000.
    assert reader.region_relation_matrix_v1()[0, 1, 2] != 1000.
    assert reader.query(0) == before


def test_sparse_surface_relation_never_trusts_rewritten_face_membership():
    _, _, reader, surface = _two_face_compiled()
    face = surface.faces[0]
    fake_probs = tuple(
        1. if i == 0 else 0.
        for i in range(len(face.region_probabilities))
    )
    forged = replace(surface, faces=(
        replace(face, region_probabilities=fake_probs),
        surface.faces[1],
    ))
    with pytest.raises(QualificationError, match="SOURCE_FACE_MEMBERSHIP_DRIFT"):
        compile_rss_vnext_sparse_face_relations_v1(
            reader, forged, requested_face_pairs=[(0, 1)],
        )


@pytest.mark.parametrize("pairs,error", [
    ([(0, 0)], "PAIR_DOMAIN_INVALID"),
    ([(0, True)], "PAIR_DOMAIN_INVALID"),
    ([(0, 9)], "PAIR_DOMAIN_INVALID"),
    ([(0, 1), (0, 1)], "PAIR_DUPLICATE"),
])
def test_rss_sparse_relation_consumer_refuses_unsupported_face_pairs(pairs, error):
    _, _, reader, surface = _two_face_compiled()
    with pytest.raises(QualificationError, match=error):
        compile_rss_vnext_sparse_face_relations_v1(
            reader, surface, requested_face_pairs=pairs,
        )


def test_rss_face_pair_compiler_requires_explicit_budget_no_quadratic_enumeration():
    _, _, reader, surface = _two_face_compiled()
    with pytest.raises(QualificationError, match="RESOURCE_BUDGET"):
        compile_rss_vnext_sparse_face_relations_v1(
            reader, surface,
            requested_face_pairs=[(0, 1)] * (reader.model_config.max_pair_queries + 1),
        )
    result = compile_rss_vnext_sparse_face_relations_v1(
        reader, surface, requested_face_pairs=[],
    )
    assert result.face_relations == ()
    assert result.candidate_mesh_sha256 == surface.candidate_mesh_sha256
