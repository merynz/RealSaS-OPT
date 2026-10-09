import numpy as np

from compiler.realsas_compiler_core.visual_contact_motion_v1 import apply_contact_projection
from compiler.realsas_compiler_core.visual_contact_v1 import (
    MATERIAL_CONTINUITY,
    contact_relation_metrics,
)
from compiler.realsas_compiler_core.visual_semantic_order_v1 import (
    compile_semantic_order,
    semantic_order_metrics,
)


def test_contact_projection_restores_cross_domain_delta_without_triangle_deformation():
    baseline = np.array([
        [0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
        [1., 0., 1.], [2., 0., 1.], [2., 1., 1.],
    ])
    repaired = baseline.copy()
    repaired[3:, 0] += 3.0
    domain = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    pair = np.array([[1, 3]], dtype=np.int64)
    projected, diag = apply_contact_projection(
        baseline_field=baseline,
        repaired_field=repaired,
        domain_id=domain,
        contact_pairs=pair,
    )
    before_edge = repaired[4] - repaired[3]
    after_edge = projected[4] - projected[3]
    assert np.allclose(before_edge, after_edge)
    assert diag["maximum_contact_delta_residual_after_px"] < 1e-6
    metrics = contact_relation_metrics(
        rest_positions=baseline[:, :2],
        baseline_positions=baseline[:, :2],
        posed_positions=projected[:, :2],
        pairs=pair,
        relation_codes=np.array([MATERIAL_CONTINUITY], dtype=np.int8),
    )
    assert metrics["qualified_contact_relations_passed"]


def test_semantic_order_creates_nonoverlapping_slot_bands_and_can_transition():
    slots = np.array([0, 0, 1, 1], dtype=np.int32)
    canonical = np.array([
        [1.0, 1.1, 2.0, 2.1],
        [2.2, 2.3, 1.0, 1.1],
    ])
    compiled = compile_semantic_order(
        canonical_depths=canonical,
        semantic_vertex_slot=slots,
    )
    metrics = semantic_order_metrics(
        effective_depths=compiled["effective_depths"],
        semantic_vertex_slot=slots,
        slot_ids=compiled["slot_ids"],
        rank_rows=compiled["rank_rows"],
    )
    assert metrics["semantic_occlusion_passed"]
    assert compiled["transition_count"] == 1
    assert np.all(compiled["effective_depths"] > 0)
