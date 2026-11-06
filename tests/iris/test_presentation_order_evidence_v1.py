from compiler.realsas_compiler_core.presentation_order_evidence_v1 import (
    compile_presentation_order_evidence_v1,
)


def src(a="OWNER:cape:00", b="OWNER:body:00", view=3, support=12, confidence=0.7):
    return {
        "relation_kind": "FRONT_OF_SOURCE_EVIDENCE",
        "a_owner_id": a,
        "b_owner_id": b,
        "view_index": view,
        "support_count": support,
        "confidence": confidence,
    }


def conflict(a="OWNER:cape:00", b="OWNER:body:00", view=3, frame=14, reversed_pixels=9, overlap=10):
    return {
        "relation_kind": "POSE_ORDER_CONFLICT",
        "a_owner_id": a,
        "b_owner_id": b,
        "view_index": view,
        "frame_index": frame,
        "support_count": reversed_pixels,
        "metadata": {"reversed_pixels": reversed_pixels, "overlap_pixels": overlap},
    }


def test_source_plus_observed_reversal_qualifies_exact_view_and_frames():
    edges, audit = compile_presentation_order_evidence_v1(
        source_relations=[src()],
        pose_conflicts=[
            conflict(frame=14, reversed_pixels=3, overlap=10),
            conflict(frame=15, reversed_pixels=9, overlap=10),
        ],
    )
    assert audit["status"] == "PASS"
    assert len(edges) == 1
    edge = edges[0]
    assert edge.front_owner_id == "OWNER:cape:00"
    assert edge.back_owner_id == "OWNER:body:00"
    assert edge.view_index == 3
    # Peak qualifies the directed edge; every frame with an observed reversal is
    # retained as the local valid frame domain.
    assert edge.valid_frame_indices == (14, 15)
    assert edge.observed_reversed_pixel_count == 12
    assert edge.peak_reversal_fraction == 0.9


def test_source_without_motion_reversal_abstains():
    edges, audit = compile_presentation_order_evidence_v1(
        source_relations=[src()], pose_conflicts=[]
    )
    assert edges == []
    assert audit["rejections"]["no_matching_reversal"] == 1


def test_low_source_support_abstains():
    edges, audit = compile_presentation_order_evidence_v1(
        source_relations=[src(support=4)], pose_conflicts=[conflict()]
    )
    assert edges == []
    assert audit["rejections"]["low_source_support"] == 1


def test_reverse_direction_contradiction_abstains_both():
    a, b = "OWNER:cape:00", "OWNER:body:00"
    edges, audit = compile_presentation_order_evidence_v1(
        source_relations=[src(a, b), src(b, a)],
        pose_conflicts=[conflict(a, b), conflict(b, a)],
    )
    assert edges == []
    assert audit["rejections"]["reverse_direction_contradiction"] == 2


def test_cycle_frame_is_removed_fail_closed():
    a, b, c = "OWNER:cape:00", "OWNER:body:00", "OWNER:sword:00"
    sources = [src(a, b), src(b, c), src(c, a)]
    conflicts = [
        conflict(a, b, frame=4),
        conflict(b, c, frame=4),
        conflict(c, a, frame=4),
    ]
    edges, audit = compile_presentation_order_evidence_v1(
        source_relations=sources, pose_conflicts=conflicts
    )
    assert edges == []
    assert audit["cyclic_domains"] == [{"view_index": 3, "frame_index": 4}]
    assert audit["rejections"]["cycle_frame"] == 3
