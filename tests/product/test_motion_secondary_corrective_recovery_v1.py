from __future__ import annotations

import math

import pytest

from compiler.realsas_compiler_core.motion_secondary import (
    build_secondary_chain,
    simulate_secondary_chain,
)
from compiler.realsas_compiler_core.motion_corrective import (
    CorrectiveDeformerSpec,
    apply_corrective_stack,
    build_corrective_stack,
)


def test_recovered_xpbd_secondary_is_deterministic_root_locked_bounded_and_dynamic():
    chain = build_secondary_chain(
        chain_id="cape_soft_chain_v1",
        component_id="CAPE_FOREGROUND",
        parent_joint_id="J:CHEST",
        rest_points=((0.0, 0.0), (0.0, -0.2), (0.0, -0.4), (0.0, -0.6)),
        source_authority_hash="assembly-authority",
    )
    roots = tuple((0.05 * math.sin(i * 0.4), 0.0) for i in range(30))
    a_frames, a_report = simulate_secondary_chain(chain, root_positions=roots, frame_dt=1.0 / 30.0)
    b_frames, b_report = simulate_secondary_chain(chain, root_positions=roots, frame_dt=1.0 / 30.0)

    assert a_frames == b_frames
    assert a_report.report_hash == b_report.report_hash
    assert a_report.passed
    assert a_report.max_root_error <= 1e-9
    assert a_report.max_offset <= chain.policy.max_offset + 1e-8
    assert any(frame[-1] != a_frames[0][-1] for frame in a_frames[1:])
    assert all(frame[0] == roots[i] for i, frame in enumerate(a_frames))


def test_recovered_xpbd_requires_exact_typed_authority_and_nonzero_segments():
    with pytest.raises(ValueError):
        build_secondary_chain(
            chain_id="cape", component_id="CAPE", parent_joint_id="J:CHEST",
            rest_points=((0.0, 0.0), (0.0, 0.0), (0.0, -0.2)), source_authority_hash="x",
        )
    with pytest.raises(ValueError):
        build_secondary_chain(
            chain_id="cape", component_id="CAPE", parent_joint_id="J:CHEST",
            rest_points=((0.0, 0.0), (0.0, -0.1), (0.0, -0.2)), source_authority_hash="",
        )


def test_recovered_corrective_stack_is_lineage_bound_bounded_and_deterministic():
    deformers = (
        CorrectiveDeformerSpec(
            "torso_squash", "SQUASH_STRETCH", (0.0, 0.0), 1.0, 0.25, (0.0, 1.0),
            source_authority_hash="motion-quality-authority",
        ),
        CorrectiveDeformerSpec(
            "cloth_lag", "VELOCITY_LAG", (0.0, 0.0), 1.2, 0.5, (1.0, 0.0),
            parameters={"source_velocity_xy": (0.5, 0.0), "lag_seconds": 0.04},
            source_authority_hash="motion-quality-authority",
        ),
    )
    stack = build_corrective_stack(stack_id="mage_corrective_v1", mesh_lineage_hash="mesh-lineage", deformers=deformers)
    source = ((-0.2, 0.0), (0.2, 0.0), (0.0, 0.5), (2.0, 2.0))
    out_a, report_a = apply_corrective_stack(stack, mesh_lineage_hash="mesh-lineage", vertices=source)
    out_b, report_b = apply_corrective_stack(stack, mesh_lineage_hash="mesh-lineage", vertices=source)

    assert out_a == out_b
    assert report_a.report_hash == report_b.report_hash
    assert report_a.passed
    assert report_a.affected_vertex_count == 3
    assert out_a[-1] == source[-1]
    assert 0.0 < report_a.max_offset < 0.35

    with pytest.raises(ValueError):
        apply_corrective_stack(stack, mesh_lineage_hash="wrong", vertices=source)


def test_corrective_stack_fails_closed_on_unbounded_result():
    spec = CorrectiveDeformerSpec(
        "warp", "WARP", (0.0, 0.0), 1.0, 2.0, (1.0, 0.0),
        parameters={"max_offset": 1.0}, source_authority_hash="authority",
    )
    stack = build_corrective_stack(stack_id="bounded", mesh_lineage_hash="mesh", deformers=(spec,))
    with pytest.raises(ValueError):
        apply_corrective_stack(stack, mesh_lineage_hash="mesh", vertices=((0.0, 0.0),), max_allowed_offset=0.1)
