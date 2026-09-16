from __future__ import annotations

import inspect

import numpy as np
import pytest

from compiler.realsas_compiler_core.motion_3d_adapter_v1 import D1CompiledClipV1
from compiler.realsas_compiler_core.motion_3d_v1 import D1JointSpec, solve_fk_v1
from compiler.realsas_compiler_core.playback_runtime_v3 import RuntimeV3FrameComposition
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.playback_v3_bridge import build_runtime_v3_clip_from_d1


def _camera():
    return {
        "view_index": 0,
        "origin": [0.0, 0.0, -10.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "half_extent": 2.0,
        "resolution": 4,
    }


def _clip():
    joint = D1JointSpec("root", None, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    poses = (solve_fk_v1((joint,), {}), solve_fk_v1((joint,), {"root": 90.0}))
    return D1CompiledClipV1(
        clip_id="turn",
        duration_seconds=1.0,
        loop=False,
        times=(0.0, 1.0),
        semantic_rotation_degrees_by_frame=({}, {"root": 90.0}),
        poses=poses,
        joint_ids=("root",),
        axis_contract_hash="a" * 64,
        source_motion_state_hash="b" * 64,
        compile_hash="c" * 64,
    )


def _compositions():
    row = {
        "V0": RuntimeV3FrameComposition(
            view_id="V0",
            draw_order_slot_ids=("body_slot",),
            active_attachment_by_slot={"body_slot": "body_attachment"},
        )
    }
    return (row, row)


def _build(**kwargs):
    dense = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    weights = np.ones((3, 1), dtype=np.float64)
    return build_runtime_v3_clip_from_d1(
        d1_clip=_clip(),
        dense_vertices=dense,
        dense_weights=weights,
        weight_joint_ids=("root",),
        cameras={"V0": _camera()},
        attachment_id="body_attachment",
        composition_by_frame=_compositions(),
        display_name="Turn",
        intent="turn",
        nominal_fps=30.0,
        required_view_ids=("V0",),
        **kwargs,
    )


def test_d1_dense_lbs_and_runtime_xyz_depth_come_from_same_posed_3d_surface():
    runtime_clip, report = _build()
    rest_payload = runtime_clip.frames[0].posed_xyz_by_mesh["V0:body_attachment"]
    posed_payload = runtime_clip.frames[1].posed_xyz_by_mesh["V0:body_attachment"]
    assert isinstance(rest_payload, np.ndarray)
    assert rest_payload.dtype == np.float32
    assert rest_payload.flags.c_contiguous
    assert posed_payload.dtype == np.float32
    assert posed_payload.flags.c_contiguous
    rest = np.asarray(rest_payload)
    posed = np.asarray(posed_payload)
    # canonical [1,0,0] -> [0,0,-1] under +90deg Y rotation.
    assert rest[0] == pytest.approx([3.0, 2.0, 10.0], abs=1e-6)
    assert posed[0] == pytest.approx([2.0, 2.0, 9.0], abs=1e-6)
    assert report.frame_count == 2
    assert report.vertex_count == 3
    assert report.max_weight_row_sum_error <= 1e-12


def test_bridge_does_not_self_promote_runtime_qualification():
    runtime_clip, report = _build()
    assert runtime_clip.runtime_qualified is False
    assert report.runtime_qualified is False

    qualified_clip, qualified_report = _build(runtime_qualified=True)
    assert qualified_clip.runtime_qualified is True
    assert qualified_report.runtime_qualified is True
    assert qualified_report.bridge_hash != report.bridge_hash


def test_dense_weight_joint_order_is_exact_not_name_set_only():
    dense = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    with pytest.raises(QualificationError, match="WEIGHT_JOINT_ORDER_DRIFT"):
        build_runtime_v3_clip_from_d1(
            d1_clip=_clip(),
            dense_vertices=dense,
            dense_weights=weights,
            weight_joint_ids=("wrong",),
            cameras={"V0": _camera()},
            attachment_id="body_attachment",
            composition_by_frame=_compositions(),
            display_name="Turn",
            intent="turn",
            nominal_fps=30.0,
            required_view_ids=("V0",),
        )


def test_composition_is_explicit_and_cannot_be_invented_by_d1_bridge():
    dense = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    with pytest.raises(QualificationError, match="COMPOSITION_VIEW_SET_MISMATCH"):
        build_runtime_v3_clip_from_d1(
            d1_clip=_clip(),
            dense_vertices=dense,
            dense_weights=weights,
            weight_joint_ids=("root",),
            cameras={"V0": _camera()},
            attachment_id="body_attachment",
            composition_by_frame=({}, {}),
            display_name="Turn",
            intent="turn",
            nominal_fps=30.0,
            required_view_ids=("V0",),
        )


def test_bridge_is_subject_agnostic():
    source = inspect.getsource(__import__(
        "compiler.realsas_compiler_services.export.playback_v3_bridge",
        fromlist=["dummy"],
    )).lower()
    assert "mage" not in source
