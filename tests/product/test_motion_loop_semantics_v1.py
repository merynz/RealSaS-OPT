from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_services.proof.motion_frame_metrics import (
    evaluate_motion_bake_metrics,
    measure_motion_bake_geometry,
)


def _frame(points):
    return SimpleNamespace(mesh_vertices_by_id=(("mesh", tuple(points)),))


def _bake(*, clip_id: str, rest, frames, loop: bool):
    return SimpleNamespace(
        clip_id=clip_id,
        loop=bool(loop),
        rest_mesh_vertices_by_id=(("mesh", tuple(rest)),),
        triangles_by_mesh_id=(("mesh", ((0, 1, 2),)),),
        frames=tuple(_frame(row) for row in frames),
    )


def test_nonrest_cyclic_pose_requires_loop_closure_but_not_rest_return():
    rest=((0.0,0.0),(1.0,0.0),(0.0,1.0))
    planted=((0.10,0.0),(1.10,0.0),(0.10,1.0))
    flight=((0.20,0.0),(1.20,0.0),(0.20,1.0))
    bake=_bake(clip_id="run",rest=rest,frames=(planted,flight,planted),loop=True)
    measured=measure_motion_bake_geometry(bake)
    result=evaluate_motion_bake_metrics(measured)

    assert measured["loop_seam_error01"] == 0.0
    assert measured["start_from_rest_error01"] > result["policy"]["max_return_to_rest_error01"]
    assert measured["return_to_rest_error01"] > result["policy"]["max_return_to_rest_error01"]
    assert result["starts_at_rest"] is False
    assert result["return_to_rest_evaluated"] is False
    assert "return_to_rest" not in result["failure_invariants"]
    assert result["passed"] is True


def test_nonrest_loop_still_fails_loop_seam_if_end_pose_drifts():
    rest=((0.0,0.0),(1.0,0.0),(0.0,1.0))
    start=((0.10,0.0),(1.10,0.0),(0.10,1.0))
    end=((0.16,0.0),(1.16,0.0),(0.16,1.0))
    bake=_bake(clip_id="bad_run",rest=rest,frames=(start,end),loop=True)
    result=evaluate_motion_bake_metrics(measure_motion_bake_geometry(bake))
    assert "loop_seam" in result["failure_invariants"]
    assert result["passed"] is False


def test_rest_starting_clip_preserves_historical_return_to_rest_gate():
    rest=((0.0,0.0),(1.0,0.0),(0.0,1.0))
    end=((0.10,0.0),(1.10,0.0),(0.10,1.0))
    bake=_bake(clip_id="one_shot",rest=rest,frames=(rest,end),loop=False)
    result=evaluate_motion_bake_metrics(measure_motion_bake_geometry(bake))
    assert result["starts_at_rest"] is True
    assert result["return_to_rest_evaluated"] is True
    assert "return_to_rest" in result["failure_invariants"]
    assert result["passed"] is False
