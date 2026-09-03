from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_services.proof.failure_signatures import derive_failure_signatures
from compiler.realsas_compiler_services.proof.motion_probe import (
    authored_motion_measurement_passes_v1,
    measure_authored_motion_v1,
)


def _product(*, collapse: bool = False):
    joint = SimpleNamespace(canonical_joint_id="J:0", parent_canonical_id=None, position=(0.0, 0.0, 0.0))
    skeleton = SimpleNamespace(joints=(joint,))
    vertices = tuple(SimpleNamespace(canonical_mesh_vertex_id=f"V:{i}", P=p) for i, p in enumerate(((0.,0.,0.), (1.,0.,0.), (0.,1.,0.))))
    mesh = SimpleNamespace(vertices=vertices, faces=(("V:0", "V:1", "V:2"),))
    rows = tuple(SimpleNamespace(canonical_mesh_vertex_id=f"V:{i}", influences=(("J:0", 1.0),)) for i in range(3))
    component = SimpleNamespace(component_id="C:0", mesh=mesh, mesh_skin=SimpleNamespace(rows=rows))
    direction = SimpleNamespace(view_index=0, components=(component,))
    mid_scale = (0.0, 0.0) if collapse else (1.0, 1.0)
    keys = (
        SimpleNamespace(time_sec=0.0, translation_xy=(0.,0.), rotation_deg=0.0, scale_xy=(1.,1.), depth_offset=0.0),
        SimpleNamespace(time_sec=0.25, translation_xy=(0.,0.), rotation_deg=4.0, scale_xy=mid_scale, depth_offset=0.0),
        SimpleNamespace(time_sec=0.75, translation_xy=(0.,0.), rotation_deg=-4.0, scale_xy=(1.,1.), depth_offset=0.0),
        SimpleNamespace(time_sec=1.0, translation_xy=(0.,0.), rotation_deg=0.0, scale_xy=(1.,1.), depth_offset=0.0),
    )
    clip = SimpleNamespace(clip_id="idle", clip_kind="PRESET", duration_sec=1.0, loop=True)
    track = SimpleNamespace(clip_id="idle", canonical_joint_id="J:0", keys=keys)
    return SimpleNamespace(
        mechanical_state=SimpleNamespace(skeleton=skeleton),
        directional_renderables=SimpleNamespace(directions=(direction,)),
        motion_state=SimpleNamespace(clips=(clip,), joint_tracks=(track,)),
    )


def test_authored_motion_dynamic_probe_passes_bounded_loop():
    m = measure_authored_motion_v1(_product())
    assert authored_motion_measurement_passes_v1(m)
    assert m["effective_clip_count"] == 1
    assert m["max_motion_normalized"] > 0.0
    assert m["max_loop_seam_normalized"] == 0.0
    assert m["causal_owner_attribution"] == "NOT_PERFORMED"


def test_authored_motion_dynamic_probe_rejects_collapse_and_localizes_without_owner():
    m = measure_authored_motion_v1(_product(collapse=True))
    assert not authored_motion_measurement_passes_v1(m)
    assert m["degenerate_triangle_instances"] > 0
    m["effective_joint_track_count"] = 1
    signatures = derive_failure_signatures("MOTION", m, status="FAIL")
    assert any(s["failure_family"] == "area_change_exceeded" for s in signatures)
    assert all(s["owner_attribution_status"] == "not_performed" for s in signatures)
