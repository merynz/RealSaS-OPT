from __future__ import annotations

"""Actual editor XPBD final positions -> native RSPOSE bytes (not product proof)."""

from dataclasses import replace
import struct

import numpy as np
import pytest

from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    qualify_camera_v3, project_points_xyz_v3,
)
from compiler.rss_vnext.workers.rss_vnext_live_mechanics_native_pose_v1 import (
    encode_rss_vnext_live_mechanics_native_pose_v1,
)
from tests.compiler.test_mechanics_xpbd_surface_editor_v1 import _editor


def _decode_positions(blob, *, faces, vertices, frames=2, views=1):
    assert blob[:8] == (b"RSPOSE01" if views == 1 else b"RSPOSE02")
    prefix = 32 if views == 1 else 36
    assert len(blob) == prefix + 12 * faces + 24 * frames * views * vertices
    return np.frombuffer(blob[prefix+12*faces:], dtype="<f8").reshape(
        frames, views, vertices, 3,
    )


def test_live_xpbd_last_surface_reaches_native_graphics_pose_bytes():
    bundle, domain, policy = _editor()
    session = ControllablePuppetSessionV1(bundle)
    session.enable_declared_xpbd_surface(domain, policy)
    initial = session.frame()
    final = session.frame(dt_seconds=1. / 60.)
    assert final.deformed.positions != initial.deformed.positions
    blob, receipt = encode_rss_vnext_live_mechanics_native_pose_v1(
        mesh=bundle.mesh, rig=bundle.rig, skin=bundle.skin,
        frames=(initial, final),
        frame_times_seconds=(0.,1./60.),
        cameras=bundle.cameras_by_view["V0"],
    )
    nverts=len(bundle.mesh.vertices)
    xyz=_decode_positions(blob,faces=len(bundle.mesh.faces),vertices=nverts)
    camera=qualify_camera_v3(
        bundle.cameras_by_view["V0"],view_id="V0",view_index=0,
    )
    expected=project_points_xyz_v3(
        np.asarray(final.deformed.positions,dtype=np.float64), camera,
    )
    np.testing.assert_allclose(xyz[1,0],expected,rtol=0,atol=0)
    assert not np.array_equal(xyz[0,0],xyz[1,0])
    assert receipt["frames"][1]["xpbd_surface_state_hash"]==final.xpbd_surface_state.state_hash
    assert receipt["source_color_or_visibility_qualified"] is False
    assert receipt["product_release_admission"] is False
    assert receipt["view_count"]==1


def test_same_live_surface_is_reprojected_to_eight_cameras():
    bundle,domain,policy=_editor()
    session=ControllablePuppetSessionV1(bundle)
    session.enable_declared_xpbd_surface(domain,policy)
    base=session.frame()
    later=session.frame(dt_seconds=1./60.)
    views=tuple(bundle.cameras_by_view[f"V{i}"] for i in range(8))
    blob,receipt=encode_rss_vnext_live_mechanics_native_pose_v1(
        mesh=bundle.mesh,rig=bundle.rig,skin=bundle.skin,
        frames=(base,later),frame_times_seconds=(0.,1./60.),
        cameras=views,
    )
    xyz=_decode_positions(
        blob,faces=len(bundle.mesh.faces),vertices=len(bundle.mesh.vertices),
        views=8,
    )
    for i in range(8):
        expected=project_points_xyz_v3(
            np.asarray(later.deformed.positions,dtype=np.float64),
            qualify_camera_v3(views[i],view_id=f"V{i}",view_index=i),
        )
        np.testing.assert_allclose(xyz[1,i],expected,rtol=0,atol=0)
    assert receipt["view_count"]==8
    assert receipt["frame_count"]==2


def test_bad_revision_lbs_or_timeline_cannot_be_accepted_as_live_frame():
    bundle,_,_=_editor()
    session=ControllablePuppetSessionV1(bundle)
    first=session.frame()
    frame=replace(first,product_revision_id="other")
    call=dict(mesh=bundle.mesh,rig=bundle.rig,skin=bundle.skin,
              cameras=bundle.cameras_by_view["V0"])
    with pytest.raises(ValueError,match="FRAME_PARENT_OR_VERTEX_DRIFT"):
        encode_rss_vnext_live_mechanics_native_pose_v1(
            **call,frames=(first,frame),frame_times_seconds=(0.,1.),
        )
    bad_position=tuple((x+0.5,y,z) for x,y,z in first.deformed.positions)
    altered=replace(first,deformed=replace(first.deformed,positions=bad_position))
    with pytest.raises(ValueError,match="LBS_REPLAY_DRIFT"):
        encode_rss_vnext_live_mechanics_native_pose_v1(
            **call,frames=(altered,),frame_times_seconds=(0.,),
        )
    with pytest.raises(ValueError,match="FRAME_TIME_INVALID"):
        encode_rss_vnext_live_mechanics_native_pose_v1(
            **call,frames=(first,first),frame_times_seconds=(0.,0.),
        )
