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


def test_native_cpp_raw_renderer_consumes_bound_live_frames_without_rewriting_rss(
    tmp_path,
):
    """Full native pixels prove new stream ABI matches the archived LBS ABI."""
    from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
        PuppetRuntimeFrameV1,
    )
    from compiler.realsas_compiler_core.playback_runtime_v3 import (
        RuntimeV3FrameComposition,
    )
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
        deform_mesh_lbs_v1, solve_control_pose_v1,
    )
    from compiler.rss_vnext.workers.rss_vnext_live_mechanics_native_pose_v1 import (
        bind_rss_vnext_live_pose_to_native_preview_v1,
    )
    from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
        reopen_rss_vnext_native_preview_package_v1,
    )
    from tests.compiler.test_rss_vnext_native_preview_package_v1 import (
        _compiled_preview_fixture, _build_player, _render,
    )
    import subprocess

    archive,_,mesh,rig,skin,controls=_compiled_preview_fixture(tmp_path)
    reopened=reopen_rss_vnext_native_preview_package_v1(archive)
    frames=[]
    for rows in controls:
        pose=solve_control_pose_v1(rig,rows)
        deformed=deform_mesh_lbs_v1(mesh,rig,skin,pose)
        frames.append(PuppetRuntimeFrameV1(
            view_id="V0",meshes=(),
            composition=RuntimeV3FrameComposition("V0",(),{}),
            pose=pose,deformed=deformed,
            product_revision_id="developer-runtime-not-product",
        ))
    pose_bytes,receipt=encode_rss_vnext_live_mechanics_native_pose_v1(
        mesh=mesh,rig=rig,skin=skin,frames=frames,
        frame_times_seconds=(0.,1./60.),
        cameras=reopened["virtual_camera"],
    )
    assert pose_bytes==reopened["pose_bytes"]
    assets,wire=bind_rss_vnext_live_pose_to_native_preview_v1(
        template_rss_path=archive,live_pose_bytes=pose_bytes,
        live_pose_receipt=receipt,
    )
    assert wire["product_release_admission"] is False
    assert wire["template_pose_remains_authoring_lbs"] is True
    for filename,data in assets.items():
        (tmp_path/filename).write_bytes(data)
    exe=_build_player(tmp_path)
    raw=tmp_path/"ephemeral-native-rgba.rgba"
    subprocess.run([
        str(exe),str(tmp_path/"graphics.bin"),
        str(tmp_path/"texture.bin"),str(tmp_path/"pose.bin"),
        "1","0","0",str(raw),
    ],check=True,capture_output=True,text=True)
    expected,_=_render(exe,archive,tmp_path,1)
    np.testing.assert_array_equal(
        np.frombuffer(raw.read_bytes(),dtype=np.uint8).reshape(expected.shape),
        expected,
    )
    forged={**receipt,"camera_policy_sha256":"0"*64}
    with pytest.raises(ValueError,match="CAMERA_POLICY_DRIFT"):
        bind_rss_vnext_live_pose_to_native_preview_v1(
            template_rss_path=archive,live_pose_bytes=pose_bytes,
            live_pose_receipt=forged,
        )
    assert archive.read_bytes().startswith(b"RSASV2R1")


def test_real_two_image_canonical_xpbd_final_pose_reaches_native_cpp_rgba(
    tmp_path,
):
    """Post-XPBD canonical pixels, not just a projected vertex/byte transport.

    Uses the existing actual raw-two-image-derived Mesh/Rig/Skin/Graphics/UV
    preview template. Native material/color are still NOT source-qualified.
    """
    import subprocess

    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
        mint_canonical_mechanics_v1,
    )
    from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
        deform_mesh_lbs_v1,solve_control_pose_v1,
    )
    from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
        DeclaredXPBDSurfaceRequestV1,XPBDSurfacePolicyV1,
        prepare_declared_xpbd_surface_domain_v1,step_declared_xpbd_surface_v1,
    )
    from compiler.realsas_compiler_core.playback_runtime_v3 import (
        RuntimeV3FrameComposition,
    )
    from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
        PuppetRuntimeFrameV1,
    )
    from compiler.rss_vnext.workers.rss_vnext_live_mechanics_native_pose_v1 import (
        bind_rss_vnext_live_pose_to_native_preview_v1,
    )
    from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
        reopen_rss_vnext_native_preview_package_v1,
    )
    from tests.compiler.test_rss_vnext_native_preview_package_v1 import (
        _compiled_preview_fixture,_build_player,
    )

    archive,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
    preserved=archive.read_bytes()
    reopened=reopen_rss_vnext_native_preview_package_v1(archive)
    # Mechanics is minted by the existing independent canonical qualifier;
    # there is no duplicate synthetic mesh, MIRA skin or preview camera.
    mechanics=mint_canonical_mechanics_v1(mesh,rig,skin)
    request=DeclaredXPBDSurfaceRequestV1(
        selected_faces=mesh.faces,
        pinned_vertex_ids=(mesh.vertices[0].canonical_mesh_vertex_id,),
    )
    domain=prepare_declared_xpbd_surface_domain_v1(
        mesh,rig,skin,mechanics,request,
    )
    policy=XPBDSurfacePolicyV1(domain_hash=domain.domain_hash)
    pose=solve_control_pose_v1(rig,())
    lbs=deform_mesh_lbs_v1(mesh,rig,skin,pose)
    first=step_declared_xpbd_surface_v1(
        mesh,lbs,domain,policy,
    )
    final=first
    # The existing solver is fixed-step (1/120 s). Every frame must
    # partition into an exact positive integer count of physics substeps.
    for _ in range(6):
        final=step_declared_xpbd_surface_v1(
            mesh,lbs,domain,policy,
            previous_state=final.state,dt_seconds=1./30.,
        )
    assert final.state.ticks>0
    assert final.deformation.canonical_mesh_vertex_ids==(
        lbs.canonical_mesh_vertex_ids
    )
    assert final.deformation.mesh_lineage_hash==mesh.mesh_lineage_hash
    assert final.deformation.positions!=lbs.positions
    assert final.deformation.positions[0]==lbs.positions[0]
    f0=PuppetRuntimeFrameV1(
        view_id="V0",meshes=(),
        composition=RuntimeV3FrameComposition("V0",(),{}),
        pose=pose,deformed=first.deformation,
        product_revision_id="developer-xpbd-graphics-not-product",
        xpbd_surface_state=first.state,
    )
    f1=replace(
        f0,deformed=final.deformation,xpbd_surface_state=final.state,
    )
    blob,receipt=encode_rss_vnext_live_mechanics_native_pose_v1(
        mesh=mesh,rig=rig,skin=skin,frames=(f0,f1),
        frame_times_seconds=(0.,6./30.),
        cameras=reopened["virtual_camera"],
    )
    assert blob!=reopened["pose_bytes"]  # final XPBD is NOT LBS replay.
    assert receipt["frames"][1]["xpbd_surface_state_hash"]==final.state.state_hash
    assert receipt["independent_xpbd_behavior_proof"] is False
    assert receipt["source_color_or_visibility_qualified"] is False
    assets,binding=bind_rss_vnext_live_pose_to_native_preview_v1(
        template_rss_path=archive,live_pose_bytes=blob,
        live_pose_receipt=receipt,
    )
    assert binding["product_release_admission"] is False
    for name,content in assets.items():
        (tmp_path/name).write_bytes(content)
    exe=_build_player(tmp_path)
    pixels=[]
    for frame in (0,1):
        target=tmp_path/f"native-xpbd-final-frame-{frame}.rgba"
        subprocess.run([
            str(exe),str(tmp_path/"graphics.bin"),
            str(tmp_path/"texture.bin"),str(tmp_path/"pose.bin"),
            str(frame),"0","0",str(target),
        ],check=True,capture_output=True,text=True)
        image=np.frombuffer(target.read_bytes(),dtype=np.uint8).reshape(
            96,96,4,
        )
        assert np.count_nonzero(image[:,:,3])>0
        pixels.append(image)
    assert not np.array_equal(pixels[0],pixels[1])
    assert archive.read_bytes()==preserved
