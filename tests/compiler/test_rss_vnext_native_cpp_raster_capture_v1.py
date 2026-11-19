from __future__ import annotations

"""Actual C++ RGB/owner/authority/UNKNOWN sidecar capture on canonical XPBD.

The test uses the real two-original-image Compiler canonical fixture, not a
source-registered camera or a learned/artist raster truth. Regressions only.
"""

from dataclasses import replace
from hashlib import sha256
import subprocess

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    mint_canonical_mechanics_v1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    solve_control_pose_v1,deform_mesh_lbs_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1,XPBDSurfacePolicyV1,
    prepare_declared_xpbd_surface_domain_v1,step_declared_xpbd_surface_v1,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    PuppetRuntimeFrameV1,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    RuntimeV3FrameComposition,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.rss_vnext_native_cpp_raster_capture_v1 import (
    AUTHORITY,
    native_cpp_capture_receipt_hash_v1,
    execute_native_cpp_candidate_raster_capture_v1,
    reopen_native_cpp_candidate_capture_v1,
)
from compiler.rss_vnext.workers.rss_vnext_live_mechanics_native_pose_v1 import (
    encode_rss_vnext_live_mechanics_native_pose_v1,
)
from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)
from tests.compiler.test_rss_vnext_native_preview_package_v1 import (
    _compiled_preview_fixture, _build_player,
)


def _source_and_native(tmp_path):
    archive,_,mesh,rig,skin,_=_compiled_preview_fixture(tmp_path)
    original_bytes=archive.read_bytes()
    exe=_build_player(tmp_path)
    renderer_sha=sha256(exe.read_bytes()).hexdigest()
    reopened=reopen_rss_vnext_native_preview_package_v1(archive)
    mechanics=mint_canonical_mechanics_v1(mesh,rig,skin)
    domain=prepare_declared_xpbd_surface_domain_v1(
        mesh,rig,skin,mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=mesh.faces,
            pinned_vertex_ids=(mesh.vertices[0].canonical_mesh_vertex_id,),
        ),
    )
    policy=XPBDSurfacePolicyV1(domain_hash=domain.domain_hash)
    pose=solve_control_pose_v1(rig,())
    lbs=deform_mesh_lbs_v1(mesh,rig,skin,pose)
    first=step_declared_xpbd_surface_v1(mesh,lbs,domain,policy)
    final=first
    for _ in range(6):
        final=step_declared_xpbd_surface_v1(
            mesh,lbs,domain,policy,previous_state=final.state,
            dt_seconds=1./30.,
        )
    assert final.deformation.positions!=lbs.positions
    frame=PuppetRuntimeFrameV1(
        view_id="V0",meshes=(),
        composition=RuntimeV3FrameComposition("V0",(),{}),
        pose=pose,deformed=first.deformation,
        product_revision_id="developer-graphics-candidate",
        xpbd_surface_state=first.state,
    )
    final_frame=replace(
        frame,deformed=final.deformation,xpbd_surface_state=final.state,
    )
    blob,pose_receipt=encode_rss_vnext_live_mechanics_native_pose_v1(
        mesh=mesh,rig=rig,skin=skin,frames=(frame,final_frame),
        frame_times_seconds=(0.,6./30.),
        cameras=reopened["virtual_camera"],
    )
    kwargs=dict(
        renderer_executable=exe,
        expected_renderer_executable_sha256=renderer_sha,
        template_rss_path=archive,
        expected_template_rss_sha256=sha256(original_bytes).hexdigest(),
        live_pose_bytes=blob,live_pose_receipt=pose_receipt,
        view_index=0,
    )
    return kwargs,original_bytes


def test_actual_cpp_xpbd_rgba_and_all_three_sidecars_from_one_final_mstar(
    tmp_path,
):
    kwargs,original=_source_and_native(tmp_path)
    start,initial=execute_native_cpp_candidate_raster_capture_v1(
        **kwargs,frame_index=0,
    )
    bundle,final=execute_native_cpp_candidate_raster_capture_v1(
        **kwargs,frame_index=1,
    )
    assert initial.frame_index==0 and final.frame_index==1
    assert initial.native_raster_bundle_schema==final.native_raster_bundle_schema
    assert final.authority==AUTHORITY
    assert final.canonical_mesh_lineage_hash == (
        kwargs["live_pose_receipt"]["mesh_lineage_hash"]
    )
    assert final.camera_policy_sha256 == (
        kwargs["live_pose_receipt"]["camera_policy_sha256"]
    )
    assert final.source_view==-1
    assert final.native_executable_was_run
    assert final.source_color_or_visibility_qualified is False
    assert final.independent_source_camera_registered is False
    assert final.product_release_admission is False
    assert final.native_pose_bytes_sha256==sha256(
        kwargs["live_pose_bytes"]
    ).hexdigest()
    arrays,proof=reopen_native_cpp_candidate_capture_v1(
        native_raster_bundle=bundle,receipt=final.to_dict(),
        expected_renderer_executable_sha256=kwargs[
            "expected_renderer_executable_sha256"
        ],
        expected_template_rss_sha256=kwargs[
            "expected_template_rss_sha256"
        ],
        expected_native_pose_bytes_sha256=final.native_pose_bytes_sha256,
    )
    assert proof==final
    assert arrays["native_rgba_u8"].shape==(96,96,4)
    assert arrays["native_front_face_i32"].shape==(96,96)
    assert arrays["native_color_authority_u8"].shape==(96,96)
    assert arrays["native_unknown_occlusion_u8"].shape==(96,96)
    assert np.count_nonzero(arrays["native_rgba_u8"][:,:,3])>0
    start_images,_=reopen_native_cpp_candidate_capture_v1(
        native_raster_bundle=start,receipt=initial.to_dict(),
        expected_renderer_executable_sha256=final.renderer_executable_sha256,
        expected_template_rss_sha256=final.source_template_rss_sha256,
        expected_native_pose_bytes_sha256=final.native_pose_bytes_sha256,
    )
    assert not np.array_equal(
        start_images["native_rgba_u8"],arrays["native_rgba_u8"],
    )
    assert kwargs["template_rss_path"].read_bytes()==original
    # Verify the same raw-mode C++ invocation, not generated test-owner pixels.
    cmd=[
        str(kwargs["renderer_executable"]),
        str(tmp_path/"graphics.bin"),str(tmp_path/"texture.bin"),
        str(tmp_path/"pose.bin"),"1","0","-1",
        str(tmp_path/"direct.rgba"),str(tmp_path/"direct.owner"),
        str(tmp_path/"direct.authority"),
        "--unknown-mask="+str(tmp_path/"direct.unknown"),
    ]
    from compiler.rss_vnext.workers.rss_vnext_live_mechanics_native_pose_v1 import (
        bind_rss_vnext_live_pose_to_native_preview_v1,
    )
    assets,_=bind_rss_vnext_live_pose_to_native_preview_v1(
        template_rss_path=kwargs["template_rss_path"],
        live_pose_bytes=kwargs["live_pose_bytes"],
        live_pose_receipt=kwargs["live_pose_receipt"],
    )
    for name,contents in assets.items():
        (tmp_path/name).write_bytes(contents)
    subprocess.run(cmd,check=True,capture_output=True)
    assert (tmp_path/"direct.rgba").read_bytes()==arrays[
        "native_rgba_u8"
    ].tobytes()
    assert (tmp_path/"direct.owner").read_bytes()==arrays[
        "native_front_face_i32"
    ].astype("<i4").tobytes()
    assert (tmp_path/"direct.authority").read_bytes()==arrays[
        "native_color_authority_u8"
    ].tobytes()
    assert (tmp_path/"direct.unknown").read_bytes()==arrays[
        "native_unknown_occlusion_u8"
    ].tobytes()


def test_rejects_mismatched_renderer_template_or_forged_product_authority(
    tmp_path,
):
    args,_=_source_and_native(tmp_path)
    with pytest.raises(QualificationError,match="PINNED_EXECUTABLE_OR_TEMPLATE"):
        execute_native_cpp_candidate_raster_capture_v1(
            **{**args,"expected_renderer_executable_sha256":"0"*64},
            frame_index=1,
        )
    with pytest.raises(QualificationError,match="PINNED_EXECUTABLE_OR_TEMPLATE"):
        execute_native_cpp_candidate_raster_capture_v1(
            **{**args,"expected_template_rss_sha256":"0"*64},
            frame_index=1,
        )
    bundle,receipt=execute_native_cpp_candidate_raster_capture_v1(
        **args,frame_index=1,
    )
    validated=dict(
        native_raster_bundle=bundle,receipt=receipt.to_dict(),
        expected_renderer_executable_sha256=receipt.renderer_executable_sha256,
        expected_template_rss_sha256=receipt.source_template_rss_sha256,
        expected_native_pose_bytes_sha256=receipt.native_pose_bytes_sha256,
    )
    forged=replace(receipt,product_release_admission=True,receipt_hash="")
    forged=replace(
        forged,receipt_hash=native_cpp_capture_receipt_hash_v1(forged),
    )
    with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
        reopen_native_cpp_candidate_capture_v1(
            **{**validated,"receipt":forged.to_dict()},
        )
    forged=replace(receipt,independent_source_camera_registered=True,receipt_hash="")
    forged=replace(
        forged,receipt_hash=native_cpp_capture_receipt_hash_v1(forged),
    )
    with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
        reopen_native_cpp_candidate_capture_v1(
            **{**validated,"receipt":forged.to_dict()},
        )
    with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
        reopen_native_cpp_candidate_capture_v1(
            **{**validated,"native_raster_bundle":bundle+b"x"},
        )
    with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
        reopen_native_cpp_candidate_capture_v1(
            **{**validated,"expected_native_pose_bytes_sha256":"0"*64},
        )


def test_rejects_unauthorized_source_camera_and_modified_live_pose(tmp_path):
    kwargs,_=_source_and_native(tmp_path)
    bad=dict(kwargs)
    receipt={**bad["live_pose_receipt"],"source_color_or_visibility_qualified":True}
    bad["live_pose_receipt"]=receipt
    with pytest.raises(QualificationError,match="INPUT_POLICY"):
        execute_native_cpp_candidate_raster_capture_v1(
            **bad,frame_index=0,
        )
    bad=dict(kwargs)
    bad["live_pose_bytes"]+=b"tampered"
    with pytest.raises(ValueError,match="TEMPLATE_OR_RECEIPT_DRIFT"):
        execute_native_cpp_candidate_raster_capture_v1(
            **bad,frame_index=0,
        )
    with pytest.raises(QualificationError,match="INPUT_POLICY"):
        execute_native_cpp_candidate_raster_capture_v1(
            **kwargs,frame_index=99,
        )
