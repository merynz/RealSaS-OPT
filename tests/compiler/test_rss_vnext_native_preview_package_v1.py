from __future__ import annotations

"""Original images → canonical mesh/skin → real Mechanics→Graphics→UV → .rss.

This is a developer PREVIEW package: source appearance and ProductRevision
independent acceptance remain false. Pose frames are computed by the real
IK/FK/LBS solver and verified again on reopening, not handwritten animation.
"""

from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    canonical_rig_from_dict_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    canonical_skin_from_dict_v2,
)
from compiler.realsas_compiler_core.graphics_reopen_v1 import (
    reopen_compiled_graphics_v1,
)
from compiler.realsas_compiler_core.graphics_runtime_v1 import (
    RuntimeGraphicsV1, RuntimeGraphicsViewV1, runtime_graphics_hash_v1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes, read_rss_v2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_graphics_capability_v1,
)
from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
    compile_rss_vnext_native_preview_package_v1,
    reopen_rss_vnext_native_preview_package_v1,
    edit_rss_vnext_native_preview_controls_v1,
    prepare_rss_vnext_virtual_preview_graphics_v1,
)
from tests.compiler.test_rss_vnext_raw_graphics_consumer_v1 import (
    _full_parent_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
PLAYER_CPP = ROOT / "runtime/realsas_cpp/src/rss_vnext_uv_presentation_player.cpp"


def _compiled_preview_fixture(tmp_path):
    cas, request, mechanics, skin_out = _full_parent_fixture(
        tmp_path, active_mechanics_courts=True,
    )
    graphics_out = execute_graphics_capability_v1(
        request, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert graphics_out["status"] == "PASS"
    def read_typed(kind):
        return json.loads(cas.read_object(next(
            x for x in request["input_artifacts"]
            if x["artifact_type"] == kind
        )))
    mesh = canonical_mesh_from_dict_v1(read_typed("RealSaS.CanonicalMeshIR"))
    rig = canonical_rig_from_dict_v2(read_typed("RealSaS.CanonicalRigIR"))
    skin = canonical_skin_from_dict_v2(read_typed("RealSaS.CanonicalSkinIR"))
    _, graphics = reopen_compiled_graphics_v1(
        json.loads(cas.read_object(graphics_out["outputs"][0])),
        face_count=len(mesh.faces),
    )
    uv = cas.read_object(next(
        ref for ref in graphics_out["outputs"]
        if ref["artifact_type"] == "RealSaS.RSSvNextMeshUVTextureAtlasCandidate"
    ))
    xyz = np.asarray([v.P for v in mesh.vertices], dtype=np.float64)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    span = max(0.5, float(np.ptp(xyz[:, :2], axis=0).max()))
    virtual_camera = {
        "view_index":0,
        "origin":(float(center[0]),float(center[1]),
                  float(xyz[:, 2].max() + max(3.,3.*span))),
        "right":(1.,0.,0.), "screen_up":(0.,1.,0.),
        "forward":(0.,0.,-1.),
        "half_extent":span*.75,
        "resolution":96,
    }
    cid = rig.controls[0].canonical_control_id
    controls = (
        (),
        (ControlTransformIRV1(
            canonical_control_id=cid, translation=(0.18,0.,0.),
        ),),
    )
    blob, manifest = compile_rss_vnext_native_preview_package_v1(
        mesh=mesh,rig=rig,skin=skin,graphics=graphics,
        uv_atlas_cas=uv,virtual_camera=virtual_camera,
        control_frames=controls,
    )
    archive = tmp_path / "real_neural_two_image_native_preview.rss"
    archive.write_bytes(blob)
    return archive, manifest, mesh,rig,skin,controls


def _build_player(tmp_path):
    exe = tmp_path / "rss_vnext_native_preview_player"
    compile_result = subprocess.run([
        "g++","-std=c++17","-O2","-Wall","-Wextra","-pedantic",
        str(PLAYER_CPP),"-o",str(exe),
    ],capture_output=True,text=True)
    assert compile_result.returncode == 0,compile_result.stderr
    return exe


def _render(exe,archive,tmp_path,frame,*,source_view=0,owners=False):
    output = tmp_path / f"native_frame_{frame}_source_{source_view}.rgba"
    cmd = [
        str(exe),"--rss",str(archive),str(frame),"0",
        str(source_view),str(output),
    ]
    owner_file=tmp_path / "native_owners.i32"
    authority_file=tmp_path / "native_authority.u8"
    if owners:
        cmd += [str(owner_file),str(authority_file)]
    result = subprocess.run(cmd,capture_output=True,text=True,check=True)
    assert "rss_archive=1" in result.stdout
    frame_rgba = np.frombuffer(output.read_bytes(),dtype=np.uint8).reshape(96,96,4)
    if owners:
        assert owner_file.stat().st_size == 96*96*4
        assert authority_file.stat().st_size == 96*96
        owners_data=np.fromfile(owner_file,dtype="<i4")
        authority=np.fromfile(authority_file,dtype=np.uint8)
        assert np.all(owners_data>=-1)
        assert set(np.unique(authority)).issubset({0,1,2})
        assert 1 in np.unique(authority)
    return frame_rgba,result.stdout


def test_real_two_original_images_reach_editable_mechanics_rss_and_native_2d_pixels(
    tmp_path,
):
    archive,manifest,mesh,rig,skin,controls = _compiled_preview_fixture(tmp_path)
    assert archive.read_bytes().startswith(b"RSASV2R1")
    assert manifest["admission"]=="DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION"
    assert manifest["product_revision_minted"]=="0"
    assert manifest["source_rgb_surface_ownership_qualified"]=="0"
    assert manifest["virtual_preview_draw_policy"] == (
        "ALL_CANONICAL_FRAGMENTS_POSED_DEPTH_ONLY"
    )
    assert manifest["virtual_preview_camera_registered_to_graphics_view"] == "0"
    assert manifest["texture_layer_draw_order_authority"] == (
        "POSED_DEPTH_ONLY_FOR_UNREGISTERED_VIRTUAL_CAMERA"
    )
    reopened = reopen_rss_vnext_native_preview_package_v1(archive)
    assert reopened["mesh"].mesh_lineage_hash==mesh.mesh_lineage_hash
    assert reopened["rig"].rig_lineage_hash==rig.rig_lineage_hash
    assert reopened["skin"].skin_lineage_hash==skin.skin_lineage_hash
    assert reopened["control_frames"]==controls
    assert not reopened["product_revision_minted"]
    assert set(read_rss_v2(archive))=={
        "manifest.txt","mesh.json","rig.json","skin.json","graphics.bin",
        "texture.bin","pose.bin","controls.json",
    }
    exe=_build_player(tmp_path)
    first,receipt0=_render(exe,archive,tmp_path,0)
    second,receipt1=_render(exe,archive,tmp_path,1)
    assert not np.array_equal(first,second)
    assert np.count_nonzero(first[:,:,3])>20
    assert np.count_nonzero(second[:,:,3])>20
    assert "source_candidate_pixels=" in receipt0
    assert "graphics_order_overrides=" in receipt1
    assert "graphics_order_overrides=0" in receipt0
    assert "graphics_order_overrides=0" in receipt1
    inferred,_=_render(exe,archive,tmp_path,0,source_view=-1)
    assert np.count_nonzero(inferred[:,:,3])>20
    assert not np.array_equal(first,inferred)
    _,_=_render(exe,archive,tmp_path,0,source_view=0,owners=True)
    # Exactly two canonical authored ROOT states have independently replayed
    # projected positions; texture color changes cannot masquerade as motion.
    projected=np.frombuffer(
        reopened["pose_bytes"][32+len(mesh.faces)*12:],
        dtype="<f8",
    ).reshape(2,len(mesh.vertices),3)
    assert not np.array_equal(projected[0],projected[1])
    assert np.any(np.abs(projected[1,:,:2]-projected[0,:,:2])>1e-4)


def test_rehashed_rss_package_cannot_hide_edits_that_no_longer_match_real_mechanics(
    tmp_path,
):
    archive,_,_,_,_,_=_compiled_preview_fixture(tmp_path)
    entries=read_rss_v2(archive)
    original=entries["pose.bin"]
    altered=bytearray(original)
    # Preserve a legal positive camera Z while altering a vertex X coordinate.
    altered[-24] ^= 1
    entries["pose.bin"]=bytes(altered)
    from hashlib import sha256
    lines=entries["manifest.txt"].decode().splitlines()
    lines=[
        f"sha256:pose.bin={sha256(entries['pose.bin']).hexdigest()}"
        if line.startswith("sha256:pose.bin=") else line for line in lines
    ]
    entries["manifest.txt"]=("\n".join(lines)+"\n").encode()
    archive.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="MECHANICS_POSE_REPLAY_DRIFT"):
        reopen_rss_vnext_native_preview_package_v1(archive)


def test_false_product_revision_flag_does_not_pass_rss_reopen_or_native_player(
    tmp_path,
):
    archive,_,_,_,_,_=_compiled_preview_fixture(tmp_path)
    entries=read_rss_v2(archive)
    entries["manifest.txt"]=entries["manifest.txt"].replace(
        b"product_revision_minted=0", b"product_revision_minted=1",
    )
    archive.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="FALSE_PRODUCT_AUTHORITY"):
        reopen_rss_vnext_native_preview_package_v1(archive)
    exe=_build_player(tmp_path)
    bad=subprocess.run([
        str(exe),"--rss",str(archive),"0","0","-1",str(tmp_path/"bad.rgba"),
    ],capture_output=True,text=True)
    assert bad.returncode!=0
    assert "FALSE_SOURCE_OR_PRODUCT_AUTHORITY" in bad.stderr



def test_actual_native_puppet_control_edit_preserves_mesh_texture_and_reopens(
    tmp_path,
):
    original,_,mesh,rig,_,_= _compiled_preview_fixture(tmp_path)
    root=next(control for control in rig.controls
              if control.control_kind=="ROOT")
    edited_frames=(
        (),
        (ControlTransformIRV1(
            canonical_control_id=root.canonical_control_id,
            translation=(0.42,0.08,0.),
        ),),
    )
    edit_bytes,edit_manifest=edit_rss_vnext_native_preview_controls_v1(
        source_package_path=original,control_frames=edited_frames,
    )
    saved=tmp_path/"puppet_after_user_control_edit.rss"
    saved.write_bytes(edit_bytes)
    reopened=reopen_rss_vnext_native_preview_package_v1(saved)
    assert reopened["control_frames"]==edited_frames
    assert reopened["product_revision_minted"] is False
    assert edit_manifest["admission"]=="DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION"
    assert reopened["mesh"].mesh_lineage_hash==mesh.mesh_lineage_hash

    a,b=read_rss_v2(original),read_rss_v2(saved)
    for immutable in (
        "mesh.json","rig.json","skin.json","graphics.bin","texture.bin",
    ):
        assert a[immutable]==b[immutable]
    assert a["controls.json"]!=b["controls.json"]
    assert a["pose.bin"]!=b["pose.bin"]
    assert a["manifest.txt"]!=b["manifest.txt"]
    assert (
        json.loads(b["controls.json"])["edit_operation"]
        =="CANONICAL_CONTROLS_RECOMPUTED_MECHANICS_LBS"
    )
    exe=_build_player(tmp_path)
    old_rgb,_=_render(exe,original,tmp_path,1,source_view=-1)
    edited_rgb,_=_render(exe,saved,tmp_path,1,source_view=-1)
    assert not np.array_equal(old_rgb,edited_rgb)
    assert np.count_nonzero(edited_rgb[:,:,3])>20

    # An unknown animator target must not create a new phantom joint/control.
    with pytest.raises(Exception,match="UNKNOWN"):
        edit_rss_vnext_native_preview_controls_v1(
            source_package_path=saved,
            control_frames=(
                (),
                (ControlTransformIRV1(
                    canonical_control_id="FORGED_CONTROL_DOES_NOT_EXIST",
                    translation=(1.,0.,0.),
                ),),
            ),
        )



def test_unregistered_virtual_camera_does_not_reuse_source_view_zero_visibility():
    """A virtual axis must not hide predicted triangles behind view-0 masks."""
    views = tuple(
        RuntimeGraphicsViewV1(
            view_index=i,
            visible_fragment_indices=(0,) if i == 0 else (0, 1),
            topological_fragment_order=(1, 0) if i == 0 else (0, 1),
            order_edges=((1, 0),) if i == 0 else (),
        ) for i in range(8)
    )
    metadata = {
        "source_visibility_coverage_complete": False,
        "pose_conditioned_order_execution": False,
        "source_visibility_requires_independent_ray_receipt": True,
    }
    original = RuntimeGraphicsV1(
        canonical_mesh_binding_hash="a" * 64,
        compiled_graphics_binding_hash="b" * 64,
        face_fragment_index=(0, 1),
        fragment_ids=("front", "rear"),
        fragment_support_band_pixels=(0, 0),
        views=views, attachments=(),
        runtime_graphics_hash="", metadata=metadata,
    )
    original = replace(
        original, runtime_graphics_hash=runtime_graphics_hash_v1(original),
    )
    converted = prepare_rss_vnext_virtual_preview_graphics_v1(
        original, canonical_mesh_hash="a" * 64, face_count=2,
    )
    assert converted.views[0].visible_fragment_indices == (0, 1)
    assert converted.views[0].order_edges == ()
    assert converted.views[0].topological_fragment_order == (0, 1)
    assert converted.views[1:] == original.views[1:]
    assert converted.compiled_graphics_binding_hash == original.compiled_graphics_binding_hash
    assert converted.runtime_graphics_hash != original.runtime_graphics_hash
    assert converted.metadata["virtual_camera_view_is_registered"] is False
    assert converted.metadata["virtual_preview_source_view_visibility_reused"] is False
    # Never promote an unregistered virtual angle into source visibility truth.
    assert converted.metadata["source_visibility_coverage_complete"] is False



def test_native_rejects_rehashed_virtual_camera_order_authority(tmp_path):
    """An unregistered camera cannot claim view-specific ordering by metadata."""
    archive, _, _, _, _, _ = _compiled_preview_fixture(tmp_path)
    entries = read_rss_v2(archive)
    original_policy = b"ALL_CANONICAL_FRAGMENTS_POSED_DEPTH_ONLY"
    assert original_policy in entries["manifest.txt"]
    entries["manifest.txt"] = entries["manifest.txt"].replace(
        original_policy, b"UNREGISTERED_VIEW_USES_LEARNED_SOURCE_OCCLUSION",
    )
    forged = tmp_path / "forged_virtual_view_authority.rss"
    forged.write_bytes(_entry_bytes(entries))
    exe = _build_player(tmp_path)
    output = tmp_path / "forged_virtual_view_authority.rgba"
    result = subprocess.run([
        str(exe), "--rss", str(forged),
        "0", "0", "-1", str(output),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "RSS_UV_PREVIEW_UNREGISTERED_VIEW_POLICY_DRIFT" in result.stderr
    assert not output.exists()
