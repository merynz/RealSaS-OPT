from __future__ import annotations

"""Original images → canonical mesh/skin → real Mechanics→Graphics→UV → .rss.

This is a developer PREVIEW package: source appearance and ProductRevision
independent acceptance remain false. Pose frames are computed by the real
IK/FK/LBS solver and verified again on reopening, not handwritten animation.
"""

from collections import OrderedDict
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
