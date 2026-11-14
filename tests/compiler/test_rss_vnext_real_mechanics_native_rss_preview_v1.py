from __future__ import annotations

"""Actual raw Product Mesh/Rig/Skin/Mechanics/Graphics → .rss → native pixels.

No invented animation or trained weights. The two frames are actual
user-authored canonical ROOT control poses executed through the production
solver and LBS; this is a preview, NOT a qualified ProductRevision.
"""

from collections import OrderedDict
from pathlib import Path
import json
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
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes, read_rss_v2,
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

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "runtime/realsas_cpp/src/rss_vnext_uv_presentation_player.cpp"


def _real_preview_fixture(tmp_path):
    cas, req, _, _ = _full_parent_fixture(
        tmp_path, active_mechanics_courts=True,
    )
    graphics_result = execute_graphics_capability_v1(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert graphics_result["status"] == "PASS"
    artifacts = req["input_artifacts"]
    def original(kind):
        refs = [x for x in artifacts if x["artifact_type"] == kind]
        assert len(refs) == 1
        return json.loads(cas.read_object(refs[0]))
    mesh = canonical_mesh_from_dict_v1(original("RealSaS.CanonicalMeshIR"))
    rig = canonical_rig_from_dict_v2(original("RealSaS.CanonicalRigIR"))
    skin = canonical_skin_from_dict_v2(original("RealSaS.CanonicalSkinIR"))
    _, graphics = reopen_compiled_graphics_v1(
        json.loads(cas.read_object(graphics_result["outputs"][0])),
        face_count=len(mesh.faces),
    )
    uv_ref = next(row for row in graphics_result["outputs"]
                  if row["artifact_type"] == "RealSaS.RSSvNextMeshUVTextureAtlasCandidate")
    root_controls = [c for c in rig.controls if c.control_kind == "ROOT"]
    assert len(root_controls) == 1
    root = root_controls[0]
    frames = [
        (),
        (ControlTransformIRV1(
            canonical_control_id=root.canonical_control_id,
            translation=(0.15, 0.05, 0.),
        ),),
    ]
    xyz = np.asarray([v.P for v in mesh.vertices], dtype=np.float64)
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = (mins + maxs) / 2.
    span = float(max(maxs[0]-mins[0], maxs[1]-mins[1], .5))
    camera = {
        "view_index": 0,
        "origin": (float(center[0]), float(center[1]),
                   float(mins[2]-2.*span-2.)),
        "right": (1.,0.,0.),
        "screen_up": (0.,1.,0.),
        "forward": (0.,0.,1.),
        "half_extent": span*.85,
        "resolution": 128,
    }
    archive, manifest = compile_rss_vnext_native_preview_package_v1(
        mesh=mesh,rig=rig,skin=skin,graphics=graphics,
        uv_atlas_cas=cas.read_object(uv_ref),
        virtual_camera=camera,
        control_frames=frames,
    )
    path=tmp_path/"real_canonical_puppet_preview.rss"
    path.write_bytes(archive)
    return path, manifest, mesh, rig, skin, frames


def _native_player(tmp_path):
    exe=tmp_path/"rss_uv_native"
    proc=subprocess.run(
        ["g++","-std=c++17","-O2","-Wall","-Wextra","-pedantic",
         str(NATIVE),"-o",str(exe)],
        capture_output=True,text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return exe


def test_actual_source_puppet_controls_roundtrip_rss_and_native_2d_pixels(tmp_path):
    path, manifest, mesh, rig, skin, authored = _real_preview_fixture(tmp_path)
    reopened=reopen_rss_vnext_native_preview_package_v1(path)
    assert reopened["product_revision_minted"] is False
    assert reopened["mesh"].mesh_lineage_hash == mesh.mesh_lineage_hash
    assert reopened["rig"].rig_lineage_hash == rig.rig_lineage_hash
    assert reopened["skin"].skin_lineage_hash == skin.skin_lineage_hash
    assert reopened["control_frames"] == tuple(tuple(x) for x in authored)
    assert len(read_rss_v2(path)) == 8
    assert manifest["admission"] == "DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION"
    player=_native_player(tmp_path)
    rendered=[]
    for frame in (0,1):
        rgba=tmp_path/f"frame_{frame}.rgba"
        owner=tmp_path/f"frame_{frame}.owner"
        authority=tmp_path/f"frame_{frame}.authority"
        run=subprocess.run(
            [str(player),"--rss",str(path),str(frame),"0","-1",
             str(rgba),str(owner),str(authority)],
            capture_output=True,text=True,
        )
        assert run.returncode == 0, run.stderr
        assert "RSS_UV_NATIVE_POSED_2D_RENDER_PASS" in run.stdout
        assert "rss_archive=1" in run.stdout
        image=np.frombuffer(rgba.read_bytes(),dtype=np.uint8).reshape(128,128,4)
        assert np.count_nonzero(image[:,:,3]) > 20
        assert authority.stat().st_size == 128*128
        assert owner.stat().st_size == 128*128*4
        rendered.append(image)
    assert not np.array_equal(*rendered)
    assert not np.array_equal(
        np.frombuffer(reopened["pose_bytes"][-len(mesh.vertices)*24:],dtype="<f8"),
        np.frombuffer(reopened["pose_bytes"][
            32+len(mesh.faces)*12:
            32+len(mesh.faces)*12+len(mesh.vertices)*24
        ],dtype="<f8"),
    )


def test_reopen_refuses_rehashed_forged_pose_even_if_manifest_sha_is_updated(tmp_path):
    path, _, *_ = _real_preview_fixture(tmp_path)
    archive=read_rss_v2(path)
    damaged=bytearray(archive["pose.bin"])
    # This changes the first projected vertex X without changing the rig.
    vertex_offset=32 + 12
    damaged[vertex_offset+5] ^= 1
    archive["pose.bin"]=bytes(damaged)
    from hashlib import sha256
    lines=archive["manifest.txt"].decode().splitlines()
    lines=[
        "sha256:pose.bin="+sha256(bytes(damaged)).hexdigest()
        if line.startswith("sha256:pose.bin=") else line
        for line in lines
    ]
    archive["manifest.txt"]=("\n".join(lines)+"\n").encode()
    path.write_bytes(_entry_bytes(archive))
    with pytest.raises(RuntimeError,match="REPLAY_DRIFT"):
        reopen_rss_vnext_native_preview_package_v1(path)


def test_native_and_python_reject_false_product_preview_authority(tmp_path):
    path, _, *_ = _real_preview_fixture(tmp_path)
    archive=read_rss_v2(path)
    archive["manifest.txt"]=archive["manifest.txt"].replace(
        b"product_revision_minted=0",b"product_revision_minted=1",
    )
    path.write_bytes(_entry_bytes(archive))
    with pytest.raises(RuntimeError,match="FALSE_PRODUCT_AUTHORITY"):
        reopen_rss_vnext_native_preview_package_v1(path)
    player=_native_player(tmp_path)
    output=tmp_path/"forged.rgba"
    run=subprocess.run(
        [str(player),"--rss",str(path),"0","0","-1",str(output)],
        capture_output=True,text=True,
    )
    assert run.returncode!=0
    assert "FALSE_SOURCE_OR_PRODUCT_AUTHORITY" in run.stderr
    assert not output.exists()
