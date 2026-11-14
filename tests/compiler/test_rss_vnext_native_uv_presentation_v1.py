from __future__ import annotations

"""Real original two-image → canonical UV → C++ posed 2D presentation.

No graphics model runs here; all CAS refs are produced by the actual current
Shape/Mesh/Rig/Skin/Mechanics/Graphics product roles in fixture form.
Self-test separately proves two overlapping deformed faces switch order
through the native Graphics partial-order/depth policy.
"""

import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.graphics_reopen_v1 import (
    reopen_compiled_graphics_v1,
)
from compiler.realsas_compiler_core.runtime_graphics_binary_v1 import (
    encode_runtime_graphics_v1,
)
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_graphics_capability_v1,
)
from compiler.rss_vnext.workers.rss_vnext_native_uv_transport_v1 import (
    encode_native_rss_vnext_uv_v1,
)
from compiler.rss_vnext.workers.rss_vnext_uv_texture_atlas_v1 import (
    reopen_rss_vnext_uv_texture_atlas_v1,
)
from tests.compiler.test_rss_vnext_raw_graphics_consumer_v1 import (
    _full_parent_fixture,
)


REPO = Path(__file__).resolve().parents[2]
NATIVE = (
    REPO / "runtime/realsas_cpp/src/rss_vnext_uv_presentation_player.cpp"
)


def _project_two_frames(mesh):
    xyz = np.array([v.P for v in mesh.vertices], dtype=np.float64)
    projected = xyz[:, :2].copy()
    lows = projected.min(axis=0)
    widths = np.maximum(projected.max(axis=0) - lows, 1e-6)
    projected = (projected - lows) / widths * 44 + 8
    frames = np.zeros((2, len(xyz), 3), dtype=np.float64)
    frames[:, :, :2] = projected
    frames[:, :, 2] = 1.5
    frames[1, :, 0] += 7.
    return frames


def _native_pose_bytes(mesh, frames, *, output_side=64):
    indices = {
        vertex.canonical_mesh_vertex_id: i
        for i, vertex in enumerate(mesh.vertices)
    }
    faces = [[indices[key] for key in face] for face in mesh.faces]
    body = bytearray(b"RSPOSE01")
    body.extend(struct.pack(
        "<IIIIII", 1, len(mesh.vertices), len(faces),
        len(frames), output_side, output_side,
    ))
    for face in faces:
        body.extend(struct.pack("<III", *face))
    body.extend(np.asarray(frames, dtype="<f8").tobytes(order="C"))
    return bytes(body)


def _build(tmp_path):
    cas, request, _, _ = _full_parent_fixture(
        tmp_path, active_mechanics_courts=True,
    )
    out = execute_graphics_capability_v1(
        request, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert out["status"] == "PASS"
    uv_ref = next(
        x for x in out["outputs"]
        if x["artifact_type"] == "RealSaS.RSSvNextMeshUVTextureAtlasCandidate"
    )
    mesh_ref = next(
        x for x in request["input_artifacts"]
        if x["artifact_type"] == "RealSaS.CanonicalMeshIR"
    )
    mesh = canonical_mesh_from_dict_v1(json.loads(cas.read_object(mesh_ref)))
    compiled, runtime = reopen_compiled_graphics_v1(
        json.loads(cas.read_object(out["outputs"][0])),
        face_count=len(mesh.faces),
    )
    assert runtime.canonical_mesh_binding_hash == mesh.mesh_lineage_hash
    atlas_blob = cas.read_object(uv_ref)
    arrays, manifest = reopen_rss_vnext_uv_texture_atlas_v1(atlas_blob)
    assert manifest["inferred_surface_texture_completion_performed"]
    assert manifest["source_rgb_values_from_original_files"]
    assert manifest["neural_graphics_rgb_is_inferred_not_observed"]

    graphics_file = tmp_path / "runtime_graphics.bin"
    texture_file = tmp_path / "canonical_uv.bin"
    pose_file = tmp_path / "two_poses.bin"
    graphics_file.write_bytes(encode_runtime_graphics_v1(runtime))
    texture_file.write_bytes(encode_native_rss_vnext_uv_v1(
        uv_atlas_cas=atlas_blob,
        canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
    ))
    pose_file.write_bytes(_native_pose_bytes(mesh, _project_two_frames(mesh)))
    player = tmp_path / "rss_uv_native_player"
    compile_result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         str(NATIVE), "-o", str(player)],
        capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, (
        "Native RSS UV C++ compile failure:\n" + compile_result.stderr
    )
    return player, graphics_file, texture_file, pose_file, arrays, manifest


def _render(player, graphics, texture, pose, tmp_path,
            *, frame=0, source_view=-1, owner=False):
    output = tmp_path / f"image_f{frame}_source{source_view}.rgba"
    args = [
        str(player), str(graphics), str(texture), str(pose),
        str(frame), "0", str(source_view), str(output),
    ]
    if owner:
        args += [str(tmp_path / "owner.i32"), str(tmp_path / "authority.u8")]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    image = np.frombuffer(output.read_bytes(), dtype=np.uint8).reshape(64, 64, 4)
    return image, result.stdout


def test_real_canonical_mesh_uv_texture_renders_two_poses_in_cpp(
    tmp_path,
):
    player, gfx, uv, poses, atlas, manifest = _build(tmp_path)
    selftest = subprocess.run(
        [str(player), "--self-test"],
        capture_output=True, text=True, check=True,
    )
    assert "RSS_UV_NATIVE_POSED_DEPTH_AND_PARTIAL_ORDER_PASS" in selftest.stdout

    front, summary = _render(
        player, gfx, uv, poses, tmp_path, frame=0, source_view=0, owner=True,
    )
    inferred, _ = _render(
        player, gfx, uv, poses, tmp_path, frame=0, source_view=-1,
    )
    moved, _ = _render(
        player, gfx, uv, poses, tmp_path, frame=1, source_view=-1,
    )
    assert "RSS_UV_NATIVE_POSED_2D_RENDER_PASS" in summary
    assert np.count_nonzero(front[:, :, 3]) > 25
    assert np.count_nonzero(inferred[:, :, 3]) > 25
    assert not np.array_equal(inferred, moved)
    assert not np.array_equal(front, inferred)
    assert np.any(
        np.all(front[:, :, :3] == np.array([180, 35, 15]), axis=-1)
        & (front[:, :, 3] == 255)
    )
    assert np.any(
        np.all(inferred[:, :, :3] == np.array([128, 128, 128]), axis=-1)
        & (inferred[:, :, 3] == 255)
    )
    owners = np.fromfile(tmp_path / "owner.i32", dtype="<i4").reshape(64, 64)
    color_authority = np.fromfile(tmp_path / "authority.u8", dtype=np.uint8).reshape(64, 64)
    assert set(np.unique(owners)).issubset({-1, 0})
    assert 1 in set(np.unique(color_authority))
    assert color_authority.max() <= 2


def test_native_rejects_texture_from_another_canonical_mesh_and_bad_pose(
    tmp_path,
):
    player, gfx, uv, poses, _, _ = _build(tmp_path)
    original = bytearray(uv.read_bytes())
    # magic[8], 5x u32[20], first digest length[4], first SHA begins at 32.
    assert original[28:32] == struct.pack("<I", 64)
    original[32] = ord("f") if original[32] != ord("f") else ord("e")
    uv.write_bytes(original)
    output = tmp_path / "forged.rgba"
    called = subprocess.run([
        str(player), str(gfx), str(uv), str(poses),
        "0", "0", "-1", str(output),
    ], capture_output=True, text=True)
    assert called.returncode != 0
    assert "GRAPHICS_MESH_POSE_DOMAIN" in called.stderr
    assert not output.exists()

    # C++ must also reject missing/fake motion frames, not infer a pose.
    uv.write_bytes(original)  # corruption already expected
    damaged = bytearray(poses.read_bytes())
    damaged[20:24] = struct.pack("<I", 0)  # frame count = 0
    poses.write_bytes(damaged)
    called = subprocess.run([
        str(player), str(gfx), str(uv), str(poses),
        "0", "0", "-1", str(output),
    ], capture_output=True, text=True)
    assert called.returncode != 0
    assert "POSE_DOMAIN" in called.stderr
