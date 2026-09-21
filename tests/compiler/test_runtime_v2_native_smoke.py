from __future__ import annotations

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.appearance_render_v2 import render_caa_reference
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    RuntimeClipV2IR,
    RuntimeProjectionV2IR,
    RuntimeViewV2IR,
    runtime_projection_hash,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_rss_v2_entries,
    write_rss_v2,
)
from dataclasses import replace


def _player() -> Path:
    raw = os.environ.get("REALSAS_RUNTIME_V2_PLAYER", "")
    if not raw:
        pytest.skip("native V2 player is only required on the self-hosted product gate")
    path = Path(raw).expanduser().resolve()
    assert path.is_file(), path
    return path


def _camera(view_index: int, resolution: int = 32):
    return {
        "view_id": f"V{view_index}",
        "view_index": view_index,
        "origin": (0.0, 0.0, -2.0),
        "right": (1.0, 0.0, 0.0),
        "screen_up": (0.0, 1.0, 0.0),
        "forward": (0.0, 0.0, 1.0),
        "half_extent": 1.0,
        "resolution": resolution,
    }


def test_native_v2_rss_smoke_matches_python_reference_byte_exact(tmp_path: Path):
    player = _player()
    vertices = np.asarray(
        [
            (-0.75, -0.75, 0.0),
            (0.75, -0.75, 0.0),
            (0.0, 0.75, 0.0),
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    face_uv = np.asarray(
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    projection_npz = tmp_path / "projection.npz"
    np.savez_compressed(
        projection_npz,
        vertices=vertices,
        faces=faces,
        face_uv=face_uv,
        clip_0_times=np.asarray([0.0], dtype=np.float64),
        clip_0_positions=vertices.reshape(1, 3, 3),
    )

    textures = []
    views = []
    for vi in range(8):
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        if vi == 0:
            # Gamma-vs-linear discriminator. At u=0.5 the black/white edge
            # must encode linear-light 0.5 as sRGB ~=188, never gamma-space 128.
            rgba[:, :8, :] = (0, 0, 0, 255)
            rgba[:, 8:, :] = (255, 255, 255, 255)
        else:
            rgba[:, :, :] = (80 + vi, 120, 160, 255)
        path = tmp_path / f"V{vi}.png"
        Image.fromarray(rgba, mode="RGBA").save(path)
        digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        textures.append(path)
        views.append(
            RuntimeViewV2IR(
                view_index=vi,
                view_id=f"V{vi}",
                camera=_camera(vi),
                texture_path=str(path),
                texture_sha256=digest,
            )
        )

    provenance = np.zeros((8, 16, 16), dtype=np.uint8)
    provenance_npz = tmp_path / "provenance.npz"
    np.savez_compressed(provenance_npz, provenance=provenance)
    sha = lambda p: __import__("hashlib").sha256(Path(p).read_bytes()).hexdigest()

    projection = RuntimeProjectionV2IR(
        complete_puppet_binding_hash="a" * 64,
        mechanical_state_binding_hash="b" * 64,
        mesh_binding_hash="c" * 64,
        dynamic_motion_binding_hash="d" * 64,
        appearance_asset_binding_hash="e" * 64,
        appearance_qualification_binding_hash="f" * 64,
        camera_set_binding_hash="1" * 64,
        visibility_contract_hash="2" * 64,
        projection_npz_path=str(projection_npz),
        projection_npz_sha256=sha(projection_npz),
        provenance_npz_path=str(provenance_npz),
        provenance_npz_sha256=sha(provenance_npz),
        views=tuple(views),
        clips=(
            RuntimeClipV2IR(
                clip_id="smoke",
                duration_seconds=1.0,
                loop=False,
                frame_count=1,
                array_prefix="clip_0",
            ),
        ),
        projection_hash="",
    )
    projection = replace(projection, projection_hash=runtime_projection_hash(projection))

    rss = tmp_path / "smoke.rss"
    entries = build_rss_v2_entries(projection)
    write_rss_v2(rss, entries)
    native_rgba = tmp_path / "native.rgba"
    native_prov = tmp_path / "native.prov"
    native_owner = tmp_path / "native.owner"
    proc = subprocess.run(
        [
            str(player),
            str(rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(native_rgba),
            "--out-provenance",
            str(native_prov),
            "--out-owner",
            str(native_owner),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "renderer=REALSAS_V2_CAA_CANONICAL_DEPTH" in proc.stdout

    mesh = SimpleNamespace(
        vertices=tuple(
            SimpleNamespace(canonical_mesh_vertex_id=f"v{i}", P=tuple(map(float, p)))
            for i, p in enumerate(vertices)
        ),
        faces=(("v0", "v1", "v2"),),
    )
    cam = CameraProjectionV3(
        view_id="V0",
        view_index=0,
        origin=(0.0, 0.0, -2.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=1.0,
        resolution=32,
    )
    reference = render_caa_reference(
        mesh=mesh,
        camera=cam,
        face_uv=face_uv.astype(np.float64),
        texture_rgba_u8=np.asarray(Image.open(textures[0]).convert("RGBA"), dtype=np.uint8),
        provenance_atlas=provenance[0],
        positions=vertices.astype(np.float64),
    )
    native = np.frombuffer(native_rgba.read_bytes(), dtype=np.uint8).reshape(32, 32, 4)
    assert np.array_equal(native, reference.straight_rgba_u8)
    midpoint = native[16, 16]
    assert 186 <= int(midpoint[0]) <= 189
    assert int(midpoint[0]) == int(midpoint[1]) == int(midpoint[2])
    assert int(midpoint[3]) == 255

    native_p = np.frombuffer(native_prov.read_bytes(), dtype=np.uint8).reshape(32, 32)
    assert np.array_equal(native_p, reference.provenance_code)
    native_o = np.frombuffer(native_owner.read_bytes(), dtype="<i4").reshape(32, 32)
    assert np.array_equal(native_o, reference.owner_face_index)

    fractional = subprocess.run(
        [
            str(player),
            str(rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0.5",
            "--out-rgba",
            str(tmp_path / "fractional.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert fractional.returncode != 0
    assert "FRAME_INTEGER_INVALID" in fractional.stderr

    tampered = entries.copy()
    tampered_manifest = tampered["manifest.txt"].replace(
        b"host_interpolation_authorized=0",
        b"host_interpolation_authorized=1",
    )
    assert tampered_manifest != tampered["manifest.txt"]
    tampered["manifest.txt"] = tampered_manifest
    tampered_rss = tmp_path / "tampered_interpolation.rss"
    write_rss_v2(tampered_rss, tampered)
    rejected = subprocess.run(
        [
            str(player),
            str(tampered_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "HOST_INTERPOLATION_MUST_BE_FORBIDDEN" in rejected.stderr
