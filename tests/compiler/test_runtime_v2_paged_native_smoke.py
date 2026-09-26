from __future__ import annotations

from dataclasses import replace
import hashlib
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _player() -> Path:
    raw = os.environ.get("REALSAS_RUNTIME_V2_PLAYER", "")
    if not raw:
        pytest.skip("native V2 player is only required on the self-hosted product gate")
    path = Path(raw).expanduser().resolve()
    assert path.is_file(), path
    return path


def _camera(view_index: int, resolution: int = 32) -> dict:
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


def test_native_v2_paged_rss_matches_python_reference_byte_exact(tmp_path: Path):
    player = _player()
    vertices = np.asarray(
        [
            (-0.75, -0.75, 0.0),
            (0.75, -0.75, 0.0),
            (0.0, 0.75, 0.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    face_uv = np.asarray(
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    face_page_index = np.asarray([1], dtype=np.int32)
    projection_npz = tmp_path / "projection.npz"
    np.savez_compressed(
        projection_npz,
        vertices=vertices,
        faces=faces,
        face_uv=face_uv,
        face_page_index=face_page_index,
        clip_0_times=np.asarray([0.0], dtype=np.float64),
        clip_0_positions=vertices.reshape(1, 3, 3),
    )

    views = []
    texture_by_view = {}
    for vi in range(8):
        pages = []
        for page in range(2):
            rgba = np.zeros((16, 16, 4), dtype=np.uint8)
            if page == 0:
                rgba[:, :, :] = (240, 20, 20, 255)
            else:
                rgba[:, :, :] = (20 + vi, 220, 40, 255)
            path = tmp_path / f"V{vi}_p{page}.png"
            Image.fromarray(rgba, mode="RGBA").save(path)
            pages.append(
                {
                    "page_index": page,
                    "path": str(path),
                    "sha256": _sha(path),
                    "width": 16,
                    "height": 16,
                }
            )
        texture_by_view[vi] = np.stack(
            [
                np.asarray(
                    Image.open(row["path"]).convert("RGBA"),
                    dtype=np.uint8,
                )
                for row in pages
            ],
            axis=0,
        )
        views.append(
            RuntimeViewV2IR(
                view_index=vi,
                view_id=f"V{vi}",
                camera=_camera(vi),
                texture_path=pages[0]["path"],
                texture_sha256=pages[0]["sha256"],
                metadata={
                    "appearance": "SEALED_CAA_V2",
                    "paged_atlas": True,
                    "page_count": 2,
                    "texture_pages": pages,
                },
            )
        )

    provenance = np.zeros((8, 2, 16, 16), dtype=np.uint8)
    source_view = np.zeros((8, 2, 16, 16), dtype=np.int16)
    for vi in range(8):
        # Page 0 is exact direct-source evidence. Page 1 deliberately carries
        # OTHER_VIEW_SOURCE from a different donor so the native test proves
        # source-view lineage indexing includes the physical page dimension.
        source_view[vi, 0, :, :] = vi
        provenance[vi, 1, :, :] = 1
        source_view[vi, 1, :, :] = (vi + 1) % 8
    provenance_npz = tmp_path / "provenance.npz"
    np.savez_compressed(
        provenance_npz,
        provenance=provenance,
        source_view=source_view,
    )

    projection = RuntimeProjectionV2IR(
        complete_puppet_binding_hash="1" * 64,
        mechanical_state_binding_hash="2" * 64,
        mesh_binding_hash="3" * 64,
        dynamic_motion_binding_hash="4" * 64,
        appearance_asset_binding_hash="5" * 64,
        appearance_qualification_binding_hash="6" * 64,
        camera_set_binding_hash="7" * 64,
        visibility_contract_hash="8" * 64,
        projection_npz_path=str(projection_npz),
        projection_npz_sha256=_sha(projection_npz),
        provenance_npz_path=str(provenance_npz),
        provenance_npz_sha256=_sha(provenance_npz),
        views=tuple(views),
        clips=(
            RuntimeClipV2IR(
                clip_id="paged",
                duration_seconds=1.0,
                loop=False,
                frame_count=1,
                array_prefix="clip_0",
            ),
        ),
        projection_hash="",
        metadata={
            "appearance_paging_contract": "FACE_INDEX_TO_FIXED_PHYSICAL_PAGE_V1"
        },
    )
    projection = replace(
        projection,
        projection_hash=runtime_projection_hash(projection),
    )

    entries = build_rss_v2_entries(projection)
    assert "textures.bin" not in entries
    assert "provenance.bin" not in entries
    assert "face_pages.bin" in entries
    assert "provenance_paged.zlib" in entries
    assert "texture_v0_p0.png" in entries
    assert "texture_v0_p1.png" in entries
    manifest = entries["manifest.txt"].decode("utf-8")
    assert "atlas_paging_contract=FACE_INDEX_TO_FIXED_PHYSICAL_PAGE_V1" in manifest
    assert "atlas_page_count=2" in manifest

    rss = tmp_path / "paged.rss"
    write_rss_v2(rss, entries)
    native_rgba = tmp_path / "native.rgba"
    native_prov = tmp_path / "native.prov"
    native_owner = tmp_path / "native.owner"
    native_source_view = tmp_path / "native.source_view"
    proc = subprocess.run(
        [
            str(player),
            str(rss),
            "--clip",
            "paged",
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
            "--out-source-view",
            str(native_source_view),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "renderer=REALSAS_V2_CAA_CANONICAL_DEPTH" in proc.stdout
    assert "atlas_pages=2" in proc.stdout

    mesh = SimpleNamespace(
        vertices=tuple(
            SimpleNamespace(
                canonical_mesh_vertex_id=f"v{i}",
                P=tuple(map(float, point)),
            )
            for i, point in enumerate(vertices)
        ),
        faces=(("v0", "v1", "v2"),),
    )
    camera = CameraProjectionV3(
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
        camera=camera,
        face_uv=face_uv,
        face_page_index=face_page_index,
        texture_rgba_u8=texture_by_view[0],
        provenance_atlas=provenance[0],
        positions=vertices,
    )
    native = np.frombuffer(
        native_rgba.read_bytes(),
        dtype=np.uint8,
    ).reshape(32, 32, 4)
    native_p = np.frombuffer(
        native_prov.read_bytes(),
        dtype=np.uint8,
    ).reshape(32, 32)
    native_o = np.frombuffer(
        native_owner.read_bytes(),
        dtype="<i4",
    ).reshape(32, 32)
    native_sv = np.frombuffer(
        native_source_view.read_bytes(),
        dtype="<i2",
    ).reshape(32, 32)
    assert np.array_equal(native, reference.straight_rgba_u8)
    assert np.array_equal(native_p, reference.provenance_code)
    assert np.array_equal(native_o, reference.owner_face_index)
    center = tuple(map(int, native[16, 16]))
    assert center == (20, 220, 40, 255)
    assert int(native_p[16, 16]) == 1
    assert int(native_sv[16, 16]) == 1
