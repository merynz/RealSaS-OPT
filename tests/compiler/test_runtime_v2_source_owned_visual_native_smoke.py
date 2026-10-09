from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_source_owned_visual_rss_v2_entries,
    write_rss_v2,
)
from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
    SourceOwnedVisualRuntimeClipV1IR,
    SourceOwnedVisualRuntimeProjectionV1IR,
    SourceOwnedVisualRuntimeViewV1IR,
    source_owned_visual_runtime_projection_hash,
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


@pytest.mark.parametrize("canonical_depth", [False, True, "slot_rigid", "motion_blend", "connected"])
def test_source_owned_visual_rss_native_smoke(tmp_path: Path, canonical_depth):
    player = _player()
    source_size = 8
    render_size = 16

    arrays = {}
    views = []
    for vi in range(8):
        uv = np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.uint32)
        rest = np.asarray(
            ((1.0, 1.0), (6.0, 1.0), (1.0, 6.0)),
            dtype=np.float64,
        )
        arrays[f"view_{vi}_uv"] = uv
        arrays[f"view_{vi}_faces"] = faces
        arrays[f"view_{vi}_rest_positions"] = rest
        arrays[f"clip_0_view_{vi}_positions"] = rest.reshape(1, 3, 2)
        if canonical_depth:
            arrays[f"clip_0_view_{vi}_depths"] = np.full((1, 3), 2., dtype=np.float64)
        if canonical_depth == "connected":
            arrays[f"view_{vi}_face_attachment_owner"] = np.zeros(1, dtype=np.int32)

        rgba = np.zeros((source_size, source_size, 4), dtype=np.uint8)
        rgba[:, :, :] = (220, 40 + vi, 20, 255)
        texture = tmp_path / f"V{vi}.png"
        Image.fromarray(rgba, mode="RGBA").save(texture)

        visual_npz = tmp_path / f"visual_V{vi}.npz"
        np.savez_compressed(visual_npz, positions=rest, faces=faces, uv=uv)

        views.append(
            SourceOwnedVisualRuntimeViewV1IR(
                view_index=vi,
                view_id=f"V{vi}",
                camera={
                    "view_id": f"V{vi}",
                    "view_index": vi,
                    "resolution": render_size,
                },
                source_width=source_size,
                source_height=source_size,
                texture_path=str(texture),
                texture_sha256=_sha(texture),
                visual_mesh_npz_path=str(visual_npz),
                visual_mesh_npz_sha256=_sha(visual_npz),
                visual_mesh_hash=hashlib.sha256(
                    f"visual-{vi}".encode("utf-8")
                ).hexdigest(),
                visual_vertex_count=3,
                visual_face_count=1,
            )
        )

    arrays["clip_0_times"] = np.asarray([0.0], dtype=np.float64)
    projection_npz = tmp_path / "source_owned_projection.npz"
    np.savez_compressed(projection_npz, **arrays)

    projection = SourceOwnedVisualRuntimeProjectionV1IR(
        complete_puppet_binding_hash="1" * 64,
        mechanical_state_binding_hash="2" * 64,
        mechanical_mesh_binding_hash="3" * 64,
        qualified_visual_presentation_binding_hash="4" * 64,
        dynamic_motion_binding_hash="5" * 64,
        appearance_asset_binding_hash="6" * 64,
        appearance_qualification_binding_hash="7" * 64,
        camera_set_binding_hash="8" * 64,
        visual_deformation_operator_id=("SOURCE_CHART_CONNECTED_2D_POSE_WITH_SLOT_OWNED_DEPTH_V6" if canonical_depth == "connected"
            else "SOURCE_CHART_CANONICAL_2D_MOTION_BLEND_WITH_SLOT_RIGID_V4" if canonical_depth == "motion_blend"
            else "SOURCE_CHART_HARMONIC_WITH_SLOT_RIGID_2D_V3" if canonical_depth == "slot_rigid"
            else "SOURCE_CHART_HARMONIC_CANONICAL_FIELD_V2" if canonical_depth
            else "REGION_LOCAL_SAFE_MECHANICAL_AFFINE_V1"),
        visual_deformation_policy_hash="9" * 64,
        projection_npz_path=str(projection_npz),
        projection_npz_sha256=_sha(projection_npz),
        views=tuple(views),
        clips=(
            SourceOwnedVisualRuntimeClipV1IR(
                clip_id="smoke",
                duration_seconds=1.0,
                loop=False,
                frame_count=1,
                array_prefix="clip_0",
            ),
        ),
        projection_hash="",
        metadata=({"target_attachments": {"attachments": [{"attachment_id": "prop", "attachment_index": 1}]},
                   "body_motion_preset": {}, "attachment_depth_operator_id": "TARGET_SLOT_CAMERA_RIGID_2_5D_FROZEN_REST_RELIEF_V1"}
                  if canonical_depth == "connected" else {}),
    )
    projection = replace(
        projection,
        projection_hash=source_owned_visual_runtime_projection_hash(projection),
    )

    entries = build_source_owned_visual_rss_v2_entries(projection)
    assert "mesh.bin" not in entries
    assert "visual_mesh_v0.bin" in entries
    assert "visual_texture_v0.png" in entries
    assert "clip_0_v0.positions.bin" in entries
    manifest = entries["manifest.txt"].decode("utf-8")
    assert "presentation_geometry_mode=SOURCE_OWNED_VISUAL_PRESENTATION_V1" in manifest
    assert "mechanical_mesh_render_authority=0" in manifest
    assert "runtime_visual_mesh_rebuild=0" in manifest

    rss = tmp_path / "source_owned_visual.rss"
    write_rss_v2(rss, entries)
    rgba_path = tmp_path / "frame.rgba"
    prov_path = tmp_path / "frame.prov"
    source_view_path = tmp_path / "frame.source_view"
    owner_path = tmp_path / "frame.owner"
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
            str(rgba_path),
            "--out-provenance",
            str(prov_path),
            "--out-source-view",
            str(source_view_path),
            "--out-owner",
            str(owner_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D" in proc.stdout

    rgba = np.frombuffer(rgba_path.read_bytes(), dtype=np.uint8).reshape(
        render_size, render_size, 4
    )
    prov = np.frombuffer(prov_path.read_bytes(), dtype=np.uint8).reshape(
        render_size, render_size
    )
    donor = np.frombuffer(
        source_view_path.read_bytes(),
        dtype="<i2",
    ).reshape(render_size, render_size)
    owner = np.frombuffer(owner_path.read_bytes(), dtype="<i4").reshape(
        render_size, render_size
    )

    visible = rgba[:, :, 3] > 0
    assert np.any(visible)
    assert np.all(rgba[:, :, 0][visible] == 220)
    assert np.all(rgba[:, :, 1][visible] == 40)
    assert np.all(rgba[:, :, 2][visible] == 20)
    assert np.all(rgba[:, :, 3][visible] == 255)
    assert np.all(prov[visible] == 0)
    assert np.all(donor[visible] == 0)
    assert np.all(owner[visible] == 0)
    assert np.all(prov[~visible] == 255)
    assert np.all(donor[~visible] == np.iinfo(np.int16).min)
    assert np.all(owner[~visible] == -1)
