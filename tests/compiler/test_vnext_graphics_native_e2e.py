from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.graphics_runtime_v1 import (
    RuntimeGraphicsV1,
    RuntimeGraphicsViewV1,
    runtime_graphics_hash_v1,
)
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    RuntimeClipV2IR,
    RuntimeProjectionV2IR,
    RuntimeViewV2IR,
    runtime_projection_hash,
    validate_runtime_projection,
)
from compiler.realsas_compiler_core.runtime_graphics_binary_v1 import (
    build_rss_vnext_graphics_entries_v1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import write_rss_v2
from compiler.realsas_compiler_core.single_m_caa_v1 import write_material_bundle
from tools.demo.render_knight_m_caa_v1 import load_inputs

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json"
MESH_BINDING = "graphics-e2e-mesh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_graphics(face_count: int, *, visible: bool) -> RuntimeGraphicsV1:
    fragment_ids = tuple(f"fragment-{index}" for index in range(face_count))
    visible_indices = tuple(range(face_count)) if visible else ()
    views = tuple(
        RuntimeGraphicsViewV1(
            view_index=view,
            visible_fragment_indices=visible_indices,
            topological_fragment_order=tuple(range(face_count)),
            order_edges=(),
        )
        for view in range(8)
    )
    value = RuntimeGraphicsV1(
        canonical_mesh_binding_hash=MESH_BINDING,
        compiled_graphics_binding_hash="graphics-e2e-compiled",
        face_fragment_index=tuple(range(face_count)),
        fragment_ids=fragment_ids,
        fragment_support_band_pixels=tuple(0 for _ in range(face_count)),
        views=views,
        attachments=(),
        runtime_graphics_hash="",
        metadata={
            "behaviorally_consumable": True,
            "hash_only_binding_forbidden": True,
            "single_canonical_mesh_authority": True,
        },
    )
    return replace(value, runtime_graphics_hash=runtime_graphics_hash_v1(value))


def _write_package(root: Path, *, visible: bool) -> tuple[Path, str]:
    _, arrays, source, cameras, caa = load_inputs(BASELINE)
    material_ref = write_material_bundle(
        root / ("material-visible.npz" if visible else "material-hidden.npz"),
        rest=arrays["rest"],
        faces=arrays["faces"],
        source=source,
        cameras=cameras,
        caa_arrays=caa,
        input_identities={"fixture": "single_m_caa_knight_v1"},
    )

    projection_npz = root / ("projection-visible.npz" if visible else "projection-hidden.npz")
    np.savez_compressed(
        projection_npz,
        vertices=np.asarray(arrays["rest"]),
        faces=np.asarray(arrays["faces"]),
        face_uv=np.zeros((len(arrays["faces"]), 3, 2), dtype=np.float64),
        clip_0_times=np.asarray(arrays["times"], dtype=np.float64),
        clip_0_positions=np.asarray(arrays["poses"], dtype=np.float64),
    )

    runtime_views = []
    for view_index, (image, camera) in enumerate(zip(source, cameras)):
        texture_path = root / f"{'visible' if visible else 'hidden'}-V{view_index}.png"
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGBA").save(texture_path)
        camera_payload = dict(vars(camera))
        camera_payload["resolution"] = 256
        runtime_views.append(
            RuntimeViewV2IR(
                view_index=view_index,
                view_id=f"V{view_index}",
                camera=camera_payload,
                texture_path=str(texture_path),
                texture_sha256=_sha256(texture_path),
            )
        )

    height, width = np.asarray(source[0]).shape[:2]
    provenance = np.zeros((8, height, width), dtype=np.uint8)
    source_view = np.empty((8, height, width), dtype=np.int16)
    for view_index in range(8):
        source_view[view_index].fill(view_index)
    provenance_npz = root / ("provenance-visible.npz" if visible else "provenance-hidden.npz")
    np.savez_compressed(
        provenance_npz,
        provenance=provenance,
        source_view=source_view,
    )

    duration = (
        float(arrays["times"][-1] - arrays["times"][0])
        if len(arrays["times"]) > 1
        else 0.0
    )
    projection = RuntimeProjectionV2IR(
        complete_puppet_binding_hash="graphics-e2e-puppet",
        mechanical_state_binding_hash="graphics-e2e-mechanics",
        mesh_binding_hash=MESH_BINDING,
        dynamic_motion_binding_hash="graphics-e2e-motion",
        appearance_asset_binding_hash="graphics-e2e-appearance",
        appearance_qualification_binding_hash="graphics-e2e-appearance-qualification",
        camera_set_binding_hash="graphics-e2e-cameras",
        visibility_contract_hash="graphics-e2e-visibility",
        projection_npz_path=str(projection_npz),
        projection_npz_sha256=_sha256(projection_npz),
        provenance_npz_path=str(provenance_npz),
        provenance_npz_sha256=_sha256(provenance_npz),
        views=tuple(runtime_views),
        clips=(
            RuntimeClipV2IR(
                clip_id="run",
                duration_seconds=duration,
                loop=True,
                frame_count=len(arrays["poses"]),
                array_prefix="clip_0",
            ),
        ),
        projection_hash="",
        metadata={"single_m_caa_material": material_ref},
    )
    projection = replace(projection, projection_hash=runtime_projection_hash(projection))
    validate_runtime_projection(projection)

    graphics = _runtime_graphics(len(arrays["faces"]), visible=visible)
    entries = build_rss_vnext_graphics_entries_v1(projection, graphics)
    path = root / ("visible.rss" if visible else "hidden.rss")
    write_rss_v2(path, entries)
    return path, graphics.runtime_graphics_hash


@pytest.mark.skipif(
    not os.environ.get("REALSAS_GRAPHICS_V1_PLAYER"),
    reason="native Graphics player is supplied by the vNext CI gate",
)
def test_graphics_bin_changes_real_native_raster_behavior(tmp_path: Path) -> None:
    player = os.environ["REALSAS_GRAPHICS_V1_PLAYER"]
    visible_package, visible_binding = _write_package(tmp_path, visible=True)
    hidden_package, hidden_binding = _write_package(tmp_path, visible=False)
    assert visible_binding != hidden_binding

    outputs = []
    for label, package, binding in (
        ("visible", visible_package, visible_binding),
        ("hidden", hidden_package, hidden_binding),
    ):
        rgba = tmp_path / f"{label}.rgba"
        proc = subprocess.run(
            [
                player,
                str(package),
                "--clip",
                "run",
                "--view",
                "V0",
                "--frame",
                "0",
                "--out-rgba",
                str(rgba),
                "--allow-layer-overflow-diagnostic",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "renderer=REALSAS_VNEXT_SINGLE_M_GRAPHICS" in proc.stdout
        assert f"graphics_binding={binding}" in proc.stdout
        outputs.append(
            np.frombuffer(rgba.read_bytes(), dtype=np.uint8).reshape(256, 256, 4)
        )

    visible, hidden = outputs
    assert np.count_nonzero(visible[..., 3]) > 0
    assert np.count_nonzero(hidden[..., 3]) == 0
    assert not np.array_equal(visible, hidden)
