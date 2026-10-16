from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.graphics_runtime_v1 import (
    RuntimeGraphicsV1,
    RuntimeGraphicsViewV1,
    runtime_graphics_hash_v1,
)
from compiler.realsas_compiler_core.runtime_graphics_binary_v1 import (
    encode_runtime_graphics_v1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _camera_payload,
    _clip_payload,
    _mesh_payload,
    write_rss_v2,
)
from compiler.realsas_compiler_core.single_m_caa_v1 import (
    MATERIAL_CONTRACT,
    load_material_bundle,
    native_material_payload,
    write_material_bundle,
)
from tools.demo.render_knight_m_caa_v1 import load_inputs

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json"


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
        canonical_mesh_binding_hash="graphics-e2e-mesh",
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
    material = load_material_bundle(
        material_ref["path"],
        expected_sha256=material_ref["sha256"],
        rest=arrays["rest"],
        faces=arrays["faces"],
    )
    views = [
        SimpleNamespace(view_index=v, camera={**vars(c), "resolution": 256})
        for v, c in enumerate(material["cameras"])
    ]
    graphics = _runtime_graphics(len(arrays["faces"]), visible=visible)
    graphics_payload = encode_runtime_graphics_v1(graphics)

    entries: OrderedDict[str, bytes] = OrderedDict()
    entries["mesh.bin"] = _mesh_payload(
        {
            "vertices": arrays["rest"],
            "faces": arrays["faces"],
            "face_uv": np.zeros((len(arrays["faces"]), 3, 2)),
        }
    )
    entries["cameras.bin"] = _camera_payload(SimpleNamespace(views=views))
    entries["run.bin"] = _clip_payload(arrays["times"], arrays["poses"])
    entries["material.bin.z"] = native_material_payload(
        material_ref["path"],
        expected_sha256=material_ref["sha256"],
        rest=arrays["rest"],
        faces=arrays["faces"],
    )
    entries["graphics.bin"] = graphics_payload
    manifest = {
        "schema": "RealSaS.RuntimePackage.v2",
        "mesh_hash": graphics.canonical_mesh_binding_hash,
        "presentation_geometry_mode": MATERIAL_CONTRACT,
        "playback_sampling_contract": "SEALED_FRAME_INDEX_ONLY",
        "view_selection_contract": "SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized": "0",
        "host_interpolation_authorized": "0",
        "presentation_state_execution_authorized": "1",
        "clipping_authorized": "0",
        "tint_order_visibility_authorized": "1",
        "texture_sampling_contract": "SOURCE_WINS_BILINEAR_SRGB__CAA_FALLBACK_LINEAR_PM",
        "pixel_coverage_contract": "SINGLE_HALF_INTEGER_TOP_LEFT_APPROVED_BASELINE",
        "pixel_coverage_sample_count": "1",
        "composition_contract": "STRAIGHT_SRGB_FRONT_TO_BACK",
        "depth_equivalence_epsilon_camera_z": "1e-9",
        "mip_generation_authorized": "0",
        "clip_count": "1",
        "clip.0.id": "run",
        "clip.0.entry": "run.bin",
        "single_m_caa_material_entry": "material.bin.z",
        "graphics_state_execution_authorized": "1",
        "graphics_state_contract": "ROLE_FREE_CANONICAL_M_GRAPHICS_V1",
        "graphics_entry": "graphics.bin",
        "graphics_binding_hash": graphics.runtime_graphics_hash,
        "graphics_face_ownership_required": "1",
        "graphics_partial_order_required": "1",
        "triangle_array_order_as_presentation_authority": "0",
        "independent_visual_mesh_authorized": "0",
    }
    entries["manifest.txt"] = (
        "\n".join(f"{key}={value}" for key, value in manifest.items()) + "\n"
    ).encode("utf-8")
    path = root / ("visible.rss" if visible else "hidden.rss")
    write_rss_v2(path, entries)
    return path, graphics.runtime_graphics_hash


@pytest.mark.skipif(
    not os.environ.get("REALSAS_GRAPHICS_V1_PLAYER"),
    reason="native Graphics player is supplied by the vNext CI gate",
)
def test_graphics_bin_changes_real_native_raster_behavior(tmp_path: Path) -> None:
    player = os.environ["REALSAS_GRAPHICS_V1_PLAYER"]
    visible_package, binding = _write_package(tmp_path, visible=True)
    hidden_package, _ = _write_package(tmp_path, visible=False)

    outputs = []
    for label, package in (("visible", visible_package), ("hidden", hidden_package)):
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
        data = np.frombuffer(rgba.read_bytes(), dtype=np.uint8).reshape(256, 256, 4)
        outputs.append(data)

    visible, hidden = outputs
    assert np.count_nonzero(visible[..., 3]) > 0
    assert np.count_nonzero(hidden[..., 3]) == 0
    assert not np.array_equal(visible, hidden)
