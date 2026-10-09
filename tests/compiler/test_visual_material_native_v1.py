from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.runtime_package_v2 import build_source_owned_visual_rss_v2_entries, write_rss_v2
from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
    SourceOwnedVisualRuntimeViewV1IR, SourceOwnedVisualRuntimeClipV1IR,
    SourceOwnedVisualRuntimeProjectionV1IR, source_owned_visual_runtime_projection_hash,
)
from compiler.realsas_compiler_core.visual_material_v1 import (
    MATERIAL_CONTRACT, DEPTH_CONTRACT, source_visual_material,
)
from compiler.realsas_compiler_core.visual_material_render_v1 import render_visual_material
from compiler.realsas_compiler_core.types import QualificationError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, reverse=False, tie=False, unsupported=False, occluded_unsupported=False):
    positions = np.tile(np.array(((0., 0.), (1., 0.), (0., 1.))), (2, 1))
    uv = np.repeat(((0., 0.), (1., 1.)), 3, axis=0)
    faces = np.array(((0, 1, 2), (3, 4, 5)), dtype=np.uint32)
    if reverse:
        faces = faces[::-1].copy()
    depth = np.repeat((1., 1. if tie else 2.), 3)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[:] = (255, 0, 0, 128)
    rgba[1, 1] = (0, 255, 0, 255)
    p = np.full((2, 2), 2, dtype=np.uint8)
    p[1, 1] = 0
    if unsupported:
        p[0, 0], rgba[0, 0] = 3, 0
    if occluded_unsupported:
        p[1, 1], rgba[1, 1], rgba[0, 0] = 3, 0, (255, 0, 0, 255)
    arrays, views = {}, []
    for vi in range(8):
        s = np.full((2, 2), -2, dtype=np.int16)
        s[1, 1] = vi
        if unsupported:
            s[0, 0] = -4
        if occluded_unsupported:
            s[1, 1] = -4
        texture = tmp_path / f"V{vi}.png"
        Image.fromarray(rgba).save(texture)
        mesh = tmp_path / f"V{vi}.npz"
        np.savez_compressed(mesh, positions=positions, faces=faces, uv=uv)
        arrays.update({f"view_{vi}_uv": uv, f"view_{vi}_faces": faces,
            f"view_{vi}_rest_positions": positions,
            f"view_{vi}_material_provenance": p,
            f"view_{vi}_material_source_view": s,
            f"clip_0_view_{vi}_positions": positions[None],
            f"clip_0_view_{vi}_depths": depth[None]})
        views.append(SourceOwnedVisualRuntimeViewV1IR(
            view_index=vi, view_id=f"V{vi}", camera={"resolution": 16},
            source_width=2, source_height=2, texture_path=str(texture), texture_sha256=sha(texture),
            visual_mesh_npz_path=str(mesh), visual_mesh_npz_sha256=sha(mesh),
            visual_mesh_hash="a"*64, visual_vertex_count=6, visual_face_count=2))
    arrays["clip_0_times"] = np.array((0.,))
    array_path = tmp_path / "projection.npz"
    np.savez_compressed(array_path, **arrays)
    projection = SourceOwnedVisualRuntimeProjectionV1IR(
        complete_puppet_binding_hash="1"*64, mechanical_state_binding_hash="2"*64,
        mechanical_mesh_binding_hash="3"*64, qualified_visual_presentation_binding_hash="4"*64,
        dynamic_motion_binding_hash="5"*64, appearance_asset_binding_hash="6"*64,
        appearance_qualification_binding_hash="7"*64, camera_set_binding_hash="8"*64,
        visual_deformation_operator_id="REGION_LOCAL_SAFE_MECHANICAL_AFFINE_V1",
        visual_deformation_policy_hash="9"*64,
        projection_npz_path=str(array_path), projection_npz_sha256=sha(array_path), views=tuple(views),
        clips=(SourceOwnedVisualRuntimeClipV1IR("probe", 1., False, 1, "clip_0"),),
        projection_hash="", metadata={"visual_material_contract": MATERIAL_CONTRACT,
            "depth_ownership_contract": DEPTH_CONTRACT})
    projection = replace(projection, projection_hash=source_owned_visual_runtime_projection_hash(projection))
    reference_args = dict(positions=positions, depths=depth, faces=faces, uv=uv, texture=rgba,
        provenance=p, source_view=arrays["view_0_material_source_view"], view_index=0, resolution=16)
    return projection, reference_args


def native(tmp_path, projection):
    raw = os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not raw:
        pytest.skip("native toolchain required")
    rss = tmp_path / "material.rss"
    write_rss_v2(rss, build_source_owned_visual_rss_v2_entries(projection))
    paths = [tmp_path / x for x in ("rgba", "prov", "owner", "donor")]
    result = subprocess.run([raw, str(rss), "--clip", "probe", "--view", "V0", "--frame", "0",
        "--out-rgba", str(paths[0]), "--out-provenance", str(paths[1]),
        "--out-owner", str(paths[2]), "--out-source-view", str(paths[3])], capture_output=True, text=True)
    return result, paths


@pytest.mark.parametrize("reverse", (False, True))
def test_completed_material_and_depth_survive_native_package(tmp_path, reverse):
    projection, args = fixture(tmp_path, reverse=reverse)
    expected = render_visual_material(**args)
    assert tuple(expected.straight_rgba_u8[5, 5]) == (188, 187, 0, 255)
    assert expected.provenance_code[5, 5] == 2
    assert expected.source_view_index[5, 5] == -2
    assert expected.owner_face_index[5, 5] == (1 if reverse else 0)
    result, paths = native(tmp_path, projection)
    assert result.returncode == 0, result.stderr
    for path, dtype, shape, target in zip(paths, (np.uint8, np.uint8, "<i4", "<i2"),
        ((16, 16, 4), (16, 16), (16, 16), (16, 16)),
        (expected.straight_rgba_u8, expected.provenance_code, expected.owner_face_index, expected.source_view_index)):
        np.testing.assert_array_equal(np.frombuffer(path.read_bytes(), dtype=dtype).reshape(shape), target)


@pytest.mark.parametrize("case,message", (("tie", "DEPTH"), ("unsupported", "UNSUPPORTED")))
def test_native_and_reference_reject_undefined_material_or_order(tmp_path, case, message):
    projection, args = fixture(tmp_path, **{case: True})
    with pytest.raises(QualificationError, match=message):
        render_visual_material(**args)
    result, _ = native(tmp_path, projection)
    assert result.returncode != 0
    assert message in result.stderr


def test_foreground_mask_cannot_manufacture_observed_opaque_art():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(QualificationError, match="FOREGROUND_WITHOUT_APPEARANCE"):
        source_visual_material(rgba, np.ones((2, 2), dtype=bool), 0)


def test_occluded_abstention_is_not_a_visible_material_failure(tmp_path):
    projection, args = fixture(tmp_path, occluded_unsupported=True)
    expected = render_visual_material(**args)
    assert tuple(expected.straight_rgba_u8[5, 5]) == (255, 0, 0, 255)
    assert expected.provenance_code[5, 5] == 2
    result, paths = native(tmp_path, projection)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.frombuffer(paths[0].read_bytes(), dtype=np.uint8).reshape(16, 16, 4),
        expected.straight_rgba_u8)


def test_package_rejects_projection_tampering_before_render(tmp_path):
    projection, _ = fixture(tmp_path)
    with open(projection.projection_npz_path, "ab") as handle:
        handle.write(b"drift")
    with pytest.raises(QualificationError, match="PROJECTION_BYTES_DRIFT"):
        build_source_owned_visual_rss_v2_entries(projection)


def test_stage45_rejects_completion_dominance_despite_native_parity(tmp_path, monkeypatch):
    from compiler.realsas_compiler_services.orchestrator.adapters import runtime_v2

    raw = os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not raw:
        pytest.skip("native toolchain required")
    projection, _ = fixture(tmp_path)
    projection = replace(projection, metadata={**projection.metadata, "appearance_quality_policy": {
        "dynamic_max_compiled_unobserved_visible_fraction": 0.02,
        "dynamic_max_frame_compiled_unobserved_visible_fraction": 0.05,
        "dynamic_max_connected_compiled_unobserved_visible_fraction": 0.01}})
    projection = replace(projection, projection_hash=source_owned_visual_runtime_projection_hash(projection))
    archive = tmp_path / "proof.rss"
    write_rss_v2(archive, build_source_owned_visual_rss_v2_entries(projection))
    package = SimpleNamespace(package_hash="a"*64, archive_path=str(archive), archive_sha256=sha(archive),
        metadata={"presentation_geometry_mode": "SOURCE_OWNED_VISUAL_PRESENTATION_V1",
            "mechanical_mesh_render_authority": False})
    playback = SimpleNamespace(package_binding_hash=package.package_hash,
        projection_binding_hash=projection.projection_hash, native_player_sha256=sha(Path(raw)),
        playback_hash="b"*64)
    monkeypatch.setattr(runtime_v2, "_native_player", lambda ctx: (Path(raw), playback.native_player_sha256))
    result = runtime_v2._prove_source_owned_visual_dynamic_integrity(
        {"run_root": tmp_path, "run_manifest": {}, "stage": {"id": "45_DYNAMIC_VISUAL_INTEGRITY_PROOF"}},
        projection=projection, package=package, playback=playback)
    assert result["status"] == "FAIL"
    report = result["diagnostics"]["qualification_report"]
    assert report["native_reference_byte_parity_passed"] is True
    assert report["material_provenance_passed"] is True
    assert report["compiled_appearance_visible_fraction"] == 1.0
    assert report["compiled_appearance_exposure_passed"] is False
    assert report["hidden_layer_material_qualified"] is False
