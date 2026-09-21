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
    source_view = np.zeros((8, 16, 16), dtype=np.int16)
    for vi in range(8):
        source_view[vi, :, :] = vi
    # V0 deliberately has one coarse OTHER_VIEW_SOURCE provenance class but
    # two exact donor identities across the bilinear footprint. Runtime must
    # preserve the mixed donor diagnostic without letting it affect RGB.
    provenance[0, :, :] = 1
    source_view[0, :, :8] = 1
    source_view[0, :, 8:] = 2
    provenance_npz = tmp_path / "provenance.npz"
    np.savez_compressed(
        provenance_npz,
        provenance=provenance,
        source_view=source_view,
    )
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
    native_source_view = tmp_path / "native.source_view"
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
            "--out-source-view",
            str(native_source_view),
            "--out-owner",
            str(native_owner),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "renderer=REALSAS_V2_CAA_CANONICAL_DEPTH" in proc.stdout
    assert "coverage=FIXED_2X2_QUARTER_SUBSAMPLES" in proc.stdout

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
    partial_alpha = native[:, :, 3][
        (native[:, :, 3] > 0) & (native[:, :, 3] < 255)
    ]
    assert len(partial_alpha) > 0
    assert set(map(int, np.unique(partial_alpha))).issubset({64, 128, 191})
    midpoint = native[16, 16]
    assert 186 <= int(midpoint[0]) <= 189
    assert int(midpoint[0]) == int(midpoint[1]) == int(midpoint[2])
    assert int(midpoint[3]) == 255

    native_p = np.frombuffer(native_prov.read_bytes(), dtype=np.uint8).reshape(32, 32)
    assert np.array_equal(native_p, reference.provenance_code)
    native_sv = np.frombuffer(
        native_source_view.read_bytes(),
        dtype="<i2",
    ).reshape(32, 32)
    visible = reference.straight_rgba_u8[:, :, 3] > 0
    assert np.any(visible)
    assert set(map(int, np.unique(native_sv[visible]))) == {-3}
    assert np.all(
        native_sv[~visible] == np.iinfo(np.int16).min
    )
    native_o = np.frombuffer(native_owner.read_bytes(), dtype="<i4").reshape(32, 32)
    assert np.array_equal(native_o, reference.owner_face_index)

    # Direction changes are discrete state selection, never cross-direction
    # blending. Every V0..V7 render must independently match its exact Python
    # reference, and returning to V0 must be byte-identical (no transition
    # hysteresis or hidden interpolation state).
    first_v0 = native_rgba.read_bytes()
    for vi in range(8):
        direction_rgba = tmp_path / f"direction_V{vi}.rgba"
        direction_proc = subprocess.run(
            [
                str(player),
                str(rss),
                "--clip",
                "smoke",
                "--view",
                f"V{vi}",
                "--frame",
                "0",
                "--out-rgba",
                str(direction_rgba),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert direction_proc.returncode == 0, direction_proc.stderr
        direction_native = np.frombuffer(
            direction_rgba.read_bytes(),
            dtype=np.uint8,
        ).reshape(32, 32, 4)
        direction_camera = CameraProjectionV3(
            view_id=f"V{vi}",
            view_index=vi,
            origin=(0.0, 0.0, -2.0),
            right=(1.0, 0.0, 0.0),
            screen_up=(0.0, 1.0, 0.0),
            forward=(0.0, 0.0, 1.0),
            half_extent=1.0,
            resolution=32,
        )
        direction_reference = render_caa_reference(
            mesh=mesh,
            camera=direction_camera,
            face_uv=face_uv.astype(np.float64),
            texture_rgba_u8=np.asarray(
                Image.open(textures[vi]).convert("RGBA"),
                dtype=np.uint8,
            ),
            provenance_atlas=provenance[vi],
            positions=vertices.astype(np.float64),
        )
        assert np.array_equal(
            direction_native,
            direction_reference.straight_rgba_u8,
        )

    return_v0 = tmp_path / "direction_return_V0.rgba"
    return_proc = subprocess.run(
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
            str(return_v0),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert return_proc.returncode == 0, return_proc.stderr
    assert return_v0.read_bytes() == first_v0

    # Exact donor lineage is diagnostic only. Replacing V0 donor IDs with
    # another valid donor must change only --out-source-view, never RGBA,
    # coarse provenance, or owner.
    donor_tampered = entries.copy()
    donor_payload = bytearray(donor_tampered["provenance.bin"])
    count = 8 * 16 * 16
    donor_offset = 12 + count
    donor_values = np.frombuffer(
        donor_payload,
        dtype="<i2",
        count=count,
        offset=donor_offset,
    ).copy()
    donor_values[: 16 * 16] = 7
    donor_payload[
        donor_offset : donor_offset + count * 2
    ] = donor_values.astype("<i2").tobytes(order="C")
    donor_tampered["provenance.bin"] = bytes(donor_payload)
    donor_rss = tmp_path / "donor_tampered.rss"
    write_rss_v2(donor_rss, donor_tampered)
    donor_rgba = tmp_path / "donor_tampered.rgba"
    donor_prov = tmp_path / "donor_tampered.prov"
    donor_sv = tmp_path / "donor_tampered.source_view"
    donor_owner = tmp_path / "donor_tampered.owner"
    donor_proc = subprocess.run(
        [
            str(player),
            str(donor_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(donor_rgba),
            "--out-provenance",
            str(donor_prov),
            "--out-source-view",
            str(donor_sv),
            "--out-owner",
            str(donor_owner),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert donor_proc.returncode == 0, donor_proc.stderr
    assert donor_rgba.read_bytes() == native_rgba.read_bytes()
    assert donor_prov.read_bytes() == native_prov.read_bytes()
    assert donor_owner.read_bytes() == native_owner.read_bytes()
    donor_sv_values = np.frombuffer(
        donor_sv.read_bytes(),
        dtype="<i2",
    ).reshape(32, 32)
    assert set(map(int, np.unique(donor_sv_values[visible]))) == {7}
    assert donor_sv.read_bytes() != native_source_view.read_bytes()

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

    view_contract_tampered = entries.copy()
    view_contract_manifest = view_contract_tampered["manifest.txt"].replace(
        b"view_selection_contract=SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        b"view_selection_contract=UNQUALIFIED_DIRECTION_BLEND",
    )
    assert view_contract_manifest != view_contract_tampered["manifest.txt"]
    view_contract_tampered["manifest.txt"] = view_contract_manifest
    view_contract_rss = tmp_path / "tampered_view_contract.rss"
    write_rss_v2(view_contract_rss, view_contract_tampered)
    view_contract_rejected = subprocess.run(
        [
            str(player),
            str(view_contract_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_view_contract.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert view_contract_rejected.returncode != 0
    assert "VIEW_SELECTION_CONTRACT_INVALID" in view_contract_rejected.stderr

    blend_tampered = entries.copy()
    blend_manifest = blend_tampered["manifest.txt"].replace(
        b"cross_direction_blending_authorized=0",
        b"cross_direction_blending_authorized=1",
    )
    assert blend_manifest != blend_tampered["manifest.txt"]
    blend_tampered["manifest.txt"] = blend_manifest
    blend_rss = tmp_path / "tampered_direction_blend.rss"
    write_rss_v2(blend_rss, blend_tampered)
    blend_rejected = subprocess.run(
        [
            str(player),
            str(blend_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_direction_blend.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert blend_rejected.returncode != 0
    assert "CROSS_DIRECTION_BLENDING_MUST_BE_FORBIDDEN" in blend_rejected.stderr

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

    capability_cases = (
        (
            b"presentation_state_execution_authorized=0",
            b"presentation_state_execution_authorized=1",
            "PRESENTATION_STATE_EXECUTION_MUST_BE_FORBIDDEN",
            "presentation_state",
        ),
        (
            b"clipping_authorized=0",
            b"clipping_authorized=1",
            "RUNTIME_CLIPPING_MUST_BE_FORBIDDEN",
            "clipping",
        ),
        (
            b"tint_order_visibility_authorized=0",
            b"tint_order_visibility_authorized=1",
            "RUNTIME_TINT_ORDER_VISIBILITY_MUST_BE_FORBIDDEN",
            "tint_order_visibility",
        ),
        (
            b"source_view_identity_render_authority=0",
            b"source_view_identity_render_authority=1",
            "SOURCE_VIEW_IDENTITY_RENDER_AUTHORITY_FORBIDDEN",
            "source_view_render_authority",
        ),
        (
            b"source_view_identity_contract=PER_TEXEL_INT16_PRESERVED__DIAGNOSTIC_ONLY",
            b"source_view_identity_contract=UNQUALIFIED_DONOR_AUTHORITY",
            "SOURCE_VIEW_IDENTITY_CONTRACT_INVALID",
            "source_view_contract",
        ),
    )
    for old, new, error, label in capability_cases:
        capability_tampered = entries.copy()
        capability_manifest = capability_tampered["manifest.txt"].replace(old, new)
        assert capability_manifest != capability_tampered["manifest.txt"]
        capability_tampered["manifest.txt"] = capability_manifest
        capability_rss = tmp_path / f"tampered_{label}.rss"
        write_rss_v2(capability_rss, capability_tampered)
        capability_rejected = subprocess.run(
            [
                str(player),
                str(capability_rss),
                "--clip",
                "smoke",
                "--view",
                "V0",
                "--frame",
                "0",
                "--out-rgba",
                str(tmp_path / f"tampered_{label}.rgba"),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert capability_rejected.returncode != 0
        assert error in capability_rejected.stderr

    mip_tampered = entries.copy()
    mip_manifest = mip_tampered["manifest.txt"].replace(
        b"mip_generation_authorized=0",
        b"mip_generation_authorized=1",
    )
    assert mip_manifest != mip_tampered["manifest.txt"]
    mip_tampered["manifest.txt"] = mip_manifest
    mip_rss = tmp_path / "tampered_mip.rss"
    write_rss_v2(mip_rss, mip_tampered)
    mip_rejected = subprocess.run(
        [
            str(player),
            str(mip_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_mip.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert mip_rejected.returncode != 0
    assert "MIP_GENERATION_MUST_BE_FORBIDDEN" in mip_rejected.stderr

    sampler_tampered = entries.copy()
    sampler_manifest = sampler_tampered["manifest.txt"].replace(
        b"texture_sampling_contract=BASE_LEVEL_BILINEAR_LINEAR_PM_ONLY",
        b"texture_sampling_contract=UNQUALIFIED_OTHER_SAMPLER",
    )
    assert sampler_manifest != sampler_tampered["manifest.txt"]
    sampler_tampered["manifest.txt"] = sampler_manifest
    sampler_rss = tmp_path / "tampered_sampler.rss"
    write_rss_v2(sampler_rss, sampler_tampered)
    sampler_rejected = subprocess.run(
        [
            str(player),
            str(sampler_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_sampler.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert sampler_rejected.returncode != 0
    assert "TEXTURE_SAMPLING_CONTRACT_INVALID" in sampler_rejected.stderr

    coverage_tampered = entries.copy()
    coverage_manifest = coverage_tampered["manifest.txt"].replace(
        b"pixel_coverage_sample_count=4",
        b"pixel_coverage_sample_count=1",
    )
    assert coverage_manifest != coverage_tampered["manifest.txt"]
    coverage_tampered["manifest.txt"] = coverage_manifest
    coverage_rss = tmp_path / "tampered_coverage.rss"
    write_rss_v2(coverage_rss, coverage_tampered)
    coverage_rejected = subprocess.run(
        [
            str(player),
            str(coverage_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_coverage.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert coverage_rejected.returncode != 0
    assert "PIXEL_COVERAGE_SAMPLE_COUNT_INVALID" in coverage_rejected.stderr

    depth_contract_tampered = entries.copy()
    depth_contract_manifest = depth_contract_tampered["manifest.txt"].replace(
        b"depth_buffer_contract=IEEE754_FLOAT64_SOFTWARE_SORT",
        b"depth_buffer_contract=UNQUALIFIED_HARDWARE_Z",
    )
    assert depth_contract_manifest != depth_contract_tampered["manifest.txt"]
    depth_contract_tampered["manifest.txt"] = depth_contract_manifest
    depth_contract_rss = tmp_path / "tampered_depth_contract.rss"
    write_rss_v2(depth_contract_rss, depth_contract_tampered)
    depth_contract_rejected = subprocess.run(
        [
            str(player),
            str(depth_contract_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_depth_contract.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert depth_contract_rejected.returncode != 0
    assert "DEPTH_BUFFER_CONTRACT_INVALID" in depth_contract_rejected.stderr

    depth_epsilon_tampered = entries.copy()
    depth_epsilon_manifest = depth_epsilon_tampered["manifest.txt"].replace(
        b"depth_equivalence_epsilon_camera_z=1e-12",
        b"depth_equivalence_epsilon_camera_z=1e-6",
    )
    assert depth_epsilon_manifest != depth_epsilon_tampered["manifest.txt"]
    depth_epsilon_tampered["manifest.txt"] = depth_epsilon_manifest
    depth_epsilon_rss = tmp_path / "tampered_depth_epsilon.rss"
    write_rss_v2(depth_epsilon_rss, depth_epsilon_tampered)
    depth_epsilon_rejected = subprocess.run(
        [
            str(player),
            str(depth_epsilon_rss),
            "--clip",
            "smoke",
            "--view",
            "V0",
            "--frame",
            "0",
            "--out-rgba",
            str(tmp_path / "tampered_depth_epsilon.rgba"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert depth_epsilon_rejected.returncode != 0
    assert "DEPTH_EQUIVALENCE_EPSILON_INVALID" in depth_epsilon_rejected.stderr
