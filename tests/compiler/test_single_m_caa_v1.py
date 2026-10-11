from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from compiler.realsas_compiler_core.appearance_compile_v2 import (
    triangular_barycentric_samples,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _mesh_payload,
    _camera_payload,
    _clip_payload,
    build_rss_v2_entries,
    read_rss_v2,
    write_rss_v2,
)
from compiler.realsas_compiler_core.single_m_caa_v1 import (
    MATERIAL_CONTRACT,
    array_hash,
    load_material_bundle,
    native_material_payload,
    prepare_caa_field,
    render_single_m_caa,
    sample_caa,
    write_material_bundle,
)
from compiler.realsas_compiler_core.types import QualificationError
from tools.demo.render_knight_m_caa_v1 import load_inputs

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json"


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("single_m")
    manifest, arrays, source, cameras, caa = load_inputs(BASELINE)
    ref = write_material_bundle(
        root / "material.npz",
        rest=arrays["rest"],
        faces=arrays["faces"],
        source=source,
        cameras=cameras,
        caa_arrays=caa,
        input_identities=manifest["inputs"],
    )
    material = load_material_bundle(
        ref["path"],
        expected_sha256=ref["sha256"],
        rest=arrays["rest"],
        faces=arrays["faces"],
    )
    return root, manifest, arrays, ref, material


def test_recorded_m_poses_keep_approved_raw_rgba_and_topology(baseline):
    _, manifest, arrays, _, material = baseline
    for view in (0, 6):
        for index in (0, 10, 20):
            result = render_single_m_caa(
                **{
                    k: material[k]
                    for k in ("rest", "faces", "source", "maps", "cameras", "caa")
                },
                posed=arrays["poses"][index],
                view=view,
            )
            assert (
                array_hash(result.straight_rgba_u8)
                == manifest["golden_rgba"][f"V{view}/{index}"]
            )
            assert result.material_fragments["caa_fragments"] > 0
            assert result.unresolved_depth_tie_count > 0  # open gate stays observable
    assert array_hash(arrays["poses"]) == manifest["pose_sha256"]


def test_material_rejects_byte_or_geometry_drift(baseline, tmp_path):
    _, _, arrays, ref, _ = baseline
    bad = arrays["rest"].copy()
    bad[0, 0] += 0.01
    with pytest.raises(QualificationError, match="REST_DRIFT"):
        load_material_bundle(ref["path"], expected_sha256=ref["sha256"], rest=bad)
    path = tmp_path / "bad.npz"
    path.write_bytes(Path(ref["path"]).read_bytes() + b"x")
    with pytest.raises(QualificationError, match="BYTES_DRIFT"):
        load_material_bundle(path, expected_sha256=ref["sha256"])


def test_existing_adaptive_caa_lattices_and_abstention_are_preserved():
    rest = np.array(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [2.0, 0.0, 2.0]]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    patterns = [triangular_barycentric_samples(n) for n in (4, 6)]
    counts = [len(x) for x in patterns]
    total = sum(counts)
    positions = np.concatenate([b @ rest[f] for b, f in zip(patterns, faces)])
    rgba = np.full((8, total, 4), 255, np.uint8)
    codes = np.zeros((8, total), np.uint8)
    donors = np.broadcast_to(np.arange(8)[:, None], codes.shape).copy()
    codes[:, counts[0] :] = 3
    donors[:, counts[0] :] = -4
    rgba[:, counts[0] :] = 0
    arrays = {
        "face_sample_offsets": np.r_[0, np.cumsum(counts)],
        "face_tile_resolutions": np.array([4, 6]),
        "sample_positions": positions,
        "sample_face_index": np.repeat(np.arange(2), counts),
        "rgba": rgba,
        "provenance": codes,
        "source_view": donors,
    }
    field = prepare_caa_field(arrays, rest, faces)
    result, provenance = sample_caa(
        field, np.array([0, 1]), np.array([[0.4, 0.3, 0.3]] * 2), 0
    )
    assert result.tolist() == [[255, 255, 255, 255], [0, 0, 0, 0]]
    assert provenance.tolist() == [0, 3]
    arrays["sample_positions"] = positions + 0.01
    with pytest.raises(QualificationError, match="ADDRESS_DRIFT"):
        prepare_caa_field(arrays, rest, faces)


@pytest.mark.xfail(
    strict=True,
    reason="Native diagonal witnesses differ; exact pixel/owner parity remains OPEN and CPU approved pixels remain canonical",
)
def test_sealed_single_m_rss_native_matches_recorded_motion(baseline):
    player = os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not player:
        pytest.skip("native player is required by the canonical gate")
    root, _, arrays, ref, material = baseline
    views = [
        SimpleNamespace(view_index=v, camera={**vars(c), "resolution": 256})
        for v, c in enumerate(material["cameras"])
    ]
    entries = OrderedDict()
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
        ref["path"],
        expected_sha256=ref["sha256"],
        rest=arrays["rest"],
        faces=arrays["faces"],
    )
    manifest = {
        "presentation_geometry_mode": MATERIAL_CONTRACT,
        "playback_sampling_contract": "SEALED_FRAME_INDEX_ONLY",
        "view_selection_contract": "SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized": "0",
        "host_interpolation_authorized": "0",
        "presentation_state_execution_authorized": "0",
        "clipping_authorized": "0",
        "tint_order_visibility_authorized": "0",
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
    }
    entries["manifest.txt"] = (
        "\n".join(f"{k}={v}" for k, v in manifest.items()) + "\n"
    ).encode()
    package = root / "single_m.rss"
    write_rss_v2(package, entries)
    assert read_rss_v2(package) == entries
    measurements = []
    for view in range(8):
        for index in (0, 10, 20):
            raw = root / f"native_v{view}_{index}.rgba"
            proc = subprocess.run(
                [
                    player,
                    str(package),
                    "--clip",
                    "run",
                    "--view",
                    f"V{view}",
                    "--frame",
                    str(index),
                    "--out-rgba",
                    str(raw),
                    "--allow-layer-overflow-diagnostic",
                ],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            assert "renderer=REALSAS_V2_SINGLE_M_SOURCE_WINS_CAA" in proc.stdout
            expected = render_single_m_caa(
                **{
                    k: material[k]
                    for k in ("rest", "faces", "source", "maps", "cameras", "caa")
                },
                posed=arrays["poses"][index],
                view=view,
            ).straight_rgba_u8
            actual = np.frombuffer(raw.read_bytes(), np.uint8).reshape(expected.shape)
            measurements.append(
                {
                    "view": view,
                    "frame": index,
                    "channel_mismatches": int(np.count_nonzero(actual != expected)),
                    "maximum_channel_error": int(
                        np.max(np.abs(actual.astype(int) - expected))
                    ),
                    "native_stdout": proc.stdout.strip(),
                }
            )
    (root / "native_parity.json").write_text(
        json.dumps(
            {
                "measurements": measurements,
                "native_parity_passed": all(
                    row["channel_mismatches"] == 0 for row in measurements
                ),
                "product_qualification_claimed": False,
            },
            indent=2,
        )
    )
    assert all(row["channel_mismatches"] == 0 for row in measurements), measurements
