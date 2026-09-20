from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_compile_v2 import (
    face_atlas_layout,
    triangular_barycentric_samples,
)
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    provenance_boundary_metrics,
    structured_holdout_metrics,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import coverage_metrics

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "canonical" / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"


def _policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_caa_policy_is_frozen_before_witness_and_atlas_capacity_is_analytic():
    policy = _policy()
    assert policy["schema"] == "RealSaS.CAAQualificationPolicy.v1"
    assert policy["status"] == "REOPENED_SUBJECT_FREE_VISUAL_FIDELITY_RECALIBRATION_V2"
    assert policy["mutable_after_witness"] is False
    compile_policy = policy["compile_policy"]
    assert compile_policy["tile_resolution"] == 8
    assert compile_policy["bleed_px"] == 2
    assert compile_policy["max_atlas_resolution"] == 2048
    assert compile_policy["max_supported_face_count"] == 28900

    passing = face_atlas_layout(
        28900,
        tile_resolution=8,
        bleed_px=2,
    )
    failing = face_atlas_layout(
        28901,
        tile_resolution=8,
        bleed_px=2,
    )
    assert passing["width"] <= 2048 and passing["height"] <= 2048
    assert failing["width"] > 2048 or failing["height"] > 2048


def _holdout_fixture(*, adversarial: bool):
    count = 400
    positions = np.column_stack(
        (
            np.linspace(0.0, 1.0, count),
            np.zeros(count),
            np.zeros(count),
        )
    )
    component = np.zeros(count, dtype=np.int32)
    direct_valid = np.ones((8, count), dtype=bool)
    source_xy = np.zeros((8, count, 2), dtype=np.float32)
    source_xy[:, :, 0] = np.arange(count, dtype=np.float32)[None, :]
    source_xy[:, :, 1] = 100.0

    rgba = np.zeros((8, count, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    if adversarial:
        for view in range(8):
            rgba[view, :, 0] = 255 if view % 2 else 0
    else:
        for view in range(8):
            rgba[view, :, 0] = view * 2
            rgba[view, :, 1] = np.arange(count, dtype=np.uint16) % 128
    return direct_valid, rgba, source_xy, positions, component


def test_structured_holdout_policy_passes_small_directional_change_and_rejects_large_patch_change():
    p = _policy()["completion_quality_policy"]
    good = structured_holdout_metrics(
        direct_valid=_holdout_fixture(adversarial=False)[0],
        direct_rgba=_holdout_fixture(adversarial=False)[1],
        source_xy=_holdout_fixture(adversarial=False)[2],
        sample_positions=_holdout_fixture(adversarial=False)[3],
        sample_component_index=_holdout_fixture(adversarial=False)[4],
        band_fraction=p["holdout_band_fraction"],
    )
    bad_args = _holdout_fixture(adversarial=True)
    bad = structured_holdout_metrics(
        direct_valid=bad_args[0],
        direct_rgba=bad_args[1],
        source_xy=bad_args[2],
        sample_positions=bad_args[3],
        sample_component_index=bad_args[4],
        band_fraction=p["holdout_band_fraction"],
    )
    assert good["sample_count"] >= p["min_structured_holdout_samples"]
    assert good["mean_rgba_l1"] <= p["max_structured_holdout_mean_rgba_l1"]
    assert good["p95_rgba_l1"] <= p["max_structured_holdout_p95_rgba_l1"]
    assert (
        bad["mean_rgba_l1"] > p["max_structured_holdout_mean_rgba_l1"]
        or bad["p95_rgba_l1"] > p["max_structured_holdout_p95_rgba_l1"]
    )


def _seam_fixture(*, adversarial: bool):
    bary = triangular_barycentric_samples(8)
    tri = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    positions = bary @ tri
    count = len(positions)
    rgba = np.zeros((8, count, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    smooth_r = np.clip(np.floor(positions[:, 0] * 180.0 + 0.5), 0, 255).astype(np.uint8)
    smooth_g = np.clip(np.floor(positions[:, 1] * 180.0 + 0.5), 0, 255).astype(np.uint8)
    for view in range(8):
        rgba[view, :, 0] = smooth_r
        rgba[view, :, 1] = smooth_g
    provenance = np.zeros((8, count), dtype=np.uint8)
    compiled = positions[:, 0] > 0.45
    provenance[:, compiled] = 2
    if adversarial:
        indices = np.arange(count)
        checker = ((indices % 2) * 255).astype(np.uint8)
        for view in range(8):
            rgba[view, compiled, 0] = checker[compiled]
            rgba[view, compiled, 1] = 255 - checker[compiled]
    return rgba, provenance, positions, np.zeros(count, dtype=np.int32)


def test_provenance_seam_policy_passes_smooth_field_and_rejects_visible_fault():
    p = _policy()["completion_quality_policy"]
    good_args = _seam_fixture(adversarial=False)
    good = provenance_boundary_metrics(
        rgba=good_args[0],
        provenance=good_args[1],
        sample_positions=good_args[2],
        sample_face_index=good_args[3],
        face_count=1,
        tile_resolution=8,
    )
    bad_args = _seam_fixture(adversarial=True)
    bad = provenance_boundary_metrics(
        rgba=bad_args[0],
        provenance=bad_args[1],
        sample_positions=bad_args[2],
        sample_face_index=bad_args[3],
        face_count=1,
        tile_resolution=8,
    )
    assert good["mean_rgba_l1"] <= p["max_provenance_boundary_mean_rgba_l1"]
    assert good["p95_rgba_l1"] <= p["max_provenance_boundary_p95_rgba_l1"]
    assert good["mean_gradient_jump"] <= p["max_provenance_boundary_mean_gradient_jump"]
    assert good["p95_gradient_jump"] <= p["max_provenance_boundary_p95_gradient_jump"]
    assert (
        bad["mean_rgba_l1"] > p["max_provenance_boundary_mean_rgba_l1"]
        or bad["p95_rgba_l1"] > p["max_provenance_boundary_p95_rgba_l1"]
        or bad["mean_gradient_jump"] > p["max_provenance_boundary_mean_gradient_jump"]
        or bad["p95_gradient_jump"] > p["max_provenance_boundary_p95_gradient_jump"]
    )


def test_rest_alpha_policy_rejects_coherent_missing_art():
    p = _policy()["completion_quality_policy"]
    width = height = 100
    source = np.zeros((height, width), dtype=np.uint8)
    source[20:80, 20:80] = 1
    perfect = source.copy()
    metrics = coverage_metrics(
        bytes(source.reshape(-1)),
        bytes(perfect.reshape(-1)),
        width=width,
        height=height,
    )
    assert metrics["recall"] >= p["rest_min_source_alpha_recall"]
    assert metrics["precision"] >= p["rest_min_source_alpha_precision"]

    damaged = source.copy()
    damaged[45:55, 30:70] = 0
    bad = coverage_metrics(
        bytes(source.reshape(-1)),
        bytes(damaged.reshape(-1)),
        width=width,
        height=height,
    )
    assert (
        bad["recall"] < p["rest_min_source_alpha_recall"]
        or bad["largest_coherent_hole_fraction"]
        > p["rest_max_largest_coherent_alpha_hole_fraction"]
        or bad["interior_uncovered_fraction"]
        > p["rest_max_alpha_interior_uncovered_fraction"]
    )


def test_dynamic_compiled_unobserved_budget_rejects_inference_dominance():
    budget = _policy()["completion_quality_policy"][
        "dynamic_max_compiled_unobserved_visible_fraction"
    ]
    assert 0.0 <= budget <= 0.02
    assert budget < 0.10
