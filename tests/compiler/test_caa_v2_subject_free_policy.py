from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.appearance_compile_v2 import (
    face_atlas_layout,
    resolve_projected_tile_resolution,
    triangular_barycentric_samples,
)
from compiler.realsas_compiler_core.appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
    source_sample_roundtrip_pm_error,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    bounded_surface_harmonic_fill,
)
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    cross_view_source_compatibility_metrics,
    provenance_boundary_metrics,
    source_feature_preservation_metrics,
    structured_holdout_metrics,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import coverage_metrics
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "canonical" / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"


def _policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _camera(view: int = 0, resolution: int = 64):
    return CameraProjectionV3(
        view_id=f"V{view}",
        view_index=view,
        origin=(0.0, 0.0, -2.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=1.0,
        resolution=resolution,
    )


def test_caa_policy_is_reopened_and_tile_density_is_art_quality_driven():
    policy = _policy()
    assert policy["schema"] == "RealSaS.CAAQualificationPolicy.v1"
    assert policy["status"] == "REOPENED_SUBJECT_FREE_VISUAL_FIDELITY_RECALIBRATION_V2"
    assert policy["mutable_after_witness"] is False
    compile_policy = policy["compile_policy"]
    assert compile_policy["tile_resolution_mode"] == "PROJECTED_SOURCE_DENSITY_V1"
    assert compile_policy["tile_resolution_candidates"] == [8, 12, 16, 24, 32]
    assert compile_policy["max_source_pixels_per_atlas_texel"] == 1.0
    assert compile_policy["bleed_px"] == 2
    assert compile_policy["max_atlas_resolution"] == 2048
    assert "tile_resolution" not in compile_policy
    assert "max_supported_face_count" not in compile_policy

    candidate = SimpleNamespace(
        vertices=(
            SimpleNamespace(candidate_vertex_id="v0", P=(-0.2, -0.2, 0.0), component_id="c0"),
            SimpleNamespace(candidate_vertex_id="v1", P=(0.2, -0.2, 0.0), component_id="c0"),
            SimpleNamespace(candidate_vertex_id="v2", P=(-0.2, 0.2, 0.0), component_id="c0"),
        ),
        faces=(("v0", "v1", "v2"),),
    )
    masks = {view: np.ones((64, 64), dtype=bool) for view in range(8)}
    evidence = resolve_projected_tile_resolution(
        candidate=candidate,
        cameras=tuple(_camera(view, 64) for view in range(8)),
        foreground_mask_by_view=masks,
        candidate_resolutions=(8, 12, 16, 24, 32),
        max_source_pixels_per_atlas_texel=1.0,
        bleed_px=2,
        max_atlas_resolution=2048,
    )
    assert evidence["mode"] == "PROJECTED_SOURCE_DENSITY_V1"
    assert evidence["selected_tile_resolution"] in {12, 16, 24, 32}
    selected = next(
        row for row in evidence["candidates"]
        if row["tile_resolution"] == evidence["selected_tile_resolution"]
    )
    assert selected["density_passed"] is True
    assert selected["capacity_passed"] is True
    for row in evidence["candidates"]:
        if row["tile_resolution"] < evidence["selected_tile_resolution"]:
            assert not (row["density_passed"] and row["capacity_passed"])

def _holdout_fixture(*, adversarial: bool):
    tile_resolution = 8
    bary = triangular_barycentric_samples(tile_resolution)
    face_count = 24
    positions = []
    face_index = []
    for face in range(face_count):
        x0 = float(face % 6) * 2.0
        y0 = float(face // 6) * 2.0
        tri = np.asarray(
            ((x0, y0, 0.0), (x0 + 1.0, y0, 0.0), (x0, y0 + 1.0, 0.0)),
            dtype=np.float64,
        )
        positions.append(bary @ tri)
        face_index.extend([face] * len(bary))
    positions = np.concatenate(positions, axis=0)
    count = len(positions)
    component = np.zeros(count, dtype=np.int32)
    direct_valid = np.ones((8, count), dtype=bool)
    source_xy = np.zeros((8, count, 2), dtype=np.float32)
    source_xy[:, :, 0] = positions[:, 0][None, :] * 20.0
    source_xy[:, :, 1] = positions[:, 1][None, :] * 20.0

    rgba = np.zeros((8, count, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    base = np.clip(np.floor((positions[:, 0] + positions[:, 1]) * 6.0), 0, 180).astype(np.uint8)
    if adversarial:
        for view in range(8):
            rgba[view, :, 0] = 255 if view % 2 else 0
            rgba[view, :, 1] = base
    else:
        for view in range(8):
            rgba[view, :, 0] = np.clip(base.astype(np.uint16) + view, 0, 255).astype(np.uint8)
            rgba[view, :, 1] = base
    return {
        "direct_valid": direct_valid,
        "direct_rgba": rgba,
        "source_xy": source_xy,
        "sample_positions": positions,
        "sample_component_index": component,
        "sample_face_index": np.asarray(face_index, dtype=np.int32),
        "face_count": face_count,
        "tile_resolution": tile_resolution,
    }


def test_structured_holdout_uses_bounded_surface_completion_and_rejects_view_conflict():
    p = _policy()["completion_quality_policy"]
    good_args = _holdout_fixture(adversarial=False)
    good = structured_holdout_metrics(
        **good_args,
        band_fraction=p["holdout_band_fraction"],
        max_region_samples=p["max_local_harmonic_region_samples"],
        max_graph_hops=p["max_local_harmonic_graph_hops"],
    )
    bad_args = _holdout_fixture(adversarial=True)
    bad = structured_holdout_metrics(
        **bad_args,
        band_fraction=p["holdout_band_fraction"],
        max_region_samples=p["max_local_harmonic_region_samples"],
        max_graph_hops=p["max_local_harmonic_graph_hops"],
    )
    assert good["mode"] == "SILHOUETTE_ADJACENT_BOUNDED_OCCLUSION_PATCHES_V3"
    assert good["sample_count"] >= p["min_structured_holdout_samples"]
    assert all(
        row["holdout_sample_count"] >= p["min_structured_holdout_samples_per_view"]
        for row in good["per_view"]
    )
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


def test_shared_canonical_edge_is_checked_even_when_provenance_class_matches():
    bary = triangular_barycentric_samples(8)
    tri_a = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    tri_b = np.asarray(
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    positions = np.concatenate((bary @ tri_a, bary @ tri_b), axis=0)
    per_face = len(bary)
    rgba = np.zeros((8, per_face * 2, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    rgba[:, :per_face, 0] = 16
    rgba[:, per_face:, 0] = 240
    provenance = np.ones((8, per_face * 2), dtype=np.uint8)  # OTHER_VIEW_SOURCE on both faces.
    source_view = np.zeros((8, per_face * 2), dtype=np.int16)
    source_view[:, :per_face] = 1
    source_view[:, per_face:] = 7
    face_index = np.concatenate(
        (
            np.zeros(per_face, dtype=np.int32),
            np.ones(per_face, dtype=np.int32),
        )
    )
    metrics = provenance_boundary_metrics(
        rgba=rgba,
        provenance=provenance,
        source_view=source_view,
        sample_positions=positions,
        sample_face_index=face_index,
        face_count=2,
        tile_resolution=8,
    )
    assert metrics["includes_shared_face_edges"] is True
    assert metrics["shared_face_edges_are_compared_even_when_provenance_matches"] is True
    assert metrics["source_view_identity_consumed"] is True
    assert metrics["donor_view_switch_pair_count"] > 0
    assert metrics["boundary_pair_count"] > 0
    assert metrics["p95_rgba_l1"] > 0.1


def test_cross_view_compatibility_measures_same_canonical_source_without_requiring_rgb_identity():
    count = 12
    valid = np.ones((8, count), dtype=bool)
    rgba = np.zeros((8, count, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    component = np.asarray([0] * 6 + [1] * 6, dtype=np.int32)
    for view in range(8):
        rgba[view, :, 0] = 80 + view
        rgba[view, :, 1] = 120
    metrics = cross_view_source_compatibility_metrics(
        direct_valid=valid,
        direct_rgba=rgba,
        sample_component_index=component,
    )
    assert metrics["pair_count"] == 8
    assert metrics["shared_direct_sample_count"] == 8 * count
    assert len(metrics["per_pair"]) == 8
    assert len(metrics["per_pair_component"]) == 16
    assert metrics["raw_rgb_equality_required"] is False
    assert metrics["measurement_is_compatibility_not_color_authority"] is True

    broken = rgba.copy()
    broken[1, :, 0] = 255
    bad = cross_view_source_compatibility_metrics(
        direct_valid=valid,
        direct_rgba=broken,
        sample_component_index=component,
    )
    assert bad["p95_premultiplied_rgba_l1"] > metrics[
        "p95_premultiplied_rgba_l1"
    ]


def test_source_pm_transport_roundtrip_has_subject_free_numerical_ceiling():
    p = _policy()["completion_quality_policy"]
    colors = (
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (16, 16, 16),
        (240, 240, 240),
        (255, 128, 0),
        (4, 64, 250),
    )
    alphas = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 255)
    weights = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    truth = []
    for rgb0 in colors:
        for rgb1 in colors:
            for alpha0 in alphas:
                for alpha1 in alphas:
                    endpoints = np.asarray(
                        [
                            (*rgb0, alpha0),
                            (*rgb1, alpha1),
                        ],
                        dtype=np.uint8,
                    )
                    pm = straight_srgb_rgba_u8_to_premultiplied_linear(
                        endpoints
                    )
                    for weight in weights:
                        truth.append(
                            (1.0 - weight) * pm[0] + weight * pm[1]
                        )
    truth = np.asarray(truth, dtype=np.float64)
    transport = premultiplied_linear_to_straight_srgb_u8(truth)
    error = source_sample_roundtrip_pm_error(transport, truth)
    assert float(np.max(error)) <= p["max_source_sample_pm_roundtrip_abs_error"]


def test_sparse_one_pixel_line_loss_is_detected_even_below_five_percent_area():
    source = np.full((64, 64, 4), 255, dtype=np.uint8)
    source[:, :, :3] = 240
    source[32, 10:54, :3] = 10
    predicted = source.copy()
    predicted[32, 10:54, :3] = 240
    foreground = np.ones((64, 64), dtype=bool)
    p = _policy()["completion_quality_policy"]
    metrics = source_feature_preservation_metrics(
        predicted_rgba=predicted,
        source_rgba=source,
        source_foreground=foreground,
        high_error_cut_rgba_l1=p["rest_feature_high_error_cut_rgba_l1"],
        edge_gradient_cut=p["rest_feature_edge_gradient_cut"],
    )
    assert metrics["high_error_fraction"] < 0.05
    assert (
        metrics["edge_recall_1px"] < p["rest_min_feature_edge_recall_1px"]
        or metrics["largest_connected_high_error_fraction"]
        > p["rest_max_largest_connected_high_error_fraction"]
        or metrics["p999_rgba_l1"] > p["rest_max_feature_p999_rgba_l1"]
    )


def test_harmonic_completion_cannot_cross_component_or_invent_without_boundary():
    rgba = np.asarray(
        [[255, 0, 0, 255], [0, 0, 0, 0], [0, 255, 0, 255], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    provenance = np.asarray([0, 255, 0, 255], dtype=np.uint8)
    source_view = np.asarray([0, -1, 0, -1], dtype=np.int16)
    missing = np.asarray([False, True, False, True])
    neighbors = ((1,), (0,), (3,), (2,))
    with pytest.raises(Exception, match="CAA_HARMONIC_REGION_WITHOUT_SOURCE_BOUNDARY"):
        bounded_surface_harmonic_fill(
            rgba=rgba,
            provenance=provenance,
            source_view=source_view,
            missing=missing,
            sample_component=("A", "A", "B", "C"),
            neighbors=neighbors,
            max_region_samples=4,
            max_graph_hops=2,
        )


def test_cross_view_policy_accepts_small_artist_variation_and_rejects_gross_contradiction():
    p = _policy()["completion_quality_policy"]
    count = 32
    valid = np.ones((8, count), dtype=bool)
    component = np.asarray([0] * 16 + [1] * 16, dtype=np.int32)
    good = np.zeros((8, count, 4), dtype=np.uint8)
    good[:, :, 3] = 255
    for view in range(8):
        good[view, :, 0] = 100 + view * 2
        good[view, :, 1] = 120
    good_metrics = cross_view_source_compatibility_metrics(
        direct_valid=valid,
        direct_rgba=good,
        sample_component_index=component,
        color_conflict_cut_rgba_l1=p["cross_view_color_conflict_cut_rgba_l1"],
        alpha_conflict_cut=p["cross_view_alpha_conflict_cut"],
    )
    assert all(
        row["shared_direct_sample_count"] >= p["cross_view_min_shared_direct_samples_per_pair"]
        and row["p95_premultiplied_rgba_l1"] <= p["cross_view_max_pair_p95_rgba_l1"]
        and row["color_conflict_fraction"] <= p["cross_view_max_pair_color_conflict_fraction"]
        and row["p95_alpha_abs"] <= p["cross_view_max_pair_p95_alpha_abs"]
        and row["alpha_conflict_fraction"] <= p["cross_view_max_pair_alpha_conflict_fraction"]
        for row in good_metrics["per_pair"]
    )

    bad = good.copy()
    bad[1, :16] = (255, 0, 255, 0)
    bad_metrics = cross_view_source_compatibility_metrics(
        direct_valid=valid,
        direct_rgba=bad,
        sample_component_index=component,
        color_conflict_cut_rgba_l1=p["cross_view_color_conflict_cut_rgba_l1"],
        alpha_conflict_cut=p["cross_view_alpha_conflict_cut"],
    )
    assert any(
        row["color_conflict_fraction"] > p["cross_view_max_component_color_conflict_fraction"]
        or row["alpha_conflict_fraction"] > p["cross_view_max_component_alpha_conflict_fraction"]
        for row in bad_metrics["per_pair_component"]
        if row["shared_direct_sample_count"] >= p["cross_view_min_component_samples_for_gate"]
    )
