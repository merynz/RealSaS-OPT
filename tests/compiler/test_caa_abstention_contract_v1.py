from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.appearance_authority_v2 import CAA_PROVENANCE
from compiler.realsas_compiler_core.appearance_bake_v2 import (
    bake_direction_atlas,
    bake_direction_source_view_atlas,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    bounded_surface_harmonic_fill,
)
from compiler.realsas_compiler_core.appearance_compile_v2 import (
    triangular_barycentric_samples,
)
from compiler.realsas_compiler_core.appearance_quality_v2 import (
    provenance_boundary_metrics,
)


def _oversized_line_fixture():
    count = 6
    rgba = np.zeros((count, 4), dtype=np.uint8)
    rgba[0] = (64, 96, 128, 255)
    provenance = np.full((count,), 255, dtype=np.uint8)
    provenance[0] = CAA_PROVENANCE["DIRECT_SOURCE"]
    source_view = np.full((count,), -1, dtype=np.int16)
    source_view[0] = 0
    missing = np.ones((count,), dtype=bool)
    missing[0] = False
    neighbors = tuple(
        tuple(
            nxt
            for nxt in (index - 1, index + 1)
            if 0 <= nxt < count
        )
        for index in range(count)
    )
    component = np.zeros((count,), dtype=np.int32)
    return rgba, provenance, source_view, missing, neighbors, component


def test_strict_harmonic_policy_still_rejects_oversized_region():
    rgba, provenance, source_view, missing, neighbors, component = (
        _oversized_line_fixture()
    )
    with pytest.raises(Exception, match="CAA_HARMONIC_REGION_TOO_LARGE"):
        bounded_surface_harmonic_fill(
            rgba=rgba,
            provenance=provenance,
            source_view=source_view,
            missing=missing,
            sample_component=component,
            neighbors=neighbors,
            max_region_samples=4,
            max_graph_hops=8,
        )


def test_shipping_completion_abstains_without_inventing_color():
    rgba, provenance, source_view, missing, neighbors, component = (
        _oversized_line_fixture()
    )
    stats = bounded_surface_harmonic_fill(
        rgba=rgba,
        provenance=provenance,
        source_view=source_view,
        missing=missing,
        sample_component=component,
        neighbors=neighbors,
        max_region_samples=4,
        max_graph_hops=8,
        abstain_on_policy_violation=True,
        abstain_provenance_code=CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"],
        abstain_source_view_value=-4,
    )
    assert stats["abstained_region_count"] == 1
    assert stats["abstained_sample_count"] == 5
    assert stats["abstained_reason_counts"]["REGION_TOO_LARGE"] == 1
    assert np.all(
        provenance[1:] == CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"]
    )
    assert np.all(source_view[1:] == -4)
    assert np.all(rgba[1:] == 0)
    assert int(provenance[0]) == CAA_PROVENANCE["DIRECT_SOURCE"]
    assert int(source_view[0]) == 0


def test_atlas_keeps_abstention_distinct_from_physical_padding():
    tile_resolution = 4
    per_face = tile_resolution * (tile_resolution + 1) // 2
    rgba = np.zeros((3 * per_face, 4), dtype=np.uint8)
    rgba[:, 3] = 255
    provenance = np.full(
        (3 * per_face,),
        CAA_PROVENANCE["DIRECT_SOURCE"],
        dtype=np.uint8,
    )
    source_view = np.zeros((3 * per_face,), dtype=np.int16)
    provenance[per_face : 2 * per_face] = CAA_PROVENANCE[
        "UNSUPPORTED_ABSTAIN"
    ]
    source_view[per_face : 2 * per_face] = -4
    rgba[per_face : 2 * per_face] = 0

    _atlas, prov, _uv, layout = bake_direction_atlas(
        face_sample_rgba=rgba,
        face_sample_provenance=provenance,
        face_count=3,
        tile_resolution=tile_resolution,
        bleed_px=2,
    )
    donor = bake_direction_source_view_atlas(
        face_sample_source_view=source_view,
        face_count=3,
        tile_resolution=tile_resolution,
        bleed_px=2,
    )
    padding = np.iinfo(np.int16).min
    assert np.any(prov == CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"])
    assert np.any(donor == -4)
    assert np.array_equal(prov == 255, donor == padding)

    stride = int(layout["tile_stride"])
    # Face 1 is the second tile in row zero.
    face1 = (
        slice(0, stride),
        slice(stride, 2 * stride),
    )
    assert np.all(
        prov[face1] == CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"]
    )
    assert np.all(donor[face1] == -4)


def test_undefined_support_is_not_scored_as_defined_appearance_seam():
    tile_resolution = 4
    bary = triangular_barycentric_samples(tile_resolution)
    tri = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    positions = bary @ tri
    count = len(positions)
    rgba = np.zeros((8, count, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    provenance = np.zeros((8, count), dtype=np.uint8)
    source_view = np.zeros((8, count), dtype=np.int16)

    unsupported = positions[:, 0] > 0.45
    provenance[:, unsupported] = CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"]
    source_view[:, unsupported] = -4
    rgba[:, unsupported, :3] = 255

    unfiltered = provenance_boundary_metrics(
        rgba=rgba,
        provenance=provenance,
        source_view=source_view,
        sample_positions=positions,
        sample_face_index=np.zeros(count, dtype=np.int32),
        face_count=1,
        tile_resolution=tile_resolution,
    )
    filtered = provenance_boundary_metrics(
        rgba=rgba,
        provenance=provenance,
        source_view=source_view,
        sample_positions=positions,
        sample_face_index=np.zeros(count, dtype=np.int32),
        face_count=1,
        tile_resolution=tile_resolution,
        excluded_provenance_codes=(
            CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"],
            255,
        ),
    )
    assert unfiltered["boundary_pair_count"] > 0
    assert filtered["boundary_pair_count"] < unfiltered["boundary_pair_count"]
    assert tuple(filtered["excluded_provenance_codes"]) == (
        CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"],
        255,
    )
