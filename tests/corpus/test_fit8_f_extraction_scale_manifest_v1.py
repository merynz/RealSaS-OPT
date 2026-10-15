from dataclasses import replace

from tools.corpus.fit8_f_extraction_scale_manifest_v1 import (
    FExtractionScaleLevelV1,
    FExtractionScaleManifestV1,
)


def _level(scale_id="L128", pitch=0.01):
    return FExtractionScaleLevelV1(
        scale_id=scale_id,
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(16, 20, 24),
        lattice_origin_xyz=(-0.8, -1.0, -1.2),
        extraction_pitch=pitch,
        provenance="synthetic extraction-scale guard",
    )


def _manifest(**kwargs):
    values = dict(
        manifest_id="FIT8_F_MATCHED_SCALE_COURT_V1",
        normalization_frame_sha256="1" * 64,
        minimum_source_margin_cells=2.0,
        levels=(_level("L_A", 0.02), _level("L_B", 0.01)),
        provenance="synthetic multiscale guard",
    )
    values.update(kwargs)
    return FExtractionScaleManifestV1(**values)


def test_manifest_is_hashable_and_builds_exact_dense_grid():
    value = _manifest()
    value.validate()
    assert len(value.sha256()) == 64
    level = value.level("L_B")
    grid = level.dense_grid()
    assert grid.res == level.res
    assert len(grid.cube_fx8) == level.cell_count
    assert grid.extraction_pitch == level.extraction_pitch


def test_resolution_number_is_not_the_scale_identity():
    value = _manifest(levels=(
        FExtractionScaleLevelV1(
            scale_id="WIDE_COARSE",
            cell_min_ijk=(-3, 2, 7),
            cell_max_exclusive_ijk=(13, 22, 31),
            lattice_origin_xyz=(-1.1, -0.9, -1.4),
            extraction_pitch=0.021,
            provenance="synthetic irregular grid",
        ),
    ))
    value.validate()
    assert value.level("WIDE_COARSE").res == (16, 20, 24)


def test_duplicate_geometry_under_different_labels_is_rejected():
    same = _level("A", 0.01)
    duplicate = replace(same, scale_id="B")
    bad = _manifest(levels=(same, duplicate))
    try:
        bad.validate()
    except ValueError as exc:
        assert "DUPLICATE_GEOMETRY" in str(exc)
    else:
        raise AssertionError("expected duplicate lattice geometry rejection")


def test_missing_normalization_frame_authority_is_rejected():
    bad = _manifest(normalization_frame_sha256="0" * 64)
    try:
        bad.validate()
    except ValueError as exc:
        assert "ZERO_SHA_FORBIDDEN:normalization_frame_sha256" in str(exc)
    else:
        raise AssertionError("expected normalization-frame authority rejection")
