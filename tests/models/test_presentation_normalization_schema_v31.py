import pytest

from models.iris.v31.presentation_objective_v1 import _normalization


def test_canonical_compiler_schema_version_is_accepted():
    center, half = _normalization({
        "schema_version": "RealSaS.NormalizationDomainIR.v1",
        "center_xyz": [1.0, 2.0, 3.0],
        "half_extent": 4.0,
    })
    assert center.tolist() == [1.0, 2.0, 3.0]
    assert half == 4.0


def test_legacy_schema_alias_remains_accepted():
    center, half = _normalization({
        "schema": "RealSaS.NormalizationDomainIR.v1",
        "center_xyz": [0.0, 0.0, 0.0],
        "half_extent": 1.0,
    })
    assert center.tolist() == [0.0, 0.0, 0.0]
    assert half == 1.0


def test_conflicting_schema_markers_fail_closed():
    with pytest.raises(ValueError, match="IRIS_V31_PRESENTATION_NORMALIZATION_SCHEMA"):
        _normalization({
            "schema": "RealSaS.NormalizationDomainIR.v1",
            "schema_version": "RealSaS.NormalizationDomainIR.v2",
            "center_xyz": [0.0, 0.0, 0.0],
            "half_extent": 1.0,
        })
