from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.runtime_package_v2 import (
    _validated_provenance_pages,
)
from compiler.realsas_compiler_core.types import QualificationError


def _projection(tmp_path, provenance, source_view):
    path = tmp_path / "provenance.npz"
    np.savez_compressed(
        path,
        provenance=np.asarray(provenance, dtype=np.uint8),
        source_view=np.asarray(source_view, dtype=np.int16),
    )
    return SimpleNamespace(provenance_npz_path=str(path))


def _base():
    provenance = np.zeros((8, 4, 4), dtype=np.uint8)
    source_view = np.zeros((8, 4, 4), dtype=np.int16)
    for view in range(8):
        source_view[view] = view
    return provenance, source_view


def test_rss_v2_accepts_explicit_unsupported_abstention_lineage(tmp_path):
    provenance, source_view = _base()
    provenance[3, 1:3, 1:3] = 3
    source_view[3, 1:3, 1:3] = -4
    value, donor = _validated_provenance_pages(
        _projection(tmp_path, provenance, source_view)
    )
    assert value.shape == (8, 1, 4, 4)
    assert donor.shape == value.shape
    assert np.all(value[3, 0, 1:3, 1:3] == 3)
    assert np.all(donor[3, 0, 1:3, 1:3] == -4)


def test_rss_v2_rejects_abstention_with_harmonic_lineage(tmp_path):
    provenance, source_view = _base()
    provenance[2, 1, 1] = 3
    source_view[2, 1, 1] = -2
    with pytest.raises(
        QualificationError,
        match="RSS_V2_UNSUPPORTED_SOURCE_VIEW_IDENTITY_DRIFT",
    ):
        _validated_provenance_pages(
            _projection(tmp_path, provenance, source_view)
        )


def test_rss_v2_keeps_padding_distinct_from_abstention(tmp_path):
    provenance, source_view = _base()
    provenance[:, 0, 0] = 255
    source_view[:, 0, 0] = np.iinfo(np.int16).min
    provenance[0, 1, 1] = 3
    source_view[0, 1, 1] = -4
    value, donor = _validated_provenance_pages(
        _projection(tmp_path, provenance, source_view)
    )
    assert np.all(value[:, 0, 0, 0] == 255)
    assert np.all(
        donor[:, 0, 0, 0] == np.iinfo(np.int16).min
    )
    assert int(value[0, 0, 1, 1]) == 3
    assert int(donor[0, 0, 1, 1]) == -4
