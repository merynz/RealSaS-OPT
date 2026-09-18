import numpy as np
import pytest

from compiler.realsas_compiler_core.drawable_surface_v1 import (
    qualify_drawable_support_binding_v1,
    qualify_drawable_surface_v1,
    transfer_skin_via_support_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64


def _drawable():
    return qualify_drawable_surface_v1(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        source_dense_lineage_hash=H1,
        conditioning_contract_hash=H2,
    )


def test_support_binding_partition_of_unity_and_skin_transfer():
    d = _drawable()
    rig = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    ids = np.asarray([[0, 1], [1, 0], [2, 0]], dtype=np.int64)
    coeff = np.asarray([[0.75, 0.25], [1.0, 0.0], [1.0, 0.0]])
    b = qualify_drawable_support_binding_v1(
        drawable=d,
        rigging_lineage_hash=H3,
        support_ids=ids,
        coeffs=coeff,
        rigging_positions=rig,
        locality_radius=1.1,
    )
    weights = np.asarray([[1, 0], [0, 1], [0.5, 0.5]], dtype=float)
    got = transfer_skin_via_support_v1(weights, b)
    assert np.allclose(got.sum(axis=1), 1.0)
    assert np.allclose(got[0], [0.75, 0.25])
    assert np.allclose(got[1], [0.0, 1.0])
    assert np.allclose(got[2], [0.5, 0.5])


def test_support_binding_rejects_nonlocal_active_support():
    d = _drawable()
    rig = np.asarray([[100, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    ids = np.asarray([[0], [1], [2]], dtype=np.int64)
    coeff = np.ones((3, 1), dtype=float)
    with pytest.raises(QualificationError):
        qualify_drawable_support_binding_v1(
            drawable=d,
            rigging_lineage_hash=H3,
            support_ids=ids,
            coeffs=coeff,
            rigging_positions=rig,
            locality_radius=2.0,
        )
