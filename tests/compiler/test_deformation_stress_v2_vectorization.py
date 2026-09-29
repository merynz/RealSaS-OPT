from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _triangle_metrics,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    _triangle_metrics_batch_exact,
)
from compiler.realsas_compiler_core.types import QualificationError


def test_vectorized_g3_triangle_metrics_match_scalar_operator():
    rest=np.asarray([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [0.2,0.8,0.1],
        [2.0,0.0,0.0],
        [2.7,0.1,0.0],
        [2.2,1.1,0.3],
    ],dtype=np.float64)
    posed=np.asarray([
        [0.1,-0.2,0.0],
        [1.15,0.05,0.1],
        [0.15,0.95,0.2],
        [1.8,-0.1,0.2],
        [2.65,0.25,0.05],
        [2.0,1.35,0.5],
    ],dtype=np.float64)
    faces=np.asarray([[0,1,2],[3,4,5]],dtype=np.int64)

    batch=_triangle_metrics_batch_exact(rest,posed,faces)
    for local,face in enumerate(faces):
        scalar=_triangle_metrics(rest[face],posed[face])
        for bi,sv in zip(batch,scalar):
            assert float(bi[local]) == pytest.approx(float(sv),abs=1e-12,rel=1e-12)


def test_vectorized_g3_preserves_scalar_nonfinite_condition_semantics():
    rest=np.asarray([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
    ],dtype=np.float64)
    posed=np.asarray([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [2.0,0.0,0.0],
    ],dtype=np.float64)
    faces=np.asarray([[0,1,2]],dtype=np.int64)

    scalar=_triangle_metrics(rest,posed)
    batch=_triangle_metrics_batch_exact(rest,posed,faces)
    assert np.isinf(scalar[1])
    assert np.isinf(batch[1][0])
    assert batch[0][0] == pytest.approx(scalar[0],abs=1e-12)
    assert batch[4][0] == pytest.approx(scalar[4],abs=1e-12)


def test_vectorized_g3_fails_closed_on_degenerate_rest_triangle():
    rest=np.asarray([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [2.0,0.0,0.0],
    ],dtype=np.float64)
    posed=rest.copy()
    with pytest.raises(QualificationError,match="G3_REST_TRIANGLE_DEGENERATE"):
        _triangle_metrics_batch_exact(
            rest,posed,np.asarray([[0,1,2]],dtype=np.int64)
        )
