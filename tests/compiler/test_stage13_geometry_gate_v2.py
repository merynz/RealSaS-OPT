from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.rest_preservation_v1 import silhouette_distance_metrics
from compiler.realsas_compiler_services.orchestrator.adapters.iris_geometry_v1 import _geometry_gate_policy_v2
from compiler.realsas_compiler_core.types import QualificationError


def _mask() -> np.ndarray:
    m=np.zeros((64,64),dtype=bool)
    m[16:48,18:46]=True
    return m


def test_stage13_shared_silhouette_metric_separates_identity_from_one_pixel_shift():
    source=_mask()
    identity=source.copy()
    shifted=np.zeros_like(source)
    shifted[:,1:]=source[:,:-1]

    assert silhouette_distance_metrics(source,identity)==pytest.approx((0.0,0.0,0.0))
    mean,p95,mx=silhouette_distance_metrics(source,shifted)
    assert mean>0.0
    assert p95==pytest.approx(1.0)
    assert mx==pytest.approx(1.0)


def test_stage13_v2_policy_requires_explicit_silhouette_threshold():
    base={
        "min_recall":0.999,
        "min_precision":0.999,
        "max_largest_coherent_hole_fraction":0.00025,
        "max_interior_uncovered_fraction":0.0005,
        "min_component_recall":0.999,
        "component_min_foreground_fraction":0.0,
    }
    assert _geometry_gate_policy_v2(base) is None

    full={**base,"max_silhouette_edge_p95_px":0.5}
    parsed=_geometry_gate_policy_v2(full)
    assert parsed is not None
    assert parsed["max_silhouette_edge_p95_px"]==pytest.approx(0.5)


def test_stage13_v2_policy_rejects_invalid_silhouette_threshold():
    cfg={
        "min_recall":0.999,
        "min_precision":0.999,
        "max_largest_coherent_hole_fraction":0.00025,
        "max_interior_uncovered_fraction":0.0005,
        "min_component_recall":0.999,
        "component_min_foreground_fraction":0.0,
        "max_silhouette_edge_p95_px":-0.1,
    }
    with pytest.raises(QualificationError,match="GEOMETRY_GATE_THRESHOLD_RANGE_INVALID"):
        _geometry_gate_policy_v2(cfg)
