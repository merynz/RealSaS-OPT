from compiler.realsas_compiler_core.mesh._historical_v05 import (
    HISTORICAL_CDT_SOURCE_BYTES,
    HISTORICAL_CDT_SOURCE_SHA256,
    historical_cdt_source_bytes,
    triangulate_production_cdt,
)


def test_historical_cdt_source_is_byte_exact():
    raw = historical_cdt_source_bytes()
    assert len(raw) == HISTORICAL_CDT_SOURCE_BYTES == 20221
    assert HISTORICAL_CDT_SOURCE_SHA256 == "dd21ae3fc570b2eb5d3c439a5c31fd25fed2ad0743d34a71119d5b391806854d"


def test_historical_cdt_kernel_executes_simple_domain():
    result = triangulate_production_cdt(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        support_points=[(0.5, 0.5)],
        target_min_angle_deg=0.0,
        max_quality_iterations=0,
    )
    assert result.success is True
    assert result.missing_constraint_count == 0
    assert len(result.triangles) >= 4
    assert result.backend_name == "realSaS.production_cdt.bowyer_watson_conforming_v1"
