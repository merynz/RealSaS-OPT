from __future__ import annotations

import hashlib
import inspect

from compiler.realsas_compiler_core.mesh._historical_v05 import (
    HISTORICAL_CDT_SOURCE_BYTES,
    HISTORICAL_CDT_SOURCE_SHA256,
    historical_cdt_source_bytes,
    load_historical_cdt_module_v05,
    triangulate_production_cdt,
)


def test_historical_cdt_payload_is_exact_hash_sealed_numerical_source():
    raw = historical_cdt_source_bytes()
    assert len(raw) == HISTORICAL_CDT_SOURCE_BYTES == 20221
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_CDT_SOURCE_SHA256
    assert HISTORICAL_CDT_SOURCE_SHA256 == "dd21ae3fc570b2eb5d3c439a5c31fd25fed2ad0743d34a71119d5b391806854d"


def test_historical_cdt_module_loads_only_as_callable_numerical_kernel():
    module = load_historical_cdt_module_v05()
    assert callable(module.triangulate_production_cdt)
    assert callable(triangulate_production_cdt)
    signature = inspect.signature(module.triangulate_production_cdt)
    assert len(signature.parameters) >= 1
