from __future__ import annotations

"""Fail-closed RSS.vNext Skin worker: no independent MIRA model fallback."""

import pytest

from compiler.realsas_compiler_services.platform_worker.rss_vnext_skin_executor_v3 import (
    EXECUTOR_CONTRACT, execute_rss_vnext_skin_capability_v3,
)


def _request():
    return {
        "kind":"MODULE","capability_id":"product.skin",
        "capability_metadata":{
            "executor_contract":EXECUTOR_CONTRACT,
            "required_models":[],
            "compiler_code_search_allowed":False,
            "compiler_code_repair_allowed":False,
            "post_hoc_skin_transfer_allowed":False,
            "carrier_skin_qualification":{
                "max_simplex_repair_l1":0.00001,
                "max_total_correction_l1":0.0001,
                "negative_tolerance":1.e-8,
                "max_influences":41,
            },
        },
        "model_releases":[],"input_artifacts":[],
    }


def _execute(request):
    return execute_rss_vnext_skin_capability_v3(
        request, read_object=lambda _:b"", put_bytes=lambda _:{},
    )


def test_skin_requires_all_canonical_parents_and_rejects_external_mira():
    request = _request()
    with pytest.raises(RuntimeError,match="PARENT_CARDINALITY:RealSaS.CanonicalMeshIR"):
        _execute(request)
    request["model_releases"] = [{"family":"MIRA"}]
    with pytest.raises(RuntimeError,match="UNIFIED_EXECUTOR_OR_MODEL_CONTRACT_DRIFT"):
        _execute(request)
    request["model_releases"] = []
    request["capability_metadata"]["required_models"] = [{"family":"AXIS"}]
    with pytest.raises(RuntimeError,match="UNIFIED_EXECUTOR_OR_MODEL_CONTRACT_DRIFT"):
        _execute(request)
    request["capability_metadata"]["required_models"] = []
    request["capability_metadata"]["compiler_code_repair_allowed"] = True
    with pytest.raises(RuntimeError,match="UNIFIED_EXECUTOR_OR_MODEL_CONTRACT_DRIFT"):
        _execute(request)
    request["capability_metadata"]["compiler_code_repair_allowed"] = False
    request["capability_metadata"]["post_hoc_skin_transfer_allowed"] = True
    with pytest.raises(RuntimeError,match="UNIFIED_EXECUTOR_OR_MODEL_CONTRACT_DRIFT"):
        _execute(request)
