from __future__ import annotations

import json

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.mainline import ready_stage_ids
from compiler.realsas_compiler_services.orchestrator.status_semantics import (
    assert_demo_only_scope,
    dependency_status_admissible,
    normalize_success_status,
)


def _plan():
    return {
        "stages": [
            {"ordinal": 1, "id": "01_A", "depends_on": []},
            {"ordinal": 2, "id": "02_B", "depends_on": ["01_A"]},
            {"ordinal": 3, "id": "03_C", "depends_on": ["02_B"]},
        ]
    }


def _ledger(execution_class: str, first_status: str = "PASS_DEMO_ONLY"):
    return {
        "execution_class": execution_class,
        "stages": [
            {"ordinal": 1, "id": "01_A", "status": first_status, "outputs": []},
            {"ordinal": 2, "id": "02_B", "status": "PENDING", "outputs": []},
            {"ordinal": 3, "id": "03_C", "status": "PENDING", "outputs": []},
        ],
    }


def test_demo_only_dependency_is_admissible_only_in_demo_witness():
    demo = _ledger("DEMO_WITNESS")
    witness = _ledger("WITNESS")
    assert dependency_status_admissible(demo, "PASS_DEMO_ONLY")
    assert not dependency_status_admissible(witness, "PASS_DEMO_ONLY")
    assert ready_stage_ids(_plan(), demo) == ("02_B",)
    assert ready_stage_ids(_plan(), witness) == ()


def test_demo_only_status_is_rejected_outside_demo_witness():
    with pytest.raises(RuntimeError, match="PASS_DEMO_ONLY_OUTSIDE_DEMO_WITNESS"):
        assert_demo_only_scope(_ledger("WITNESS"))


def test_demo_only_taint_propagates_transitively():
    plan = _plan()
    ledger = _ledger("DEMO_WITNESS")

    status_b = normalize_success_status(
        plan=plan,
        ledger=ledger,
        stage_id="02_B",
        reported_status="PASS",
    )
    assert status_b == "PASS_DEMO_ONLY"
    ledger["stages"][1]["status"] = status_b

    status_c = normalize_success_status(
        plan=plan,
        ledger=ledger,
        stage_id="03_C",
        reported_status="PASS",
    )
    assert status_c == "PASS_DEMO_ONLY"


def test_normal_lineage_remains_normal_pass():
    plan = _plan()
    ledger = _ledger("DEMO_WITNESS", first_status="PASS")
    assert normalize_success_status(
        plan=plan,
        ledger=ledger,
        stage_id="02_B",
        reported_status="PASS",
    ) == "PASS"


def test_adapter_payload_accepts_demo_only_only_in_demo_witness(tmp_path):
    payload = {"schema": "Schema.Test.v1", "value": 1}
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    output = {
        "path": str(artifact),
        "sha256": digest,
        "schema": "Schema.Test.v1",
        "authority_class": "TEST",
    }
    demo = {
        "ledger": {
            "execution_class": "DEMO_WITNESS",
            "stages": [
                {
                    "id": "01_A",
                    "status": "PASS_DEMO_ONLY",
                    "outputs": [output],
                }
            ],
        }
    }
    assert stage_output_payload(demo, "01_A", "Schema.Test.v1") == payload

    witness = {
        "ledger": {
            "execution_class": "WITNESS",
            "stages": [
                {
                    "id": "01_A",
                    "status": "PASS_DEMO_ONLY",
                    "outputs": [output],
                }
            ],
        }
    }
    with pytest.raises(QualificationError, match="ADAPTER_UPSTREAM_NOT_PASS"):
        stage_output_payload(witness, "01_A", "Schema.Test.v1")
