from __future__ import annotations

import hashlib
import json

import pytest

from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _static_mesh_for_appearance,
)


def _artifact(tmp_path, *, report: dict):
    value = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash="c" * 64,
        surface_addressing_binding_hash="a" * 64,
        appearance_domain_binding_hash="d" * 64,
        geometry_gate_binding_hash="g" * 64,
        partition_binding_hash="p" * 64,
        qualification_report=dict(report),
        qualification_hash="",
    )
    from dataclasses import replace

    value = replace(value, qualification_hash=static_mesh_qualification_hash(value))
    path = tmp_path / "static.json"
    path.write_text(json.dumps(value.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "schema": "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        "authority_class": "TEST_STATIC",
    }


def _ctx(output, *, execution_class: str, ledger_status: str, demo: dict | None = None):
    return {
        "ledger": {
            "execution_class": execution_class,
            "stages": [
                {
                    "id": "19_STATIC_CANONICAL_MESH_QUALIFIED",
                    "status": ledger_status,
                    "outputs": [output],
                }
            ],
        },
        "run_manifest": {"demo_execution": dict(demo or {})},
    }


def test_product_static_mesh_status_is_admitted(tmp_path):
    out = _artifact(
        tmp_path,
        report={"status": "PASS_STATIC_CANONICAL_CARRIER"},
    )
    value = _static_mesh_for_appearance(
        _ctx(out, execution_class="WITNESS", ledger_status="PASS")
    )
    assert value.qualification_report["status"] == "PASS_STATIC_CANONICAL_CARRIER"


def test_demo_measurement_is_admitted_only_with_explicit_scope(tmp_path):
    report = {
        "status": "DEMO_ONLY_MEASURED_STATIC_CANONICAL_CARRIER__P999_FAIL",
        "demo_geometry_lineage": True,
        "product_authority_claimed": False,
        "demo_only_source_fidelity_admission": True,
    }
    out = _artifact(tmp_path, report=report)
    value = _static_mesh_for_appearance(
        _ctx(
            out,
            execution_class="DEMO_WITNESS",
            ledger_status="PASS_DEMO_ONLY",
            demo={
                "stage13_scientific_pass": False,
                "product_authority_claimed": False,
            },
        )
    )
    assert value.qualification_report["product_authority_claimed"] is False


def test_demo_measurement_is_rejected_in_product_witness(tmp_path):
    out = _artifact(
        tmp_path,
        report={
            "status": "DEMO_ONLY_MEASURED_STATIC_CANONICAL_CARRIER__P999_FAIL",
            "demo_geometry_lineage": True,
            "product_authority_claimed": False,
            "demo_only_source_fidelity_admission": True,
        },
    )
    with pytest.raises(QualificationError, match="CAA_STATIC_MESH_NOT_ADMISSIBLE"):
        _static_mesh_for_appearance(
            _ctx(out, execution_class="WITNESS", ledger_status="PASS")
        )
