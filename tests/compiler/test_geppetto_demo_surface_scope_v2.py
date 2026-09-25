from __future__ import annotations

from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.learned_mechanics_v2 import (
    _assert_geppetto_surface_scope,
)


def _ctx(*, execution_class="DEMO_WITNESS", ledger_status="PASS_DEMO_ONLY", demo=None):
    return {
        "ledger": {
            "execution_class": execution_class,
            "stages": [
                {
                    "id": "15_RIGGING_SURFACE_QUALIFIED",
                    "status": ledger_status,
                }
            ],
        },
        "run_manifest": {
            "demo_execution": dict(
                {
                    "allow_stage14_scientific_fail_for_demo": True,
                    "product_authority_claimed": False,
                }
                if demo is None
                else demo
            )
        },
    }


def _qualification(report):
    return SimpleNamespace(qualification_report=dict(report))


def test_geppetto_accepts_explicit_demo_surface_scope():
    _assert_geppetto_surface_scope(
        _ctx(),
        _qualification(
            {
                "status": "DEMO_ONLY_RIGGING_SURFACE_ADMISSION__SCIENTIFIC_ADEQUACY_FAIL",
                "substrate_adequacy_passed": False,
                "demo_fallback_admitted": True,
                "product_authority_claimed": False,
            }
        ),
    )


def test_geppetto_rejects_demo_surface_in_product_witness():
    with pytest.raises(QualificationError, match="GEPPETTO_STAGE15_SURFACE_NOT_ADMISSIBLE"):
        _assert_geppetto_surface_scope(
            _ctx(execution_class="WITNESS", ledger_status="PASS"),
            _qualification(
                {
                    "status": "DEMO_ONLY_RIGGING_SURFACE_ADMISSION__SCIENTIFIC_ADEQUACY_FAIL",
                    "substrate_adequacy_passed": False,
                    "demo_fallback_admitted": True,
                    "product_authority_claimed": False,
                }
            ),
        )


def test_geppetto_rejects_demo_surface_if_product_authority_claimed():
    with pytest.raises(QualificationError, match="GEPPETTO_STAGE15_SURFACE_NOT_ADMISSIBLE"):
        _assert_geppetto_surface_scope(
            _ctx(),
            _qualification(
                {
                    "status": "DEMO_ONLY_RIGGING_SURFACE_ADMISSION__SCIENTIFIC_ADEQUACY_FAIL",
                    "substrate_adequacy_passed": False,
                    "demo_fallback_admitted": True,
                    "product_authority_claimed": True,
                }
            ),
        )
