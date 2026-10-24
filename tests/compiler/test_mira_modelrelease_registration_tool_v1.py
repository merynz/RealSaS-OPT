from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "register_mira_v55_modelrelease_v1.py"
SPEC = importlib.util.spec_from_file_location("register_mira_v55_modelrelease_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _canonical_receipt() -> Path:
    return ROOT / "canonical" / "MIRA_V55_RUNTIME_RECOVERY_V1.json"


def test_canonical_recovery_receipt_builds_draft_only_registration_payload():
    receipt = MODULE._load_receipt(_canonical_receipt())
    runtime = receipt["runtime_package"]
    payload = MODULE._registration_payload(
        receipt,
        checkpoint_artifact_id="11111111-1111-1111-1111-111111111111",
        engine_release_id="22222222-2222-2222-2222-222222222222",
        source_code_sha="a" * 40,
        actor="ci",
        name=MODULE._model_name(runtime["content_sha256"]),
        evidence_artifact_id="33333333-3333-3333-3333-333333333333",
    )

    assert payload["family"] == "MIRA"
    assert payload["architecture_id"] == "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"
    assert payload["checkpoint_content_sha256"] == "b5ad2d56a9006058cba30c0b123d4e111f3eea3b78ed42f3542cbd095983139a"
    assert payload["initial_state"] == "DRAFT"
    assert payload["metadata"]["compute_contract_id"] == "CUDA_BF16_BACKBONE_FP32_READOUT_TF32_OFF"
    assert payload["metadata"]["cross_device_parity_proven"] is False
    assert payload["metadata"]["cpu_fp32_product_execution_admitted"] is False
    assert payload["metadata"]["new_scientific_pass_minted_by_recovery"] is False
    assert payload["metadata"]["generalization_claimed"] is False
    assert payload["metadata"]["product_skin_authority_claimed"] is False


def test_recovery_receipt_rejects_new_scientific_pass_claim(tmp_path: Path):
    data = json.loads(_canonical_receipt().read_text(encoding="utf-8"))
    data["runtime_package"]["scientific_pass_minted"] = True
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="RECOVERY_MINTED_SCIENTIFIC_PASS"):
        MODULE._load_receipt(path)


def test_recovery_receipt_rejects_cpu_execution_or_cross_device_claim(tmp_path: Path):
    original = json.loads(_canonical_receipt().read_text(encoding="utf-8"))
    for key in ("cpu_fp32_product_execution_admitted", "cross_device_parity_proven"):
        data = json.loads(json.dumps(original))
        data["historical_compute_contract"][key] = True
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RuntimeError):
            MODULE._load_receipt(path)


def test_model_name_is_content_addressed():
    runtime_sha = "b5ad2d56a9006058cba30c0b123d4e111f3eea3b78ed42f3542cbd095983139a"
    assert MODULE._model_name(runtime_sha) == "mira-v5.5-axis541-b5ad2d56a900"
