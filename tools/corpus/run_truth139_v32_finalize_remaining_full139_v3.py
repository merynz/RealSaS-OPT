#!/usr/bin/env python3
"""Hardened V3 entrypoint for Truth139 V3.2 remaining-truth finalization.

V3 keeps the V2 cohort/split/readiness logic, but adds three release invariants:
1. the one-subject Response SVD numeric rescue is explicitly rebound to the
   selected 139/139 Response receipt;
2. the Python truth-materialization runtime is frozen and verified;
3. TESSA producer/court subprocesses are routed through the explicit 3D-Rtree
   entrypoints.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

from tools.corpus import run_truth139_v32_finalize_remaining_full139_v2 as impl

RESCUE_STATUS = "PASS_SINGLE_SUBJECT_SVD_COURT_NUMERIC_RESCUE"
RESCUE_SUBJECT = "truth139_0fc13c59558762dccc11"
RESCUE_VALIDATOR_SHA256 = "90a1ebd246e0254b8e7bac704a0d3a65a6ba0fbe202c2ebc804d87f64c53f39c"
RUNTIME_STATUS = "PASS_TRUTH139_V32_FINALIZE_RUNTIME_FROZEN"
EXPECTED_PACKAGES = {
    "torch": "2.8.0+cpu",
    "torchvision": "0.23.0+cpu",
    "numpy": "2.1.3",
    "Pillow": "11.3.0",
    "trimesh": "5.2.0",
    "Rtree": "1.4.1",
    "fast-simplification": "0.2.0",
    "pytest": "8.4.2",
}


def _load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{code}:{path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def _extract_value(argv: list[str], flag: str) -> str:
    hits = [i for i, value in enumerate(argv) if value == flag]
    if len(hits) != 1 or hits[0] + 1 >= len(argv):
        raise ValueError("V32_FINAL_V3_ARGUMENT:" + flag)
    return argv[hits[0] + 1]


def _strip_pair(argv: list[str], flag: str) -> tuple[list[str], str]:
    value = _extract_value(argv, flag)
    i = argv.index(flag)
    return argv[:i] + argv[i + 2 :], value


def _validate_runtime(path: Path) -> dict:
    value = _load_json(path, "V32_FINAL_RUNTIME_FINGERPRINT")
    if value.get("status") != RUNTIME_STATUS:
        raise ValueError("V32_FINAL_RUNTIME_STATUS")
    requirements = dict(value.get("requirements") or {})
    req_path = Path(str(requirements.get("path") or "")).resolve()
    if not req_path.is_file() or impl.sha256(req_path) != requirements.get("sha256"):
        raise ValueError("V32_FINAL_RUNTIME_REQUIREMENTS_DRIFT")
    declared = dict(value.get("packages") or {})
    if declared != EXPECTED_PACKAGES:
        raise ValueError("V32_FINAL_RUNTIME_DECLARATION_DRIFT")
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if actual != EXPECTED_PACKAGES:
        raise ValueError("V32_FINAL_RUNTIME_INSTALLED_PACKAGE_DRIFT:" + repr(actual))
    return value


def _validate_response_rescue(rescue_path: Path, response_path: Path) -> dict:
    rescue_path = Path(rescue_path).resolve()
    response_path = Path(response_path).resolve()
    if rescue_path.parent != response_path.parent:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_BATCH_DRIFT")
    rescue = _load_json(rescue_path, "V32_FINAL_RESPONSE_RESCUE")
    response = _load_json(response_path, "V32_FINAL_RESPONSE_RECEIPT")
    if rescue.get("status") != RESCUE_STATUS or rescue.get("subject_id") != RESCUE_SUBJECT:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_STATUS")
    if rescue.get("base_repository_commit") != response.get("code_commit_sha"):
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_BASE_COMMIT")
    if rescue.get("patched_validator_sha256") != RESCUE_VALIDATOR_SHA256:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_VALIDATOR_DRIFT")
    diagnosis = dict(rescue.get("diagnosis") or {})
    if diagnosis.get("producer_truth_changed") is not False:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_PRODUCER_CHANGED")
    if diagnosis.get("teacher_truth_changed") is not False:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_TEACHER_CHANGED")
    if diagnosis.get("tolerance_relaxed") is not False:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_TOLERANCE_RELAXED")
    if diagnosis.get("new_independent_formulation") != "REST_TANGENT_SVD_GENERALIZED_PRINCIPAL_STRETCH":
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_FORMULATION")

    bindings = dict(response.get("subject_output_bindings") or {})
    binding = dict(bindings.get(RESCUE_SUBJECT) or {})
    chain_path = Path(str(binding.get("chain_path") or "")).resolve()
    if not chain_path.is_file() or impl.sha256(chain_path) != binding.get("chain_sha256"):
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_CHAIN_DRIFT")
    chain = _load_json(chain_path, "V32_FINAL_RESPONSE_RESCUE_CHAIN")
    court = dict(rescue.get("court") or {})
    court_ref = dict((chain.get("outputs") or {}).get("response_court") or {})
    if court_ref.get("sha256") != court.get("sha256"):
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_COURT_SHA_DRIFT")
    if (chain.get("bindings") or {}).get("response_court_identity_sha256") != court.get("court_identity_sha256"):
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_COURT_IDENTITY_DRIFT")
    if float(court.get("maximum_nonrigid_abs_error", 1.0)) >= 1.0e-8:
        raise ValueError("V32_FINAL_RESPONSE_RESCUE_NUMERIC_ERROR")
    return rescue


def _route_tessa_v2(cmd, log=None):
    routed = []
    for item in cmd:
        text = str(item)
        if text.endswith("tools/training/build_truth139_v32_mechanical_tessa_teacher_v1.py"):
            text = str(impl.ROOT / "tools/training/build_truth139_v32_mechanical_tessa_teacher_v2.py")
        elif text.endswith("tools/training/validate_truth139_v32_mechanical_tessa_teacher_v1.py"):
            text = str(impl.ROOT / "tools/training/validate_truth139_v32_mechanical_tessa_teacher_v2.py")
        routed.append(text)
    return _ORIGINAL_RUN(routed, log)


def main() -> None:
    argv = list(sys.argv)
    argv, runtime_raw = _strip_pair(argv, "--runtime-fingerprint")
    argv, rescue_raw = _strip_pair(argv, "--response-svd-rescue")
    response_raw = _extract_value(argv, "--response-receipt")
    output_raw = _extract_value(argv, "--output-root")

    runtime_path = Path(runtime_raw).resolve()
    rescue_path = Path(rescue_raw).resolve()
    response_path = Path(response_raw).resolve()
    output_root = Path(output_raw).resolve()

    runtime = _validate_runtime(runtime_path)
    rescue = _validate_response_rescue(rescue_path, response_path)

    # V2 argparse must see only its own declared arguments.
    sys.argv = argv
    impl.run = _route_tessa_v2
    impl.main()

    ledger_path = output_root / "V32_TRAINING_READINESS_RECEIPT.json"
    ledger = _load_json(ledger_path, "V32_FINAL_V3_LEDGER")
    if ledger.get("status") != impl.FINAL_PASS:
        raise ValueError("V32_FINAL_V3_LEDGER_NOT_PASS")

    ledger.pop("ledger_identity_sha256", None)
    ledger["runner_revision"] = 3
    ledger["runtime_fingerprint"] = {
        "path": str(runtime_path),
        "sha256": impl.sha256(runtime_path),
        "identity_sha256": runtime.get("runtime_identity_sha256"),
        "status": runtime.get("status"),
    }
    ledger["supplemental_response_numeric_rescue"] = {
        "path": str(rescue_path),
        "sha256": impl.sha256(rescue_path),
        "receipt_identity_sha256": rescue.get("receipt_identity_sha256"),
        "status": rescue.get("status"),
        "subject_id": rescue.get("subject_id"),
        "patched_validator_sha256": rescue.get("patched_validator_sha256"),
    }
    ledger["ledger_identity_sha256"] = impl.digest(ledger)
    impl.atomic(ledger_path, ledger)

    print(json.dumps({
        "status": ledger["status"],
        "runner_revision": 3,
        "subject_count": ledger["subject_count"],
        "runtime_fingerprint_sha256": ledger["runtime_fingerprint"]["sha256"],
        "response_numeric_rescue_sha256": ledger["supplemental_response_numeric_rescue"]["sha256"],
        "ledger_identity_sha256": ledger["ledger_identity_sha256"],
        "receipt_path": str(ledger_path),
    }, sort_keys=True, indent=2), flush=True)


_ORIGINAL_RUN = impl.run


if __name__ == "__main__":
    main()
