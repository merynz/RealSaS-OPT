#!/usr/bin/env python3
"""Prepare or apply exact recovered MIRA V5.5 DRAFT ModelRelease registration.

The operator is intentionally narrow:
- current canonical main must already be the deployed Platform source;
- recovery receipt owns the recovered runtime identity;
- dry-run performs no Platform mutation;
- apply seals/reuses one current RESEARCH EngineRelease, then delegates all
  Artifact/ModelRelease/evaluation writes to register_mira_v55_modelrelease_v1.py;
- no promotion policy is created, no channel is selected and product.skin is not
  activated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_services.orchestrator import mainline
from tools import register_mira_v55_modelrelease_v1 as registration
from tools.platform_release_snapshot import snapshot

SCHEMA = "RealSaS.MIRAV55ModelReleaseOperatorReceipt.v1"
RELEASE_NAME = "MIRA_V55_RUNTIME_REGISTRATION__RESEARCH_RELEASE"
OPERATION = "MIRA_V55_DRAFT_MODELRELEASE_REGISTER"
DEFAULT_RECEIPT = "canonical/MIRA_V55_RUNTIME_RECOVERY_V1.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _require_within(path: Path, root: Path, code: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(code) from exc


def _deployment_code_sha(platform_root: Path) -> str:
    marker = platform_root / "deployment.json"
    if not marker.is_file():
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_DEPLOYMENT_MARKER_MISSING")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.DeveloperDeployment.v1":
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_DEPLOYMENT_SCHEMA_DRIFT")
    code_sha = str(value.get("code_sha") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_DEPLOYED_SHA_INVALID")
    return code_sha


def _release_request(actor: str) -> dict[str, Any]:
    plan = mainline.load_json(mainline.PLAN_PATH)
    return snapshot(
        plan,
        name=RELEASE_NAME,
        purpose="RESEARCH",
        created_by=actor,
    )


def _release_id(response: dict[str, Any]) -> str:
    value = str(response.get("ReleaseID") or response.get("release_id") or "").strip()
    if not value:
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_RELEASE_ID_MISSING")
    return value


def _run_registration(
    *,
    repo: Path,
    api_root: str,
    runtime_path: Path,
    recovery_receipt: Path,
    engine_release_id: str,
    source_code_sha: str,
    actor: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo / "tools/register_mira_v55_modelrelease_v1.py"),
        "--api-root",
        api_root,
        "--runtime-checkpoint",
        str(runtime_path),
        "--recovery-receipt",
        str(recovery_receipt),
        "--engine-release-id",
        engine_release_id,
        "--source-code-sha",
        source_code_sha,
        "--actor",
        actor,
        "--apply",
    ]
    completed = subprocess.run(
        command,
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_REGISTRATION_RECEIPT_INVALID") from exc
    if result.get("schema") != "RealSaS.MIRAV55ModelReleaseRegistrationPlan.v1":
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_REGISTRATION_SCHEMA_DRIFT")
    if result.get("apply") is not True or result.get("promotion_performed") is not False:
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_REGISTRATION_AUTHORITY_DRIFT")
    if result.get("lifecycle_state") != "DRAFT":
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_NOT_DRAFT")
    if str(result.get("engine_release_id")) != engine_release_id:
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_ENGINE_RELEASE_DRIFT")
    if str(result.get("source_code_sha")) != source_code_sha:
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_SOURCE_SHA_DRIFT")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--platform-root", type=Path, default=Path("/home/monster/realsas_platform"))
    parser.add_argument("--authority-root", type=Path, default=Path("/home/monster/realsas_authority"))
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--source-code-sha", required=True)
    parser.add_argument("--actor", default="mira-v55-modelrelease-operator")
    parser.add_argument("--recovery-receipt", default=DEFAULT_RECEIPT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    platform_root = args.platform_root.resolve()
    authority_root = args.authority_root.resolve()
    source_code_sha = args.source_code_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_code_sha):
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_SOURCE_SHA_INVALID")
    if _deployment_code_sha(platform_root) != source_code_sha:
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_DEPLOYED_MAIN_DRIFT")

    receipt_path = Path(args.recovery_receipt)
    if not receipt_path.is_absolute():
        receipt_path = repo / receipt_path
    receipt_path = receipt_path.resolve()
    _require_within(receipt_path, repo, "MIRA_MODELRELEASE_OPERATOR_RECEIPT_OUTSIDE_REPO")
    receipt = registration._load_receipt(receipt_path)

    runtime_path = Path(str(receipt["runtime_package"]["authority_host_path"])).resolve()
    _require_within(runtime_path, authority_root, "MIRA_MODELRELEASE_OPERATOR_RUNTIME_OUTSIDE_AUTHORITY_ROOT")
    if not runtime_path.is_file():
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_RUNTIME_MISSING")
    runtime_sha, runtime_size = registration._sha256_file(runtime_path)
    expected_runtime = receipt["runtime_package"]
    if runtime_sha != expected_runtime["content_sha256"] or runtime_size != int(expected_runtime["size_bytes"]):
        raise RuntimeError("MIRA_MODELRELEASE_OPERATOR_RUNTIME_IDENTITY_DRIFT")

    release_request = _release_request(args.actor)
    release_request_sha = _sha256_json(release_request)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": OPERATION,
        "status": "DRY_RUN_VALIDATED" if not args.apply else "APPLY_PENDING",
        "apply": bool(args.apply),
        "source_code_sha": source_code_sha,
        "deployment_code_sha": source_code_sha,
        "api_root": args.api_root,
        "release_name": RELEASE_NAME,
        "release_request_sha256": release_request_sha,
        "pipeline_plan_sha256": release_request["manifest"]["pipeline_plan_sha256"],
        "runtime_content_sha256": runtime_sha,
        "runtime_size_bytes": runtime_size,
        "runtime_path": str(runtime_path),
        "recovery_receipt": str(receipt_path),
        "promotion_performed": False,
        "promotion_policy_created": False,
        "product_skin_activated": False,
        "new_scientific_pass_minted": False,
        "platform_mutation_performed": False,
    }

    if args.apply:
        release = registration._request_json(args.api_root, "POST", "/v1/releases", release_request)
        engine_release_id = _release_id(release)
        registration._require_canonical_source(args.api_root, engine_release_id, source_code_sha)
        registered = _run_registration(
            repo=repo,
            api_root=args.api_root,
            runtime_path=runtime_path,
            recovery_receipt=receipt_path,
            engine_release_id=engine_release_id,
            source_code_sha=source_code_sha,
            actor=args.actor,
        )
        result.update(
            {
                "status": "DRAFT_MODELRELEASE_REGISTERED",
                "engine_release_id": engine_release_id,
                "model_release_id": registered["model_release_id"],
                "runtime_artifact_id": registered["runtime_artifact_id"],
                "recovery_evidence_artifact_id": registered["recovery_evidence_artifact_id"],
                "lifecycle_state": registered["lifecycle_state"],
                "evaluations": registered["evaluations"],
                "platform_mutation_performed": True,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(_json_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
