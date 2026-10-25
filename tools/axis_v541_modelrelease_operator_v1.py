#!/usr/bin/env python3
from __future__ import annotations

"""Package, register, evaluate and RESEARCH-promote exact AXIS V5.4.1.

The operator preserves the narrow historical claim boundary: one sealed FIT1
terminal PASS plus deterministic runtime packaging/load evidence. It does not
train, fit, mint a new scientific PASS, claim FIT8/LOFO/unseen generalization,
or confer product rig authority. Compiler remains canonical rig authority.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_services.orchestrator import mainline
from models.axis.runtime_checkpoint_v1 import (
    AXIS_V541_ARCHITECTURE_ID,
    load_axis_runtime_model_v1,
    save_axis_runtime_checkpoint_v1,
)
from tools import register_axis_v541_modelrelease_v1 as registration
from tools.platform_release_snapshot import snapshot

SCHEMA = "RealSaS.AXISV541ModelReleaseOperatorReceipt.v1"
OPERATION = "AXIS_V541_RESEARCH_MODELRELEASE_PROMOTE"
RELEASE_NAME = "AXIS_V541_RUNTIME_REGISTRATION__RESEARCH_RELEASE"
POLICY_ID = "9389310f-9555-5a42-ba35-a98f531ca725"
POLICY_SHA256 = "e715f018de5c03f5561989829f70a0cff8eeff0a692b2c73fc67de7300e7cfb9"
POLICY_SCOPES = [
    "RUNTIME_PACKAGE_IDENTITY_V1",
    "RUNTIME_SELF_CONTAINED_LOAD_V1",
    "HISTORICAL_FIT1_TERMINAL_V541",
    "AXIS_RUNTIME_AUTHORITY_CONTRACT_V1",
]

SOURCE_REL = Path("model_runtime_recovery/axis_v541/sources")
RUNTIME_REL = Path("model_runtime_recovery/axis_v541/runtime")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _deployment_code_sha(platform_root: Path) -> str:
    marker = platform_root / "deployment.json"
    if not marker.is_file():
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_DEPLOYMENT_MARKER_MISSING")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.DeveloperDeployment.v1":
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_DEPLOYMENT_SCHEMA_DRIFT")
    sha = str(value.get("code_sha") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_DEPLOYED_SHA_INVALID")
    return sha


def _require_within(path: Path, root: Path, code: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(code) from exc


def _release_request(actor: str) -> dict[str, Any]:
    plan = mainline.load_json(mainline.PLAN_PATH)
    return snapshot(plan, name=RELEASE_NAME, purpose="RESEARCH", created_by=actor)


def _release_id(response: dict[str, Any]) -> str:
    value = str(response.get("ReleaseID") or response.get("release_id") or "").strip()
    if not value:
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_RELEASE_ID_MISSING")
    return value


def _package(authority_root: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    source_root = (authority_root / SOURCE_REL).resolve()
    runtime_root = (authority_root / RUNTIME_REL).resolve()
    _require_within(source_root, authority_root, "AXIS_MODELRELEASE_OPERATOR_SOURCE_OUTSIDE_AUTHORITY")
    _require_within(runtime_root, authority_root, "AXIS_MODELRELEASE_OPERATOR_RUNTIME_OUTSIDE_AUTHORITY")
    base = source_root / "GEPPETTO_KNIGHT_CHECKPOINT.pt"
    v53 = source_root / "AXIS41_TREE_GRAMMAR_PROGRESS_V53.pt"
    v541 = source_root / "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_COMPOSITE_V541.pt"
    transport = source_root / "SOURCE_TRANSPORT_RECEIPT.json"
    for path in (base, v53, v541, transport):
        if not path.is_file():
            raise RuntimeError(f"AXIS_MODELRELEASE_OPERATOR_SOURCE_MISSING:{path.name}")

    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime = runtime_root / "AXIS_RUNTIME_CHECKPOINT_V1.pt"
    receipt_path = runtime_root / "AXIS_RUNTIME_PACKAGE_RECEIPT_V1.json"
    receipt = save_axis_runtime_checkpoint_v1(
        output_path=runtime,
        base_checkpoint_path=base,
        v53_progress_path=v53,
        v541_composite_path=v541,
    ).to_dict()
    receipt.update({
        "runtime_size_bytes": runtime.stat().st_size,
        "self_contained_load_verified": False,
        "training_executed": False,
        "fit_executed": False,
        "scientific_pass_minted": False,
        "genericity_claimed": False,
        "product_rig_authority_claimed": False,
        "compiler_remains_canonical_rig_authority": True,
    })
    model, payload = load_axis_runtime_model_v1(runtime, device="cpu")
    del model
    if payload.get("architecture_id") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_RUNTIME_ARCHITECTURE_DRIFT")
    receipt["self_contained_load_verified"] = True
    receipt_path.write_bytes(_json_bytes(receipt))
    return runtime, receipt_path, v541, transport, receipt


def _existing_axis_research_channel(api_root: str) -> dict[str, Any] | None:
    for row in registration._request_json(api_root, "GET", "/v1/model-channels"):
        if row.get("family") == "AXIS" and row.get("channel") == "RESEARCH":
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--platform-root", type=Path, default=Path("/home/monster/realsas_platform"))
    parser.add_argument("--authority-root", type=Path, default=Path("/home/monster/realsas_authority"))
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--source-code-sha", required=True)
    parser.add_argument("--actor", default="axis-v541-modelrelease-operator")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    platform_root = args.platform_root.resolve()
    authority_root = args.authority_root.resolve()
    source_sha = args.source_code_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_SOURCE_SHA_INVALID")
    if _deployment_code_sha(platform_root) != source_sha:
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_DEPLOYED_MAIN_DRIFT")

    runtime, package_receipt, historical, transport, package = _package(authority_root)
    runtime_sha, runtime_size = registration._sha256_file(runtime)
    if runtime_sha != package["content_sha256"] or runtime_size != package["runtime_size_bytes"]:
        raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_RUNTIME_IDENTITY_DRIFT")

    release_request = _release_request(args.actor)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": OPERATION,
        "status": "DRY_RUN_VALIDATED" if not args.apply else "APPLY_PENDING",
        "apply": bool(args.apply),
        "source_code_sha": source_sha,
        "deployment_code_sha": source_sha,
        "architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "release_name": RELEASE_NAME,
        "release_request_sha256": _sha256_json(release_request),
        "pipeline_plan_sha256": release_request["manifest"]["pipeline_plan_sha256"],
        "runtime_content_sha256": runtime_sha,
        "runtime_size_bytes": runtime_size,
        "runtime_path": str(runtime),
        "package_receipt_path": str(package_receipt),
        "promotion_policy_id": POLICY_ID,
        "promotion_policy_sha256": POLICY_SHA256,
        "promotion_policy_required_scopes": POLICY_SCOPES,
        "promotion_channel": "RESEARCH",
        "promotion_performed": False,
        "platform_mutation_performed": False,
        "runtime_packaging_performed": True,
        "new_scientific_pass_minted": False,
        "fit8_qualified": False,
        "lofo_qualified": False,
        "unseen_generalization_qualified": False,
        "product_rig_authority_claimed": False,
        "compiler_remains_canonical_rig_authority": True,
    }

    if args.apply:
        release = registration._request_json(args.api_root, "POST", "/v1/releases", release_request)
        engine_release_id = _release_id(release)
        registration._require_canonical_source(args.api_root, engine_release_id, source_sha)
        command = [
            sys.executable, str(repo / "tools/register_axis_v541_modelrelease_v1.py"),
            "--api-root", args.api_root,
            "--runtime-checkpoint", str(runtime),
            "--package-receipt", str(package_receipt),
            "--historical-composite", str(historical),
            "--source-transport-receipt", str(transport),
            "--engine-release-id", engine_release_id,
            "--source-code-sha", source_sha,
            "--actor", args.actor,
            "--apply",
        ]
        import subprocess
        completed = subprocess.run(command, cwd=repo, check=True, text=True, capture_output=True, timeout=1800)
        registered = json.loads(completed.stdout)
        if registered.get("schema") != "RealSaS.AXISV541ModelReleaseRegistrationPlan.v1":
            raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_REGISTRATION_SCHEMA_DRIFT")
        model_id = str(registered.get("model_release_id") or "")
        if not model_id:
            raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_MODELRELEASE_ID_MISSING")
        evaluations = {row["scope"]: row["result"] for row in registered.get("evaluations", [])}
        for scope in POLICY_SCOPES:
            if evaluations.get(scope) != "PASS":
                raise RuntimeError(f"AXIS_MODELRELEASE_OPERATOR_POLICY_SCOPE_NOT_PASS:{scope}")
        for scope in ("FIT8", "LOFO", "UNSEEN_GENERALIZATION"):
            if evaluations.get(scope) != "NOT_RUN":
                raise RuntimeError(f"AXIS_MODELRELEASE_OPERATOR_GENERALIZATION_SCOPE_DRIFT:{scope}")

        channel = _existing_axis_research_channel(args.api_root)
        if channel is None:
            promoted = registration._request_json(args.api_root, "POST", "/v1/model-releases/promote", {
                "family": "AXIS",
                "channel": "RESEARCH",
                "model_release_id": model_id,
                "expected_lock_version": 0,
                "actor": args.actor,
            })
            if str(promoted.get("model_release_id")) != model_id or promoted.get("channel") != "RESEARCH":
                raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_PROMOTION_RESPONSE_DRIFT")
            promotion_performed = True
            channel = promoted
        else:
            if str(channel.get("model_release_id")) != model_id:
                raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_RESEARCH_CHANNEL_ALREADY_OWNED")
            promotion_performed = False

        final = registration._request_json(args.api_root, "GET", f"/v1/model-releases/{model_id}")
        if final.get("lifecycle_state") != "RESEARCH_QUALIFIED":
            raise RuntimeError("AXIS_MODELRELEASE_OPERATOR_NOT_RESEARCH_QUALIFIED")
        result.update({
            "status": "RESEARCH_MODELRELEASE_SELECTED",
            "engine_release_id": engine_release_id,
            "model_release_id": model_id,
            "runtime_artifact_id": registered["runtime_artifact_id"],
            "package_evidence_artifact_id": registered["package_evidence_artifact_id"],
            "historical_evidence_artifact_id": registered["historical_evidence_artifact_id"],
            "source_transport_evidence_artifact_id": registered["source_transport_evidence_artifact_id"],
            "lifecycle_state": final["lifecycle_state"],
            "evaluations": final.get("evaluations", []),
            "channel": channel,
            "promotion_performed": promotion_performed,
            "platform_mutation_performed": True,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(_json_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
