#!/usr/bin/env python3
"""Real API/outbox/Temporal/Engine/Registry smoke, not Knight scientific proof.

Runs only SourceLicense in a two-node research graph. Removes its independent
SourceBytes sibling in the second release and proves exact artifact reuse.
No direct Registry writes, fabricated execution versions or product promotion.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_release_snapshot import snapshot

API = "http://127.0.0.1:8080"
ACTOR = "platform-deployment-smoke"


def request(path, value=None, *, method=None, raw=None):
    body = raw if raw is not None else (json.dumps(value).encode() if value is not None else None)
    req = urllib.request.Request(API + path, data=body, method=method,
                                 headers={"Content-Type": "application/octet-stream" if raw is not None else "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"PLATFORM_API_{error.code}:{error.read(2048).decode()}") from error


def smoke_plans():
    baseline = mainline.load_json(mainline.PLAN_PATH)
    baseline["stages"] = baseline["stages"][:2]
    baseline["stage_count"] = 2
    baseline["stages"][1]["depends_on"] = []
    candidate = deepcopy(baseline)
    candidate["stages"] = [candidate["stages"][1]]
    candidate["stages"][0]["ordinal"] = 1
    candidate["stage_count"] = 1
    return baseline, candidate


def wait_attempt(attempt_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request(f"/v1/attempts/{attempt_id}")
        if state["state"] == "COMPLETED":
            return state
        if state["state"] != "OPEN":
            raise RuntimeError(f"SMOKE_ATTEMPT_FAILED:{json.dumps(state)}")
        time.sleep(1)
    raise RuntimeError(f"SMOKE_ATTEMPT_TIMEOUT:{attempt_id}; inspect stored events, do not resubmit blindly")


def registry_read(root, sql):
    from pgserver._commands import POSTGRES_BIN_PATH
    result = subprocess.run([str(POSTGRES_BIN_PATH / "psql"), "-h", str(root / "data/socket"),
                             "-p", "55432", "-d", "postgres", "-Atc", sql],
                            check=True, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def artifact_for_attempt(root, attempt_id, stage_id):
    attempt_id = str(uuid.UUID(attempt_id))
    if stage_id != "02_SOURCE_LICENSE_PROVENANCE":
        raise RuntimeError("SMOKE_UNEXPECTED_STAGE_ID")
    # Read-only proof of the Registry binding. All mutations went through API.
    sql = ("SELECT a.id||'|'||a.content_sha256||'|'||a.storage_key FROM attempt_artifacts aa "
           "JOIN artifacts a ON a.id=aa.artifact_id WHERE aa.attempt_id='" + attempt_id +
           "' AND aa.role='stage:" + stage_id + "';")
    fields = registry_read(root, sql).split("|")
    if len(fields) != 3:
        raise RuntimeError("SMOKE_REGISTRY_ARTIFACT_BINDING_MISSING")
    data = (root / "artifacts" / fields[2]).read_bytes()
    if hashlib.sha256(data).hexdigest() != fields[1]:
        raise RuntimeError("SMOKE_REGISTRY_CAS_HASH_DRIFT")
    return {"artifact_id": fields[0], "content_sha256": fields[1], "storage_key": fields[2]}


def run_smoke(root, timeout):
    started = time.monotonic()
    deployment = json.loads((root / "deployment.json").read_text())
    os.environ["REALSAS_AUTHORITY_ROOT"] = deployment["authority_root"]
    subject = request("/v1/subjects", {"slug": "platform-deployment-smoke-v1",
                       "display_name": "Platform deployment engineering smoke", "created_by": ACTOR})["subject_id"]
    license_section = {"source_pack": "subject-free-service-smoke", "license_name": "engineering-fixture",
                       "license_ref": "urn:realsas:platform:smoke:no-scientific-authority"}
    raw = json.dumps(license_section, sort_keys=True).encode()
    sha = hashlib.sha256(raw).hexdigest()
    obj = request(f"/v1/artifacts/bytes?sha256={sha}", method="PUT", raw=raw)
    imported = request("/v1/artifacts/import", {"artifact_type": "RealSaS.SourceLicenseInput",
                       "schema_version": "v1", "object": obj,
                       "source_uri": "urn:realsas:platform:deployment-smoke", "created_by": ACTOR})
    inputs = request("/v1/subject-inputs", {"subject_id": subject,
                     "bindings": [{"role": "manifest:source_license", "artifact_id": imported["artifact_id"]}],
                     "created_by": ACTOR})
    baseline, candidate = smoke_plans()
    # Release identity depends on consumed sections, never this run's UUID.
    identity = {"source_license": license_section}
    releases = [request("/v1/releases", snapshot(plan, identity, name=f"platform-deployment-smoke-{index}-v1",
                                               purpose="RESEARCH", created_by=ACTOR))["ReleaseID"]
                for index, plan in enumerate((baseline, candidate))]
    target = baseline["stages"][1]["id"]
    removed = baseline["stages"][0]["id"]
    attempts, states, artifacts = [], [], []
    first_license_status = None
    for index, (plan, release_id) in enumerate(zip((baseline, candidate), releases)):
        run_id = "PLATFORM_SERVICE_SMOKE_" + uuid.uuid4().hex.upper()
        manifest_path = mainline.run_manifest_path(run_id)
        if manifest_path.exists():
            raise RuntimeError("SMOKE_RUN_COLLISION")
        manifest = {"run_id": run_id, "subject_id": "SUBJECT_FREE_PLATFORM_DEPLOYMENT",
                    "execution_class": "IMPLEMENTATION_AUDIT", **identity}
        mainline.atomic_json(manifest_path, manifest)
        attempt_request = {"subject_id": subject,
                          "baseline_engine_release_id": releases[0], "candidate_engine_release_id": release_id,
                          "parent_attempt_id": attempts[0] if attempts else None, "created_by": ACTOR}
        if index == 1:
            attempt_request["intervention"] = {"direct_changed_stage_ids": [removed],
                                                "preserved_stage_ids": [target]}
        attempt = request("/v1/research/attempts", attempt_request)
        attempt_id = attempt["AttemptID"]
        attempts.append(attempt_id)
        if index == 1:
            impact = attempt["Impact"]
            if removed not in (impact.get("RemovedStageIDs") or ()) or target not in (impact.get("UnchangedStageIDs") or ()):
                raise RuntimeError("SMOKE_GRAPH_REMOVAL_IMPACT_DRIFT")
        request("/v1/research/compile", {"research_attempt_id": attempt_id, "subject_id": subject,
                "engine_release_id": release_id, "subject_input_id": inputs["subject_input_id"],
                "target_stage_id": target, "compiler_run_id": run_id,
                "run_manifest_path": str(manifest_path), "run_ledger_path": str(mainline.run_ledger_path(run_id)),
                "pipeline_plan_sha256": mainline.content_sha256(plan),
                "idempotency_key": f"platform-smoke:{run_id}", "requested_by": ACTOR})
        state = wait_attempt(attempt_id, timeout)
        states.append(state)
        artifacts.append(artifact_for_attempt(root, attempt_id, target))
        if index == 0:
            first_license_status = "EXECUTED" if any(event["type"] == "STAGE_EXECUTION_PASSED" for event in state["events"]) else "REUSED"
            if first_license_status == "EXECUTED":
                ledger = mainline.load_json(mainline.run_ledger_path(run_id))
                if ledger["stages"][0]["status"] != "PENDING" or ledger["stages"][1]["status"] != "PASS":
                    raise RuntimeError("SMOKE_ENGINE_EXECUTED_CANONICAL_INSTEAD_OF_RELEASED_GRAPH")
    second_events = states[1]["events"]
    if not any(event["type"] == "STAGE_REUSED" for event in second_events) or any(event["type"] == "STAGE_EXECUTION_STARTED" for event in second_events):
        raise RuntimeError("SMOKE_INDEPENDENT_ARTIFACT_WAS_REBUILT")
    if artifacts[0] != artifacts[1]:
        raise RuntimeError("SMOKE_ARTIFACT_IDENTITY_CHANGED_AFTER_UNRELATED_NODE_REMOVAL")
    checks = [event["payload"] for event in second_events if event["type"] == "INTERVENTION_REUSE_VERIFIED"]
    if len(checks) != 1:
        raise RuntimeError("SMOKE_CONTROLLED_INTERVENTION_RECEIPT_MISSING")
    control = checks[0]
    preserved = control["receipt"]["preserved_artifacts"]
    if (control["baseline_subject_input_id"] != inputs["subject_input_id"]
            or control["candidate_subject_input_id"] != inputs["subject_input_id"]
            or control["receipt"]["execute_stage_ids"]
            or len(preserved) != 1 or preserved[0]["stage_id"] != target
            or preserved[0]["artifact_id"] != artifacts[0]["artifact_id"]):
        raise RuntimeError("SMOKE_CONTROLLED_INTERVENTION_IDENTITY_DRIFT")
    if registry_read(root, "SELECT count(*) FROM product_revisions WHERE subject_id='" + str(uuid.UUID(subject)) + "'") != "0":
        raise RuntimeError("SMOKE_RESEARCH_MINTED_PRODUCT_REVISION")
    receipt = {"schema": "RealSaS.DeveloperDeploymentSmoke.v1", "status": "PASS",
               "scope": "ENGINEERING_ONLY__NOT_KNIGHT_OR_PRODUCT_PROOF", "code_sha": deployment["code_sha"],
               "attempt_ids": attempts, "target_stage_id": target, "removed_stage_id": removed,
               "first_target_action": first_license_status, "second_target_action": "REUSE",
               "exact_shared_artifact": artifacts[0], "product_revision_minted": False,
               "controlled_intervention_verified": True, "intervention_receipt": control,
               "wall_seconds": round(time.monotonic() - started, 3)}
    output = root / "receipts" / f"smoke-{attempts[1]}.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    run_smoke(args.root, args.timeout)


if __name__ == "__main__":
    main()
