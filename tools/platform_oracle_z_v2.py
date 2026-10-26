#!/usr/bin/env python3
"""Run Oracle-Z V2 as one Platform Experiment -> Attempt -> CAS execution.

GitHub Actions may invoke this operator, but Platform Experiment/Attempt/Artifact
identity is the durable scientific authority. A timeout resumes the same host
checkpoint and never silently creates another Attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_deployment_smoke import request, wait_attempt_terminal, registry_read
from tools.platform_release_snapshot import snapshot

ACTOR = "alfred-oracle-z-v2"
SLUG = "iris-tessa-v3-oracle-z-representation-v2"
STAGE_ID = "R01_ORACLE_Z_REPRESENTATION_V2"
PROBE = ROOT / "experiments/iris_tessa_v3_apparatus_audit_20261025/oracle_z_representation_probe_v2.py"
PREREG = ROOT / "experiments/iris_tessa_v3_apparatus_audit_20261025/ORACLE_Z_REPRESENTATION_PREREG_V2.md"
EXPERIMENT_REQUEST = ROOT / "experiments/iris_tessa_v3_apparatus_audit_20261025/ORACLE_Z_REPRESENTATION_EXPERIMENT_V2.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def canonical_checkout(deployment: dict) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True).split()[0]
    if head != remote or head != str(deployment.get("code_sha") or ""):
        raise RuntimeError("ORACLE_Z_REQUIRES_EXACT_DEPLOYED_CANONICAL_MAIN")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("ORACLE_Z_SOURCE_OVERLAY_FORBIDDEN")
    return head


def oracle_plan() -> dict:
    plan = mainline.load_json(mainline.PLAN_PATH)
    plan["stages"] = [{
        "ordinal": 1,
        "id": STAGE_ID,
        "title": "Oracle-Z representation V2 external evidence consumption",
        "group": "research_apparatus",
        "depends_on": [],
        "adapter": "compiler.realsas_compiler_services.orchestrator.adapters.oracle_z_external_v2:consume_oracle_z_report",
        "manifest_keys": ["oracle_z_spec"],
        "policy": {
            "fail_closed": True,
            "cacheable": False,
            "output_hash_required": True,
            "product_pass_authority": False,
        },
    }]
    plan["stage_count"] = 1
    mainline.validate_plan(plan, require_product_pass_authority=False)
    return plan


def stage_parameters() -> dict:
    return {
        "schema": "RealSaS.StageReleaseParameters.v1",
        "stages": {STAGE_ID: {"external_evidence_roles": ["oracle_z_report"]}},
    }


def put_import(data: bytes, *, artifact_type: str, schema_version: str, source_uri: str) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    obj = request(f"/v1/artifacts/bytes?sha256={digest}", method="PUT", raw=data)
    return request("/v1/artifacts/import", {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "object": obj,
        "source_uri": source_uri,
        "created_by": ACTOR,
    })


def experiment_record(created: dict) -> dict:
    record = created.get("experiment") if isinstance(created, dict) else None
    if not isinstance(record, dict) or not record.get("id"):
        raise RuntimeError("ORACLE_Z_EXPERIMENT_CREATE_RESPONSE_INVALID")
    return record


def transition(experiment_id: str, state: str) -> dict:
    return request("/v1/experiments/transition", {
        "experiment_id": experiment_id,
        "to_state": state,
        "actor": ACTOR,
    })


def close_session(state: dict, terminal: dict) -> None:
    if state.get("handoff") or not state.get("ticket"):
        return
    scope = state["ticket"]["scope"]
    payload = {
        "session_id": scope["session_id"],
        "attempt_id": state["attempt_id"],
        "scope_sha256": state["ticket"]["scope_sha256"],
        "created_by": ACTOR,
        "next_action": "Review immutable Oracle-Z V2 report; any retry must create a new Platform Attempt under the same preregistered Experiment",
        "summary": json.dumps({
            "attempt_state": terminal.get("state"),
            "report_sha256": state.get("report_sha256"),
            "scientific_verdict_inferred": False,
        }, sort_keys=True),
    }
    state["handoff"] = request("/v1/agents/exit", payload)


def run(root: Path, timeout: float) -> dict:
    deployment = json.loads((root / "deployment.json").read_text())
    os.environ["REALSAS_AUTHORITY_ROOT"] = deployment["authority_root"]
    code_sha = canonical_checkout(deployment)

    work = root / "research-results" / SLUG
    checkpoint = work / "api-checkpoint.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}

    if state.get("code_sha") and state["code_sha"] != code_sha:
        raise RuntimeError("ORACLE_Z_CHECKPOINT_CODE_CHANGED__DO_NOT_REUSE_ATTEMPT")
    if state.get("handoff"):
        terminal = state.get("terminal") or {}
        print(json.dumps({"already_finalized_attempt": state.get("attempt_id"), "state": terminal.get("state")}, sort_keys=True))
        return state

    exp_request = json.loads(EXPERIMENT_REQUEST.read_text())
    if not state.get("experiment_id"):
        created = request("/v1/experiments", exp_request)
        exp = experiment_record(created)
        state.update(experiment_id=exp["id"], experiment_spec_sha256=exp["spec_sha256"], code_sha=code_sha)
        atomic_json(checkpoint, state)
    experiment_id = state["experiment_id"]

    if not state.get("subject_id"):
        subject = request("/v1/subjects", {
            "slug": SLUG + "-apparatus",
            "display_name": "Subject-independent Oracle-Z V2 apparatus identity",
            "created_by": ACTOR,
        })
        state["subject_id"] = subject["subject_id"]
        atomic_json(checkpoint, state)

    identity = {
        "schema": "RealSaS.OracleZExecutionSpec.v2",
        "experiment_id": experiment_id,
        "experiment_spec_sha256": state["experiment_spec_sha256"],
        "canonical_code_sha": code_sha,
        "preregistration_sha256": sha256(PREREG),
        "probe_sha256": sha256(PROBE),
        "product_authority_changed": False,
        "truth139_training_authorized": False,
        "fit8_training_authorized": False,
    }
    if not state.get("subject_input_id"):
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        artifact = put_import(raw, artifact_type="RealSaS.OracleZExecutionSpec", schema_version="v2",
                              source_uri="urn:realsas:oracle-z-v2:execution-spec")
        sealed = request("/v1/subject-inputs", {
            "subject_id": state["subject_id"],
            "bindings": [{"role": "manifest:oracle_z_spec", "artifact_id": artifact["artifact_id"]}],
            "created_by": ACTOR,
        })
        state["subject_input_id"] = sealed["subject_input_id"]
        atomic_json(checkpoint, state)

    plan = oracle_plan()
    if not state.get("release_id"):
        release_request = snapshot(plan, identity, name="ORACLE_Z_REPRESENTATION_V2", purpose="RESEARCH",
                                   created_by=ACTOR, stage_parameters=stage_parameters())
        state["release_id"] = request("/v1/releases", release_request)["ReleaseID"]
        atomic_json(checkpoint, state)

    if not state.get("attempt_id"):
        attempt = request("/v1/research/attempts", {
            "subject_id": state["subject_id"],
            "baseline_engine_release_id": state["release_id"],
            "candidate_engine_release_id": state["release_id"],
            "created_by": ACTOR,
        })
        state["attempt_id"] = attempt["AttemptID"]
        state["run_id"] = "ORACLE_Z_V2_" + uuid.uuid4().hex.upper()
        request("/v1/experiments/link", {
            "experiment_id": experiment_id,
            "target_kind": "ATTEMPT",
            "target_id": state["attempt_id"],
            "actor": ACTOR,
        })
        exp = request("/v1/experiments/" + experiment_id)
        if exp["lifecycle_state"] == "PREREGISTERED":
            transition(experiment_id, "RUNNING")
        elif exp["lifecycle_state"] != "RUNNING":
            raise RuntimeError("ORACLE_Z_EXPERIMENT_NOT_RUNNABLE:" + exp["lifecycle_state"])
        atomic_json(checkpoint, state)

    context = request("/v1/agents/context/" + state["subject_id"])
    if not state.get("ticket"):
        if context["active_session"] is not None:
            raise RuntimeError("ORACLE_Z_EXISTING_SESSION__RESUME_EXACT_SCOPE")
        state["ticket"] = request("/v1/agents/enter", {
            "attempt_id": state["attempt_id"],
            "subject_input_id": state["subject_input_id"],
            "target_stage_id": STAGE_ID,
            "canonical_code_sha": code_sha,
            "expected_handoff_sha256": (context.get("last_handoff") or {}).get("handoff_sha256", ""),
            "created_by": ACTOR,
        })
        atomic_json(checkpoint, state)

    manifest_path = mainline.run_manifest_path(state["run_id"])
    manifest = {"run_id": state["run_id"], "subject_id": state["subject_id"],
                "execution_class": "WITNESS", "oracle_z_spec": identity}
    if manifest_path.exists() and mainline.load_json(manifest_path) != manifest:
        raise RuntimeError("ORACLE_Z_RUN_MANIFEST_DRIFT")
    mainline.atomic_json(manifest_path, manifest)

    if not state.get("command"):
        scope = state["ticket"]["scope"]
        state["command"] = request("/v1/research/compile", {
            "research_attempt_id": state["attempt_id"],
            "subject_id": state["subject_id"],
            "agent_session_id": scope["session_id"],
            "engine_release_id": state["release_id"],
            "subject_input_id": state["subject_input_id"],
            "target_stage_id": STAGE_ID,
            "compiler_run_id": state["run_id"],
            "run_manifest_path": str(manifest_path),
            "run_ledger_path": str(mainline.run_ledger_path(state["run_id"])),
            "pipeline_plan_sha256": mainline.content_sha256(plan),
            "idempotency_key": SLUG + ":" + state["attempt_id"],
            "requested_by": ACTOR,
        })
        atomic_json(checkpoint, state)

    report_path = work / state["attempt_id"] / "oracle_z_report.json"
    if not state.get("report_sha256"):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "REALSAS_EXPERIMENT_ID": experiment_id,
            "REALSAS_ATTEMPT_ID": state["attempt_id"],
            "REALSAS_ENGINE_RELEASE_ID": state["release_id"],
            "REALSAS_CANONICAL_CODE_SHA": code_sha,
        })
        subprocess.run([sys.executable, str(PROBE), "--mode", "scientific", "--out", str(report_path)],
                       cwd=ROOT, env=env, check=True, timeout=timeout)
        report = json.loads(report_path.read_text())
        expected_context = {key: env[key] for key in (
            "REALSAS_EXPERIMENT_ID", "REALSAS_ATTEMPT_ID", "REALSAS_ENGINE_RELEASE_ID", "REALSAS_CANONICAL_CODE_SHA")}
        if report.get("schema") != "RealSaS.OracleZRepresentationCourt.v2" or report.get("mode") != "PLATFORM_SCIENTIFIC":
            raise RuntimeError("ORACLE_Z_LOCAL_REPORT_CONTRACT_INVALID")
        if report.get("platform_context") != expected_context:
            raise RuntimeError("ORACLE_Z_LOCAL_REPORT_PLATFORM_CONTEXT_DRIFT")
        raw = report_path.read_bytes()
        state["report_sha256"] = hashlib.sha256(raw).hexdigest()
        artifact = put_import(raw, artifact_type="RealSaS.OracleZRepresentationCourt", schema_version="v2",
                              source_uri="urn:realsas:oracle-z-v2:attempt:" + state["attempt_id"])
        state["report_artifact_id"] = artifact["artifact_id"]
        atomic_json(checkpoint, state)

    if not state.get("external_binding"):
        state["external_binding"] = request(
            f"/v1/research/attempts/{state['attempt_id']}/stages/{STAGE_ID}/external-evidence",
            {"bindings": [{"role": "oracle_z_report", "artifact_id": state["report_artifact_id"]}], "bound_by": ACTOR},
        )
        atomic_json(checkpoint, state)

    terminal = wait_attempt_terminal(state["attempt_id"], timeout)
    state["terminal"] = terminal
    if terminal["state"] == "COMPLETED":
        exp = request("/v1/experiments/" + experiment_id)
        if exp["lifecycle_state"] == "RUNNING":
            state["experiment_terminal"] = transition(experiment_id, "COMPLETED")
    close_session(state, terminal)
    atomic_json(checkpoint, state)

    product_count = registry_read(root, "SELECT count(*) FROM product_revisions WHERE subject_id='" +
                                  str(uuid.UUID(state["subject_id"])) + "'")
    if product_count != "0":
        raise RuntimeError("ORACLE_Z_RESEARCH_MINTED_PRODUCT_REVISION")
    if terminal["state"] != "COMPLETED":
        raise RuntimeError("ORACLE_Z_ATTEMPT_NOT_COMPLETED__CHECKPOINT_PRESERVED")

    report = json.loads(report_path.read_text())
    result = {
        "experiment_id": experiment_id,
        "attempt_id": state["attempt_id"],
        "engine_release_id": state["release_id"],
        "canonical_code_sha": code_sha,
        "report_sha256": state["report_sha256"],
        "report_status": report.get("status"),
        "scientific_verdict_inferred": False,
        "product_revision_count": 0,
    }
    print(json.dumps(result, sort_keys=True))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    run(args.root, args.timeout)
