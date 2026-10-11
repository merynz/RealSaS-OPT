#!/usr/bin/env python3
"""One authorized authored-2D oracle via Go session/Attempt, never direct stages.

This operator's JSON checkpoint only retains API-returned IDs for recovery.
Attempt/events/Registry remain the sole durable execution authority. A timeout
does not create another Attempt. Source must be the exact deployed live main.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_artifact_store import hydrate_inventory, cas_path
from tools.platform_deployment_smoke import request, wait_attempt_terminal, registry_read
from tools.platform_release_snapshot import snapshot

ACTOR = "alfred-authored-visual-oracle-v1"


def finalize_terminal(root, export, checkpoint, state, terminal):
    """Export partial evidence and close only the original Go-owned scope.

    May run after main advances. Never constructs inputs, enters a session or
    submits execution. Go revalidates scope and refuses pending commands.
    """
    if terminal["attempt_id"] != state["attempt_id"] or terminal["state"] not in {
            "COMPLETED", "FAILED", "PASS", "FAIL", "ABSTAIN", "CANCELED", "CANCELLED", "ABORTED"}:
        raise RuntimeError("ORACLE_FINALIZE_REQUIRES_ORIGINAL_TERMINAL_ATTEMPT")
    ticket = state["ticket"]
    scope = ticket["scope"]
    expected = {"attempt_id": state["attempt_id"], "subject_id": state["subject_id"],
                "subject_input_id": state["subject_input_id"], "engine_release_id": state["release_id"],
                "canonical_code_sha": state["code_sha"], "target_stage_id": "49_VISUAL_ORACLE_MEASURE"}
    if any(scope.get(k) != v for k, v in expected.items()):
        raise RuntimeError("ORACLE_CHECKPOINT_SCOPE_DRIFT")
    context = request("/v1/agents/context/" + state["subject_id"])
    prior = context.get("last_handoff") or {}
    retry = state.get("exit_request") or {}
    exact_saved_retry = all(retry.get(k) == v for k, v in {
        "session_id": scope["session_id"], "attempt_id": state["attempt_id"],
        "scope_sha256": ticket["scope_sha256"], "created_by": ACTOR}.items())
    if context.get("active_session") != scope and not (
            prior.get("scope") == scope and prior.get("scope_sha256") == ticket["scope_sha256"]) and not exact_saved_retry:
        raise RuntimeError("ORACLE_GO_CONTEXT_SCOPE_DRIFT")
    state["terminal_state"] = terminal
    mainline.atomic_json(export / "terminal-attempt.json", terminal)
    errors, verified = [], []
    ledger_path = mainline.run_ledger_path(state["run_id"])
    try:
        ledger = mainline.load_json(ledger_path) if ledger_path.is_file() else None
    except (OSError, ValueError) as error:
        ledger = None
        errors.append({"error": "ENGINE_LEDGER_UNREADABLE", "detail": str(error)})
    if ledger is None:
        errors.append({"error": "ENGINE_LEDGER_MISSING"})
    else:
        mainline.atomic_json(export / "engine-ledger.json", ledger)
        for stage in ledger["stages"]:
            for output in stage.get("outputs", []):
                try:
                    path = Path(output["path"])
                    if mainline.sha256_file(path) != output["sha256"]:
                        raise RuntimeError("ORACLE_EXPORT_OUTPUT_DRIFT")
                    # Stage directories prevent partial outputs with equal
                    # basenames from silently overwriting each other.
                    destination = export / "outputs" / stage["id"] / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    verified.append({"stage_id": stage["id"], "path": str(destination), "sha256": output["sha256"]})
                except (OSError, RuntimeError) as error:
                    errors.append({"stage_id": stage["id"], "path": output["path"], "error": str(error)})
    measurement = next((row for row in verified if row["stage_id"] == "49_VISUAL_ORACLE_MEASURE"
                        and Path(row["path"]).name == "measurement.json"), None)
    if measurement:
        shutil.copyfile(measurement["path"], export / "measurement.json")
    elif terminal["state"] == "COMPLETED":
        errors.append({"error": "COMPLETED_ATTEMPT_MEASUREMENT_MISSING"})
    manifest_path = mainline.run_manifest_path(state["run_id"])
    if manifest_path.is_file():
        mainline.atomic_json(export / "run-manifest.json", mainline.load_json(manifest_path))
    mainline.atomic_json(export / "hydration.json", state.get("hydration", {}))
    receipt = {"attempt_state": terminal["state"], "verified_outputs": verified, "export_errors": errors,
               "scientific_pass_inferred": False}
    mainline.atomic_json(export / "export-receipt.json", receipt)
    state["export_receipt"] = receipt
    # Persist the exact retry body before sending. A lost HTTP reply cannot
    # change an exit summary on retry and cause idempotency drift.
    if not state.get("exit_request"):
        state["exit_request"] = {"session_id": scope["session_id"], "attempt_id": state["attempt_id"],
            "scope_sha256": ticket["scope_sha256"], "created_by": ACTOR,
            "next_action": "Review this exact oracle's terminal diagnostics and measurements; any new execution needs a separately declared scope",
            "summary": json.dumps({"attempt_state": terminal["state"], "verified_output_count": len(verified),
                                   "export_error_count": len(errors), "scientific_pass_inferred": False}, sort_keys=True)}
    mainline.atomic_json(checkpoint, state)
    if not state.get("handoff"):
        state["handoff"] = request("/v1/agents/exit", state["exit_request"])
        mainline.atomic_json(checkpoint, state)
    product_count = registry_read(root, "SELECT count(*) FROM product_revisions WHERE subject_id='"+str(uuid.UUID(state["subject_id"]))+"'")
    if product_count != "0":
        raise RuntimeError("ORACLE_RESEARCH_MINTED_PRODUCT_REVISION")
    if terminal["state"] != "COMPLETED" or errors:
        raise RuntimeError("ORACLE_TERMINAL_FAILURE_OR_EXPORT_GAP__DIAGNOSTICS_AND_HANDOFF_PRESERVED")
    return receipt


def oracle_plan():
    plan = mainline.load_json(mainline.PLAN_PATH)
    rows = [
        ("47_AUTHORED_VISUAL_ORACLE", [], "reference", "produce_reference", ["oracle_source"]),
        ("48_NATIVE_VISUAL_ORACLE", ["47_AUTHORED_VISUAL_ORACLE"], "native", "render_reference", ["oracle_render"]),
        ("49_VISUAL_ORACLE_MEASURE", ["47_AUTHORED_VISUAL_ORACLE", "48_NATIVE_VISUAL_ORACLE"], "measure", "measure_reference", ["oracle_source", "oracle_measure"])]
    plan["stages"] = [{"ordinal": i, "id": sid, "title": sid, "group": "research_visual_oracle",
        "depends_on": deps, "adapter": f"compiler.realsas_compiler_services.orchestrator.adapters.visual_oracle_{module}_v1:{function}",
        "manifest_keys": keys, "policy": {"fail_closed": True, "cacheable": True, "output_hash_required": True, "product_pass_authority": False}}
        for i, (sid, deps, module, function, keys) in enumerate(rows, 1)]
    plan["stage_count"] = len(rows)
    return plan


def canonical_checkout(deployment):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True).split()[0]
    if head != remote or head != deployment["code_sha"]:
        raise RuntimeError("ORACLE_REQUIRES_EXACT_DEPLOYED_LIVE_MAIN")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("ORACLE_SOURCE_OVERLAY_FORBIDDEN")


def source_inputs(spec, root, *, rclone_remote):
    hydration = hydrate_inventory(spec["inventory"], cas_root=root / "external-cas", rclone_remote=rclone_remote)
    if not hydration["all_input_bytes_verified"]:
        raise RuntimeError("ORACLE_SOURCE_HYDRATION_FAILED")
    archive = root / "external-cas" / "assembled" / (spec["source"]["archive_sha256"] + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with temporary.open("wb") as out:
            for part in spec["inventory"]["required_files"]:
                with cas_path(root / "external-cas", part["sha256"]).open("rb") as source:
                    shutil.copyfileobj(source, out)
        temporary.replace(archive)
    if archive.stat().st_size != spec["archive_size_bytes"] or mainline.sha256_file(archive) != spec["source"]["archive_sha256"]:
        raise RuntimeError("ORACLE_ASSEMBLED_ARCHIVE_DRIFT")
    player = root / "oracle-binaries" / spec["experiment_id"] / "realsas_runtime_v2_caa_reference"
    player.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
        str(ROOT / "runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp"), "-lpng", "-lz", "-o", str(player)], check=True, timeout=120)
    return {"oracle_source": {**spec["source"], "archive_path": str(archive)},
            "oracle_render": {"player_path": str(player), "player_sha256": mainline.sha256_file(player),
                "native_source_sha256": mainline.sha256_file(ROOT / "runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp"),
                "variants": spec["variants"]}, "oracle_measure": spec["measurement"]}, hydration


def run(root, timeout, *, rclone_remote=None, finalize_existing=False):
    spec = mainline.load_json(ROOT / "canonical/AUTHORED_VISUAL_ORACLE_V1_20261010.json")
    deployment = mainline.load_json(root / "deployment.json")
    os.environ["REALSAS_AUTHORITY_ROOT"] = deployment["authority_root"]
    export = root / "oracle-results" / spec["experiment_id"]
    export.mkdir(parents=True, exist_ok=True)
    checkpoint = export / "api-checkpoint.json"
    state = mainline.load_json(checkpoint) if checkpoint.exists() else {}
    def save():
        mainline.atomic_json(checkpoint, state)
    if finalize_existing:
        if not state.get("ticket"):
            raise RuntimeError("ORACLE_FINALIZE_REQUIRES_EXISTING_CHECKPOINT_AND_TICKET")
        terminal = request("/v1/attempts/" + state["attempt_id"])
        finalize_terminal(root, export, checkpoint, state, terminal)
        return
    canonical_checkout(deployment)
    if state.get("handoff"):
        print(json.dumps({"already_finalized_attempt": state["attempt_id"], "export": str(export)}), flush=True)
        if state.get("terminal_state", {}).get("state") != "COMPLETED" or state.get("export_receipt", {}).get("export_errors"):
            raise RuntimeError("ORACLE_TERMINAL_FAILURE_OR_EXPORT_GAP__DIAGNOSTICS_AND_HANDOFF_PRESERVED")
        return
    if not state:
        if not rclone_remote:
            raise RuntimeError("ORACLE_NEW_EXECUTION_REQUIRES_RCLONE_REMOTE")
        identity, hydration = source_inputs(spec, root, rclone_remote=rclone_remote)
        subject = request("/v1/subjects", {"slug": spec["experiment_id"].lower(),
            "display_name": "Authored 2D appearance oracle; independent of mechanical Knight", "created_by": ACTOR})["subject_id"]
        state.update(subject_id=subject, identity=identity, hydration=hydration, code_sha=deployment["code_sha"])
        save()
    if state["code_sha"] != deployment["code_sha"]:
        raise RuntimeError("ORACLE_RECOVERY_CODE_CHANGED__INSPECT_GO_CONTEXT_AND_DECLARE_CHILD_INTERVENTION")
    plan, identity = oracle_plan(), state["identity"]
    if not state.get("subject_input_id"):
        bindings = []
        for key, value in identity.items():
            data = json.dumps(value, sort_keys=True).encode()
            digest = hashlib.sha256(data).hexdigest()
            obj = request(f"/v1/artifacts/bytes?sha256={digest}", method="PUT", raw=data)
            artifact = request("/v1/artifacts/import", {"artifact_type": "RealSaS.VisualOracleInputSection",
                "schema_version": "v1", "object": obj, "source_uri": "urn:realsas:authored-visual-oracle:"+key, "created_by": ACTOR})
            bindings.append({"role": "manifest:"+key, "artifact_id": artifact["artifact_id"]})
        state["subject_input_id"] = request("/v1/subject-inputs", {"subject_id": state["subject_id"],
            "bindings": bindings, "created_by": ACTOR})["subject_input_id"]
        save()
    if not state.get("release_id"):
        state["release_id"] = request("/v1/releases", snapshot(plan, identity, name=spec["experiment_id"], purpose="RESEARCH", created_by=ACTOR))["ReleaseID"]
        save()
    if not state.get("attempt_id"):
        state["attempt_id"] = request("/v1/research/attempts", {"subject_id": state["subject_id"],
            "baseline_engine_release_id": state["release_id"], "candidate_engine_release_id": state["release_id"], "created_by": ACTOR})["AttemptID"]
        state["run_id"] = "AUTHORED_VISUAL_ORACLE_" + uuid.uuid4().hex.upper()
        save()
    context = request("/v1/agents/context/" + state["subject_id"])
    state["entry_context"] = context; save()
    if not state.get("ticket"):
        if context["active_session"] is not None:
            raise RuntimeError("ORACLE_EXISTING_SESSION__RESUME_EXACT_SCOPE")
        state["ticket"] = request("/v1/agents/enter", {"attempt_id": state["attempt_id"],
            "subject_input_id": state["subject_input_id"], "target_stage_id": plan["stages"][-1]["id"],
            "canonical_code_sha": deployment["code_sha"], "expected_handoff_sha256":
                (context["last_handoff"] or {}).get("handoff_sha256", ""), "created_by": ACTOR})
        save()
    manifest_path = mainline.run_manifest_path(state["run_id"])
    manifest = {"run_id": state["run_id"], "subject_id": "AUTHORED_2D_REFERENCE_ORACLE",
                "execution_class": "WITNESS", **identity}
    if manifest_path.exists() and mainline.load_json(manifest_path) != manifest:
        raise RuntimeError("ORACLE_EXISTING_MANIFEST_DRIFT")
    mainline.atomic_json(manifest_path, manifest)
    ticket = state["ticket"]
    # Identical idempotency key/request recovers existing command; never resubmit
    # a timed-out run as a fresh command or new Attempt.
    state["command"] = request("/v1/research/compile", {"research_attempt_id": state["attempt_id"],
        "subject_id": state["subject_id"], "agent_session_id": ticket["scope"]["session_id"],
        "engine_release_id": state["release_id"], "subject_input_id": state["subject_input_id"],
        "target_stage_id": plan["stages"][-1]["id"], "compiler_run_id": state["run_id"],
        "run_manifest_path": str(manifest_path), "run_ledger_path": str(mainline.run_ledger_path(state["run_id"])),
        "pipeline_plan_sha256": mainline.content_sha256(plan), "idempotency_key": spec["experiment_id"]+":"+state["run_id"], "requested_by": ACTOR})
    save()
    print(json.dumps({"attempt_id": state["attempt_id"], "run_id": state["run_id"], "code_sha": state["code_sha"]}), flush=True)
    terminal = wait_attempt_terminal(state["attempt_id"], timeout)
    print(json.dumps(finalize_terminal(root, export, checkpoint, state, terminal)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--rclone-remote", help="Existing authenticated host Drive remote; required for new execution")
    parser.add_argument("--finalize-existing", action="store_true", help="Only export and exit the original terminal scope, including after main advances; no execution")
    args = parser.parse_args()
    run(args.root, args.timeout, rclone_remote=args.rclone_remote, finalize_existing=args.finalize_existing)
