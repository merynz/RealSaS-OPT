#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Corpus370 truth campaign observer and CAS exporter.

All original Attempts are preserved. PASS, FAIL, BLOCKED, ABSTAIN and cancellation
are scientific outcomes. The observer never retries or drops a failed subject. It
exports exact Registry-bound Stage47/48 outputs when present and stops only when all
370 Attempts are terminal or the bounded observation window ends.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import time

from tools.platform_cohort_bootstrap_v1 import api
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256

PREPARATION_SCHEMA = "RealSaS.Corpus370TruthCampaignPreparation.v1"
PROJECTION_SCHEMA = "RealSaS.Corpus370TruthCampaignPreparation.v2"
STAGES = ("47_SOURCE_RESPONSE_CALIBRATION", "48_PRESENTATION_RESPONSE_TRUTH", "49_SUBJECT_TRUTH_FAMILIES")
TERMINAL = {"FAILED", "FAIL", "ABSTAIN", "CANCELED", "CANCELLED", "ABORTED", "COMPLETED", "PASS", "BLOCKED"}
SUCCESS = {"COMPLETED", "PASS"}


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verified_cas(root: Path, ref: dict) -> Path:
    key = PurePosixPath(ref["storage_key"])
    require(not key.is_absolute() and ".." not in key.parts, "CORPUS370_OBSERVE_CAS_PATH_ESCAPE")
    path = (root / key).resolve()
    require(path.is_relative_to(root.resolve()), "CORPUS370_OBSERVE_CAS_PATH_ESCAPE")
    require(path.is_file() and path.stat().st_size == ref["size_bytes"] and sha256(path) == ref["content_sha256"], "CORPUS370_OBSERVE_CAS_BYTES_DRIFT")
    return path


def export_stage(member: dict, stage: str, *, api_root: str, artifact_root: Path, output: Path):
    records = api(api_root, "GET", f"/v1/subjects/{member['subject_id']}/artifacts?artifact_type=RealSaS.StageResultManifest&schema_version=v1&limit=500")
    rows = [
        row for row in records
        if row["source_kind"] == "ATTEMPT"
        and row["source_id"] == member["attempt_id"]
        and row["role"] == "stage:" + stage
    ]
    require(len(rows) <= 1, "CORPUS370_OBSERVE_DUPLICATE_STAGE_BINDING:" + stage)
    if not rows:
        return None
    binding = rows[0]
    record = api(api_root, "GET", "/v1/artifacts/" + binding["artifact_id"])
    require(
        record["id"] == binding["artifact_id"]
        and record["content_sha256"] == binding["content_sha256"]
        and record["producer_contract"] == stage,
        "CORPUS370_OBSERVE_REGISTRY_BINDING_DRIFT:" + stage,
    )
    manifest_path = verified_cas(artifact_root, record)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == "RealSaS.StageResultManifest.v1" and manifest.get("stage_id") == stage, "CORPUS370_OBSERVE_STAGE_MANIFEST:" + stage)
    target = output / member["attempt_id"] / stage
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "REGISTRY_BINDING.json", {"binding": binding, "artifact": record})
    shutil.copyfile(manifest_path, target / "STAGE_RESULT.json")
    outputs = []
    for ref in manifest.get("outputs") or ():
        relative = PurePosixPath(ref["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == ref["relative_path"], "CORPUS370_OBSERVE_OUTPUT_PATH_ESCAPE")
        source = verified_cas(artifact_root, ref)
        destination = target / "outputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, temporary)
        require(sha256(temporary) == ref["content_sha256"], "CORPUS370_OBSERVE_COPY_DRIFT")
        temporary.replace(destination)
        outputs.append({
            "path": str(destination),
            "sha256": ref["content_sha256"],
            "size_bytes": ref["size_bytes"],
            "schema": ref["payload_schema"],
            "authority_class": ref["authority_class"],
        })
    return {
        "stage_id": stage,
        "compiler_status": manifest.get("compiler_status"),
        "registry_stage_artifact_id": binding["artifact_id"],
        "outputs": outputs,
    }


def export_diagnostics(preparation: dict, output: Path) -> list[dict]:
    bootstrap_ref = preparation["bootstrap_receipt"]
    bootstrap_path = Path(bootstrap_ref["path"]).resolve()
    require(bootstrap_path.is_file() and sha256(bootstrap_path) == bootstrap_ref["sha256"], "CORPUS370_OBSERVE_BOOTSTRAP_DRIFT")
    bindings = json.loads(bootstrap_path.read_text())["run_bindings"]
    result = []
    for binding in bindings:
        target = output / "diagnostics" / str(binding["subject_id"])
        target.mkdir(parents=True, exist_ok=True)
        row = {"subject_id": binding["subject_id"], "files": [], "missing": []}
        ledger = Path(binding["run_ledger_path"]).resolve()
        manifest = Path(binding["run_manifest_path"]).resolve()
        candidates = [ledger, manifest]
        for stage in STAGES:
            stage_root = ledger.parent / "artifacts" / stage
            candidates.append(stage_root / "failure_diagnostics.json")
            if stage_root.is_dir():
                candidates.extend(sorted(p for p in stage_root.iterdir() if p.name.endswith((".blender.log", ".job.json")))[:64])
        for source in candidates:
            if not source.is_file():
                row["missing"].append(str(source))
                continue
            # Large NPZ/blend payloads belong in Registry/CAS export, not diagnostics.
            if source.stat().st_size > 16 * 1024 * 1024:
                continue
            destination = target / source.name
            if destination.exists() and sha256(destination) != sha256(source):
                destination = target / (source.parent.name + "__" + source.name)
            shutil.copyfile(source, destination)
            row["files"].append({"path": str(destination), "sha256": sha256(destination), "size_bytes": destination.stat().st_size})
        result.append(row)
    return result


def observe(
    preparation: dict,
    *,
    api_root: str,
    artifact_root: Path,
    output: Path,
    maximum_seconds: float,
    interval_seconds: float,
) -> dict:
    require(preparation.get("schema") in {PREPARATION_SCHEMA, PROJECTION_SCHEMA}, "CORPUS370_OBSERVE_PREPARATION_SCHEMA")
    member_count = int(preparation.get("member_count", -1))
    if preparation["schema"] == PROJECTION_SCHEMA:
        require(preparation.get("candidate_subject_count") == 370
                and 0 < member_count <= 370
                and member_count + int(preparation.get("excluded_subject_count", -1)) == 370,
                "CORPUS370_OBSERVE_MEMBER_ACCOUNTING")
    else:
        require(member_count == 370, "CORPUS370_OBSERVE_MEMBER_COUNT")
    require(0 < interval_seconds <= 60 and maximum_seconds >= 0, "CORPUS370_OBSERVE_POLL_BOUNDS")
    output.mkdir(parents=True, exist_ok=True)
    initial_campaign = api(api_root, "GET", "/v1/research/campaigns/" + preparation["campaign_id"])
    initial = {row["attempt_id"]: row for row in initial_campaign["members"]}
    require(len(initial) == member_count, "CORPUS370_OBSERVE_CAMPAIGN_MEMBER_COUNT")
    started = time.monotonic()
    exported: dict[str, dict] = {}
    while True:
        campaign = api(api_root, "GET", "/v1/research/campaigns/" + preparation["campaign_id"])
        require({row["attempt_id"] for row in campaign["members"]} == set(initial), "CORPUS370_OBSERVE_CAMPAIGN_MEMBER_DRIFT")
        attempts = []
        all_terminal = True
        for member in campaign["members"]:
            saved = initial[member["attempt_id"]]
            require(all(member[key] == saved[key] for key in ("subject_id", "role")), "CORPUS370_OBSERVE_MEMBER_SCOPE_DRIFT")
            attempt = api(api_root, "GET", "/v1/attempts/" + member["attempt_id"])
            state = str(attempt["state"])
            all_terminal &= state in TERMINAL
            write_json(output / member["attempt_id"] / "ATTEMPT.json", attempt)
            attempts.append({
                "attempt_id": member["attempt_id"],
                "subject_id": member["subject_id"],
                "role": member["role"],
                "state": state,
                "last_event": attempt["events"][-1] if attempt.get("events") else None,
            })
            row = exported.setdefault(member["attempt_id"], {"attempt_id": member["attempt_id"], "subject_id": member["subject_id"], "stages": {}})
            for stage in STAGES:
                if stage not in row["stages"]:
                    value = export_stage(member, stage, api_root=api_root, artifact_root=artifact_root, output=output)
                    if value is not None:
                        row["stages"][stage] = value
        elapsed = time.monotonic() - started
        timed_out = not all_terminal and elapsed >= maximum_seconds
        if all_terminal or timed_out:
            diagnostics = export_diagnostics(preparation, output)
            counts = {}
            for row in attempts:
                counts[row["state"]] = counts.get(row["state"], 0) + 1
            result = {
                "schema": "RealSaS.Corpus370TruthObservation.v1",
                "campaign_id": preparation["campaign_id"],
                "experiment_id": preparation["experiment_id"],
                "source_code_sha": preparation["source_code_sha"],
                "campaign": campaign,
                "attempts": attempts,
                "attempt_state_counts": dict(sorted(counts.items())),
                "exports": list(exported.values()),
                "bound_run_diagnostics": diagnostics,
                "all_370_attempts_terminal": all_terminal and member_count == 370,
                "all_projected_attempts_terminal": all_terminal,
                "member_count": member_count,
                "candidate_subject_count": 370,
                "excluded_subject_count": preparation.get("excluded_subject_count", 0),
                "observation_timed_out": timed_out,
                "failed_attempt_count": sum(count for state, count in counts.items() if state not in SUCCESS),
                "execution_resubmitted": False,
                "training_submitted": False,
                "truth_qualification_inferred": False,
                "final_family_matrix_minted": False,
            }
            write_json(output / "CORPUS370_TRUTH_OBSERVATION_V1.json", result)
            print(json.dumps({"all_terminal": all_terminal, "state_counts": result["attempt_state_counts"], "failed_attempt_count": result["failed_attempt_count"], "timed_out": timed_out}, sort_keys=True), flush=True)
            return result
        print(json.dumps({"event": "CORPUS370_TRUTH_OBSERVE", "terminal": sum(row["state"] in TERMINAL for row in attempts), "total": member_count, "elapsed_seconds": elapsed}, sort_keys=True), flush=True)
        time.sleep(min(interval_seconds, max(0.0, maximum_seconds - elapsed)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--maximum-seconds", type=float, default=21600)
    parser.add_argument("--interval-seconds", type=float, default=15)
    args = parser.parse_args()
    result = observe(
        json.loads(args.preparation.read_text()),
        api_root=args.api_root,
        artifact_root=args.artifact_root,
        output=args.output,
        maximum_seconds=args.maximum_seconds,
        interval_seconds=args.interval_seconds,
    )
    if result["observation_timed_out"]:
        raise SystemExit("CORPUS370_TRUTH_OBSERVATION_WINDOW_ENDED__ORIGINAL_CAMPAIGN_PRESERVED")


if __name__ == "__main__":
    main()
