#!/usr/bin/env python3
"""Read-only export of exact Stage48 Platform truth bytes.

The exporter never retries, resubmits, qualifies or promotes.  It selects the
StageResultManifest bound to each exact Attempt, verifies Registry/CAS identity,
and copies Stage48 outputs into a deterministic local directory for the cohort
qualifier and later LOFO input seal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import time

from tools.platform_cohort_bootstrap_v1 import api
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256

STAGE = "48_PRESENTATION_RESPONSE_TRUTH"
TERMINAL = {"FAILED", "FAIL", "ABSTAIN", "CANCELED", "CANCELLED", "ABORTED", "COMPLETED", "PASS"}


def require(value, code):
    if not value:
        raise ValueError(code)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _cas(root, ref):
    key = PurePosixPath(ref["storage_key"])
    require(not key.is_absolute() and ".." not in key.parts, "PRESENTATION_OBSERVE_CAS_PATH_ESCAPE")
    path = (Path(root) / key).resolve()
    require(path.is_relative_to(Path(root).resolve()), "PRESENTATION_OBSERVE_CAS_PATH_ESCAPE")
    require(
        path.is_file()
        and path.stat().st_size == ref["size_bytes"]
        and sha256(path) == ref["content_sha256"],
        "PRESENTATION_OBSERVE_CAS_BYTES_DRIFT",
    )
    return path


def export_member(member, *, api_root, artifact_root, output):
    records = api(
        api_root,
        "GET",
        f"/v1/subjects/{member['subject_id']}/artifacts?artifact_type=RealSaS.StageResultManifest&schema_version=v1&limit=500",
    )
    rows = [
        row
        for row in records
        if row["source_kind"] == "ATTEMPT"
        and row["source_id"] == member["attempt_id"]
        and row["role"] == "stage:" + STAGE
    ]
    require(len(rows) <= 1, "PRESENTATION_OBSERVE_DUPLICATE_STAGE_BINDING")
    if not rows:
        return None
    binding = rows[0]
    record = api(api_root, "GET", "/v1/artifacts/" + binding["artifact_id"])
    require(
        record["id"] == binding["artifact_id"]
        and record["content_sha256"] == binding["content_sha256"]
        and record["producer_contract"] == STAGE,
        "PRESENTATION_OBSERVE_REGISTRY_BINDING_DRIFT",
    )
    manifest_path = _cas(artifact_root, record)
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema") == "RealSaS.StageResultManifest.v1"
        and manifest.get("stage_id") == STAGE
        and manifest.get("compiler_status") == "PASS",
        "PRESENTATION_OBSERVE_STAGE_MANIFEST_DRIFT",
    )
    target = Path(output) / member["attempt_id"]
    target.mkdir(parents=True, exist_ok=True)
    _write(target / "REGISTRY_BINDING.json", {"binding": binding, "artifact": record})
    shutil.copyfile(manifest_path, target / "STAGE_RESULT.json")
    exported = []
    for ref in manifest["outputs"]:
        relative = PurePosixPath(ref["relative_path"])
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and str(relative) == ref["relative_path"],
            "PRESENTATION_OBSERVE_OUTPUT_PATH_ESCAPE",
        )
        source = _cas(artifact_root, ref)
        destination = target / "outputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, temp)
        require(sha256(temp) == ref["content_sha256"], "PRESENTATION_OBSERVE_COPY_BYTES_DRIFT")
        temp.replace(destination)
        exported.append(
            {
                "path": str(destination.resolve()),
                "sha256": ref["content_sha256"],
                "size_bytes": ref["size_bytes"],
                "schema": ref["payload_schema"],
                "authority_class": ref["authority_class"],
            }
        )
    return {
        "attempt_id": member["attempt_id"],
        "subject_id": member["subject_id"],
        "role": member["role"],
        "registry_stage_artifact_id": binding["artifact_id"],
        "outputs": exported,
    }


def observe(preparation, *, api_root, artifact_root, output, maximum_seconds=1800, interval_seconds=30):
    require(preparation.get("schema") == "RealSaS.PresentationTruthCampaignPreparation.v1", "PRESENTATION_OBSERVE_PREPARATION_SCHEMA")
    require(preparation.get("target_stage_id") == STAGE, "PRESENTATION_OBSERVE_TARGET_STAGE")
    require(0 < interval_seconds <= 60 and maximum_seconds >= 0, "PRESENTATION_OBSERVE_POLL_BOUNDS")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initial = {row["attempt_id"]: row for row in preparation["members"]}
    exports = {}
    started = time.monotonic()
    while True:
        campaign = api(api_root, "GET", "/v1/research/campaigns/" + preparation["campaign_id"])
        require({row["attempt_id"] for row in campaign["members"]} == set(initial), "PRESENTATION_OBSERVE_CAMPAIGN_MEMBERS_DRIFT")
        states = []
        errors = []
        for member in campaign["members"]:
            saved = initial[member["attempt_id"]]
            require(all(member[key] == saved[key] for key in ("subject_id", "role")), "PRESENTATION_OBSERVE_MEMBER_SCOPE_DRIFT")
            attempt = api(api_root, "GET", "/v1/attempts/" + member["attempt_id"])
            states.append({"attempt_id": member["attempt_id"], "subject_id": member["subject_id"], "role": member["role"], "state": attempt["state"]})
            _write(output / member["attempt_id"] / "ATTEMPT.json", attempt)
            if member["attempt_id"] not in exports:
                try:
                    value = export_member(member, api_root=api_root, artifact_root=artifact_root, output=output)
                    if value is not None:
                        exports[member["attempt_id"]] = value
                except (OSError, ValueError, KeyError) as error:
                    errors.append({"attempt_id": member["attempt_id"], "error": str(error)})
        elapsed = time.monotonic() - started
        failed = any(row["state"] in TERMINAL - {"COMPLETED", "PASS"} for row in states)
        complete = len(exports) == len(initial) and campaign.get("goal_complete") is True
        stop = complete or failed or bool(errors) or elapsed >= maximum_seconds
        result = {
            "schema": "RealSaS.PresentationTruthCampaignObservation.v1",
            "campaign_id": preparation["campaign_id"],
            "experiment_id": preparation["experiment_id"],
            "source_code_sha": preparation["source_code_sha"],
            "campaign": campaign,
            "attempts": states,
            "exports": list(exports.values()),
            "errors": errors,
            "all_stage48_outputs_exported": len(exports) == len(initial),
            "goal_complete": campaign.get("goal_complete") is True,
            "timed_out": not complete and not errors and not failed and elapsed >= maximum_seconds,
            "execution_resubmitted": False,
            "training_submitted": False,
            "qualification_minted": False,
        }
        _write(output / "OBSERVATION.json", result)
        print(f"[PRESENTATION_OBSERVE] exported={len(exports)}/{len(initial)} elapsed_s={elapsed:.1f} errors={len(errors)}", flush=True)
        if stop:
            return result
        time.sleep(min(interval_seconds, max(0.0, maximum_seconds - elapsed)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "artifact-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--maximum-seconds", type=float, default=1800)
    parser.add_argument("--interval-seconds", type=float, default=30)
    args = parser.parse_args()
    result = observe(
        json.loads(Path(args.preparation).read_text()),
        api_root=args.api_root,
        artifact_root=args.artifact_root,
        output=args.output,
        maximum_seconds=args.maximum_seconds,
        interval_seconds=args.interval_seconds,
    )
    if not result["all_stage48_outputs_exported"] or not result["goal_complete"]:
        raise SystemExit("PRESENTATION_TRUTH_NOT_YET_EXPORTED__ORIGINAL_CAMPAIGN_PRESERVED")
