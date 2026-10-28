#!/usr/bin/env python3
"""Prepare/submit the finite FIT8 PresentationResponseTeacherV1 campaign.

This coordinator reuses the exact sealed observation-rest calibration request and
runtime as immutable parents, extends the released DAG only to Stage48, and adds
presentation-truth policy to the run manifest.  It does not launch Blender or
training directly; scientific execution remains Platform Attempt-owned.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from tools import platform_cohort_bootstrap_v1 as bootstrap
from tools.observation.pinned_evaluator_runtime_v1 import launch_options
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256, _bound_path
from tools.truth_foundry.platform_calibration_operator_v1 import immutable_json


def _require(value, code):
    if not value:
        raise ValueError(code)


def prepare(request, *, repo, platform_root, authority_root, expected_sha, output, api_root, rclone_remote, submit):
    _require(
        request.get("schema") == "RealSaS.PresentationTruthOperatorRequest.v1"
        and request.get("enabled") is True
        and request.get("training_allowed") is False,
        "PRESENTATION_OPERATOR_SCOPE_REQUIRED",
    )
    repo, platform, authority, output = map(
        lambda p: Path(p).resolve(), (repo, platform_root, authority_root, output)
    )
    _require(json.loads((platform / "deployment.json").read_text())["code_sha"] == expected_sha, "PRESENTATION_OPERATOR_DEPLOYED_MAIN_DRIFT")
    _require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == expected_sha, "PRESENTATION_OPERATOR_CHECKOUT_DRIFT")
    _require(subprocess.check_output(["git", "-C", str(repo), "ls-remote", "origin", "refs/heads/main"], text=True).split()[0] == expected_sha, "PRESENTATION_OPERATOR_REMOTE_MAIN_DRIFT")

    base_path = _bound_path(repo, request["base_calibration_request_repo_path"])
    base = json.loads(base_path.read_text())
    _require(base.get("schema") == "RealSaS.TeacherCalibrationOperatorRequest.v2", "PRESENTATION_OPERATOR_BASE_REQUEST_SCHEMA")
    _require(base.get("training_allowed") is False and base.get("native_baseline_authority_for_this_rest") is False, "PRESENTATION_OPERATOR_BASE_AUTHORITY")

    campaign = json.loads(_bound_path(repo, request["campaign_contract_repo_path"]).read_text())
    _require(
        campaign.get("required_target_stage_id") == "48_PRESENTATION_RESPONSE_TRUTH"
        and int(campaign.get("required_stage_count", 0)) == 10,
        "PRESENTATION_OPERATOR_FINITE_TARGET_REQUIRED",
    )
    plan = json.loads(_bound_path(repo, campaign["compiler_plan"]).read_text())
    _require(
        len(plan.get("stages") or ()) == 10
        and plan["stages"][-1]["id"] == "48_PRESENTATION_RESPONSE_TRUTH",
        "PRESENTATION_OPERATOR_PLAN_TARGET_DRIFT",
    )

    baseline_rows = []
    for row in base["native_baselines"]:
        path = _bound_path(authority, row["relative"])
        _require(path.is_file() and sha256(path) == row["sha256"], "PRESENTATION_OPERATOR_NATIVE_BASELINE_BYTES_DRIFT")
        baseline_rows.append({"subject_key": row["subject_key"], "native_teacher": {"path": str(path), "sha256": row["sha256"]}})
    equivalence_path = _bound_path(authority, base["source_equivalence_report"]["relative"])
    _require(equivalence_path.is_file() and sha256(equivalence_path) == base["source_equivalence_report"]["sha256"], "PRESENTATION_OPERATOR_EQUIVALENCE_REPORT_DRIFT")
    equivalence = json.loads(equivalence_path.read_text())
    _require(equivalence.get("status") == "EXACT_SEALED_REPLAY" and all(row.get("exact_arrays_equal") for row in equivalence.get("subjects") or ()), "PRESENTATION_OPERATOR_BASE_REPLAY_NOT_EQUIVALENT")

    executable = Path(base["evaluator_executable"]).expanduser().resolve()
    version = subprocess.check_output(
        [str(executable), "--version"], text=True,
        **launch_options(executable, base.get("evaluator_runtime")),
    ).splitlines()[0]
    _require(version == base["evaluator_version"] and sha256(executable) == base["expected_evaluator_binary_sha256"], "PRESENTATION_OPERATOR_EVALUATOR_DRIFT")

    durable = authority / "truth_foundry_inputs"
    policy_path = _bound_path(repo, base["policy_repo_path"])
    policy = json.loads(policy_path.read_text())
    _require(policy.get("schema") == "RealSaS.TeacherKinematicCalibrationPolicy.v2", "PRESENTATION_OPERATOR_CALIBRATION_POLICY")
    policy_ref = immutable_json(durable, "POLICY.json", policy)
    baseline_ref = immutable_json(
        durable,
        "NATIVE_BASELINES.json",
        {"schema": "RealSaS.NativeTeacherBaselineBindings.v1", "subjects": baseline_rows},
    )
    truth_cfg = {
        "mode": "OBSERVATION_BIND_REST_KINEMATIC_CALIBRATION_ONLY",
        "policy": policy_ref,
        "native_baselines": baseline_ref,
        "source_equivalence_report": {"path": str(equivalence_path), "sha256": base["source_equivalence_report"]["sha256"]},
        "evaluator_binary": {"path": str(executable), "sha256": base["expected_evaluator_binary_sha256"]},
        "evaluator_version": version,
        "maximum_seconds_per_subject": base["maximum_seconds_per_subject"],
        "producer_source_commit": expected_sha,
        "native_baseline_role": "HISTORICAL_PROVENANCE_ONLY__NOT_REST_OR_TEMPORAL_AUTHORITY",
        "observation_runtime_replacement": base["observation_runtime_replacement"],
        "evaluator_runtime": base["evaluator_runtime"],
    }
    presentation = dict(request.get("presentation_truth") or {})
    _require(
        presentation.get("mode") == "OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1"
        and presentation.get("semantic_part_taxonomy") == "FORBIDDEN"
        and presentation.get("action_measure") == "CONTROL_BALANCED__EQUAL_MASS_PER_CONTROL_THEN_PER_SEALED_SIGN_AMPLITUDE_ROW",
        "PRESENTATION_OPERATOR_POLICY_REQUIRED",
    )

    descriptor = json.loads(_bound_path(repo, request["input_descriptor_repo_path"]).read_text())
    descriptor["campaign_contract"] = request["campaign_contract_repo_path"]
    descriptor["campaign_name"] = request["campaign_name"]
    descriptor["actor"] = "presentation-truth-fit8-coordinator"
    descriptor["additional_manifest_sections"] = {
        "truth_foundry": truth_cfg,
        "presentation_truth": presentation,
    }
    descriptor["regression_additional_manifest_sections"] = {
        "truth_foundry": {"mode": "SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS", "new_truth_claimed": False},
        "presentation_truth": {"mode": "SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS", "new_truth_claimed": False},
    }
    stage_parameters = json.loads(_bound_path(repo, request["stage_parameters_repo_path"]).read_text())
    _require(stage_parameters.get("schema") == "RealSaS.StageReleaseParameters.v1", "PRESENTATION_OPERATOR_STAGE_PARAMETER_SCHEMA")
    descriptor["additional_stage_parameters"] = stage_parameters["stages"]
    descriptor_ref = immutable_json(durable, "PRESENTATION_BOOTSTRAP_DESCRIPTOR.json", descriptor)

    output.mkdir(parents=True, exist_ok=True)
    bootstrap_path = output / "CAMPAIGN_BOOTSTRAP.json"
    experiment_request = {
        "slug": request["experiment_slug"],
        "name": request["experiment_name"],
        "research_question": request["research_question"],
        "hypothesis": request["hypothesis"],
        "initial_state": "PREREGISTERED",
        "created_by": descriptor["actor"],
        "metadata": {
            "scope": "FIT8_OBSERVATION_REST_GENERIC_PRESENTATION_RESPONSE_TRUTH_ONLY",
            "calibration_policy_sha256": policy_ref["sha256"],
            "presentation_policy": presentation,
            "producer_source_commit": expected_sha,
            "training_allowed": False,
            "product_authority_changed": False,
        },
    }
    experiment = bootstrap.api(api_root, "POST", "/v1/experiments", experiment_request)["experiment"]
    with (output / "bootstrap.log").open("w") as log, contextlib.redirect_stdout(log):
        bootstrap.main(
            SimpleNamespace(
                repo=repo,
                descriptor=Path(descriptor_ref["path"]),
                platform_root=platform,
                authority_root=authority,
                api=api_root,
                rclone_remote=rclone_remote,
                out=bootstrap_path,
                no_submit=True,
            )
        )
    prepared = json.loads(bootstrap_path.read_text())
    campaign_id = prepared["campaign_id"]
    status = bootstrap.api(api_root, "GET", "/v1/research/campaigns/" + campaign_id)
    _require(
        status.get("target_stage_id") == "48_PRESENTATION_RESPONSE_TRUTH"
        and status.get("execution_scope") == "SOURCE_ENTRY_TO_RESEARCH_COMPONENT",
        "PRESENTATION_OPERATOR_PLATFORM_TARGET_DRIFT",
    )
    bootstrap.api(api_root, "POST", "/v1/experiments/link", {"experiment_id": experiment["id"], "target_kind": "CAMPAIGN", "target_id": campaign_id, "actor": descriptor["actor"]})
    for member in status["members"]:
        bootstrap.api(api_root, "POST", "/v1/experiments/link", {"experiment_id": experiment["id"], "target_kind": "ATTEMPT", "target_id": member["attempt_id"], "actor": descriptor["actor"]})

    result = {
        "schema": "RealSaS.PresentationTruthCampaignPreparation.v1",
        "source_code_sha": expected_sha,
        "experiment_id": experiment["id"],
        "campaign_id": campaign_id,
        "engine_release_id": prepared["release_id"],
        "target_stage_id": status["target_stage_id"],
        "bootstrap_receipt": {"path": str(bootstrap_path), "sha256": sha256(bootstrap_path)},
        "members": status["members"],
        "measurement_submitted": False,
        "training_submitted": False,
        "truth_qualification_minted": False,
        "product_authority_changed": False,
    }
    result_path = output / "PREPARATION.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if submit:
        if experiment["lifecycle_state"] == "PREREGISTERED":
            bootstrap.api(api_root, "POST", "/v1/experiments/transition", {"experiment_id": experiment["id"], "to_state": "RUNNING", "actor": descriptor["actor"]})
        _require(status["state"] == "READY", "PRESENTATION_OPERATOR_CAMPAIGN_NOT_READY")
        result["campaign_run"] = bootstrap.api(api_root, "POST", "/v1/research/campaigns/run", {
            "campaign_id": campaign_id,
            "members": prepared["run_bindings"],
            "pipeline_plan_sha256": prepared["pipeline_plan_sha256"],
            "requested_by": descriptor["actor"],
        })
        result["measurement_submitted"] = True
        result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("experiment_id", "campaign_id", "engine_release_id", "target_stage_id", "measurement_submitted")}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "repo", "platform-root", "authority-root", "expected-main-sha", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--api-root", default="http://127.0.0.1:8080")
    p.add_argument("--rclone-remote", default="gdrive")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    prepare(
        json.loads(Path(a.request).read_text()),
        repo=a.repo,
        platform_root=a.platform_root,
        authority_root=a.authority_root,
        expected_sha=a.expected_main_sha,
        output=a.output,
        api_root=a.api_root,
        rclone_remote=a.rclone_remote,
        submit=a.submit,
    )
