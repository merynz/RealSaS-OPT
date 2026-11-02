#!/usr/bin/env python3
from __future__ import annotations

"""Prepare and submit the finite Corpus370 truth campaign through Platform.

The operator requires an already sealed 370-member Platform truth input bundle.
It creates the Experiment, uses the training-independent truth bootstrap to create
one released Campaign and its Attempts, links those Attempts to the Experiment,
and only then submits the campaign. It never trains a model or mints final family
admission/readiness.
"""

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from tools import platform_truth_cohort_bootstrap_v1 as bootstrap
from tools import platform_observation_truth_cohort_bootstrap_v2 as observation_bootstrap
from tools.observation.pinned_evaluator_runtime_v1 import launch_options
from tools.platform_artifact_store import sha256_file

SCHEMA = "RealSaS.Corpus370TruthOperatorRequest.v1"
PROJECTION_SCHEMA = "RealSaS.Corpus370ObservationTruthOperatorRequest.v2"


def projected_member_count(preflight: dict, projected: bool) -> int:
    if not projected:
        return 370
    count = int(preflight.get("cohort_subject_count", -1))
    excluded = int(preflight.get("excluded_subject_count", -1))
    require(preflight.get("candidate_subject_count") == 370
            and 0 < count <= 370 and count + excluded == 370
            and len(preflight.get("excluded_subjects") or []) == excluded,
            "CORPUS370_TRUTH_OPERATOR_PROJECTION_ACCOUNTING")
    return count


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def prepare(
    request: dict,
    *,
    repo: Path,
    platform_root: Path,
    authority_root: Path,
    expected_sha: str,
    output: Path,
    api_root: str,
    rclone_remote: str,
    submit: bool,
) -> dict:
    require(
        request.get("schema") in {SCHEMA, PROJECTION_SCHEMA}
        and request.get("enabled") is True
        and request.get("training_allowed") is False,
        "CORPUS370_TRUTH_OPERATOR_SCOPE",
    )
    repo = repo.resolve()
    platform_root = platform_root.resolve()
    authority_root = authority_root.resolve()
    output = output.resolve()
    projected = request["schema"] == PROJECTION_SCHEMA
    require(load(platform_root / "deployment.json")["code_sha"] == expected_sha, "CORPUS370_TRUTH_OPERATOR_DEPLOYED_MAIN_DRIFT")
    require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == expected_sha, "CORPUS370_TRUTH_OPERATOR_CHECKOUT_DRIFT")
    require(subprocess.check_output(["git", "-C", str(repo), "ls-remote", "origin", "refs/heads/main"], text=True).split()[0] == expected_sha, "CORPUS370_TRUTH_OPERATOR_REMOTE_MAIN_DRIFT")

    campaign_path = (repo / request["campaign_contract_repo_path"]).resolve()
    campaign = load(campaign_path)
    require(
        campaign.get("schema") == ("RealSaS.Corpus370ObservationTruthCampaign.v2" if projected else "RealSaS.Corpus370TruthCampaign.v1")
        and campaign.get("required_target_stage_id") == ("49_SUBJECT_TRUTH_FAMILIES" if projected else "48_PRESENTATION_RESPONSE_TRUTH")
        and int(campaign.get("candidate_universe_subject_count" if projected else "expected_subject_count", 0)) == 370
        and int(campaign.get("required_stage_count", 0)) == (11 if projected else 10),
        "CORPUS370_TRUTH_OPERATOR_CAMPAIGN_CONTRACT",
    )
    plan = load((repo / campaign["compiler_plan"]).resolve())
    require(len(plan.get("stages") or ()) == campaign["required_stage_count"] and plan["stages"][-1]["id"] == campaign["required_target_stage_id"], "CORPUS370_TRUTH_OPERATOR_PLAN_TARGET")

    bundle = dict(request.get("cohort_bundle") or {})
    require(bundle.get("provider") == "google_drive", "CORPUS370_TRUTH_OPERATOR_BUNDLE_PROVIDER")
    require(bool(bundle.get("file_id")) and len(str(bundle.get("sha256") or "")) == 64 and int(bundle.get("size_bytes") or 0) > 0, "CORPUS370_TRUTH_OPERATOR_BUNDLE_IDENTITY")
    roster_sha = str(request.get("exact_source_roster_sha256") or "")
    require(len(roster_sha) == 64, "CORPUS370_TRUTH_OPERATOR_ROSTER_IDENTITY")

    evaluator = Path(request["evaluator_executable"]).resolve()
    version = subprocess.check_output([str(evaluator), "--version"], text=True, **launch_options(evaluator, request.get("evaluator_runtime"))).splitlines()[0]
    require(version == request["evaluator_version"], "CORPUS370_TRUTH_OPERATOR_EVALUATOR_VERSION")
    require(sha256_file(evaluator) == request["expected_evaluator_binary_sha256"], "CORPUS370_TRUTH_OPERATOR_EVALUATOR_SHA")
    policy_path = (repo / request["policy_repo_path"]).resolve()
    policy = load(policy_path)
    require(policy.get("schema") == "RealSaS.TeacherKinematicCalibrationPolicy.v2", "CORPUS370_TRUTH_OPERATOR_POLICY")
    stage_parameters = load((repo / request["stage_parameters_repo_path"]).resolve())
    require(stage_parameters.get("schema") == "RealSaS.StageReleaseParameters.v1", "CORPUS370_TRUTH_OPERATOR_STAGE_PARAMETERS")

    truth_cfg = {
        "mode": "OBSERVATION_BIND_REST_KINEMATIC_CALIBRATION_ONLY",
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        "evaluator_binary": {"path": str(evaluator), "sha256": request["expected_evaluator_binary_sha256"]},
        "evaluator_version": version,
        "maximum_seconds_per_subject": int(request["maximum_seconds_per_subject"]),
        "producer_source_commit": expected_sha,
        "observation_runtime_replacement": request["observation_runtime_replacement"],
        "evaluator_runtime": request["evaluator_runtime"],
    }
    presentation = dict(request.get("presentation_truth") or {})
    require(
        presentation.get("mode") == "OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1"
        and presentation.get("semantic_part_taxonomy") == "FORBIDDEN"
        and presentation.get("action_measure") == "CONTROL_BALANCED__EQUAL_MASS_PER_CONTROL_THEN_PER_SEALED_SIGN_AMPLITUDE_ROW",
        "CORPUS370_TRUTH_OPERATOR_PRESENTATION_POLICY",
    )
    descriptor = {
        "schema": observation_bootstrap.DESCRIPTOR_SCHEMA if projected else "RealSaS.PlatformTruthCohortBootstrapDescriptor.v1",
        "purpose": "RESEARCH",
        "training_allowed": False,
        "campaign_name": request["campaign_name"],
        "campaign_contract": request["campaign_contract_repo_path"],
        "actor": "corpus370-truth-coordinator",
        "cohort_bundle": bundle,
        "exact_source_roster_sha256": roster_sha,
        "additional_manifest_sections": {
            "truth_foundry": truth_cfg,
            "presentation_truth": presentation,
        },
        "additional_stage_parameters": stage_parameters.get("stages") or {},
    }
    output.mkdir(parents=True, exist_ok=True)
    descriptor_path = output / "BOOTSTRAP_DESCRIPTOR.json"
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True, indent=2) + "\n")

    experiment_request = {
        "slug": request["experiment_slug"],
        "name": request["experiment_name"],
        "research_question": request["research_question"],
        "hypothesis": request["hypothesis"],
        "initial_state": "PREREGISTERED",
        "created_by": descriptor["actor"],
        "metadata": {
            "scope": "CORPUS370_SOURCE_RESPONSE_FAMILY_COURTS_ONLY" if projected else "CORPUS370_SOURCE_RESPONSE_PRESENTATION_TRUTH_ONLY",
            "expected_subject_count": 370,
            "exact_source_roster_sha256": roster_sha,
            "bundle_sha256": bundle["sha256"],
            "producer_source_commit": expected_sha,
            "training_allowed": False,
            "model_promotion_allowed": False,
            "product_authority_changed": False,
        },
    }
    previous = [row for row in bootstrap.api(api_root, "GET", "/v1/experiments")
                if row.get('slug') == experiment_request['slug']]
    require(len(previous) <= 1, 'CORPUS370_TRUTH_OPERATOR_AMBIGUOUS_EXPERIMENT')
    if previous:
        experiment = previous[0]
        require(all(experiment.get(key) == experiment_request[key]
                    for key in ('name', 'research_question', 'hypothesis', 'metadata')),
                'CORPUS370_TRUTH_OPERATOR_EXISTING_EXPERIMENT_SCOPE_DRIFT')
    else:
        experiment = bootstrap.api(api_root, "POST", "/v1/experiments", experiment_request)["experiment"]
    bootstrap_path = output / "CAMPAIGN_BOOTSTRAP.json"
    preflight = load(bootstrap_path) if bootstrap_path.is_file() else (observation_bootstrap if projected else bootstrap).main(
        SimpleNamespace(
            repo=repo,
            descriptor=descriptor_path,
            platform_root=platform_root,
            authority_root=authority_root,
            api=api_root,
            rclone_remote=rclone_remote,
            out=bootstrap_path,
            no_submit=True,
        )
    )
    campaign_id = preflight["campaign_id"]
    require(preflight.get('bundle_sha256') == bundle['sha256'],
            'CORPUS370_TRUTH_OPERATOR_RESUME_BUNDLE_DRIFT')
    member_count = projected_member_count(preflight, projected)
    status = bootstrap.api(api_root, "GET", "/v1/research/campaigns/" + campaign_id)
    require(
        status.get("target_stage_id") == campaign["required_target_stage_id"]
        and status.get("execution_scope") == "SOURCE_ENTRY_TO_RESEARCH_COMPONENT"
        and len(status.get("members") or ()) == member_count,
        "CORPUS370_TRUTH_OPERATOR_PLATFORM_CAMPAIGN_SCOPE",
    )
    bootstrap.api(api_root, "POST", "/v1/experiments/link", {
        "experiment_id": experiment["id"], "target_kind": "CAMPAIGN", "target_id": campaign_id, "actor": descriptor["actor"]
    })
    for member in status["members"]:
        bootstrap.api(api_root, "POST", "/v1/experiments/link", {
            "experiment_id": experiment["id"], "target_kind": "ATTEMPT", "target_id": member["attempt_id"], "actor": descriptor["actor"]
        })

    result = {
        "schema": "RealSaS.Corpus370TruthCampaignPreparation.v2" if projected else "RealSaS.Corpus370TruthCampaignPreparation.v1",
        "source_code_sha": expected_sha,
        "experiment_id": experiment["id"],
        "campaign_id": campaign_id,
        "engine_release_id": preflight["release_id"],
        "target_stage_id": preflight["target_stage_id"],
        "bundle_sha256": bundle["sha256"],
        "exact_source_roster_sha256": roster_sha,
        "member_count": len(status["members"]),
        "candidate_subject_count": 370,
        "excluded_subject_count": preflight.get("excluded_subject_count", 0),
        "excluded_subjects": preflight.get("excluded_subjects", []),
        "bootstrap_receipt": {"path": str(bootstrap_path), "sha256": sha256_file(bootstrap_path)},
        "measurement_submitted": False,
        "training_submitted": False,
        "truth_qualification_minted": False,
        "final_family_matrix_minted": False,
    }
    result_path = output / "PREPARATION.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if submit and status['state'] == 'READY':
        if experiment["lifecycle_state"] == "PREREGISTERED":
            bootstrap.api(api_root, "POST", "/v1/experiments/transition", {
                "experiment_id": experiment["id"], "to_state": "RUNNING", "actor": descriptor["actor"]
            })
        require(status["state"] == "READY", "CORPUS370_TRUTH_OPERATOR_CAMPAIGN_NOT_READY")
        result["campaign_run"] = bootstrap.api(api_root, "POST", "/v1/research/campaigns/run", {
            "campaign_id": campaign_id,
            "members": preflight["run_bindings"],
            "pipeline_plan_sha256": preflight["pipeline_plan_sha256"],
            "requested_by": descriptor["actor"],
        })
        result["measurement_submitted"] = True
    elif submit:
        require(status['state'] in {'RUNNING', 'COMPLETED', 'FAILED', 'BLOCKED', 'ABSTAIN'},
                'CORPUS370_TRUTH_OPERATOR_RESUME_CAMPAIGN_STATE')
        result['measurement_submitted'] = True
        result['existing_campaign_resumed_without_resubmission'] = True
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("experiment_id", "campaign_id", "engine_release_id", "member_count", "measurement_submitted", "training_submitted")}, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--platform-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--rclone-remote", default="gdrive")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    prepare(
        load(args.request),
        repo=args.repo,
        platform_root=args.platform_root,
        authority_root=args.authority_root,
        expected_sha=args.expected_main_sha,
        output=args.output,
        api_root=args.api_root,
        rclone_remote=args.rclone_remote,
        submit=args.submit,
    )
