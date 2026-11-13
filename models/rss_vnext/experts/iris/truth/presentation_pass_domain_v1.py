"""Fail-closed admission of a measured Stage48 PASS-ray subdomain.

The historical renderer intervention court deliberately records per-ray PASS,
ABSTAIN and FAIL outcomes.  Stage48's preregistered subject policy is not
"every sampled ray must factorize"; it is a minimum number of admitted PASS rays
per subject.  This helper binds that subject-level decision to the exact Attempt
run manifest without changing a measurement tolerance or erasing failed rays.
"""
from __future__ import annotations


def _require(value: bool, code: str) -> None:
    if not value:
        raise ValueError(code)


def admit_presentation_pass_domain_v1(candidate: dict, run_manifest: dict) -> dict:
    cfg = dict(run_manifest.get("presentation_truth") or {})
    _require(
        cfg.get("mode") == "OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1",
        "PRESENTATION_PASS_DOMAIN_MODE",
    )
    _require(
        candidate.get("schema") == "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"
        and candidate.get("authority") == "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS"
        and candidate.get("qualification_minted") is False,
        "PRESENTATION_PASS_DOMAIN_CANDIDATE_AUTHORITY",
    )

    maximum = int(cfg.get("measured_rays_per_subject", 0))
    minimum = int(cfg.get("minimum_pass_rays_per_subject", 0))
    selected = int(candidate.get("selected_ray_count", -1))
    passed = int(candidate.get("pass_ray_count", -1))
    failed = int(candidate.get("fail_ray_count", -1))
    abstained = int(candidate.get("abstain_ray_count", -1))

    _require(maximum > 0 and 0 < minimum <= maximum, "PRESENTATION_PASS_DOMAIN_POLICY_BOUNDS")
    _require(0 < selected <= maximum, "PRESENTATION_PASS_DOMAIN_SELECTED_COUNT")
    _require(
        min(passed, failed, abstained) >= 0 and passed + failed + abstained == selected,
        "PRESENTATION_PASS_DOMAIN_COUNT_PARTITION",
    )
    _require(len(candidate.get("rays") or ()) == selected, "PRESENTATION_PASS_DOMAIN_RAY_CARDINALITY")
    _require(passed >= minimum, "PRESENTATION_PASS_DOMAIN_MINIMUM_NOT_MET")

    return {
        "schema": "RealSaS.PresentationPassDomainAdmission.v1",
        "minimum_pass_rays_per_subject": minimum,
        "measured_rays_per_subject": maximum,
        "selected_ray_count": selected,
        "pass_ray_count": passed,
        "fail_ray_count": failed,
        "abstain_ray_count": abstained,
        "failed_rays_retained_in_raw_measurement": failed > 0,
        "nonpass_rays_admitted_to_source_reference_domain": False,
        "measurement_tolerance_changed": False,
        "status": "PASS_PREREGISTERED_MINIMUM_PASS_RAY_DOMAIN",
    }
