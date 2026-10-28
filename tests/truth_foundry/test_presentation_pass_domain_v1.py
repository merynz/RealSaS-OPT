import copy

import pytest

from models.iris.truth.presentation_pass_domain_v1 import admit_presentation_pass_domain_v1


def _manifest(minimum=16, maximum=32):
    return {
        "presentation_truth": {
            "mode": "OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1",
            "measured_rays_per_subject": maximum,
            "minimum_pass_rays_per_subject": minimum,
            "measurement_tolerance": 0.0075,
        }
    }


def _candidate(passed=27, failed=1, abstained=4):
    selected = passed + failed + abstained
    return {
        "schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1",
        "authority": "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS",
        "qualification_minted": False,
        "selected_ray_count": selected,
        "pass_ray_count": passed,
        "fail_ray_count": failed,
        "abstain_ray_count": abstained,
        "rays": [{"ray_index": index} for index in range(selected)],
    }


def test_preregistered_minimum_admits_george_shape_without_erasing_failed_ray():
    receipt = admit_presentation_pass_domain_v1(_candidate(), _manifest())
    assert receipt["status"] == "PASS_PREREGISTERED_MINIMUM_PASS_RAY_DOMAIN"
    assert receipt["pass_ray_count"] == 27
    assert receipt["fail_ray_count"] == 1
    assert receipt["abstain_ray_count"] == 4
    assert receipt["failed_rays_retained_in_raw_measurement"] is True
    assert receipt["nonpass_rays_admitted_to_source_reference_domain"] is False
    assert receipt["measurement_tolerance_changed"] is False


def test_minimum_is_fail_closed_and_not_posthoc_relaxed():
    with pytest.raises(ValueError, match="PRESENTATION_PASS_DOMAIN_MINIMUM_NOT_MET"):
        admit_presentation_pass_domain_v1(_candidate(passed=15, failed=1, abstained=16), _manifest())


def test_count_partition_and_policy_bounds_are_fail_closed():
    bad = _candidate()
    bad["selected_ray_count"] = 31
    with pytest.raises(ValueError, match="PRESENTATION_PASS_DOMAIN_COUNT_PARTITION"):
        admit_presentation_pass_domain_v1(bad, _manifest())
    with pytest.raises(ValueError, match="PRESENTATION_PASS_DOMAIN_POLICY_BOUNDS"):
        admit_presentation_pass_domain_v1(_candidate(), _manifest(minimum=33))


def test_candidate_authority_is_required():
    bad = copy.deepcopy(_candidate())
    bad["qualification_minted"] = True
    with pytest.raises(ValueError, match="PRESENTATION_PASS_DOMAIN_CANDIDATE_AUTHORITY"):
        admit_presentation_pass_domain_v1(bad, _manifest())
