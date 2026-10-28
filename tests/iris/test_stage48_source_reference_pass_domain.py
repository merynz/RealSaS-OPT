from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PASS = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def test_source_reference_replays_only_admitted_pass_valid_rays():
    source = (ROOT / "tools/truth_foundry/blender_source_ray_appearance_reference_v1.py").read_text()
    assert "measurement = dict(ray.get(\"measurement\") or {})" in source
    assert f'measurement.get("status") != "{PASS}"' in source
    assert 'measurement.get("valid") is not True' in source
    assert 'continue' in source[source.index('measurement = dict(ray.get("measurement") or {})'):source.index('for hit_index, hit in enumerate', source.index('measurement = dict(ray.get("measurement") or {})'))]
    assert 'SOURCE_RAY_REFERENCE_NONPASS_PACKED_RAY' not in source
    assert 'SOURCE_RAY_REFERENCE_EMPTY_PASS_DOMAIN' in source


def test_source_reference_pass_domain_matches_subject_qualification_semantics():
    qualification = (ROOT / "models/iris/truth/presentation_qualification_v1.py").read_text()
    assert 'measurement.get("status") != PASS or measurement.get("valid") is not True' in qualification
    assert 'continue' in qualification


def test_stage48_r5_activation_identity_is_immutable_and_distinct():
    import json
    request = json.loads((ROOT / "canonical/PRESENTATION_TRUTH_OPERATOR_REQUEST_V1.json").read_text())
    assert request["activation_revision"] == 5
    assert request["experiment_slug"] == "fit8-presentation-response-truth-v1-r5"
