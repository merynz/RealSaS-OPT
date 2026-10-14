from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "platform_cohort_operator.yml"


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_cohort_operator_uses_generic_executor_and_self_hosted_canonical_main():
    text = workflow_text()
    assert "workflow_dispatch:" in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "deployment['code_sha'] == os.environ['EXPECTED_SHA']" in text
    assert "tools/platform_cohort_bootstrap_v1.py" in text
    assert '--descriptor "$DESCRIPTOR"' in text
    assert '--authority-root "$AUTHORITY_ROOT"' in text


def test_cohort_operator_defaults_to_preflight_and_requires_explicit_submit():
    text = workflow_text()
    assert "submit:" in text
    assert "default: false" in text
    assert "args+=(--no-submit)" in text
    assert "COHORT_OPERATOR_PREFLIGHT_SUBMITTED_EXECUTION" in text
    assert "COHORT_OPERATOR_SUBMIT_RECEIPT_MISSING" in text


def test_cohort_operator_contains_no_subject_specific_execution_branch():
    text = workflow_text().lower()
    for forbidden in (
        "fit8_snake_v1",
        "fit8_bat_v1",
        "fit8_trex_v1",
        "fit8_horse_v1",
        "fit8_spider_v1",
        "fit8_greenblob_v1",
        "fit8_shark_v1",
        "fit8_george_v1",
        "subject_count == 8",
        "len(subjects) == 8",
    ):
        assert forbidden not in text


def test_cohort_operator_rejects_absolute_and_parent_descriptor_paths():
    text = workflow_text()
    assert "COHORT_OPERATOR_DESCRIPTOR_MUST_BE_REPO_RELATIVE" in text
    assert "COHORT_OPERATOR_DESCRIPTOR_PATH_ESCAPE" in text
    assert "resolved.relative_to(repo)" in text
