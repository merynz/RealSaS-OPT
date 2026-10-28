import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "platform_cohort_operator.yml"
TRIGGER = ROOT / "canonical" / "COHORT_OPERATOR_TRIGGER_V1.json"


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_cohort_operator_uses_generic_executor_and_self_hosted_canonical_main():
    text = workflow_text()
    assert "workflow_dispatch:" in text
    assert "workflow_run:" in text
    triggers = yaml.load(text, Loader=yaml.BaseLoader)["on"]["workflow_run"]["workflows"]
    assert "platform-developer-deployment" in triggers
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
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


def test_deployment_completion_only_activates_changed_enabled_trigger():
    text = workflow_text()
    assert "canonical/COHORT_OPERATOR_TRIGGER_V1.json" in text
    assert "WORKFLOW_RUN_CONCLUSION" in text
    assert "WORKFLOW_RUN_HEAD_BRANCH" in text
    assert "git', 'diff-tree'" in text
    assert "trigger_path in changed" in text
    assert "trigger['enabled']" in text
    assert "needs.activation.outputs.should_run == 'true'" in text


def test_trigger_is_generic_operator_data():
    trigger = json.loads(TRIGGER.read_text(encoding="utf-8"))
    assert trigger["schema"] == "RealSaS.CohortOperatorTrigger.v1"
    assert isinstance(trigger["enabled"], bool)
    assert isinstance(trigger["submit"], bool)
    assert isinstance(trigger["descriptor"], str) and trigger["descriptor"]
    assert isinstance(trigger["nonce"], str) and trigger["nonce"]


def test_cohort_storage_transport_is_configurable_with_current_host_default():
    text = workflow_text()
    assert "vars.REALSAS_RCLONE_DRIVE_REMOTE || 'gdrive'" in text
    assert 'grep -Fx "${RCLONE_REMOTE}:"' in text
    assert '--rclone-remote "$RCLONE_REMOTE"' in text
    assert "realsas-drive:" not in text


def test_cohort_operator_exposes_repo_root_for_python_imports():
    text = workflow_text()
    assert "PYTHONPATH: ${{ github.workspace }}" in text
    assert '"$PLATFORM_ROOT/current/bin/realsas-python" "${args[@]}"' in text


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
