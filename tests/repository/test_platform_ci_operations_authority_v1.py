"""Canonical CI/runner/GC ownership is enforced regardless of the active agent."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / "canonical/PLATFORM_CI_OPERATIONS_AUTHORITY_V1.json").read_text(encoding="utf-8"))
WF = ROOT / ".github" / "workflows"


def test_one_main_owned_execution_policy_and_workflow_budget():
    assert POLICY["schema"] == "RealSaS.PlatformCIAndCacheOperationsAuthority.v1"
    assert POLICY["owner"] == "main"
    items = list(WF.glob("*.yml")) + list(WF.glob("*.yaml"))
    assert len(items) <= POLICY["workflow_budget"]["max_workflow_files"], len(items)
    assert POLICY["workflow_budget"]["pr_concurrency_and_cancel_required"] is True
    assert POLICY["workflow_budget"]["max_workflow_files"] <= 430


def test_all_pr_workflows_cancel_superseded_heads():
    for path in WF.glob("*.yml"):
        source = path.read_text(encoding="utf-8")
        if not re.search(r"(?m)^  pull_request:", source):
            continue
        assert re.search(r"(?m)^concurrency:", source), path.name
        assert re.search(r"(?m)^  cancel-in-progress:\s*(true|\$\{\{ github.event_name == 'pull_request' \}\})$", source), path.name


def test_exact_main_deployment_and_hosted_contract_remain_separate():
    source = (WF / "platform_developer_deploy.yml").read_text(encoding="utf-8")
    assert "name: platform-developer-deployment" in source
    assert "  workflow_dispatch:" in source
    assert "  pull_request:" not in source and "  push:" not in source
    assert "if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'" in source
    assert "runs-on: [self-hosted, linux, x64, realsas]" in source
    hosted = (WF / "platform_developer_contracts_hosted.yml").read_text(encoding="utf-8")
    assert "name: platform-developer-contracts-hosted" in hosted
    assert "runs-on: ubuntu-latest" in hosted
    assert "github.event.repository.private == false" in hosted
    assert "self-hosted" not in hosted


def test_gc_owner_not_aliased_to_github_actions_cleanup():
    gc = POLICY["gc"]
    assert (ROOT / gc["engine"]).is_file()
    assert (ROOT / gc["tests"]).is_file()
    assert gc["lock_safe"] and gc["atomic_install"] and gc["minimum_free_disk"]
    assert gc["verified_CAS_authority_not_automatically_deleted"] is True
    assert POLICY["queue"]["automation_must_not_bulk_cancel_or_delete"] is True
    legacy = WF / "rss_vnext_pr448_queue_drain.yml"
    if legacy.exists():
        assert "  pull_request:" not in legacy.read_text(encoding="utf-8"), "Unsafe cleanup is automatic"


def test_queue_sentinel_is_read_only_and_not_a_pr_job():
    source = (WF / "ci_queue_sentinel_v1.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in source and "schedule:" in source
    assert "actions: read" in source
    assert "runs-on: ubuntu-latest" in source
    assert "listJobsForWorkflowRun" in source
    assert "cancelWorkflowRun" not in source and "deleteWorkflowRun" not in source
    assert "DELETE /repos" not in source
    assert "  pull_request:" not in source and "  push:" not in source
