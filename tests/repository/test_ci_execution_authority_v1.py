"""Machine-enforced RSS.vNext CI execution owner; no unreviewed per-commit court fan-out."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / "canonical/CI_EXECUTION_AUTHORITY_V1.json").read_text(encoding="utf-8"))
WF = ROOT / ".github" / "workflows"


def _source(name: str) -> str:
    path = WF / name
    assert path.is_file(), name
    return path.read_text(encoding="utf-8")


def _pr_block(source: str) -> str:
    match = re.search(r"(?ms)^  pull_request:\s*\n(.*?)(?=^  [A-Za-z_][\w-]*:|^(?:permissions|concurrency|jobs):|\Z)", source)
    assert match, "Missing pull_request event"
    return match.group(1)


def test_exact_one_shared_authority_for_rss_vnext_fast_and_full():
    assert POLICY["schema"] == "RealSaS.CIExecutionAuthority.v1"
    p = POLICY["pr_policy"]
    fast = p["fast_on_synchronize"]
    full = p["full_on_open_reopen_ready_or_explicit_dispatch"]
    assert len(fast) == p["max_automatic_sync_workflows_in_rss_vnext_court"] == 10
    assert len(full) == 16
    assert len(set(fast + full)) == 26
    assert set(fast).isdisjoint(full)
    assert len(fast) == len(set(fast)) and len(full) == len(set(full))


def test_full_courts_never_relaunch_on_every_synchronize():
    for name in POLICY["pr_policy"]["full_on_open_reopen_ready_or_explicit_dispatch"]:
        content = _source(name)
        assert re.search(r"(?m)^  workflow_dispatch:\s*$", content), name
        block = _pr_block(content)
        assert re.search(r"(?m)^    types: \[opened, reopened, ready_for_review\]$", block), name
        assert "synchronize" not in block, name
        assert "concurrency:" in content and "cancel-in-progress:" in content, name


def test_fast_pr_checks_retain_cancel_and_latest_head_revalidation():
    for name in POLICY["pr_policy"]["fast_on_synchronize"]:
        content = _source(name)
        block = _pr_block(content)
        types = re.search(r"(?m)^    types:\s*\[([^]]+)\]", block)
        assert not types or "synchronize" in types.group(1), name
        assert "concurrency:" in content and "cancel-in-progress:" in content, name


def test_full_admission_is_sha_exact_and_runner_execution_is_separate():
    admission = _source("ci_pr_full_admission_v1.yml")
    assert "workflow_dispatch:" in admission
    assert "github.rest.pulls.get" in admission
    assert "head.sha" in admission and "head_sha:" in admission
    assert "full_on_open_reopen_ready_or_explicit_dispatch" in admission
    assert "fast_on_synchronize" in admission
    assert "github.rest.actions.listWorkflowRunsForRepo" in admission
    assert "run.conclusion === 'success'" in admission
    assert "actions: read" in admission
    assert "runs-on: ubuntu-latest" in admission
    assert "pull_request:" not in admission
    physical = _source("platform_developer_deploy.yml")
    assert "name: platform-developer-deployment" in physical
    assert "  workflow_dispatch:" in physical
    assert "  pull_request:" not in physical and "  push:" not in physical
    assert "if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'" in physical
    assert "runs-on: [self-hosted, linux, x64, realsas]" in physical


def test_zero_job_orphan_monitor_is_read_only_and_never_an_execution_owner():
    monitor = _source("ci_queue_sentinel_v1.yml")
    assert "schedule:" in monitor and "workflow_dispatch:" in monitor
    assert "actions: read" in monitor
    assert "runs-on: ubuntu-latest" in monitor
    assert "listWorkflowRunsForRepo" in monitor and "listJobsForWorkflowRun" in monitor
    assert "cancelWorkflowRun" not in monitor and "deleteWorkflowRun" not in monitor
    assert "DELETE /repos" not in monitor
    assert "  pull_request:" not in monitor and "  push:" not in monitor
    assert not (WF / "rss_vnext_pr448_queue_drain.yml").exists(), "Delete unsafe legacy cleanup"


def test_gc_and_hosted_first_historical_owners_exist():
    assert (ROOT / POLICY["gc"]["owner"]).is_file()
    assert (ROOT / POLICY["gc"]["tests"]).is_file()
    assert (ROOT / "tests/repository/test_self_hosted_execution_policy.py").is_file()
    assert (ROOT / "docs/platform/CI_EXECUTION_AUTHORITY.md").is_file()
    assert POLICY["historical_provenance"]["hosted_and_gc"].startswith("1888d9")
    assert POLICY["historical_provenance"]["physical_deploy_manual"].startswith("9f9f53")
