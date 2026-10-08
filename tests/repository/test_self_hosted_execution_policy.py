"""Mixed execution policy for hosted CPU CI and explicit self-hosted witnesses."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
HOSTED_FIRST = {
    "current_mainline_self_hosted_ci.yml",
    "model_mainline_source_gate.yml",
    "live_authority_map.yml",
    "completion_audit_contract.yml",
    "architecture_freeze_source_gate.yml",
    "iris_v2_source_contract.yml",
    "living_compile_v4.yml",
    "platform_go_contract_v1.yml",
    "platform_throughput_hydration_contract.yml",
}
AUTHORIZED_SELF_HOSTED = {
    "[self-hosted, linux, x64, realsas]",
    "[self-hosted, realsas, gpu]",
}


def test_workflow_runner_kinds_are_explicit_and_authorized():
    checked = 0
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for value in re.findall(r"^\s*runs-on:\s*(.+)$", text, re.MULTILINE):
            checked += 1
            value = value.strip()
            assert value == "ubuntu-latest" or value in AUTHORIZED_SELF_HOSTED, (
                f"Implicit or unauthorized runner: {path.name}: {value}"
            )
    assert checked > 0


def test_host_independent_current_workflows_are_hosted_and_public_only():
    for name in sorted(HOSTED_FIRST):
        path = ROOT / ".github/workflows" / name
        text = path.read_text(encoding="utf-8")
        assert "runs-on: ubuntu-latest" in text, name
        for authorized in AUTHORIZED_SELF_HOSTED:
            assert authorized not in text, name
        assert "github.event.repository.private == false" in text, name


def test_automatic_pr_checks_cancel_superseded_heads():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^  pull_request:", text, re.MULTILINE):
            assert re.search(r"^concurrency:", text, re.MULTILINE), path.name
            value = re.search(r"^  cancel-in-progress: (.+)$", text, re.MULTILINE)
            assert value and value.group(1) in (
                "true", "${{ github.event_name == 'pull_request' }}"
            ), path.name


def test_frozen_r6_training_replays_are_explicit():
    paths = list((ROOT / ".github/workflows").glob("r6_oracle_substrate_*.yml"))
    assert len(paths) == 4
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "  workflow_dispatch:" in text, path.name
        assert not re.search(r"^  (pull_request|push):", text, re.MULTILINE), path.name
