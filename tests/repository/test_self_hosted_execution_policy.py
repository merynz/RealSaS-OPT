"""No hosted runner fallback in any repository workflow, including old replay."""
from pathlib import Path
import re


def test_all_workflows_use_the_authorized_self_hosted_runner():
    root = Path(__file__).resolve().parents[2]
    checked = 0
    for path in (root / ".github/workflows").glob("*.yml"):
        for value in re.findall(r"^\s*runs-on:\s*(.+)$", path.read_text(), re.MULTILINE):
            checked += 1
            assert "self-hosted" in value, f"Hosted or implicit runner: {path.name}: {value}"
    assert checked > 0


def test_automatic_pr_checks_cancel_superseded_heads():
    root = Path(__file__).resolve().parents[2]
    for path in (root / ".github/workflows").glob("*.yml"):
        text = path.read_text()
        if re.search(r"^  pull_request:", text, re.MULTILINE):
            assert re.search(r"^concurrency:", text, re.MULTILINE), path.name
            value = re.search(r"^  cancel-in-progress: (.+)$", text, re.MULTILINE)
            assert value and value.group(1) in (
                "true", "${{ github.event_name == 'pull_request' }}"
            ), path.name


def test_frozen_r6_training_replays_are_explicit():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / ".github/workflows").glob("r6_oracle_substrate_*.yml"))
    assert len(paths) == 4
    for path in paths:
        text = path.read_text()
        assert "  workflow_dispatch:" in text, path.name
        assert not re.search(r"^  (pull_request|push):", text, re.MULTILINE), path.name
