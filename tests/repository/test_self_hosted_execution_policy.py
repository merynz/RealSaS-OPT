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
