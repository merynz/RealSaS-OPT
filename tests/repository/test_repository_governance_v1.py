from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]

REQUIRED = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "pyproject.toml",
    "requirements/mainline-ci.txt",
    "requirements/torch-cpu.txt",
    "requirements/dev.txt",
    "requirements/README.md",
    "docs/repository/DEPENDENCY_AND_IP_POLICY.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
)

CURRENT_SELF_HOSTED_WORKFLOWS = (
    ".github/workflows/current_mainline_self_hosted_ci.yml",
    ".github/workflows/model_mainline_source_gate.yml",
    ".github/workflows/live_authority_map.yml",
    ".github/workflows/completion_audit_contract.yml",
)


def test_required_governance_files_exist() -> None:
    for rel in REQUIRED:
        assert (ROOT / rel).is_file(), rel


def test_proprietary_license_is_explicit_and_does_not_relicense_third_party() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "All Rights Reserved" in text
    assert "NO LICENSE OR PERMISSION IS GRANTED" in text
    assert "third-party" in text.lower()


def test_direct_model_dependency_notice_binds_dino_and_cleanroom_boundary() -> None:
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "facebookresearch/dinov2" in text
    assert "7764ea0f912e53c92e82eb78a2a1631e92725fc8" in text
    assert "Apache License 2.0" in text
    assert "External research references are not dependencies by default" in text


def test_current_ci_profiles_are_exactly_pinned() -> None:
    for rel in ("requirements/mainline-ci.txt", "requirements/torch-cpu.txt", "requirements/dev.txt"):
        for raw in (ROOT / rel).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            assert "==" in line, f"unpinned dependency in {rel}: {line}"


def test_current_authority_workflows_are_self_hosted_only() -> None:
    for rel in CURRENT_SELF_HOSTED_WORKFLOWS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "runs-on: [self-hosted, linux, x64, realsas]" in text, rel
        assert "runs-on: ubuntu-latest" not in text, rel


def test_current_model_ci_consumes_pinned_profiles() -> None:
    for rel in (
        ".github/workflows/current_mainline_self_hosted_ci.yml",
        ".github/workflows/model_mainline_source_gate.yml",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "requirements/mainline-ci.txt" in text, rel
        assert "requirements/torch-cpu.txt" in text, rel
        assert re.search(r"pip install[^\n]*-r requirements/mainline-ci\.txt", text), rel
        assert re.search(r"pip install[^\n]*-r requirements/torch-cpu\.txt", text), rel


def test_codeowners_protects_science_and_authority_spine() -> None:
    text = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
    for token in ("/CURRENT_STATE.md", "/canonical/", "/models/", "/compiler/", "/.github/"):
        assert token in text
