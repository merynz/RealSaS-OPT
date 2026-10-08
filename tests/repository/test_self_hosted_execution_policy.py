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
    'appearance_pixel_center_behavioral_v1.yml',
    'appearance_source_contract.yml',
    'arachne_codec_v2_source_contract.yml',
    'arachne_shipping_boundary.yml',
    'architecture_v3_svg.yml',
    'architecture_v4_contract.yml',
    'candidate_stack_v1.yml',
    'complete_e2e_source_contract.yml',
    'current_runtime_self_hosted_ci.yml',
    'export_runtime_source_contract.yml',
    'geppetto_r6_teacher_projection_v1.yml',
    'geppetto_v2_source_contract.yml',
    'motion_deformation_behavioral_v1.yml',
    'motion_source_contract.yml',
    'mwb2_directional_behavioral_v1.yml',
    'mwb2_source_contract.yml',
    'native_runtime_source_gate.yml',
    'proof_engine_source_contract.yml',
    'proof_service_promotion_gate.yml',
    'quaternius_motion_external_gate.yml',
    'repair_child_attempt_gate.yml',
    'runtime_deploy_bake_promotion_gate.yml',
    'single_family_data_contract_v1.yml',
    'skin_field_codec_shipping_capacity.yml',
    'tessa_knight_static_survivor_replay_v1.yml',
}


def _runner_values(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(r"^\s*runs-on:\s*(.+)$", text, re.MULTILINE)
    ]


def _is_explicit_self_hosted(value: str) -> bool:
    return value.startswith("[") and value.endswith("]") and "self-hosted" in value


def test_workflow_runner_kinds_are_explicit():
    checked = 0
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for value in _runner_values(text):
            checked += 1
            assert value == "ubuntu-latest" or _is_explicit_self_hosted(value), (
                f"Implicit or unsupported runner expression: {path.name}: {value}"
            )
    assert checked > 0


def test_host_independent_current_workflows_are_hosted_and_public_only():
    for name in sorted(HOSTED_FIRST):
        path = ROOT / ".github/workflows" / name
        text = path.read_text(encoding="utf-8")
        values = _runner_values(text)
        assert values, name
        assert all(value == "ubuntu-latest" for value in values), name
        assert "github.event.repository.private == false" in text, name


def test_automatic_self_hosted_pr_checks_reject_public_fork_code():
    """Every automatic PR workflow reaching a physical runner must reject fork code.

    Report the complete debt set in one failure so migrations are fixed as one wave
    instead of discovering one historical workflow per CI run.
    """
    missing = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if not re.search(r"^  pull_request:", text, re.MULTILINE):
            continue
        if not any(_is_explicit_self_hosted(value) for value in _runner_values(text)):
            continue
        if path.name == "platform_developer_deploy.yml" and "    if: github.event_name == 'push' && github.ref == 'refs/heads/main'" in text:
            continue
        if "github.event.pull_request.head.repo.full_name == github.repository" not in text:
            missing.append(path.name)
    assert not missing, "Self-hosted PR workflows lacking same-repository firewall: " + ", ".join(missing)


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


def test_developer_deployment_does_not_queue_physical_host_on_pr():
    text = (ROOT / ".github/workflows/platform_developer_deploy.yml").read_text()
    physical_job = text.split("  developer-host:", 1)[1]
    assert "if: github.event_name == 'push' && github.ref == 'refs/heads/main'" in physical_job
    assert "  developer-contract:" in text
    assert "runs-on: ubuntu-latest" in text


def test_historical_family_selector_preserves_frozen_checkout_boundary():
    text = (ROOT / ".github/workflows/post_freeze_family_selector_v1.yml").read_text()
    assert "  workflow_dispatch:" in text
    assert "ref: ${{ inputs.frozen_ref }}" in text
    assert not re.search(r"^  (pull_request|push):", text, re.MULTILINE)
