from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/platform_developer_deploy.yml"


def test_deployment_prewarm_is_exact_persistent_and_before_service_install():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("Prewarm exact deployment wheelhouse") == 1
    assert "Check installed release before dependency prewarm (read-only)" in text
    assert "if: steps.installed-release.outputs.reuse != 'true'" in text
    assert text.index("Check installed release before dependency prewarm") < text.index(
        "Prewarm exact deployment wheelhouse"
    )
    assert text.count("timeout-minutes: 30") == 1
    assert text.count("wheelhouse_root=/home/monster/realsas_platform/wheelhouse") == 1
    assert text.count('echo "PIP_NO_INDEX=1" >> "$GITHUB_ENV"') == 1
    assert text.count('echo "PIP_FIND_LINKS=$wheelhouse" >> "$GITHUB_ENV"') == 1
    for requirement in (
        "requirements/platform-deployment.txt",
        "requirements/mainline-ci.txt",
        "requirements/platform-worker.txt",
        "requirements/torch-cpu.txt",
        "requirements/truth-family-courts.txt",
    ):
        assert requirement in text

    assert "  developer-host:\n" not in text
    manual = text.split("  developer-host-manual:", 1)[1]
    assert manual.index("Prewarm exact deployment wheelhouse") < manual.index(
        "Install persistent developer services from exact canonical main"
    )
    assert 'find "$wheelhouse_root" -mindepth 1 -maxdepth 1 -type d ! -name "$key"' in manual
    assert "python -m pip cache purge || true" in manual
    assert "2147483648" in manual

def test_persistent_platform_uses_cpu_parent_runtime_not_mira_cuda_profile():
    deployment = (ROOT / "requirements/platform-deployment.txt").read_text(encoding="utf-8")
    assert "-r torch-cpu.txt" in deployment
    assert "torch-cuda.txt" not in deployment
    assert "-r truth-family-courts.txt" in deployment

    hosted = (ROOT / ".github/workflows/platform_developer_contracts_hosted.yml").read_text(
        encoding="utf-8"
    )
    # CUDA profile changes still trigger hosted contract checks. The manual
    # physical install's persistent wheelhouse is CPU-only.
    assert "- 'requirements/torch-cpu.txt'" in hosted
    physical = WORKFLOW.read_text(encoding="utf-8")
    for block in physical.split("paths = (", 1)[1:]:
        assert "requirements/torch-cpu.txt" in block


def test_deployment_job_allows_network_prefetch_without_relaxing_inner_install_guard():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("timeout-minutes: 45") == 1
    deploy = (ROOT / "tools/platform_deploy.py").read_text(encoding="utf-8")
    assert "provision_shared_venv(root, source, release)" in deploy
    shared = (ROOT / "tools/platform_shared_python_env_v1.py").read_text(encoding="utf-8")
    assert 'directory / "platform-deployment.txt"' in shared
    assert "ensure_venv(cache_root, [], requirement_files=requirements)" in shared
    # The shared environment builder receives the original offline pip wheelhouse
    # variables from the workflow, rather than invoking an unguarded pip install.
    bootstrap = (ROOT / "tools/platform_ci_bootstrap.py").read_text(encoding="utf-8")
    assert "env = os.environ.copy()" in bootstrap
    assert "subprocess.run(command, check=True, env=env)" in bootstrap
