from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/platform_developer_deploy.yml"


def test_deployment_prewarm_is_exact_persistent_and_before_service_install():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("Prewarm exact deployment wheelhouse") == 2
    assert text.count("timeout-minutes: 30") == 2
    assert text.count("wheelhouse_root=/home/monster/realsas_platform/wheelhouse") == 2
    assert text.count('echo "PIP_NO_INDEX=1" >> "$GITHUB_ENV"') == 2
    assert text.count('echo "PIP_FIND_LINKS=$wheelhouse" >> "$GITHUB_ENV"') == 2
    for requirement in (
        "requirements/platform-deployment.txt",
        "requirements/mainline-ci.txt",
        "requirements/platform-worker.txt",
        "requirements/torch-cpu.txt",
    ):
        assert requirement in text

    automatic, manual = text.split("  developer-host-manual:", 1)
    for job in (automatic, manual):
        assert job.index("Prewarm exact deployment wheelhouse") < job.index(
            "Install persistent developer services from exact canonical main"
        )
        assert "find \"$wheelhouse_root\" -mindepth 1 -maxdepth 1 -type d ! -name \"$key\"" in job
        assert "python -m pip cache purge || true" in job
        assert "2147483648" in job


def test_persistent_platform_uses_cpu_parent_runtime_not_mira_cuda_profile():
    deployment = (ROOT / "requirements/platform-deployment.txt").read_text(encoding="utf-8")
    assert "-r torch-cpu.txt" in deployment
    assert "torch-cuda.txt" not in deployment

    text = WORKFLOW.read_text(encoding="utf-8")
    # CUDA profile changes may still exercise deployment contracts, but the
    # persistent wheelhouse identity is bound only to the admitted CPU runtime.
    assert "- 'requirements/torch-cpu.txt'" in text
    for block in text.split("paths = (", 1)[1:]:
        assert "requirements/torch-cpu.txt" in block


def test_deployment_job_allows_network_prefetch_without_relaxing_inner_install_guard():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("timeout-minutes: 45") == 2
    deploy = (ROOT / "tools/platform_deploy.py").read_text(encoding="utf-8")
    assert 'source / "requirements/platform-deployment.txt"], timeout=600)' in deploy
