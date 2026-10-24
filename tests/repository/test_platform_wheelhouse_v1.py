from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/platform_developer_deploy.yml"


def test_deployment_prewarm_is_exact_persistent_and_before_service_install():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("Prewarm exact deployment wheelhouse") == 2
    assert text.count("timeout-minutes: 30") == 2
    assert text.count("/home/monster/realsas_platform/wheelhouse/$key") == 2
    assert text.count('echo "PIP_NO_INDEX=1" >> "$GITHUB_ENV"') == 2
    assert text.count('echo "PIP_FIND_LINKS=$wheelhouse" >> "$GITHUB_ENV"') == 2
    for requirement in (
        "requirements/platform-deployment.txt",
        "requirements/mainline-ci.txt",
        "requirements/platform-worker.txt",
        "requirements/torch-cuda.txt",
    ):
        assert requirement in text

    automatic, manual = text.split("  developer-host-manual:", 1)
    for job in (automatic, manual):
        assert job.index("Prewarm exact deployment wheelhouse") < job.index(
            "Install persistent developer services from exact canonical main"
        )


def test_deployment_job_allows_network_prefetch_without_relaxing_inner_install_guard():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("timeout-minutes: 45") == 2
    deploy = (ROOT / "tools/platform_deploy.py").read_text(encoding="utf-8")
    assert 'source / "requirements/platform-deployment.txt"], timeout=600)' in deploy
