from pathlib import Path


def test_developer_deployment_installs_existing_pinned_cpu_learned_runtime():
    root = Path(__file__).resolve().parents[2]
    deployment = (root / "requirements/platform-deployment.txt").read_text(encoding="utf-8")
    cpu = (root / "requirements/torch-cpu.txt").read_text(encoding="utf-8")

    assert "-r torch-cpu.txt" in deployment
    assert "torch-cuda.txt" not in deployment
    assert "https://download.pytorch.org/whl/cpu" in cpu
    assert "torch==2.8.0+cpu" in cpu
    assert "torchvision==0.23.0+cpu" in cpu


def test_generic_runtime_profile_does_not_claim_mira_cuda_bf16_admission():
    root = Path(__file__).resolve().parents[2]
    deployment = (root / "requirements/platform-deployment.txt").read_text(encoding="utf-8")
    assert "CUDA/BF16" in deployment
    assert "MIRA" in deployment
    assert "own execution lane" in deployment
