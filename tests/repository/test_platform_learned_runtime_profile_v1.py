from pathlib import Path


def test_developer_deployment_installs_pinned_cuda_learned_runtime():
    root = Path(__file__).resolve().parents[2]
    deployment = (root / "requirements/platform-deployment.txt").read_text(encoding="utf-8")
    cuda = (root / "requirements/torch-cuda.txt").read_text(encoding="utf-8")

    assert "-r torch-cuda.txt" in deployment
    assert "https://download.pytorch.org/whl/cu128" in cuda
    assert "torch==2.8.0+cu128" in cuda
    assert "torchvision==0.23.0+cu128" in cuda


def test_runtime_profile_does_not_claim_mira_bf16_hardware_admission():
    root = Path(__file__).resolve().parents[2]
    cuda = (root / "requirements/torch-cuda.txt").read_text(encoding="utf-8")
    assert "MIRA BF16 admission" in cuda
    assert "checked separately" in cuda
