import json
import subprocess

import pytest

from tools.platform_render_checkpoint import RenderCheckpoint, checkout_identity


def test_resume_reuses_exact_frame_and_rejects_changed_bytes(tmp_path):
    root = tmp_path / "render"
    checkpoint = RenderCheckpoint(root, {"M": "same", "CAA": "v2"})
    frame = root / "frame.png"
    frame.write_bytes(b"actual rendered frame")
    checkpoint.commit("RUN_0", frame, {"overflow": 0})
    resumed = RenderCheckpoint(root, {"M": "same", "CAA": "v2"})
    assert resumed.read("RUN_0") == (frame, {"overflow": 0})
    frame.write_bytes(b"different frame")
    with pytest.raises(RuntimeError, match="FRAME_BYTES_DRIFT"):
        resumed.read("RUN_0")


@pytest.mark.parametrize("key", ["M", "CAA", "motion", "renderer", "camera"])
def test_independent_identity_drift_cannot_resume(tmp_path, key):
    contract = {k: "original" for k in ("M", "CAA", "motion", "renderer", "camera")}
    RenderCheckpoint(tmp_path, contract)
    with pytest.raises(RuntimeError, match="RESUME_CONTRACT_DRIFT"):
        RenderCheckpoint(tmp_path, {**contract, key: "changed"})


def test_resume_rejects_frame_escape(tmp_path):
    root = tmp_path / "render"
    checkpoint = RenderCheckpoint(root, {})
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    checkpoint.value["frames"]["escaped"] = {"file": "../outside.png", "sha256": "irrelevant"}
    with pytest.raises(RuntimeError, match="FRAME_BYTES_DRIFT"):
        checkpoint.read("escaped")


def test_checkout_rejects_overlay_even_with_same_commit(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    file = tmp_path / "operator.py"
    file.write_text("sealed")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "sealed")
    sha = git("rev-parse", "HEAD")
    assert checkout_identity(tmp_path, sha) == sha
    with pytest.raises(RuntimeError, match="SHA_MISMATCH"):
        checkout_identity(tmp_path, "another commit")
    file.write_text("research overlay")
    with pytest.raises(RuntimeError, match="SOURCE_OVERLAY_FORBIDDEN"):
        checkout_identity(tmp_path, sha)
