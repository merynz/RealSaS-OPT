"""Synthetic exact-SHA reuse: no duplicate source/bin/venv under repeated dispatch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import platform_deploy as deploy
from tools import platform_release_idempotency_v1 as idem


def make_complete_release(tmp_path: Path, *, sha: str = "a" * 40):
    root = tmp_path / "home" / "monster" / "realsas_platform"
    releases = root / "releases"
    release = releases / (sha + "-12345678")
    (release / "source").mkdir(parents=True)
    for path in ("venv/bin/python", "bin/realsas-api",
                 "bin/realsas-control-worker", "bin/temporal", "bin/realsas-python"):
        p = release / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("mock")
    (root / "current").symlink_to(release)
    authority = tmp_path / "home" / "monster" / "realsas_authority"
    authority.mkdir()
    marker = {
        "schema": "RealSaS.DeveloperDeployment.v1", "root": str(root),
        "state": "SERVICES_ACTIVE", "code_sha": sha,
        "authority_root": str(authority), "current_release": str(release),
        "previous_release": None,
    }
    (root / "deployment.json").write_text(json.dumps(marker))
    return root, authority, release


def test_exact_current_sha_reused(tmp_path):
    root, authority, release = make_complete_release(tmp_path)
    assert idem.matching_current_release(root, "a" * 40, authority) == release
    assert idem.matching_current_release(root, "b" * 40, authority) is None


def test_same_sha_healthy_dispatch_noop_without_cloning(tmp_path, monkeypatch, capsys):
    root, authority, release = make_complete_release(tmp_path)
    monkeypatch.setattr(deploy, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(deploy, "existing_release_ready", lambda: True)
    deploy.install(root, "a" * 40, authority)
    receipt = json.loads(capsys.readouterr().out.strip())
    assert receipt["status"] == "NOOP_IDENTICAL_RELEASE"
    assert receipt["current_release"] == str(release)
    assert list((root / "releases").iterdir()) == [release]


def test_same_sha_unhealthy_does_not_duplicate(tmp_path, monkeypatch):
    root, authority, release = make_complete_release(tmp_path)
    monkeypatch.setattr(deploy, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(deploy, "existing_release_ready", lambda: False)
    with pytest.raises(RuntimeError, match="MATCHING_SHA_PLATFORM_NOT_HEALTHY"):
        deploy.install(root, "a" * 40, authority)
    assert list((root / "releases").iterdir()) == [release]


@pytest.mark.parametrize("field,value,error", [
    ("authority_root", "/different/authority", "AUTHORITY_ROOT_DRIFT"),
    ("current_release", "/different/release", "ACTIVE_RELEASE_ID_DRIFT"),
    ("state", "STARTING", "PREVIOUS_DEPLOY_NOT_COMPLETE"),
])
def test_matching_sha_untrusted_receipts_fail(tmp_path, field, value, error):
    root, authority, _ = make_complete_release(tmp_path)
    p = root / "deployment.json"
    data = json.loads(p.read_text())
    data[field] = value
    p.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match=error):
        idem.matching_current_release(root, "a" * 40, authority)


def test_missing_binary_fail_closed(tmp_path):
    root, authority, release = make_complete_release(tmp_path)
    (release / "bin/realsas-api").unlink()
    with pytest.raises(RuntimeError, match="RELEASE_INCOMPLETE"):
        idem.matching_current_release(root, "a" * 40, authority)


def test_non_matching_new_sha_not_reused(tmp_path):
    root, authority, _ = make_complete_release(tmp_path)
    assert idem.matching_current_release(root, "b" * 40, authority) is None


def test_preflight_skips_budget_only_for_matching_complete_release():
    import inspect
    src = inspect.getsource(deploy.preflight)
    assert src.index("matching_current_release(root, expected_sha, authority)") < src.index(
        "require_release_budget(root)")
    assert "if identical is None:" in src
