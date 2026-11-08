"""Synthetic, non-destructive developer deploy retention and host disk guard."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import platform_deploy
from tools import platform_release_budget_v1 as budget


def test_release_count_is_advisory_not_a_dead_end(tmp_path, capsys):
    root = tmp_path / "platform"
    releases = root / "releases"
    releases.mkdir(parents=True)
    for i in range(12):
        (releases / (f"{i:040x}-{i:08x}")).mkdir()
    budget.require_release_budget(root)
    assert "REALSAS_RELEASE_RETENTION_WARNING" in capsys.readouterr().err


def test_six_releases_safe_under_budget(tmp_path):
    root = tmp_path / "platform"
    releases = root / "releases"
    releases.mkdir(parents=True)
    for i in range(6):
        (releases / (f"{i:040x}-{i:08x}")).mkdir()
    budget.require_release_budget(root)


def test_unmanaged_directory_not_counted_as_release(tmp_path):
    root = tmp_path / "platform"
    releases = root / "releases"
    releases.mkdir(parents=True)
    for i in range(4):
        (releases / (f"{i:040x}-{i:08x}")).mkdir()
    (releases / "do-not-delete-science").mkdir()
    budget.require_release_budget(root)


def test_windows_backing_disk_low_space_blocks(tmp_path, monkeypatch):
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(Path, "is_mount", lambda self: str(self) == "/mnt/c")
    monkeypatch.setattr(budget.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=99 * (1 << 20)
                            if str(path) == "/mnt/c" else 100 * (1 << 30)))
    with pytest.raises(RuntimeError, match="WINDOWS_C_FREE_BELOW_10_GIB"):
        budget.require_release_budget(root)


def test_linux_disk_low_space_blocks(tmp_path, monkeypatch):
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(Path, "is_mount", lambda self: False)
    monkeypatch.setattr(budget.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=2 * (1 << 30)))
    with pytest.raises(RuntimeError, match="LINUX_FREE_BELOW_8_GIB"):
        budget.require_release_budget(root)


def test_deployment_preflight_budget_only_for_new_materialization():
    import inspect
    source = inspect.getsource(platform_deploy.preflight)
    assert source.index('marker = root / "deployment.json"') < source.index(
        "matching_current_release(root, expected_sha, authority)")
    assert source.index("if identical is None:") < source.index(
        "require_release_budget(root)")
    assert "force_rebuild" in source
