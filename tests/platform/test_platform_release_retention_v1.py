"""Synthetic-only GC tests. No real Platform roots or research artifacts touched."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from tools import platform_release_retention_v1 as m


@pytest.fixture(autouse=True)
def simulate_nonroot(monkeypatch):
    # GitHub CI can run as root in a container; live operator execution cannot.
    monkeypatch.setattr(m.os, "geteuid", lambda: 1000)


def fake_platform(tmp_path, count=10):
    root = tmp_path / "home" / "monster" / "realsas_platform"
    releases = root / "releases"
    releases.mkdir(parents=True)
    dirs = []
    for i in range(count):
        p = releases / f"{i:040x}-{i:08x}"
        p.mkdir()
        (p / "source").mkdir()
        (p / "source" / "immutable.txt").write_text("source")
        dirs.append(p)
    (root / "current").symlink_to(dirs[0])
    (root / "deployment.json").write_text(json.dumps({
        "schema": "RealSaS.DeveloperDeployment.v1",
        "root": str(root), "current_release": str(dirs[0]),
        "previous_release": str(dirs[1]), "state": "SERVICES_ACTIVE"}))
    (root / "artifacts").mkdir()
    (root / "artifacts" / "science.dat").write_text("must-preserve")
    (root / "data").mkdir()
    (root / "data" / "postgres").mkdir()
    (root / "data" / "postgres" / "PG_VERSION").write_text("17")
    (root / "receipts").mkdir()
    (root / "research-results").mkdir()
    (root.parent / "realsas_authority").mkdir()
    return root, dirs



def reference_index(root, referenced=()):
    authority = root.parent / "realsas_authority"
    authority.mkdir(parents=True, exist_ok=True)
    report = authority / "historical_reference_probe.json"
    report.write_text("\n".join(str(x) for x in referenced) or "no old reference")
    index = root.parent / "reference_file_list.txt"
    index.write_text(str(report) + "\n")
    return index

def test_plan_preserves_current_previous_six_most_recent(tmp_path):
    root, dirs = fake_platform(tmp_path)
    out = m.plan(root, 5)
    assert out["release_count"] == 10
    assert out["keep_count"] == 7
    assert out["candidate_count"] == 3
    assert str(dirs[0]) not in out["candidates"]
    assert str(dirs[1]) not in out["candidates"]


def test_confirmation_missing_fails_without_deletion(tmp_path):
    root, dirs = fake_platform(tmp_path)
    with pytest.raises(RuntimeError, match="EXPLICIT_CONFIRMATION"):
        m.apply(root, 5, "no")
    assert all(p.is_dir() for p in dirs)


def test_active_platform_blocks_without_deletion(tmp_path, monkeypatch):
    root, dirs = fake_platform(tmp_path)
    def blocked(): raise RuntimeError("PLATFORM_ACTIVE")
    monkeypatch.setattr(m, "target_inactive", blocked)
    with pytest.raises(RuntimeError, match="PLATFORM_ACTIVE"):
        m.apply(root, 5, m.CONFIRM, reference_index(root))
    assert all(p.exists() for p in dirs)


def test_apply_only_prunes_unreferenced_release_directories(tmp_path, monkeypatch):
    root, dirs = fake_platform(tmp_path)
    monkeypatch.setattr(m, "target_inactive", lambda: None)
    monkeypatch.setattr(m, "check_no_references", lambda _: None)
    monkeypatch.setattr(m, "check_no_mounted_subtrees", lambda _: None)
    report = m.apply(root, 5, m.CONFIRM, reference_index(root))
    assert report == {"removed": 3, "remaining": 7}
    assert dirs[0].exists() and dirs[1].exists()
    assert (root / "artifacts" / "science.dat").read_text() == "must-preserve"
    assert (root / "data" / "postgres" / "PG_VERSION").read_text() == "17"


def test_marker_drift_aborts(tmp_path):
    root, dirs = fake_platform(tmp_path)
    path = root / "deployment.json"
    data = json.loads(path.read_text())
    data["current_release"] = str(dirs[2])
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="ACTIVE_RELEASE_MARKER_MISMATCH"):
        m.plan(root)


def test_unmanaged_directory_is_never_removed(tmp_path, monkeypatch):
    root, _ = fake_platform(tmp_path)
    p = root / "releases" / "user-source-not-a-managed-release"
    p.mkdir()
    (p / "keep.txt").write_text("yes")
    monkeypatch.setattr(m, "target_inactive", lambda: None)
    monkeypatch.setattr(m, "check_no_references", lambda _: None)
    monkeypatch.setattr(m, "check_no_mounted_subtrees", lambda _: None)
    m.apply(root, 5, m.CONFIRM, reference_index(root))
    assert (p / "keep.txt").read_text() == "yes"


def test_symlinked_root_refused(tmp_path):
    root, _ = fake_platform(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    with pytest.raises(RuntimeError, match="UNSAFE_ROOT"):
        m.plan(alias)


def test_retention_below_minimum_refused(tmp_path):
    root, _ = fake_platform(tmp_path)
    with pytest.raises(RuntimeError, match="RETENTION_COUNT_OUT_OF_SAFE_RANGE"):
        m.plan(root, 2)

def test_historical_release_reference_protected_while_unreferenced_can_prune(tmp_path, monkeypatch):
    root, dirs = fake_platform(tmp_path)
    index = reference_index(root, [dirs[3]])
    preview = m.plan(root, 5, index)
    assert preview["inspected_historical_reference_records"] == 1
    assert preview["protected_historical_release_count"] == 1
    assert str(dirs[3]) not in preview["candidates"]
    assert preview["candidate_count"] == 2
    monkeypatch.setattr(m, "target_inactive", lambda: None)
    monkeypatch.setattr(m, "check_no_references", lambda _: None)
    monkeypatch.setattr(m, "check_no_mounted_subtrees", lambda _: None)
    result = m.apply(root, 5, m.CONFIRM, index)
    assert result["removed"] == 2
    assert dirs[3].exists()


def test_apply_requires_reference_court_even_if_all_services_stopped(tmp_path, monkeypatch):
    root, dirs = fake_platform(tmp_path)
    monkeypatch.setattr(m, "target_inactive", lambda: None)
    with pytest.raises(RuntimeError, match="HISTORICAL_REFERENCE_REVIEW_REQUIRED"):
        m.apply(root, 5, m.CONFIRM)
    assert all(p.exists() for p in dirs)


def test_missing_historical_reference_record_fails_closed(tmp_path):
    root, dirs = fake_platform(tmp_path)
    idx = root.parent / "reference_file_list.txt"
    idx.write_text(str(root.parent / "realsas_authority" / "missing.json") + "\n")
    with pytest.raises(RuntimeError, match="RECORD_MISSING_OR_UNTRUSTED"):
        m.plan(root, 5, idx)
