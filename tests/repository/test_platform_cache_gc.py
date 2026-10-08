import os
import time
import fcntl
import subprocess
import sys
from pathlib import Path

from tools.platform_cache_gc import collect
from tools.platform_cache_lock import cache_lease


def _venv(root, name, payload, mtime):
    path = root / "venvs" / name
    path.mkdir(parents=True)
    (path / "payload.bin").write_bytes(payload)
    stamp = path / ".realsas-env.json"
    stamp.write_bytes(b"")
    os.utime(stamp, (mtime, mtime))
    return path


def _high_free(monkeypatch):
    monkeypatch.setattr(
        "tools.platform_cache_gc.shutil.disk_usage",
        lambda root: type("U", (), {"free": 10_000})(),
    )


def test_gc_removes_temp_and_expired_venv_without_touching_fresh(tmp_path, monkeypatch):
    now = time.time()
    old = _venv(tmp_path, "old", b"o" * 10, now - 1000)
    fresh = _venv(tmp_path, "fresh", b"f" * 10, now)
    temp_root = tmp_path / ".tmp"
    temp_root.mkdir()
    temp = temp_root / "download.part"
    temp.write_bytes(b"t" * 10)
    _high_free(monkeypatch)

    report = collect(
        tmp_path,
        soft_max_bytes=1000,
        target_max_bytes=900,
        max_age_seconds=100,
        min_free_bytes=0,
        dry_run=False,
        now=now,
    )
    reasons = {Path(row["path"]).name: row["reason"] for row in report["deleted"]}
    assert reasons == {
        "download.part": "TEMPORARY_OR_INCOMPLETE",
        "old": "VENV_MAX_AGE_EXPIRED",
    }
    assert not old.exists()
    assert fresh.exists()
    assert report["schema"] == "RealSaS.PlatformCacheGC.v2"


def test_gc_lru_evicts_whole_venvs_to_target_watermark(tmp_path, monkeypatch):
    now = time.time()
    for idx in range(3):
        _venv(tmp_path, str(idx), b"x" * 10, now + idx)
    _high_free(monkeypatch)

    report = collect(
        tmp_path,
        soft_max_bytes=20,
        target_max_bytes=10,
        max_age_seconds=10_000,
        min_free_bytes=0,
        dry_run=False,
        now=now + 5,
    )
    assert report["after_bytes_estimate"] <= 10
    assert [Path(row["path"]).name for row in report["deleted"]] == ["0", "1"]
    assert all(row["kind"] == "venv" for row in report["deleted"])
    assert (tmp_path / "venvs" / "2").is_dir()


def test_gc_dry_run_reports_without_deleting(tmp_path, monkeypatch):
    temp_root = tmp_path / ".tmp"
    temp_root.mkdir()
    path = temp_root / "x.part"
    path.write_bytes(b"x")
    _high_free(monkeypatch)

    report = collect(
        tmp_path,
        soft_max_bytes=100,
        target_max_bytes=50,
        max_age_seconds=100,
        min_free_bytes=0,
        dry_run=True,
    )
    assert report["deleted_count"] == 1
    assert path.exists()
    assert report["after_bytes_actual"] is None


def test_gc_evicts_go_module_namespace_atomically_under_pressure(tmp_path, monkeypatch):
    module_file = tmp_path / "go-mod" / "github.com" / "example" / "mod@v1" / "source.go"
    module_file.parent.mkdir(parents=True)
    module_file.write_bytes(b"x" * 32)
    _high_free(monkeypatch)

    report = collect(
        tmp_path,
        soft_max_bytes=16,
        target_max_bytes=0,
        max_age_seconds=10_000,
        min_free_bytes=0,
        dry_run=False,
    )
    assert not (tmp_path / "go-mod").exists()
    assert report["deleted_count"] == 1
    assert report["deleted"][0]["kind"] == "namespace:go-mod"
    assert report["deleted"][0]["reason"] == "LRU_OR_DISK_PRESSURE"


def test_gc_does_not_delete_live_cache_namespaces(tmp_path, monkeypatch):
    _high_free(monkeypatch)
    path = tmp_path / "pip" / "active.whl"
    path.parent.mkdir()
    path.write_bytes(b"active")
    with cache_lease(tmp_path):
        report = collect(tmp_path, soft_max_bytes=0, target_max_bytes=0,
                         max_age_seconds=0, min_free_bytes=0, dry_run=False)
    assert report["status"] == "LIVE_CACHE_LEASE_HELD"
    assert not report["safe_to_execute"]
    assert path.read_bytes() == b"active"


def test_gc_holds_venv_lock_until_deletion_finishes(tmp_path, monkeypatch):
    _high_free(monkeypatch)
    path = _venv(tmp_path, "old", b"payload", 0)
    from tools import platform_cache_gc as gc
    remove = gc._remove_tree

    def remove_with_contending_process(target):
        script = '''import fcntl, sys
with open(sys.argv[1], 'a+b') as handle:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(23)
'''
        result = subprocess.run([sys.executable, "-c", script, str(path)+".lock"])
        assert result.returncode == 23
        remove(target)

    monkeypatch.setattr(gc, "_remove_tree", remove_with_contending_process)
    report = collect(tmp_path, soft_max_bytes=100, target_max_bytes=50,
                     max_age_seconds=1, min_free_bytes=0, dry_run=False, now=100)
    assert report["deleted_count"] == 1
    assert not path.exists()


def test_gc_retains_environment_with_legacy_live_job_lock(tmp_path, monkeypatch):
    _high_free(monkeypatch)
    path = _venv(tmp_path, "active", b"payload", 0)
    with Path(str(path)+".lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        report = collect(tmp_path, soft_max_bytes=0, target_max_bytes=0,
                         max_age_seconds=0, min_free_bytes=0, dry_run=False, now=100)
    assert path.exists()
    assert report["deleted_count"] == 0
    assert not report["safe_to_execute"]
