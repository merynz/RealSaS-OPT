import os
import time
from pathlib import Path

from tools.platform_cache_gc import collect


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
