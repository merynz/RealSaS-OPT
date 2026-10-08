import os
import time
from pathlib import Path

from tools.platform_cache_gc import collect


def test_gc_removes_temp_then_expired_without_touching_fresh(tmp_path, monkeypatch):
    old = tmp_path / "old.bin"
    fresh = tmp_path / "fresh.bin"
    temp = tmp_path / "download.part"
    old.write_bytes(b"o" * 10)
    fresh.write_bytes(b"f" * 10)
    temp.write_bytes(b"t" * 10)
    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(fresh, (now, now))
    os.utime(temp, (now, now))
    monkeypatch.setattr("tools.platform_cache_gc.shutil.disk_usage", lambda root: type("U", (), {"free": 10_000})())
    report = collect(tmp_path, soft_max_bytes=1000, target_max_bytes=900,
                     max_age_seconds=100, min_free_bytes=0, dry_run=False, now=now)
    reasons = {Path(row["path"]).name: row["reason"] for row in report["deleted"]}
    assert reasons == {"download.part": "TEMPORARY_OR_INCOMPLETE", "old.bin": "MAX_AGE_EXPIRED"}
    assert fresh.exists()


def test_gc_lru_hits_target_watermark(tmp_path, monkeypatch):
    now = time.time()
    for idx in range(3):
        path = tmp_path / f"{idx}.bin"
        path.write_bytes(b"x" * 10)
        os.utime(path, (now + idx, now + idx))
    monkeypatch.setattr("tools.platform_cache_gc.shutil.disk_usage", lambda root: type("U", (), {"free": 10_000})())
    report = collect(tmp_path, soft_max_bytes=20, target_max_bytes=10,
                     max_age_seconds=10_000, min_free_bytes=0, dry_run=False, now=now + 5)
    assert report["after_bytes_estimate"] <= 10
    assert [Path(row["path"]).name for row in report["deleted"]] == ["0.bin", "1.bin"]


def test_gc_dry_run_reports_without_deleting(tmp_path, monkeypatch):
    path = tmp_path / "x.part"
    path.write_bytes(b"x")
    monkeypatch.setattr("tools.platform_cache_gc.shutil.disk_usage", lambda root: type("U", (), {"free": 10_000})())
    report = collect(tmp_path, soft_max_bytes=100, target_max_bytes=50,
                     max_age_seconds=100, min_free_bytes=0, dry_run=True)
    assert report["deleted_count"] == 1
    assert path.exists()
