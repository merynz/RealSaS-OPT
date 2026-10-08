import hashlib

import pytest

from tools.platform_artifact_store import cas_path, hydrate_inventory, validate_inventory


def inventory(data=b"exact"):
    return {
        "schema": "RealSaS.ExternalInputInventory.v1",
        "subject": "TEST",
        "required_files": [{
            "name": "evidence.bin",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "source": {"provider": "google_drive", "file_id": "drive-id"},
        }],
    }


def test_hydration_verifies_bytes_installs_cas_and_reuses(tmp_path):
    calls = []

    def fetcher(*, file_id, destination, remote):
        calls.append((file_id, remote))
        destination.write_bytes(b"exact")

    manifest = inventory()
    first = hydrate_inventory(manifest, cas_root=tmp_path, fetcher=fetcher, rclone_remote="drive")
    assert first["all_input_bytes_verified"]
    assert first["files"][0]["status"] == "HYDRATED"
    assert not first["qualification_minted"]
    assert not first["stage_cache_hit_minted"]
    second = hydrate_inventory(manifest, cas_root=tmp_path, fetcher=fetcher, rclone_remote="drive")
    assert second["files"][0]["status"] == "CAS_HIT"
    assert calls == [("drive-id", "drive")]


def test_wrong_remote_bytes_fail_closed_and_are_not_installed(tmp_path):
    def fetcher(*, file_id, destination, remote):
        destination.write_bytes(b"wrong")

    manifest = inventory()
    with pytest.raises(RuntimeError, match="HYDRATED_BYTES_INVALID"):
        hydrate_inventory(manifest, cas_root=tmp_path, fetcher=fetcher)
    assert not cas_path(tmp_path, manifest["required_files"][0]["sha256"]).exists()


def test_offline_miss_does_not_fetch(tmp_path):
    def fetcher(**kwargs):
        raise AssertionError("must not fetch")

    report = hydrate_inventory(inventory(), cas_root=tmp_path, fetcher=fetcher, allow_remote=False)
    assert not report["all_input_bytes_verified"]
    assert report["files"][0]["status"] == "MISS_REMOTE_DISABLED"


def test_inventory_requires_stable_google_drive_locator():
    bad = inventory()
    bad["required_files"][0]["source"] = {"provider": "google_drive", "file_id": ""}
    with pytest.raises(ValueError, match="SOURCE_FILE_ID_INVALID"):
        validate_inventory(bad)


def test_non_regular_cas_object_fails_closed(tmp_path):
    manifest = inventory()
    target = cas_path(tmp_path, manifest["required_files"][0]["sha256"])
    target.parent.mkdir(parents=True)
    real = tmp_path / "elsewhere.bin"
    real.write_bytes(b"exact")
    target.symlink_to(real)

    def fetcher(**kwargs):
        raise AssertionError("must not fetch over host drift")

    with pytest.raises(RuntimeError, match="CAS_NON_REGULAR_OBJECT"):
        hydrate_inventory(manifest, cas_root=tmp_path, fetcher=fetcher)
    assert target.is_symlink()
