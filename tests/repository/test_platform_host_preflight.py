import hashlib

from tools.platform_host_preflight import inventory_report


def inventory(data=b"exact"):
    return {"schema": "RealSaS.ExternalInputInventory.v1", "required_files": [
        {"name": "evidence.json", "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}


def test_inventory_hashes_exact_bytes_without_minting_qualification(tmp_path):
    (tmp_path / "evidence.json").write_bytes(b"exact")
    report = inventory_report(tmp_path, inventory())
    assert report["input_bytes_available"]
    assert not report["qualification_minted"]
    assert not report["scientific_pass_claimed"]


def test_wrong_bytes_and_external_symlinks_are_not_inputs(tmp_path):
    root = tmp_path / "authority"
    root.mkdir()
    (root / "evidence.json").write_bytes(b"wrong")
    assert not inventory_report(root, inventory())["input_bytes_available"]
    external = tmp_path / "outside"
    external.mkdir()
    (external / "evidence.json").write_bytes(b"exact")
    (root / "linked").symlink_to(external, target_is_directory=True)
    assert not inventory_report(root, inventory())["input_bytes_available"]


def test_incomplete_scan_is_not_a_missing_or_available_claim(tmp_path):
    (tmp_path / "evidence.json").write_bytes(b"exact")
    report = inventory_report(tmp_path, inventory(), max_files=0)
    assert not report["scan_complete"]
    assert not report["input_bytes_available"]
