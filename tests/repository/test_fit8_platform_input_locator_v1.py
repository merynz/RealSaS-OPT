from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_platform_bundle_locator_binds_current_fit8_selection():
    locator = json.loads((ROOT / "canonical/FIT8_PLATFORM_STAGE01_INPUT_BUNDLE_20261013.json").read_text())
    selection = ROOT / locator["selection"]["path"]
    assert locator["schema"] == "RealSaS.Fit8PlatformStage01InputBundleLocator.v1"
    assert locator["subject_count"] == 8
    assert locator["selection"]["sha256"] == sha(selection)
    assert len(locator["drive"]["sha256"]) == 64
    assert locator["drive"]["size_bytes"] > 0
    assert locator["drive"]["upload_roundtrip_sha256_verified"] is True
    assert locator["authority"]["fit8_pass_claimed"] is False
    assert locator["authority"]["product_pass_claimed"] is False
