from __future__ import annotations

import json
from pathlib import Path

from tools.corpus.launch_full139_truth_family_sweep_drive_v1 import discover, compatible_pairs


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(root: Path, suffix: str = "a"):
    subjects = {}
    cache_rows = []
    required = {
        "geometry": {"path": "/x", "sha256": "1" * 64},
        "geometry_court": {"path": "/x", "sha256": "2" * 64},
        "rgb": {"path": "/x", "sha256": "3" * 64},
        "rgb_court": {"path": "/x", "sha256": "4" * 64},
        "stage48": {"path": "/x", "sha256": "5" * 64},
        "tessa_teacher": {"path": "/x", "sha256": "6" * 64},
        "tessa_court": {"path": "/x", "sha256": "7" * 64},
    }
    for i in range(139):
        sid = f"truth139_{i:020x}"
        subjects[sid] = dict(required)
        cache_rows.append({"subject_id": sid, "valid": True, "corpus_relative_path": f"raw/{sid}.gltf"})
    rp = root / suffix / "V32_TRAINING_READINESS_RECEIPT.json"
    cp = root / suffix / "FULL_SOURCE_CACHE_MANIFEST.json"
    write(rp, {"schema": "RealSaS.Truth139V32PretrainTruthLedger.v2", "status": "PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY", "subject_count": 139, "subjects": subjects, "training_authorized": True})
    write(cp, {"schema": "RealSaS.Truth139FullSourceCache.v1", "status": "PASS_TRUTH139_FULL139_SOURCE_BYTES_CACHED__TRUTH_NOT_PERFORMED", "subject_count": 139, "source_failures": {}, "subjects": cache_rows})
    return rp, cp


def test_drive_discovery_finds_exactly_one_compatible_pair(tmp_path: Path):
    fixture(tmp_path)
    readiness, caches = discover(tmp_path)
    pairs = compatible_pairs(readiness, caches)
    assert len(readiness) == 1
    assert len(caches) == 1
    assert len(pairs) == 1


def test_drive_discovery_exposes_ambiguity_instead_of_picking_latest(tmp_path: Path):
    fixture(tmp_path, "a")
    fixture(tmp_path, "b")
    readiness, caches = discover(tmp_path)
    pairs = compatible_pairs(readiness, caches)
    assert len(readiness) == 2
    assert len(caches) == 2
    assert len(pairs) == 4


def test_invalid_historical_receipt_is_not_a_candidate(tmp_path: Path):
    fixture(tmp_path, "good")
    bad = tmp_path / "bad" / "V32_TRAINING_READINESS_RECEIPT.json"
    write(bad, {"schema": "RealSaS.Truth139V32PretrainTruthLedger.v2", "status": "PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY", "subject_count": 138, "subjects": {}})
    readiness, caches = discover(tmp_path)
    assert len(readiness) == 1
    assert len(caches) == 1
