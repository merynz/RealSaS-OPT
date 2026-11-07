"""Synthetic identity-court protections. No private source files in CI."""
import csv
import hashlib
from pathlib import Path

import pytest

from tools.corpus.audit_cross_corpus_source_identity_v1 import run


def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as file:
        w = csv.DictWriter(file, data[0].keys())
        w.writeheader()
        w.writerows(data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case(tmp_path, split="EXTERNAL_HOLDOUT", duplicate_core=False):
    content = b"fake-char-payload"  # Synthetic only.
    source_sha = hashlib.sha256(content).hexdigest()
    token = source_sha[:24]
    master = tmp_path / "master.csv"
    core = tmp_path / "core.csv"
    source = tmp_path / (token + "_source.fbx")
    source.write_bytes(content)
    write_csv(master, [{
        "canonical_asset_id": "asset_1",
        "candidate_id": "quaternius_cc0:" + token,
        "provider": "quaternius_cc0",
        "historical_split": split,
        "training_authorized": "False",
    }])
    core_data = [{
        "subject_id": "core_1",
        "source_sha256": source_sha,
        "source_relative_path": "original/source.fbx",
        "training_authorized": "False",
    }]
    if duplicate_core:
        core_data.append(dict(core_data[0], subject_id="core_2"))
    write_csv(core, core_data)
    return master, core, source


def check(tmp_path, master, core, files=(), count_core=1):
    return run(
        master, core, files, tmp_path / "out",
        expected_master=1, expected_core=count_core,
        master_expected_sha=digest(master), core_expected_sha=digest(core)
    )


def test_exact_source_sha_proves_one_candidate_and_masks_holdout(tmp_path):
    m, c, source = case(tmp_path)
    report = check(tmp_path, m, c, [source])
    assert report["matched_sha_prefix_pairs"] == 1
    assert report["fully_byte_sha_verified_overlap_count"] == 1
    assert report["evaluation_and_holdout_split_risk_pairs"] == 1
    assert report["training_authorized"] is False
    out = (tmp_path / "out" / "CROSS_CORPUS_SOURCE_IDENTITY_CANDIDATES_V1.csv")
    with out.open(newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["identity_evidence"] == "FULL_SHA256_BYTE_VERIFIED"
    assert row["priority"] == "P0_EVALUATION_HOLDOUT_SPLIT_RISK"
    assert row["loss_mask_valid"] == "False"
    assert row["training_authorized"] == "False"
    assert row["dedup_finalized"] == "False"


def test_prefix_alone_is_never_fully_verified(tmp_path):
    m, c, _ = case(tmp_path, split="FIT")
    report = check(tmp_path, m, c)
    assert report["matched_sha_prefix_pairs"] == 1
    assert report["fully_byte_sha_verified_overlap_count"] == 0
    assert report["prefix_overlap_requires_full_source_rehash_count"] == 1


def test_source_byte_tamper_must_fail(tmp_path):
    m, c, original = case(tmp_path)
    original.write_bytes(b"tampered" + original.read_bytes())
    with pytest.raises(ValueError, match="SOURCE_BYTES_TOKEN_DISAGREE"):
        check(tmp_path, m, c, [original])


def test_source_manifest_tamper_must_fail(tmp_path):
    m, c, source = case(tmp_path)
    pin = digest(m)
    m.write_text(m.read_text().replace("EXTERNAL_HOLDOUT", "FIT"))
    with pytest.raises(ValueError, match="MASTER_SOURCE_MANIFEST_SHA"):
        run(m, c, [source], tmp_path / "out",
            expected_master=1, expected_core=1,
            master_expected_sha=pin, core_expected_sha=digest(c))


def test_duplicate_core_sha_cannot_collapse_many_subjects(tmp_path):
    m, c, _ = case(tmp_path, duplicate_core=True)
    with pytest.raises(ValueError, match="AMBIGUOUS_IDENTITY_MANY_TO_MANY"):
        check(tmp_path, m, c, count_core=2)


def test_training_authority_must_fail(tmp_path):
    m, c, _ = case(tmp_path)
    m.write_text(m.read_text().replace("False", "True"))
    with pytest.raises(ValueError, match="UNEXPECTED_CURRENT_AUTHORITY"):
        check(tmp_path, m, c)
