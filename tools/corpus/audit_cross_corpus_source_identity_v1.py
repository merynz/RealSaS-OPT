#!/usr/bin/env python3
"""Strict read-only cross-corpus identity/potential split-leakage court.

Important: 96-bit prefix matches are *candidates*, NOT verified source SHA.
A source-file argument is verified against its contents and 64-byte SHA.
No source or cohort is merged, deleted, resplit or training admitted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "RealSaS.CrossCorpusSourceIdentityCourt.v1"
PROVIDERS = frozenset(("quaternius_cc0", "kaykit_cc0"))
PROTECTED = frozenset(("DEV", "TUNE", "CAL", "EXTERNAL_HOLDOUT"))
MASTER_SOURCE_SHA = "15e90dbd5ffd470737fb230d9c34138a63b98fcc5057a09450d3239984db20d6"
CORE_SOURCE_SHA = "2c149c72a64f3a57d957b88f31a7becf4215beff46f3330ef537dae0b7d0d307"


def require(flag, code):
    if not flag:
        raise ValueError("CROSS_CORPUS_IDENTITY_" + code)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def dump(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha(path)


def run(master_path, core_path, source_files, output,
        *, expected_master=3993, expected_core=370,
        master_expected_sha=MASTER_SOURCE_SHA, core_expected_sha=CORE_SOURCE_SHA):
    require(sha(master_path) == master_expected_sha, "MASTER_SOURCE_MANIFEST_SHA")
    require(sha(core_path) == core_expected_sha, "CORE_SOURCE_MANIFEST_SHA")
    master = read_rows(master_path)
    core = read_rows(core_path)
    require(len(master) == expected_master and len(core) == expected_core,
            "SOURCE_COUNTS")
    require(len({r["canonical_asset_id"] for r in master}) == len(master)
            and len({r["subject_id"] for r in core}) == len(core),
            "SUBJECT_UNIQUE")
    require(all(r["training_authorized"] == "False" for r in master + core),
            "UNEXPECTED_CURRENT_AUTHORITY")
    require(all(re.fullmatch(r"[a-f0-9]{64}", r["source_sha256"]) for r in core),
            "CORE_SOURCE_SHA_FORMAT")
    core_by_prefix = defaultdict(list)
    for row in core:
        core_by_prefix[row["source_sha256"][:24]].append(row)
    byte_receipts = {}
    for file in source_files:
        file = Path(file)
        token = re.match(r"^([a-f0-9]{24})_", file.name)
        require(token is not None, "SOURCE_FILE_TOKEN_ABSENT")
        digest = sha(file)  # actual bytes, not filename, prove the hash
        require(digest.startswith(token.group(1)), "SOURCE_BYTES_TOKEN_DISAGREE")
        require(token.group(1) not in byte_receipts, "DUPLICATE_BYTE_RECEIPT")
        byte_receipts[token.group(1)] = {
            "sha256": digest, "byte_size": file.stat().st_size}
    pairs = []
    seen_master = set()
    seen_core = set()
    for row in master:
        provider = row["provider"]
        if provider not in PROVIDERS:
            continue
        split = row["historical_split"]
        require(split in {"FIT", "TUNE", "CAL", "DEV", "EXTERNAL_HOLDOUT"},
                "HISTORICAL_SPLIT")
        identity = row["candidate_id"].split(":", 1)
        require(len(identity) == 2 and identity[0] == provider
                and re.fullmatch(r"[a-f0-9]{24}", identity[1]),
                "CANDIDATE_SHA_TOKEN_FORMAT")
        token = identity[1]
        for hit in core_by_prefix.get(token, []):
            mid, cid = row["canonical_asset_id"], hit["subject_id"]
            require(mid not in seen_master and cid not in seen_core,
                    "AMBIGUOUS_IDENTITY_MANY_TO_MANY")
            seen_master.add(mid)
            seen_core.add(cid)
            receipt = byte_receipts.get(token)
            if receipt:
                require(receipt["sha256"] == hit["source_sha256"],
                        "ACTUAL_SOURCE_SHA_CONTRADICTS_CORE")
            pairs.append({
                "master_asset_id": mid,
                "master_candidate_id": row["candidate_id"],
                "master_provider": provider,
                "master_original_split": split,
                "core_subject_id": cid,
                "core_source_sha256_historical": hit["source_sha256"],
                "core_source_path": hit["source_relative_path"],
                "identity_evidence": "FULL_SHA256_BYTE_VERIFIED" if receipt
                    else "96_BIT_SHA_PREFIX_CANDIDATE_ONLY",
                "real_source_sha256": receipt["sha256"] if receipt else "",
                "priority": "P0_EVALUATION_HOLDOUT_SPLIT_RISK" if split in PROTECTED
                    else "P1_CROSS_CORPUS_SPLIT_UNRECONCILED",
                "disposition": "BLOCK_HEAD_ADMISSION_BOTH_SIDES_PENDING_IDENTITY_COURT",
                "loss_mask_valid": False,
                "training_authorized": False,
                "dedup_finalized": False,
            })
    pairs.sort(key=lambda r: (r["priority"], r["master_original_split"],
                              r["master_asset_id"]))
    require(len(pairs) == len(seen_master) == len(seen_core), "PAIR_UNIQUENESS")
    require(all(x["training_authorized"] is False and
                x["loss_mask_valid"] is False and
                x["dedup_finalized"] is False for x in pairs), "PROMOTION_FORBIDDEN")
    output.mkdir(parents=True, exist_ok=True)
    filename = "CROSS_CORPUS_SOURCE_IDENTITY_CANDIDATES_V1.csv"
    file_hash = dump(output / filename, pairs) if pairs else ""
    splits = Counter(x["master_original_split"] for x in pairs)
    provider = Counter(x["master_provider"] for x in pairs)
    report = {
        "schema": SCHEMA, "status": "REVIEW_REQUIRED_NO_DEDUP_OR_HEAD_ADMISSION",
        "master_count": len(master), "core_count": len(core),
        "matched_sha_prefix_pairs": len(pairs),
        "matched_by_provider": dict(sorted(provider.items())),
        "matched_by_master_split": dict(sorted(splits.items())),
        "evaluation_and_holdout_split_risk_pairs":
            sum(splits[x] for x in PROTECTED),
        "fully_byte_sha_verified_overlap_count": sum(
            x["identity_evidence"] == "FULL_SHA256_BYTE_VERIFIED" for x in pairs),
        "prefix_overlap_requires_full_source_rehash_count": sum(
            x["identity_evidence"] == "96_BIT_SHA_PREFIX_CANDIDATE_ONLY" for x in pairs),
        "cross_format_semantic_identity_done": False,
        "certified_unique_character_count": None,
        "training_authorized": False, "materialization_authorized": False,
        "input_master_sha256": sha(master_path),
        "input_core_sha256": sha(core_path),
        "pair_file_sha256": file_hash,
        "actual_source_file_receipts": byte_receipts,
    }
    (output / "CROSS_CORPUS_SOURCE_IDENTITY_SUMMARY_V1.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ("master", "core", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--source-files", nargs="*", type=Path, default=[])
    a = p.parse_args()
    report = run(a.master, a.core, a.source_files, a.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
