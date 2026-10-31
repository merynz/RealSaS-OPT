#!/usr/bin/env python3
from __future__ import annotations

"""Failure-tolerant entrypoint for Full139 family sweep V1 execution.

V1 contains the full subject orchestration.  V2 replaces only aggregate behavior so
`--continue-on-error` can preserve failure rows without trying to read family
candidates from them.  This is the canonical execution entrypoint.
"""

from tools.corpus import run_full139_truth_family_sweep_v1 as impl


def aggregate(receipts: dict[str, dict], *, historical_ref: dict, source_cache_ref: dict, code_sha: str) -> dict:
    family_counts: dict[str, dict[str, int]] = {}
    failures = {}
    completed = {}
    for subject, receipt in receipts.items():
        status = str(receipt.get("status") or "")
        if status.startswith("FAIL_"):
            failures[subject] = receipt
            continue
        completed[subject] = receipt
        for family, family_status in dict(receipt.get("canonical_family_candidates") or {}).items():
            by_status = family_counts.setdefault(family, {})
            by_status[family_status] = by_status.get(family_status, 0) + 1
    return {
        "schema": "RealSaS.Full139TruthFamilySweepReceipt.v2",
        "status": (
            "PASS_FULL139_TRUTH_FAMILY_SWEEP_MEASURED__FINAL_ADMISSION_PENDING"
            if not failures and len(receipts) == impl.EXPECTED_SUBJECTS
            else "INCOMPLETE_FULL139_TRUTH_FAMILY_SWEEP"
        ),
        "code_commit_sha": code_sha,
        "subject_count": len(receipts),
        "completed_subject_count": len(completed),
        "failed_subject_count": len(failures),
        "historical_readiness_inventory": historical_ref,
        "source_cache_inventory": source_cache_ref,
        "family_candidate_counts": family_counts,
        "failures": failures,
        "authority": {
            "historical_split_inherited": False,
            "historical_training_authorization_inherited": False,
            "final_subject_disposition_minted": False,
            "final_split_minted": False,
            "training_authorized": False,
        },
        "pending_global_gates": [
            "ANIMATION_CANONICAL_REPLAY_COURT",
            "GRAPHICS_STUDENT_OBSERVABILITY_ADMISSION_AFTER_FINAL_ADMITTED_CORPUS",
            "FINAL_ADMIT_CORE_MASKED_QUARANTINE_DROP",
            "DUPLICATE_AND_LINEAGE_FAMILY_REBUILD",
            "FINAL_TRAIN_UNSEEN_SPLIT",
            "TRAINING_READINESS_RECEIPT",
        ],
    }


def main() -> None:
    impl.aggregate = aggregate
    impl.main()


if __name__ == "__main__":
    main()
