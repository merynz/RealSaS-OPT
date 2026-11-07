#!/usr/bin/env python3
"""Strict source-selection coverage matrix over actual imported control receipts.

A missing control receipt is an UNKNOWN, not zero IK/rig/controls. The historical
Master selection and imported observations never mint training readiness.
Only reads local private files. No network, no changing source truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.corpus.validate_corpus4008_deep_diagnostic_v1 import verify_control

SCHEMA = "RealSaS.Corpus4008DeepControlCoverageMatrix.v1"


def require(cond, reason):
    if not cond:
        raise ValueError("CORPUS4008_DEEP_MATRIX_" + reason)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def run(selection, receipts, output, *, expected_count=3993):
    raw = selection.read_bytes()
    data = json.loads(raw)
    require(isinstance(data, dict) and len(data) == expected_count,
            "CANONICAL_SELECTION_UNIVERSE")
    docs = {}
    for file in sorted(receipts.rglob("*.control.json")):
        d = json.loads(file.read_text())
        verify_control(d)
        asset = d.get("asset_id")
        require(isinstance(asset, str) and asset not in docs,
                "DUPLICATE_OR_MALFORMED_CONTROL_ASSET_ID")
        docs[asset] = (d, file_sha(file))
    require(set(docs).issubset(set(data)), "UNSELECTED_CONTROL_PROFILE")
    out = []
    statuses = Counter()
    providers = Counter()
    splits = Counter()
    decisions = Counter()
    for aid, item in sorted(data.items()):
        require(item.get("canonical_asset_id") == aid,
                "MASTER_ASSET_ID_DRIFT")
        candidate = item.get("candidate_id")
        require(isinstance(candidate, str) and ":" in candidate,
                "MASTER_CANDIDATE")
        provider = item.get("source_registry_id")
        require(candidate.startswith(provider + ":"), "PROVIDER_CANDIDATE")
        split = item.get("split")
        require(split in {"FIT", "TUNE", "DEV", "CAL", "EXTERNAL_HOLDOUT"},
                "SOURCE_SPLIT")
        status = "NO_IMPORTED_CONTROL_OBSERVATION"
        ctrl = None
        receipt_sha = None
        if aid in docs:
            d, receipt_sha = docs[aid]
            require(d.get("candidate_id") == candidate, "CANDIDATE_ID_DRIFT")
            actual_sha = d.get("original_sha256", "")
            token = candidate.split(":", 1)[1]
            require(len(token) in {24, 64} and
                    actual_sha.startswith(token), "SOURCE_SHA_DRIFT")
            status = "IMPORTED_CONTROL_OBSERVATION_ONLY"
            ctrl = d["imported_control"]
        statuses[status] += 1
        providers[provider] += 1
        splits[split] += 1
        decisions[item.get("decision")] += 1
        row = {
            "asset_id": aid,
            "provider": provider,
            "candidate_id": candidate,
            "historical_decision_not_truth": item.get("decision"),
            "historical_split": split,
            "holdout_training_excluded": split == "EXTERNAL_HOLDOUT",
            "control_profile_status": status,
            "source_control_receipt_sha256": receipt_sha,
            "source_sha256_verified": (docs[aid][0]["original_sha256"]
                                      if ctrl is not None else None),
            "imported_armature_count": ctrl["imported_armature_count"] if ctrl else None,
            "imported_bone_count": ctrl["imported_pose_bone_count"] if ctrl else None,
            "imported_nondeform_bone_count": (
                ctrl["imported_nondeform_bone_count"] if ctrl else None),
            "imported_ik_constraint_count": ctrl["imported_ik_constraint_count"] if ctrl else None,
            "imported_driver_count": ctrl["imported_driver_count"] if ctrl else None,
            "imported_action_count": ctrl["imported_action_count"] if ctrl else None,
            "imported_action_curve_count": ctrl["imported_action_curve_count"] if ctrl else None,
            "imported_nla_strip_count": ctrl["imported_nla_strip_count"] if ctrl else None,
            "character_vs_prop": "UNKNOWN",
            "morphology": "UNKNOWN",
            "source_ik_functional": "UNKNOWN",
            "source_controller_roles": "UNKNOWN",
            "source_ik_absence": "UNKNOWN",
            "source_rest_pose_parity": "UNKNOWN",
            "source_anatomical_front": "UNKNOWN",
            "source_anatomical_up": "UNKNOWN",
            "source_camera_courted": False,
            "source_rgb_courted": False,
            "stylized_render_courted": False,
            "attachment_response_courted": False,
            "deformation_response_courted": False,
            "semantic_truth_courted": False,
            "axis_target_admitted": False,
            "mira_target_admitted": False,
            "tessa_target_admitted": False,
            "iris_target_admitted": False,
            "training_authorized": False,
        }
        out.append(row)
    require(len(out) == expected_count, "OUTPUT_COUNT")
    output.mkdir(parents=True, exist_ok=True)
    matrix = {
        "schema": SCHEMA,
        "status": "EXACT_SELECTION_CONTROL_COVERAGE_NOT_TRAINING",
        "selection_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_master_count": expected_count,
        "excluded_qa_only_outside_this_selection": 15 if expected_count == 3993 else None,
        "control_receipt_status_counts": dict(sorted(statuses.items())),
        "provider_counts": dict(sorted(providers.items())),
        "historical_split_counts": dict(sorted(splits.items())),
        "historical_decision_counts": dict(sorted(decisions.items())),
        "all_unmeasured_control_values_are_null_not_zero": True,
        "all_new_family_admissions_closed": True,
        "training_authorized": False,
        "rows": out,
    }
    (output / "CORPUS4008_DEEP_CONTROL_MATRIX_V1.json").write_text(
        json.dumps(matrix, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with (output / "CORPUS4008_DEEP_CONTROL_MATRIX_V1.csv").open(
            "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return {k: matrix[k] for k in (
        "schema", "selected_master_count", "control_receipt_status_counts",
        "provider_counts", "historical_split_counts", "training_authorized",
    )}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--control-receipts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.selection, args.control_receipts, args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
