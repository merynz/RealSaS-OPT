#!/usr/bin/env python3
"""Exact private source-hit oracle-consumer replay on sealed 3-subject pilot.

Restores no authoring/part names to student inputs. All per-pixel source and
barycentric addresses remain private teacher evidence. Replays independent V1,
V2 and visibility courts and demonstrates bounded hidden-edit preservation.
Does NOT mint same/different drawable, mechanical ownership or training truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.court_corpus370_visibility_oracle_consumer_v1 import consume

BATCH_SCHEMA = "RealSaS.Corpus370SourceHitOracleConsumerBatch.v1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok: bool, reason: str):
    if not ok:
        raise ValueError(reason)


def run(source_root: Path, visibility_root: Path, output_root: Path,
        expected_summary_sha: str):
    src_summary = source_root/"CORPUS370_V5_PALETTE_INTERVENTION_PILOT_V1.json"
    vis_summary = visibility_root/"CORPUS370_HOSTED_SOURCE_HIT_VISIBILITY_BATCH_V1.json"
    require(sha(src_summary) == expected_summary_sha,
            "ORACLE_BATCH_SOURCE_SUMMARY_HASH_DRIFT")
    source = json.loads(src_summary.read_text())
    vis = json.loads(vis_summary.read_text())
    require(source.get("schema") == "RealSaS.Corpus370V5SourcePaletteInterventionPilot.v1"
            and source.get("training_authorized") is False
            and source.get("graphics_family_authority_minted") is False
            and source.get("failure_count") == 0,
            "ORACLE_BATCH_SOURCE_SCOPE")
    require(vis.get("schema") == "RealSaS.Corpus370BoundedSourceHitVisibilityBatchCourt.v1"
            and vis.get("training_authorized") is False
            and vis.get("graphics_family_authority_minted") is False
            and vis.get("parent_summary_sha256") == expected_summary_sha,
            "ORACLE_BATCH_VISIBILITY_SCOPE")
    ids = source.get("exact_subject_ids") or []
    require(len(ids) == len(set(ids)) == 3
            and set(ids) == set(vis["subjects"]) == set(source["subjects"]),
            "ORACLE_BATCH_SUBJECT_SET_DRIFT")
    counts = {"pixels": 0, "recoverable": 0, "abstain": 0}
    max_replay_error = 0.0
    subjects = {}
    for sid in ids:
        src = source_root/"subjects"/sid/"SOURCE_PALETTE_INTERVENTION_V1.json"
        visibility = visibility_root/"subjects"/sid/"SOURCE_HIT_VISIBILITY_CANDIDATE_COURT_V1.json"
        require(src.is_file() and visibility.is_file(),
                "ORACLE_BATCH_PRIVATE_SUBJECT_PARENT_MISSING")
        require(sha(visibility) == vis["subjects"][sid]["receipt_sha256"],
                "ORACLE_BATCH_VISIBILITY_RECEIPT_HASH_DRIFT")
        render = json.loads(src.read_text())
        qualified = json.loads(visibility.read_text())
        require(qualified.get("parents", {}).get("source_manifest_sha256") == sha(src)
                and qualified.get("parents", {}).get("source_pilot_summary_sha256") == expected_summary_sha,
                "ORACLE_BATCH_RENDER_PARENT_HASH_DRIFT")
        require(render.get("subject_id") == sid
                and render.get("source_sha256") == source["subjects"][sid]["source_sha256"],
                "ORACLE_BATCH_SUBJECT_IDENTITY_DRIFT")
        product = consume(render, qualified)
        require(product["source_sha256"] == render["source_sha256"]
                and product["graphics_owner_truth_minted"] is False
                and product["training_authorized"] is False
                and product["qualified_same_drawable_positive_pairs"] == 0
                and product["qualified_different_drawable_negative_pairs"] == 0,
                "ORACLE_BATCH_UNQUALIFIED_OWNER_PROMOTION")
        product["parents"] = {
            "source_renderer_manifest_sha256": sha(src),
            "source_hit_visibility_receipt_sha256": sha(visibility),
            "source_pilot_summary_sha256": expected_summary_sha,
        }
        dest = output_root/"subjects"/sid/"SOURCE_HIT_ORACLE_CONSUMER_V1.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(product, sort_keys=True, indent=2,
                                   allow_nan=False)+"\n")
        counts["pixels"] += product["evaluated_pixel_count"]
        counts["recoverable"] += product["occluded_edit_sources_recoverable"]
        counts["abstain"] += product["abstain_pixel_count"]
        max_replay_error = max(max_replay_error,
                               product["max_full_rgba_replay_error"])
        subjects[sid] = {
            "receipt_sha256": sha(dest),
            "pixels": product["evaluated_pixel_count"],
            "recovered_occluded_edit_targets": product["occluded_edit_sources_recoverable"],
            "abstain": product["abstain_pixel_count"],
            "owner_truth_minted": False,
            "training_authorized": False,
        }
    require(counts["pixels"] == 6
            and counts["recoverable"] + counts["abstain"] == counts["pixels"],
            "ORACLE_BATCH_COMPLETE_TWO_VIEW_PIXEL_CENSUS")
    report = {
        "schema": BATCH_SCHEMA,
        "status": "BOUNDED_PRIVATE_SOURCE_VISIBILITY_ORACLE__DRAWABLE_OWNER_UNKNOWN",
        "subject_count": 3,
        "source_pilot_summary_sha256": expected_summary_sha,
        "visibility_batch_sha256": sha(vis_summary),
        "evaluated_pixel_count": counts["pixels"],
        "recoverable_occluded_edit_sources": counts["recoverable"],
        "abstain_pixel_count": counts["abstain"],
        "max_full_rgba_replay_error": max_replay_error,
        "first_hit_only_baseline_potential_hidden_source_losses": counts["recoverable"],
        "qualified_same_drawable_positive_pairs": 0,
        "qualified_different_drawable_negative_pairs": 0,
        "teacher_oracle_only": True,
        "graphics_owner_truth_minted": False,
        "training_authorized": False,
        "subjects": subjects,
    }
    out = output_root/"CORPUS370_SOURCE_HIT_ORACLE_CONSUMER_BATCH_V1.json"
    output_root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2,
                              allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="subjects"},
                     sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--visibility-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    args = parser.parse_args()
    run(args.source_root, args.visibility_root, args.output_root,
        args.expected_source_summary_sha256)


if __name__ == "__main__":
    main()
