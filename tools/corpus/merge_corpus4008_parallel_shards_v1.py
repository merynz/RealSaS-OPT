#!/usr/bin/env python3
"""Global SHA/identity/split ledger from 32 independently audited shard CSVs.

This stage validates every shard output against SHA-pinned canonical Master
and rejects cross-shard duplicate origins. The evidence is imported source
observation only; no semantic/Graphics/IK truth or training PASS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.corpus.audit_corpus4008_parallel_shard_v1 import (
    SCHEMA as SHARD_SCHEMA,
    file_hash,
    read_csv,
    require,
)
# Pinned to the source-availability Master3993 SHA, rechecked on readback.
MASTER_SELECTION_SHA = "af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"

SCHEMA = "RealSaS.Corpus4008Parallel1024ImportedSceneMasterLedger.v1"
HEADS = ("IRIS_GEOMETRY", "IRIS_SEMANTIC", "IRIS_GRAPHICS",
         "TESSA", "AXIS", "MIRA")
OBSERVED_STATUS = "IMPORTED_SOURCE_OBSERVATIONS_ONLY"


def val(x):
    if x in (None, "", "None"):
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        raise ValueError("C4008_PARALLEL_GLOBAL_INVALID_COUNT")


def merge(selection: Path, results_root: Path, output: Path, *,
          expected_master_count=3993,
          expected_master_sha=MASTER_SELECTION_SHA):
    require(selection.is_file()
            and file_hash(selection) == expected_master_sha,
            "MASTER_SELECTION_SHA")
    master = json.loads(selection.read_text())
    require(isinstance(master, dict)
            and len(master) == expected_master_count,
            "MASTER_SELECTION_CARDINALITY")
    unique_ids = set()
    unique_sources = set()
    status_counts = Counter()
    imported_family_candidates = Counter()
    observed_features = Counter()
    formats = Counter()
    rows = []
    shard_receipts = []
    for shard in range(32):
        base = results_root / f"shard-{shard}"
        summary_file = base / "SHARD_READBACK_SUMMARY.json"
        csv_file = base / "SHARD_TYPED_OBSERVATIONS.csv"
        require(summary_file.is_file()
                and csv_file.is_file(),
                "SHARD_NOT_READY_" + str(shard))
        summary = json.loads(summary_file.read_text())
        require(summary.get("schema") == SHARD_SCHEMA
                and summary.get("shard") == shard
                and summary.get("training_authorized") is False
                and summary.get("per_head_training_admitted_count") == 0
                and summary.get("source_original_bytes_refetched") is False
                and summary.get("full_import_receipts_sha_verified") is True,
                "SHARD_COURT_AUTHORITY")
        require(file_hash(csv_file)
                == summary.get("typed_observations_csv_sha256"),
                "SHARD_CSV_DRIFT")
        current = read_csv(csv_file)
        require(len(current)
                == summary.get("total_unique_source_count")
                == summary.get("prior_count") +
                summary.get("additional_wave_count"),
                "SHARD_ROW_NUMBER_DRIFT")
        require(summary.get("prior_count") == 8 and
                1 <= summary.get("additional_wave_count", 0) <= 24,
                "SHARD_WAVE_SIZE")
        local_shas = {r["source_sha256"] for r in current}
        local_ids = {r["canonical_asset_id"] for r in current}
        require(len(local_shas) == len(local_ids) == len(current)
                and sorted(local_shas) == summary.get("original_sha256_list")
                and sorted(local_ids) == summary.get("asset_id_list"),
                "SHARD_IDENTITY_DRIFT")
        local_status = Counter()
        for r in current:
            aid = r["canonical_asset_id"]
            source = r["source_sha256"]
            require(aid not in unique_ids and
                    source not in unique_sources,
                    "CROSS_SHARD_SOURCE_IDENTITY_DUPLICATE")
            unique_ids.add(aid)
            unique_sources.add(source)
            require(aid in master
                    and master[aid]["candidate_id"] == r["candidate_id"]
                    and master[aid]["split"] == "FIT"
                    and r["split"] == "FIT"
                    and r["source_shard"] == str(shard),
                    "MASTER_LINEAGE_OR_SPLIT_MISMATCH")
            token = r["candidate_id"].split(":", 1)[-1]
            require(len(token) in (24, 64)
                    and source.startswith(token),
                    "SOURCE_SHA_TOKEN_MISMATCH")
            require(r["training_authorized"] == "False"
                    and r["head_admission_count"] == "0"
                    and r["functional_native_ik_fk"] == "UNKNOWN"
                    and r["source_character_vs_prop"] == "UNKNOWN"
                    and r["graphics_source_bound_2d_art_qualified"] == "UNKNOWN",
                    "FALSE_TEACHER_PROMOTION")
            wave = val(r["import_wave"])
            require(wave == (1 if shard % 4 == 0 else 2)
                    or wave == 3,
                    "WRONG_SHARD_WAVE")
            st = r["source_import_observation_status"]
            local_status[st] += 1
            status_counts[st] += 1
            formats[r["source_format"]] += 1
            if st == OBSERVED_STATUS:
                if (val(r["mesh_object_count"]) or 0) > 0:
                    for h in ("IRIS_GEOMETRY","IRIS_SEMANTIC","IRIS_GRAPHICS"):
                        imported_family_candidates[h] += 1
                armature = (val(r["armature_count"]) or 0)
                if armature > 0:
                    for h in ("TESSA","AXIS"):
                        imported_family_candidates[h] += 1
                if armature > 0 and (
                    (val(r["skin_rows_sampled"]) or 0) >
                    (val(r["skin_rows_zero_weights"]) or 0)
                ):
                    imported_family_candidates["MIRA"] += 1
                if (val(r["observed_ik_constraint_count"]) or 0) > 0:
                    observed_features["imported_ik_constraint_sources"] += 1
                if (val(r["observed_action_count"]) or 0) > 0:
                    observed_features["imported_animation_action_sources"] += 1
                if (val(r["image_dependency_missing_count"]) or 0) > 0:
                    observed_features["missing_external_image_sources"] += 1
                if (val(r["mesh_boundary_edges"]) or 0) > 0:
                    observed_features["nonclosed_mesh_sources"] += 1
                observed_features["source_import_success"] += 1
            r["global_original_sha_covered"] = True
            rows.append(r)
        require(dict(sorted(local_status.items())) ==
                summary.get("source_import_status_counts"),
                "SHARD_STATUS_RECOUNT_DRIFT")
        shard_receipts.append({
            "shard": shard,
            "summary_sha256": file_hash(summary_file),
            "csv_sha256": file_hash(csv_file),
            "source_count": len(current),
        })
    require(len(unique_ids) == len(unique_sources) == len(rows)
            and 0 < len(rows) <= 1024,
            "GLOBAL_SOURCE_COUNT")
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "CORPUS4008_1024_SHARD_READBACK_MATRIX_V1.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["canonical_asset_id"]))
    report = {
        "schema": SCHEMA,
        "source_master_selection_sha256": file_hash(selection),
        "total_master_selected_candidates": len(master),
        "unique_source_candidates_measured": len(rows),
        "per_format_source_counts": dict(sorted(formats.items())),
        "import_status_counts": dict(sorted(status_counts.items())),
        "imported_source_features_not_truth":dict(sorted(observed_features.items())),
        "head_observation_candidate_counts_not_training": {
            h:imported_family_candidates[h] for h in HEADS
        },
        "head_training_admitted_counts":{h:0 for h in HEADS},
        "shard_receipts":shard_receipts,
        "typed_source_matrix_sha256":file_hash(dest),
        "source_original_rights_courted":False,
        "source_anatomical_pose_courted":False,
        "source_face_semantics_courted":False,
        "working_IK_FK_courted":False,
        "source_skin_bind_and_deformation_courted":False,
        "graphics_2d_art_courted":False,
        "normalization_vertex_bone_skin_roundtrip_courted":False,
        "materialization_authorized":False,
        "training_authorized":False,
        "claim":"Import coverage/candidate evidence ONLY; no head supervision PASS.",
    }
    (output / "CORPUS4008_1024_SHARD_READBACK_SUMMARY_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection",type=Path,required=True)
    ap.add_argument("--results-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    x=merge(a.selection,a.results_root,a.output)
    print(json.dumps({
        "unique_source_candidates":x["unique_source_candidates_measured"],
        "head_observation_candidates":x["head_observation_candidate_counts_not_training"],
        "per_source_status":x["import_status_counts"],
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
