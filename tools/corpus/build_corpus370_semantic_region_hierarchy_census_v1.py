#!/usr/bin/env python3
"""Independent readback of real 370 source groups into parent semantic domains.

Consumes SHA-sealed V1 PRIVATE teacher candidate receipts. No Blender, source
import, color invention, or model training. Parent domains are explicitly
not artist components, drawable IDs, or qualified semantic labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from models.iris.truth.semantic_region_hierarchy_v1 import build

SCHEMA = "RealSaS.Corpus370SemanticParentRegionCensus.v1"
SOURCE_INDEX_SCHEMA = "RealSaS.Corpus370SemanticSourceGroupSweep.v1"
EXPECTED_SOURCE_INDEX_SHA = "f06db9a6214b836e1c2eda3240ad614a23df689f0ca69841cf4b07c01c92505a"


def _sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError("CORPUS370_SEMANTIC_PARENT_CENSUS_" + code)


def run(*, source_index: Path, source_receipts: Path, prior_policy: Path,
        output: Path) -> dict:
    _require(source_index.is_file() and _sha(source_index) == EXPECTED_SOURCE_INDEX_SHA,
             "PINNED_REAL_370_SOURCE_INDEX_REQUIRED")
    index = json.loads(source_index.read_text())
    policy = json.loads(prior_policy.read_text())
    _require(index.get("schema") == SOURCE_INDEX_SCHEMA
             and index.get("subject_count") == 370
             and index.get("attempted_group_subject_count") == 352
             and index.get("status_counts") ==
                 {"MEASURED_SOURCE_GROUP_CANDIDATE_ONLY": 352,
                  "OBSERVATION_EXCLUDED": 18}
             and index.get("training_authorized") is False,
             "REAL_370_CENSUS_STATUS")
    records = index["subjects"]
    _require(len(records) == 370 and len({r["subject_id"] for r in records}) == 370,
             "SUBJECT_UNIVERSE")
    outcome=Counter()
    concept=Counter()
    hints=Counter()
    all_islands=all_parents=all_faces=0
    rows=[]
    for i,row in enumerate(records,1):
        sid=row["subject_id"]
        if row["status"]=="OBSERVATION_EXCLUDED":
            out={"subject_id":sid,"source_sha256":row["source_sha256"],
                 "status":"OBSERVATION_EXCLUDED", "parent_region_count":0}
        else:
            _require(row["status"]=="MEASURED_SOURCE_GROUP_CANDIDATE_ONLY",
                     "UNEXPECTED_SUBJECT_STATUS")
            source_path=source_receipts/"subjects"/sid/"SEMANTIC_SOURCE_GROUPS_CANDIDATE_V1.json"
            _require(source_path.is_file() and
                     _sha(source_path)==row["private_receipt"]["sha256"],
                     "PRIVATE_SOURCE_PARENT_BYTES_DRIFT:"+sid)
            source=json.loads(source_path.read_text())
            _require(source.get("subject_id")==sid
                     and source.get("source_sha256")==row["source_sha256"]
                     and source.get("source_face_count")==row["source_face_count"]
                     and source.get("group_count")==row["group_count"],
                     "SOURCE_CANDIDATE_IDENTITY:"+sid)
            parent=build(source_groups=source,semantic_policy=policy)
            _require(parent["source_face_count"]==source["source_face_count"]
                     and parent["child_island_count"]==source["group_count"]
                     and parent["training_authorized"] is False,
                     "PARENT_REGION_COURT")
            p=output/"subjects"/sid/"SEMANTIC_PARENT_REGION_CANDIDATE_V1.json"
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(parent,sort_keys=True,indent=2,allow_nan=False)+"\n")
            out={"subject_id":sid,"source_sha256":row["source_sha256"],
                 "status":"PARENT_REGION_CANDIDATE_ONLY",
                 "source_island_count":parent["child_island_count"],
                 "parent_region_count":parent["parent_region_count"],
                 "source_face_count":parent["source_face_count"],
                 "private_receipt_sha256":_sha(p)}
            all_islands+=parent["child_island_count"]
            all_faces+=parent["source_face_count"]
            all_parents+=parent["parent_region_count"]
            hints.update(parent["hint_status_counts_per_parent_region"])
            for item in parent["regions"]:
                concept[item["semantic_concept_hint"]]+=1
        rows.append(out)
        outcome[out["status"]]+=1
        if i%30==0 or i==370:
            print(json.dumps({"done":i,"total":370,"states":dict(outcome)},
                             sort_keys=True),flush=True)
    _require(outcome["PARENT_REGION_CANDIDATE_ONLY"]==352
             and outcome["OBSERVATION_EXCLUDED"]==18
             and all_islands==151477 and all_faces==1419315,
             "REAL_V1_CHILD_FACE_ACCOUNTING")
    report={
        "schema":SCHEMA,
        "status":"ALL_370_SEMANTIC_REGION_HIERARCHY_CANDIDATES_ACCOUNTED__NO_TRAINING",
        "subject_count":370,
        "candidate_source_subject_count":352,
        "excluded_observation_subject_count":18,
        "source_island_count":all_islands,
        "parent_semantic_region_candidate_count":all_parents,
        "total_source_face_count":all_faces,
        "parent_region_hint_status_counts":dict(sorted(hints.items())),
        "parent_region_concept_hint_counts":dict(sorted(concept.items())),
        "subject_status_counts":dict(sorted(outcome.items())),
        "source_v1_index_sha256":EXPECTED_SOURCE_INDEX_SHA,
        "semantic_prior_policy_sha256":_sha(prior_policy),
        "subjects":rows,
        "source_names_teacher_only":True,
        "true_semantic_component_identity_qualified":False,
        "semantic_class_training_authorized":False,
        "training_authorized":False,
    }
    out=output/"CORPUS370_SEMANTIC_PARENT_REGION_CENSUS_V1.json"
    out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return report


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-index",required=True,type=Path)
    p.add_argument("--source-receipts",required=True,type=Path)
    p.add_argument("--semantic-policy",required=True,type=Path)
    p.add_argument("--output-root",required=True,type=Path)
    args=p.parse_args()
    r=run(source_index=args.source_index,source_receipts=args.source_receipts,
          prior_policy=args.semantic_policy,output=args.output_root)
    print(json.dumps({k:r[k] for k in (
        "status","source_island_count","parent_semantic_region_candidate_count",
        "parent_region_hint_status_counts","parent_region_concept_hint_counts",
        "training_authorized"
    )},sort_keys=True))


if __name__=="__main__":
    main()
