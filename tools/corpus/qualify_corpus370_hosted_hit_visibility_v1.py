#!/usr/bin/env python3
"""Recheck private renderer evidence as bounded source-hit visibility candidates.

Runs after SHA-sealed two-view Blender V2 receipts. Downloads are performed by
the authenticated workflow, NOT this script. Only PRIVATE Drive receives
per-pixel, source-triangle or provenance details; public logs expose counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.court_corpus370_source_locus_visibility_v1 import qualify
from tools.corpus.generic_source_locus_reciprocal_palette_v2 import court_document


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def require(ok,code):
    if not ok:raise ValueError(code)

def run(summary_path,source_dir,output_dir,expected_sha):
    summary_path=Path(summary_path)
    source_dir=Path(source_dir)
    output_dir=Path(output_dir)
    require(sha(summary_path)==expected_sha,"VIS_BATCH_PARENT_SUMMARY_SHA_DRIFT")
    x=json.loads(summary_path.read_text())
    require(x.get("schema")=="RealSaS.Corpus370V5SourcePaletteInterventionPilot.v1"
            and x.get("status")=="ALL_THREE_SOURCE_PALETTE_PILOT_SUBJECTS_ACCOUNTED__ADMISSION_SEPARATE"
            and x.get("failure_count")==0
            and x.get("v2_two_causal_locus_rows")==6
            and x.get("graphics_family_authority_minted") is False
            and x.get("training_authorized") is False,
            "VIS_BATCH_EXACT_V2_SOURCE_PILOT_REQUIRED")
    ids=x.get("exact_subject_ids") or []
    require(len(ids)==3 and len(set(ids))==3 and set(ids)==set(x["subjects"]),
            "VIS_BATCH_EXACT_THREE_SOURCE_IDS")
    total_positive=total_negative=total_abstain=0
    report={
        "schema":"RealSaS.Corpus370BoundedSourceHitVisibilityBatchCourt.v1",
        "status":"PRIVATE_SOURCE_HIT_VISIBILITY_CANDIDATES__GRAPHICS_OWNER_SEPARATE",
        "parent_summary_sha256":expected_sha,"subject_count":len(ids),
        "subjects":{},"qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
        "graphics_family_authority_minted":False,"training_authorized":False,
    }
    for sid in ids:
        base=source_dir/"subjects"/sid
        src=base/"SOURCE_PALETTE_INTERVENTION_V1.json"
        prior_path=base/"SOURCE_PALETTE_RECIPROCAL_CAUSAL_COURT_V2.json"
        require(src.is_file() and prior_path.is_file(),
                "VIS_BATCH_PRIVATE_PARENT_EVIDENCE_MISSING")
        prior_sha=x["subjects"][sid]["reciprocal_causal_court_v2"]["sha256"]
        require(sha(prior_path)==prior_sha,
                "VIS_BATCH_PRIOR_INDEPENDENT_COURT_SHA_DRIFT")
        prior=json.loads(prior_path.read_text())
        require(prior.get("parent",{}).get("sha256")==sha(src),
                "VIS_BATCH_SOURCE_MANIFEST_SHA_DRIFT")
        source=json.loads(src.read_text())
        require(source.get("subject_id")==sid and
                source.get("source_sha256")==x["subjects"][sid]["source_sha256"]
                and source.get("sampled_view_indices")==[0,4]
                and source.get("sample_policy")=="GEOMETRY_ONLY_OPPOSITE_VIEW_DIVERSE_V1",
                "VIS_BATCH_TWO_VIEW_SOURCE_BINDING")
        independent=court_document(source)
        require({k:v for k,v in prior.items() if k!="parent"}==independent,
                "VIS_BATCH_PRIOR_V2_INDEPENDENT_COURT_DRIFT")
        qualified=qualify(source)
        require(qualified["training_authorized"] is False
                and qualified["qualified_same_drawable_positives"]==0
                and qualified["qualified_different_drawable_negatives"]==0,
                "VIS_BATCH_OWNER_OR_TRAINING_PROMOTION")
        dest=output_dir/"subjects"/sid/"SOURCE_HIT_VISIBILITY_CANDIDATE_COURT_V1.json"
        dest.parent.mkdir(parents=True,exist_ok=True)
        qualified["parents"]={
            "source_manifest_sha256":sha(src),
            "prior_independent_court_sha256":prior_sha,
            "source_pilot_summary_sha256":expected_sha,
        }
        dest.write_text(json.dumps(qualified,sort_keys=True,indent=2,allow_nan=False)+"\n")
        total_positive+=qualified["qualified_visible_hit_positives"]
        total_negative+=qualified["qualified_occluded_hit_negatives"]
        total_abstain+=qualified["abstain_hit_count"]
        report["subjects"][sid]={
            "status":qualified["status"],
            "receipt_sha256":sha(dest),
            "positive_visible":qualified["qualified_visible_hit_positives"],
            "negative_occluded":qualified["qualified_occluded_hit_negatives"],
            "abstain":qualified["abstain_hit_count"],
            "training_authorized":False,
        }
    report["visible_source_hit_positives"]=total_positive
    report["opaque_occluded_source_hit_negatives"]=total_negative
    report["abstain_source_hits"]=total_abstain
    report["evaluated_source_hit_count"]=total_positive+total_negative+total_abstain
    require(report["evaluated_source_hit_count"]==12,
            "VIS_BATCH_TWO_VIEWS_BY_TWO_LOCI_BY_THREE_SUBJECTS")
    out=output_dir/"CORPUS370_HOSTED_SOURCE_HIT_VISIBILITY_BATCH_V1.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "source_hits":report["evaluated_source_hit_count"],
        "visible_positive":total_positive,
        "opaque_occluded_negative":total_negative,
        "abstain":total_abstain,
        "drawable_owner_positive_negative":"0/0_NOT_QUALIFIED",
        "training_authorized":False,
    },sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--expected-summary-sha256",required=True)
    a=p.parse_args()
    run(a.summary,a.input_root,a.output_root,a.expected_summary_sha256)

if __name__=="__main__":main()
