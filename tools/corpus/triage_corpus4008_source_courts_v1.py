#!/usr/bin/env python3
"""Evidence-directed QA scheduling from 1024 imported source observations.

This scores REVIEW URGENCY, not source validity or model training suitability.
Source original permissions, morphology, original texture truth, semantic
teacher, deform bind and 2D art remain unreviewed unless separately courted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

SOURCE_SCHEMA="RealSaS.Corpus4008Parallel1024ImportedSceneMasterLedger.v1"
SCHEMA="RealSaS.Corpus4008SourceCourtReviewTriage.v1"


def require(x,why):
    if not x: raise ValueError("C4008_TRIAGE_"+why)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def iv(x):
    if x in (None,"","None"):return None
    try:
        y=int(x)
        require(y>=0,"NEGATIVE_OBSERVED_COUNT")
        return y
    except ValueError:
        raise ValueError("C4008_TRIAGE_BAD_SOURCE_COUNT")


def triage(summary_path:Path,matrix_path:Path,out:Path,*,count=1024):
    s=json.loads(summary_path.read_text())
    require(s.get("schema")==SOURCE_SCHEMA
            and s.get("training_authorized") is False
            and s.get("unique_source_candidates_measured")==count
            and s.get("typed_source_matrix_sha256")==sha(matrix_path),
            "SOURCE_READBACK_SHA_OR_TRUTH")
    with matrix_path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==count,"SOURCE_MATRIX_COUNT")
    outrows=[]
    ids=set()
    originals=set()
    flags=Counter()
    groups=defaultdict(list)
    for r in rows:
        aid=r["canonical_asset_id"]
        src=r["source_sha256"]
        require(aid not in ids and src not in originals,
                "DUPLICATE_SOURCE")
        ids.add(aid);originals.add(src)
        require(r["split"]=="FIT"
                and r["training_authorized"]=="False"
                and r["head_admission_count"]=="0"
                and r["functional_native_ik_fk"]=="UNKNOWN"
                and r["source_character_vs_prop"]=="UNKNOWN"
                and r["graphics_source_bound_2d_art_qualified"]=="UNKNOWN",
                "FALSE_SOURCE_TRUTH_OR_HOLDOUT")
        status=r["source_import_observation_status"]
        has_import=(status=="IMPORTED_SOURCE_OBSERVATIONS_ONLY")
        mesh=iv(r["mesh_object_count"])
        arm=iv(r["armature_count"])
        bones=iv(r["observed_bone_count"])
        meshcomponents=iv(r["mesh_disconnected_components"])
        boundary=iv(r["mesh_boundary_edges"])
        manifold=iv(r["mesh_nonmanifold_edges"])
        missing_texture=iv(r["image_dependency_missing_count"])
        bad_weights=iv(r["skin_rows_bad_sum"])
        zero_weights=iv(r["skin_rows_zero_weights"])
        skin_sample=iv(r["skin_rows_sampled"])
        ik=iv(r["observed_ik_constraint_count"])
        drivers=iv(r["observed_driver_channel_count"])
        actions=iv(r["observed_action_count"])
        require(not has_import or all(
            v is not None for v in
            (mesh,arm,bones,meshcomponents,boundary,manifold,
             missing_texture,bad_weights,zero_weights,skin_sample,ik,drivers,actions)),
            "PARTIAL_IMPORTED_COUNTERS")
        warnings=[]
        if not has_import:
            warnings.append("SOURCE_IMPORT_EXCEPTION")
        else:
            if (mesh or 0)==0:
                warnings.append("EMPTY_IMPORTED_MESH")
            if (arm or 0)==0:
                warnings.append("NO_IMPORTED_ARMATURE")
            if (bones or 0)==0:
                warnings.append("NO_IMPORTED_BONES")
            if (skin_sample or 0)==0 or (zero_weights or 0)>0:
                warnings.append("ZERO_OR_MISSING_IMPORTED_VERTEX_WEIGHTS")
            if (bad_weights or 0)>0:
                warnings.append("IMPORTED_SKIN_WEIGHT_ROW_SUM_ANOMALY")
            if (missing_texture or 0)>0:
                warnings.append("EXTERNAL_TEXTURES_NOT_RESOLVED")
            if (boundary or 0)>0 or (manifold or 0)>0:
                warnings.append("OPEN_OR_NONMANIFOLD_SOURCE_MESH")
            if (meshcomponents or 0)>32:
                warnings.append("HIGHLY_FRAGMENTED_SOURCE_MESH")
            if (ik or 0)==0:
                warnings.append("NATIVE_IK_NOT_OBSERVED_AFTER_EXPORT")
        warnings+=["SOURCE_LICENSE_RIGHTS_UNKNOWN",
                   "SOURCE_CHARACTER_MORPHOLOGY_UNKNOWN",
                   "2D_GAME_ART_PRESENTATION_NOT_COURTED"]
        for w in warnings: flags[w]+=1
        group=("IMPORT_EXCEPTION" if not has_import else
               "RIG_AND_SKIN_DAMAGE_RISK" if any(z in warnings for z in (
                   "NO_IMPORTED_ARMATURE",
                   "NO_IMPORTED_BONES",
                   "ZERO_OR_MISSING_IMPORTED_VERTEX_WEIGHTS",
                   "IMPORTED_SKIN_WEIGHT_ROW_SUM_ANOMALY")) else
               "TEXTURE_OR_TOPOLOGY_RISK" if any(z in warnings for z in (
                   "EXTERNAL_TEXTURES_NOT_RESOLVED",
                   "OPEN_OR_NONMANIFOLD_SOURCE_MESH")) else
               "SOURCE_LOOKS_STRUCTURALLY_MEASURABLE_BUT_UNREVIEWED")
        groups[group].append(src)
        outrows.append({
            "canonical_asset_id":aid,
            "original_source_sha256":src,
            "source_import_status":status,
            "source_review_queue":group,
            "source_observed_mesh_objects":mesh,
            "source_observed_bones":bones,
            "source_observed_actions":actions,
            "source_observed_IK_constraints":ik,
            "source_observed_driver_channels":drivers,
            "source_observed_skin_rows":skin_sample,
            "source_observed_skin_bad_sum":bad_weights,
            "source_observed_skin_no_weights":zero_weights,
            "source_observed_boundary_edges":boundary,
            "source_observed_nonmanifold_edges":manifold,
            "source_observed_unresolved_external_images":missing_texture,
            "source_review_warnings":";".join(warnings),
            "source_morphology":"UNKNOWN",
            "original_commercial_ML_rights":"UNKNOWN",
            "native_control_functionality":"UNKNOWN",
            "native_skin_deformation":"UNKNOWN",
            "semantic_source_face_regions":"UNKNOWN",
            "graphics_2D_game_art_target":"REVIEW_REQUIRED",
            "materialization_authorized":False,
            "training_authorized":False,
        })
    require(len(ids)==len(originals)==count,"MISMATCHED_UNIQUE_SOURCE_COUNT")
    out.mkdir(parents=True,exist_ok=True)
    matrix=out/"CORPUS4008_1024_SOURCE_REVIEW_TRIAGE_V1.csv"
    with matrix.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(outrows[0]))
        w.writeheader()
        w.writerows(sorted(outrows,key=lambda x:x["canonical_asset_id"]))
    summary={
        "schema":SCHEMA,
        "source_count":count,
        "source_readback_summary_sha256":sha(summary_path),
        "source_readback_matrix_sha256":sha(matrix_path),
        "review_warning_counts":dict(sorted(flags.items())),
        "review_queue_counts":dict(sorted((k,len(v)) for k,v in groups.items())),
        "review_queue_exemplar_original_sha256":{
            k:sorted(v)[:8] for k,v in sorted(groups.items())
        },
        "triage_matrix_sha256":sha(matrix),
        "source_graphic_style_2d_art_admitted":0,
        "source_truth_rights_courted":0,
        "per_head_training_admitted":0,
        "triage_is_not_independent_source_truth":True,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    (out/"CORPUS4008_1024_SOURCE_REVIEW_TRIAGE_SUMMARY_V1.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(triage(a.summary,a.matrix,a.output),sort_keys=True))


if __name__=="__main__":
    main()
