#!/usr/bin/env python3
"""Fail-closed pre-Materialization PRECHECK, NOT an admission authority.

This validates immutable normalization/multi-head evidence *claims* and their
file SHA identities. It cannot independently certify an author/reviewer,
license, source provenance, or render correctness; a separate trusted
admission authority must seal those substantive courts.

No output from this script authorizes materialization or training by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.build_corpus4008_normalization_plan_v1 import (
    SCHEMA as NORM_SCHEMA, issha, require as normalization_require
)

SCHEMA="RealSaS.Corpus4008PreMaterializationPrecheck.v1"
MANIFEST_SCHEMA="RealSaS.Corpus4008StandardizedSourceTargetCandidate.v1"
COURT_SCHEMA="RealSaS.IndependentSourceCourt.v1"
SPLITS={"FIT","TUNE","DEV","CAL","EXTERNAL_HOLDOUT"}
HEADS={
 "IRIS_GEOMETRY":("SOURCE_BYTES","COMMERCIAL_ML_RIGHTS",
                  "LINEAGE_DEDUP_SPLIT","SOURCE_IMPORT_PARITY",
                  "NORMALIZED_SOURCE_ROUNDTRIP","SOURCE_CAMERA"),
 "IRIS_SEMANTIC":("SOURCE_BYTES","COMMERCIAL_ML_RIGHTS",
                  "LINEAGE_DEDUP_SPLIT","SOURCE_IMPORT_PARITY",
                  "NORMALIZED_SOURCE_ROUNDTRIP","REVIEWED_SEMANTIC_SOURCE_SUPPORT"),
 "IRIS_GRAPHICS":("SOURCE_BYTES","COMMERCIAL_ML_RIGHTS",
                  "LINEAGE_DEDUP_SPLIT","SOURCE_IMPORT_PARITY",
                  "NORMALIZED_SOURCE_ROUNDTRIP","SOURCE_CAMERA",
                  "SOURCE_RGBA_PARITY","STYLIZED_2D_GAME_ART",
                  "CORE370_2D_ART_DOMAIN_PARITY"),
 "TESSA":("SOURCE_BYTES","COMMERCIAL_ML_RIGHTS",
                  "LINEAGE_DEDUP_SPLIT","SOURCE_IMPORT_PARITY",
                  "NORMALIZED_SOURCE_ROUNDTRIP","TOPOLOGY_DEFORMATION"),
 "AXIS":("SOURCE_BYTES","COMMERCIAL_ML_RIGHTS",
                  "LINEAGE_DEDUP_SPLIT","SOURCE_IMPORT_PARITY",
                  "NORMALIZED_SOURCE_ROUNDTRIP","RIG_KINEMATIC_PARITY"),
 "MIRA":("SOURCE_BYTES","COMMERCIAL_ML_RIGHTS",
                  "LINEAGE_DEDUP_SPLIT","SOURCE_IMPORT_PARITY",
                  "NORMALIZED_SOURCE_ROUNDTRIP","RIG_SKIN_WEIGHT_DEFORMATION"),
}
STATES={"PASS","FAIL","MASKED","UNKNOWN","QUARANTINED","NOT_APPLICABLE"}


def require(v,code):
    if not v:
        raise ValueError("C4008_PREMATERIALIZATION_"+code)


def sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda:fp.read(4<<20),b""):
            h.update(block)
    return h.hexdigest()


def bound_ref(root,ref):
    require(isinstance(ref,dict) and
            isinstance(ref.get("path"),str) and
            ref["path"] and
            issha(ref.get("sha256")),"EVIDENCE_REF")
    path=Path(ref["path"])
    require(not path.is_absolute() and ".." not in path.parts,
            "EVIDENCE_PATH_ESCAPE")
    f=(root/path).resolve()
    require(f.is_relative_to(root.resolve()),"EVIDENCE_SYMLINK_ESCAPE")
    require(f.is_file(),"EVIDENCE_MISSING")
    require(sha(f)==ref["sha256"],"EVIDENCE_SHA_DRIFT")
    return f


def validate(manifest_file:Path):
    root=manifest_file.parent.resolve()
    manifest=json.loads(manifest_file.read_text())
    require(manifest.get("schema")==MANIFEST_SCHEMA,"MANIFEST_SCHEMA")
    require(manifest.get("training_authorized") is False and
            manifest.get("materialization_authorized") is False,
            "PREAUTHORIZED_INPUT_FORBIDDEN")
    aid=manifest.get("canonical_asset_id")
    source=manifest.get("source_sha256")
    head=manifest.get("model_head")
    split=manifest.get("split")
    require(isinstance(aid,str) and aid and issha(source),
            "IDENTITY")
    require(head in HEADS,"HEAD")
    require(split in SPLITS,"SPLIT")
    require(isinstance(manifest.get("source_identity_cluster"),str)
            and len(manifest["source_identity_cluster"])>=12,
            "SOURCE_LINEAGE")
    norm_file=bound_ref(root,manifest.get("normalization_plan"))
    norm=json.loads(norm_file.read_text())
    require(norm.get("schema")==NORM_SCHEMA and
            norm.get("status")==
            "REVERSIBLE_PLAN_NUMERICALLY_CHECKED__NOT_SOURCE_PARITY" and
            norm.get("canonical_asset_id")==aid and
            norm.get("original_source_sha256")==source and
            norm.get("materialization_authorized") is False and
            norm.get("training_authorized") is False and
            norm.get("source_mesh_vertex_roundtrip_courted") is False,
            "NORMALIZATION_PLAN_NOT_SOURCE_TRUTH")
    require(norm.get("common_global_scale_for_all_views_and_poses") is True and
            norm.get("per_view_recenter_permitted") is False,
            "NORMALIZATION_VIEW_LEAKAGE")
    states=manifest.get("family_courts")
    require(isinstance(states,dict),"FAMILY_COURTS")
    required=set(HEADS[head])
    # No phantom PASS from a prior model/other head; inspect all present
    # declared court references but count only head-specific requirements.
    blockers=[]
    looked_up=[]
    for name in sorted(required):
        entry=states.get(name)
        if not isinstance(entry,dict):
            blockers.append("MISSING_REQUIRED_COURT:"+name)
            continue
        status=entry.get("status")
        require(status in STATES,"COURT_STATUS")
        if status!="PASS":
            require(entry.get("receipt") is None,
                    "NONPASS_COURT_CANNOT_HAVE_TARGET_PAYLOAD")
            blockers.append("REQUIRED_COURT_NOT_PASS:"+name+":"+status)
            continue
        # PASS must have a sha-pinned independent court doc, with exact identity.
        file=bound_ref(root,entry.get("receipt"))
        court=json.loads(file.read_text())
        require(court.get("schema")==COURT_SCHEMA and
                court.get("asset_id")==aid and
                court.get("source_sha256")==source and
                court.get("court_kind")==name and
                court.get("status")=="PASS" and
                court.get("training_authorized") is False and
                court.get("independent_review") is True,
                "COURT_PAYLOAD_IDENTITY_AND_REVIEW")
        if name=="STYLIZED_2D_GAME_ART":
            require(court.get("target_presentation")=="STYLIZED_2D_GAME_ART" and
                    court.get("raw_3d_rgb_direct_training_forbidden") is True and
                    court.get("source_bound_view_and_alpha_court")=="PASS",
                    "RAW_3D_GRAPHICS_NOT_ALLOWED")
        looked_up.append(name)
    if head=="IRIS_GRAPHICS":
        require(manifest.get("graphics_target_domain")==
                "STYLIZED_2D_GAME_ART",
                "GRAPHICS_TARGET_DOMAIN")
    else:
        require(manifest.get("graphics_target_domain") in (None,"NOT_A_GRAPHICS_TARGET"),
                "UNEXPECTED_GRAPHICS_TARGET")
    targets=manifest.get("target_family_status")
    require(isinstance(targets,dict) and targets,"TARGET_FAMILY_STATUSES")
    target_receipts=manifest.get("target_payloads",{})
    require(isinstance(target_receipts,dict),"TARGET_PAYLOADS")
    for target,status in targets.items():
        require(isinstance(target,str) and target and status in STATES,
                "TARGET_STATUS")
        if status=="PASS":
            # Unknown/dummy zeros not permitted for asserted supervision.
            require(target in target_receipts,"PASS_WITHOUT_TARGET_PAYLOAD")
            bound_ref(root,target_receipts[target])
        else:
            require(target not in target_receipts,
                    "UNKNOWN_MASKED_TARGET_MUST_BE_ABSENT")
    require(set(target_receipts).issubset(targets),"FOREIGN_TARGET_PAYLOAD")
    if not blockers:
        blockers.append("EXTERNAL_TRUSTED_ADMISSION_FREEZE_REQUIRED")
    if split=="EXTERNAL_HOLDOUT":
        blockers.append("PROTECTED_HOLDOUT_CANNOT_ENTER_TRAIN")
    return {
        "schema":SCHEMA,
        "canonical_asset_id":aid,
        "source_sha256":source,
        "model_head":head,
        "split":split,
        "normalization_plan_sha256":sha(norm_file),
        "required_court_kinds":list(HEADS[head]),
        "receipt_sha_validated_courts":looked_up,
        "blocked_reasons":blockers,
        "preflight_structurally_complete":len(looked_up)==len(required)
            and not any(b.startswith("REQUIRED_COURT_NOT_PASS") for b in blockers),
        "status":"BLOCKED_PENDING_TRUSTED_ADMISSION",
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=validate(a.candidate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":result["status"],
        "materialization_authorized":False,
        "blockers":result["blocked_reasons"],
    },sort_keys=True))


if __name__=="__main__":
    main()
