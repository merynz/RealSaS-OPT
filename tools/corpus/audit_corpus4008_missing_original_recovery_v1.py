"""Original Objaverse recovery plan for MISSING historical Corpus4008 sources.

Uses exact manifest identities and exact byte-pinned upstream annotation tables.
Produces links/retrieval plans in a PRIVATE receipt only. The public summary
contains no source names, file identifiers, URLs or model-training authority.

This does not download objects, adjudicate their current rights, or claim
original source bytes are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

MASTER_SCHEMA = "RealSaS.Corpus4008RemoteOriginalSourceInventory.v1"
SCHEMA = "RealSaS.Corpus4008MissingObjaverseRecoveryPlan.v1"
SOURCE_CANDIDATES_SHA = "5439237f8a62c143cd081a564e11aef7b2c2dd176566d9144e987f5f3a75a25e"
LINKED_ANNOTATIONS_SHA = "1bd139233f756b6acbee88aa7ca59c21a186df17967c1cdf15bb16a8eba8f822"


def require(ok, code):
    if not ok:
        raise ValueError("CORPUS4008_ORIGINAL_RECOVERY_" + code)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(8<<20),b""):
            h.update(part)
    return h.hexdigest()


def safe_string(obj):
    return obj if isinstance(obj,str) else ""


def retrieval_route(url, source):
    host=urlparse(url).hostname or ""
    host=host.lower()
    if host in {"github.com", "raw.githubusercontent.com"}:
        if host=="github.com" and "/blob/" in url:
            return "PINNED_GITHUB_BLOB_OR_RAW__EXACT_BYTE_HASH_REQUIRED"
        return "GITHUB_LINK__MANUAL_FILE_RESOLUTION"
    if host in {"sketchfab.com","www.sketchfab.com"}:
        return "SKETCHFAB_OBJAVERSE_OFFICIAL_FETCH__EXACT_BYTE_HASH_REQUIRED"
    if host:
        return "OTHER_PROVIDER__REVIEW"
    if safe_string(source).lower() in {"github","sketchfab"}:
        return "PROVIDER_IDENTIFIER_ONLY__RECONSTRUCT_OFFICIAL_URL"
    return "NO_RETRIEVAL_ROUTE_RESOLVED"


def build(inventory_path, candidates_path, annotations_path):
    import pandas as pd

    require(sha(candidates_path)==SOURCE_CANDIDATES_SHA, "SOURCE_CANDIDATES_SHA")
    require(sha(annotations_path)==LINKED_ANNOTATIONS_SHA, "LINKED_ANNOTATIONS_SHA")
    doc=json.loads(inventory_path.read_text())
    require(doc.get("schema")==MASTER_SCHEMA and doc.get("summary",{}).get("subject_count")==4008,
            "SOURCE_INVENTORY_SCHEMA")
    require(doc["summary"].get("statuses",{}).get("REMOTE_ORIGINAL_NOT_FOUND")==3349,
            "SOURCE_INVENTORY_EXPECTED_MISSING_3349")
    require(doc["summary"].get("source_bytes_verified_count")==0 and
            doc["summary"].get("training_authorized") is False, "SOURCE_INVENTORY_NO_ADMISSION")
    all_subjects=doc["subjects_private"]
    require(len(all_subjects)==4008, "SOURCE_INVENTORY_FULL_ROWS")
    missing=[x for x in all_subjects if x["source_status"]=="REMOTE_ORIGINAL_NOT_FOUND"]
    require(len(missing)==3349 and all(x["provider"]=="objaverse_animated_originals" for x in missing),
            "SOURCE_INVENTORY_ORIGINAL_PROVIDER")
    cand=pd.read_parquet(candidates_path)
    linked=pd.read_parquet(annotations_path)
    needed={"candidate_id","canonical_asset_id","source_registry_id","file_identifier",
            "source_url","objaverse_source","source_type","legal_decision",
            "object_license","source_record_id"}
    require(needed<=set(cand.columns), "SOURCE_CANDIDATE_FIELDS")
    require({"fileIdentifier","license","sha256","source"}<=set(linked.columns),
            "UPSTREAM_LINKED_FIELDS")
    ids={x["candidate_id"] for x in missing}
    rows=cand.loc[cand["candidate_id"].isin(ids)].to_dict("records")
    candidate_index=defaultdict(list)
    for r in rows:
        candidate_index[r["candidate_id"]].append(r)
    # Linkage is advisory only. The original candidate_id and expected source
    # SHA must be reverified against acquired source bytes, never annotation label.
    link_index=defaultdict(list)
    relevant_names={safe_string(r.get("file_identifier")) for r in rows}
    for r in linked.loc[linked["fileIdentifier"].isin(relevant_names)].to_dict("records"):
        link_index[safe_string(r.get("fileIdentifier"))].append(r)
    report=[]
    for x in sorted(missing,key=lambda s:s["canonical_asset_id"]):
        found=candidate_index.get(x["candidate_id"], [])
        detail=found[0] if len(found)==1 else {}
        file_id=safe_string(detail.get("file_identifier"))
        url=safe_string(detail.get("source_url")) or file_id
        source=safe_string(detail.get("objaverse_source"))
        links=link_index.get(file_id,[])
        route=retrieval_route(url,source) if len(found)==1 else (
            "NO_LINKED_CANDIDATE" if not found else "DUPLICATED_CANDIDATE_METADATA")
        report.append({
            "canonical_asset_id":x["canonical_asset_id"],
            "candidate_id":x["candidate_id"],
            "source_expected_sha256":x["candidate_id"].split(":",1)[1],
            "historical_split":x["historical_split"],
            "source_candidate_metadata_count":len(found),
            "upstream_annotation_matches":len(links),
            "historical_legal_decision":safe_string(detail.get("legal_decision")) or "UNKNOWN",
            "historical_object_license_private":safe_string(detail.get("object_license")),
            "file_identifier_private":file_id,
            "source_url_private":url,
            "source_provider_private":source,
            "source_type_private":safe_string(detail.get("source_type")),
            "retrieval_route":route,
            "raw_original_reacquired":False,
            "current_license_revalidated":False,
            "source_sha256_verified":False,
            "mesh_rest_pose_revalidated":False,
            "anatomical_camera_courted":False,
            "semantic_truth_courted":False,
            "training_authorized":False,
        })
    require(len(report)==3349,"RECOVERY_ROW_COUNT")
    routes=Counter(r["retrieval_route"] for r in report)
    license=Counter(r["historical_legal_decision"] for r in report)
    coverage={
        "schema":SCHEMA,
        "status":"MISSING_ORIGINAL_RECOVERY_METADATA_ONLY__NO_SOURCE_BYTES",
        "missing_original_count":3349,
        "source_candidate_unique_count":sum(r["source_candidate_metadata_count"]==1 for r in report),
        "source_candidate_missing_count":sum(r["source_candidate_metadata_count"]==0 for r in report),
        "source_candidate_ambiguous_count":sum(r["source_candidate_metadata_count"]>1 for r in report),
        "upstream_annotation_linked_count":sum(r["upstream_annotation_matches"]>0 for r in report),
        "retrieval_routes":dict(sorted(routes.items())),
        "historical_legal_decisions":dict(sorted(license.items())),
        "heldout_count":sum(r["historical_split"]=="EXTERNAL_HOLDOUT" for r in report),
        "sources_reacquired":0,
        "source_sha256_reverified":0,
        "current_license_revalidated":0,
        "training_authorized":False,
        "inventory_sha256":sha(inventory_path),
        "source_candidates_sha256":sha(candidates_path),
        "linked_annotations_sha256":sha(annotations_path),
    }
    return {"schema":SCHEMA,"summary":coverage,"missing_originals_private":report}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-inventory-private",type=Path,required=True)
    p.add_argument("--source-candidates",type=Path,required=True)
    p.add_argument("--linked-annotations",type=Path,required=True)
    p.add_argument("--output-private",type=Path,required=True)
    p.add_argument("--output-public",type=Path,required=True)
    a=p.parse_args()
    doc=build(a.source_inventory_private,a.source_candidates,a.linked_annotations)
    a.output_private.parent.mkdir(parents=True,exist_ok=True)
    a.output_public.parent.mkdir(parents=True,exist_ok=True)
    a.output_private.write_text(json.dumps(doc,indent=2,sort_keys=True,default=str)+"\n")
    a.output_public.write_text(json.dumps(doc["summary"],indent=2,sort_keys=True)+"\n")
    print(json.dumps(doc["summary"],sort_keys=True))


if __name__=="__main__":
    main()
