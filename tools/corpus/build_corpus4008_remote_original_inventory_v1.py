#!/usr/bin/env python3
"""SHA-pinned remote metadata preflight for the 4,008-source Master Corpus.

The source inventory is a PRIVATE Drive receipt. Only aggregate provider/status
counts (without object names or IDs) are safe for public Actions summaries.

This is remote file METADATA: it proves neither original bytes nor character
class, rest/up/front, source graphics, source attachments, or model admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

SCHEMA = "RealSaS.Corpus4008RemoteOriginalSourceInventory.v1"
COUNTS = {
    "objaverse_animated_originals": 3765,
    "quaternius_cc0": 191,
    "kaykit_cc0": 37,
    "khronos_gltf_qa": 15,
}
SOURCE_EXTS = {".glb", ".gltf", ".fbx", ".blend", ".obj"}
MASTER_SHA = "475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913"
SELECTION_SHA = "af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("CORPUS4008_REMOTE_ORIGINAL_" + reason)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_inventory(master: Path, selection: Path, remote_list: Path, *,
                   expected_master_sha=MASTER_SHA, expected_selection_sha=SELECTION_SHA):
    require(digest(master) == expected_master_sha, "MASTER_MANIFEST_SHA")
    require(digest(selection) == expected_selection_sha, "SELECTION_SHA")
    source_rows = [json.loads(line) for line in master.read_text().splitlines() if line.strip()]
    chosen = json.loads(selection.read_text())
    require(len(source_rows) == 4008 and len(chosen) == 3993, "EXACT_MASTER_COUNTS")
    require(len({r["candidate_id"] for r in source_rows}) == 4008 and
            len({r["canonical_asset_id"] for r in source_rows}) == 4008, "MASTER_IDENTITY_UNIQUE")
    counts = Counter(r["source_registry_id"] for r in source_rows)
    require(dict(counts) == COUNTS, "PROVIDER_COUNTS")
    require(sum(r["split"] == "EXTERNAL_HOLDOUT" for r in source_rows) == 169, "HOLDOUT_COUNT")
    require(sum(r["split"] == "QA_ONLY" for r in source_rows) == 15, "QA_COUNT")
    require(set(chosen) == {r["canonical_asset_id"] for r in source_rows
                           if r["split"] != "QA_ONLY"}, "SELECTION_SET")
    remote = json.loads(remote_list.read_text())
    require(isinstance(remote, list), "RCLONE_JSON_ARRAY")
    by_identity = defaultdict(list)
    total_files = 0
    for file in remote:
        if file.get("IsDir") is True:
            continue
        relative = str(file.get("Path") or "")
        parts = PurePosixPath(relative).parts
        if len(parts) < 2 or parts[0] not in COUNTS:
            continue
        name = parts[-1]
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in SOURCE_EXTS:
            continue
        token = name.split("_",1)[0]
        if not re.fullmatch(r"[a-f0-9]{24}|[a-f0-9]{64}", token):
            continue
        size = file.get("Size")
        require(type(size) is int and size >= 0, "SOURCE_REMOTE_SIZE")
        require(".." not in parts and not relative.startswith("/"), "REMOTE_PATH_ESCAPE")
        by_identity[(parts[0],token)].append({
            "source_relative_path_private": relative,
            "source_size_bytes": size,
            "source_extension": suffix,
            "remote_md5_present": bool(file.get("Hashes",{}).get("MD5")),
        })
        total_files += 1
    rows = []
    for row in sorted(source_rows,key=lambda x:x["canonical_asset_id"]):
        provider = row["source_registry_id"]
        candidate = row["candidate_id"]
        require(candidate.startswith(provider+":"), "CANDIDATE_PROVIDER")
        token = candidate.split(":",1)[1]
        require(re.fullmatch(r"[a-f0-9]{24}|[a-f0-9]{64}", token) is not None,
                "SOURCE_RECORD_TOKEN")
        matches = sorted(by_identity.get((provider,token), []),
                         key=lambda x:x["source_relative_path_private"])
        status = ("REMOTE_ORIGINAL_NOT_FOUND" if not matches
                  else "AMBIGUOUS_REMOTE_ORIGINAL_VARIANTS" if len(matches)>1
                  else "REMOTE_ORIGINAL_METADATA_MATCH__BYTES_NOT_VERIFIED")
        rows.append({
            "canonical_asset_id":row["canonical_asset_id"],
            "candidate_id":candidate,
            "provider":provider,
            "historical_split":row["split"],
            "historical_decision":row["decision"],
            "source_matches_private":matches,
            "source_file_count":len(matches),
            "source_status":status,
            "source_bytes_verified":False,
            "source_character_class":"UNKNOWN",
            "source_pose_class":"UNKNOWN",
            "anatomical_axes_courted":False,
            "native_camera_courted":False,
            "native_appearance_courted":False,
            "attachment_courted":False,
            "semantic_truth_courted":False,
            "training_authorized":False,
        })
    statuses = Counter(x["source_status"] for x in rows)
    by_provider = {
        provider: dict(sorted(Counter(x["source_status"] for x in rows
                                      if x["provider"]==provider).items()))
        for provider in COUNTS
    }
    summary = {
        "schema":SCHEMA,
        "status":"REMOTE_METADATA_CENSUS__ORIGINAL_BYTE_PARITY_OPEN",
        "master_sha256":digest(master),
        "selection_sha256":digest(selection),
        "subject_count":4008,
        "provider_counts":dict(counts),
        "remote_files_with_candidate_style_names":total_files,
        "statuses":dict(sorted(statuses.items())),
        "by_provider_statuses":by_provider,
        "historical_external_holdout_count":169,
        "qa_only_count":15,
        "source_bytes_verified_count":0,
        "source_camera_courted_count":0,
        "source_semantic_courted_count":0,
        "training_authorized":False,
    }
    require(sum(statuses.values()) == 4008, "STATUS_SUM")
    return {"schema":SCHEMA,"summary":summary,"subjects_private":rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--master",required=True,type=Path)
    ap.add_argument("--selection",required=True,type=Path)
    ap.add_argument("--remote-file-list",required=True,type=Path)
    ap.add_argument("--output-private",required=True,type=Path)
    ap.add_argument("--output-public",required=True,type=Path)
    args=ap.parse_args()
    doc=make_inventory(args.master,args.selection,args.remote_file_list)
    args.output_private.parent.mkdir(parents=True,exist_ok=True)
    args.output_public.parent.mkdir(parents=True,exist_ok=True)
    args.output_private.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    args.output_public.write_text(json.dumps(doc["summary"],indent=2,sort_keys=True)+"\n")
    # stdout and public Actions logs contain aggregate counts only
    print(json.dumps(doc["summary"],sort_keys=True))


if __name__=="__main__":
    main()
