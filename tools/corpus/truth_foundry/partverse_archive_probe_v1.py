#!/usr/bin/env python3
"""PartVerse-XL source annotation *archive* streaming probe (no full 823GB download).

This phase inspects the pinned 8.97GB anno_infos.tar.gz by streaming the head
of the archive; stages bounded part annotation JSON and a few source
segmented-GLB examples without retaining the entire archive. Every entry is
UNQUALIFIED; source rights, semantic concepts, exact original SHA and
drawable necessity are NOT approved by finding face2label entries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

UID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{8,90}$")
MAX_SINGLE_JSON = 32*1024*1024
MAX_SINGLE_GLB = 80*1024*1024


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def safe_archive_path(name):
    s = PurePosixPath(name)
    return bool(not s.is_absolute() and ".." not in s.parts and
                not str(name).startswith(("./", "../")) and "\x00" not in name)


def parse_face2label(raw, max_faces=500000):
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("FACE2LABEL_NOT_DICT")
    if len(obj) > max_faces:
        raise ValueError("FACE2LABEL_OVERSIZE")
    ids, labels = set(), Counter()
    for k, v in obj.items():
        if not str(k).isdigit() or int(k) < 0:
            raise ValueError("INVALID_FACE_KEY")
        if isinstance(v, bool) or not isinstance(v, (int, str)):
            raise ValueError("INVALID_PART_LABEL")
        if isinstance(v, str) and not v.isdigit():
            raise ValueError("NON_NUMERIC_PART_LABEL")
        i = int(k)
        if i in ids:
            raise ValueError("DUPLICATE_FACE")
        ids.add(i)
        labels[int(v)] += 1
    if not ids or not labels:
        raise ValueError("EMPTY_FACE_MAP")
    return {"labeled_faces":len(ids), "unique_part_ids":len(labels),
            "face_id_max":max(ids), "part_face_counts":dict(sorted(labels.items()))}


def inspect_tar_stream(fileobj, out_dir: Path, *,
                       max_members=5000,
                       max_total_staged_bytes=160*1024*1024,
                       max_json_files=60, max_glbs=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    written, parsed, rows, stats = 0, 0, [], Counter()
    staged = defaultdict(set)
    members = 0
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
        for member in tf:
            members += 1
            if members > max_members:
                break
            stats["members_scanned"] += 1
            name = member.name
            if not safe_archive_path(name):
                stats["unsafe_paths_skipped"] += 1
                continue
            if not member.isfile():
                continue
            filename = PurePosixPath(name).name
            suffix = None
            if filename.endswith("_face2label.json"):
                suffix = "face2label"
                uid = filename[: -len("_face2label.json")]
                max_size = MAX_SINGLE_JSON
                if stats["json_staged"] >= max_json_files:
                    continue
            elif filename.endswith("_segmented.glb"):
                suffix = "segmented_glb"
                uid = filename[: -len("_segmented.glb")]
                max_size = MAX_SINGLE_GLB
                if stats["glb_staged"] >= max_glbs:
                    continue
            elif filename.endswith("_info.json"):
                suffix = "info_json"
                uid = filename[:-len("_info.json")]
                max_size = MAX_SINGLE_JSON
                if stats["json_staged"] >= max_json_files:
                    continue
            else:
                stats["other_file_members_seen"] += 1
                continue
            if not UID.fullmatch(uid):
                stats["unsupported_uid"] += 1
                continue
            if member.size > max_size or member.size < 1 or written + member.size > max_total_staged_bytes:
                stats["size_quarantined"] += 1
                continue
            src = tf.extractfile(member)
            if src is None:
                stats["missing_member_payload"] += 1
                continue
            payload = src.read(member.size+1)
            if len(payload) != member.size:
                stats["size_mismatch"] += 1
                continue
            record = {
                "archive_path":name,
                "archive_uid":uid,
                "kind":suffix,
                "bytes":member.size,
                "sha256":sha256_bytes(payload),
                "independent_source_rights_pass":False,
                "original_source_sha256_court":"UNKNOWN",
                "source_face_parity_court":"UNKNOWN",
                "semantic_concept_review":"UNKNOWN",
                "graphics_drawable_court":"UNKNOWN",
                "training_authorized":False,
            }
            if suffix == "face2label":
                try:
                    record.update(parse_face2label(payload))
                    parsed += 1
                except (ValueError, json.JSONDecodeError) as exc:
                    record["parse_error"]=str(exc)
                    stats["face2label_invalid"] += 1
                    rows.append(record)
                    continue
            subdir = out_dir/"sampled_annotation_data"/uid
            subdir.mkdir(parents=True, exist_ok=True)
            target = subdir/filename
            if target.exists():
                stats["duplicate_archive_uid_filename"] += 1
                continue
            target.write_bytes(payload)
            rows.append(record)
            staged[uid].add(suffix)
            written += member.size
            stats["json_staged" if suffix != "segmented_glb" else "glb_staged"] += 1
    summary={
        "schema":"RealSaS.PartVerseXLArchiveProbe.v1",
        "archive_member_limit":max_members,
        "observed_archive_members":members,
        "staged_total_bytes":written,
        "sampled_source_uids":len(staged),
        "face2label_records_parsed":parsed,
        "source_rights_approved":0,
        "source_geometry_parity_pass":0,
        "source_face_semantic_pass":0,
        "drawable_layer_pass":0,
        "training_authorized":False,
        "full_archive_processed":False,
        "statistics":dict(stats),
        "sampled_records":rows,
    }
    (out_dir/"PROBE_SUMMARY_V1.json").write_text(json.dumps(summary,indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--max-members",type=int,default=5000)
    parser.add_argument("--max-staged-mib",type=int,default=160)
    parser.add_argument("--max-json-files",type=int,default=60)
    parser.add_argument("--max-glbs",type=int,default=5)
    args=parser.parse_args()
    assert re.fullmatch("[a-f0-9]{40}",args.revision), "pinned revision mandatory"
    assert 0<args.max_members<=20000
    assert 0<args.max_staged_mib<=512
    import requests
    url=f"https://huggingface.co/datasets/dscdyc/partversexl/resolve/{args.revision}/anno_infos.tar.gz"
    with requests.get(url,stream=True,timeout=(30,240),headers={"User-Agent":"RealSaS-TruthFoundry/1.0"}) as response:
        response.raise_for_status()
        response.raw.decode_content=False
        s=inspect_tar_stream(response.raw,args.out,max_members=args.max_members,
                             max_total_staged_bytes=args.max_staged_mib*1024*1024,
                             max_json_files=args.max_json_files,max_glbs=args.max_glbs)
        s["upstream_repo"]="dscdyc/partversexl"
        s["pinned_revision"]=args.revision
        (args.out/"PROBE_SUMMARY_V1.json").write_text(json.dumps(s,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in s.items() if k!="sampled_records"},indent=2))


if __name__=="__main__":
    main()
