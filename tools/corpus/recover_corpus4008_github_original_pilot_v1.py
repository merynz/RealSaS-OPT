"""Conservative, exact-commit original GitHub source-byte recovery PILOT.

Only approved GitHub blob URLs at a 40-hex commit; full SHA256 equality is
mandatory. Verified objects are staged in a PRIVATE quarantine, never in the
Master's canonical source root and never in public Actions artifacts.

The original dataset legal PASS is historical: it does not grant current
training, render, RGB, semantic, or product authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse, quote

SCHEMA = "RealSaS.Corpus4008PinnedGitHubOriginalRecoveryPilot.v1"
SHA = re.compile(r"^[a-f0-9]{64}$")
COMMIT = re.compile(r"^[a-f0-9]{40}$")
PART = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_BYTES_DEFAULT = 64 * 1024 * 1024
ALLOWED_SUFFIXES = {".fbx", ".glb", ".gltf"}


def require(cond, code):
    if not cond:
        raise ValueError("C4008_PINNED_GITHUB_RECOVERY_" + code)


def pinned_raw_url(blob_url):
    u = urlparse(blob_url)
    require(u.scheme == "https" and u.netloc.lower() == "github.com" and
            not u.username and not u.password and not u.query and not u.fragment,
            "SOURCE_MUST_BE_PUBLIC_GITHUB_BLOB")
    parts = [unquote(part) for part in u.path.split("/") if part]
    require(len(parts) >= 5 and parts[2] == "blob" and COMMIT.fullmatch(parts[3]),
            "EXACT_COMMIT_REQUIRED")
    owner,repo=parts[0],parts[1]
    require(PART.fullmatch(owner) and PART.fullmatch(repo) and
            all(x and x not in (".","..") and "\\" not in x and "\x00" not in x
                for x in parts[4:]),"UNSAFE_SOURCE_BLOB_PATH")
    suffix=Path(parts[-1]).suffix.lower()
    require(suffix in ALLOWED_SUFFIXES,"SOURCE_EXTENSION_NOT_SUPPORTED")
    url = "https://raw.githubusercontent.com/" + "/".join(
        [quote(owner,safe=""),quote(repo,safe=""),quote(parts[3],safe="")] +
        [quote(p,safe="") for p in parts[4:]]
    )
    return url, suffix


def choose_pilot(rows, per_format):
    require(type(per_format) is int and 1<=per_format<=12,"PILOT_CAP")
    by_type=defaultdict(list)
    for row in rows:
        if row["historical_split"] == "EXTERNAL_HOLDOUT":
            continue
        if row["historical_legal_decision"] != "PASS":
            continue
        if row["source_candidate_metadata_count"] != 1:
            continue
        try:
            _,ext=pinned_raw_url(row["source_url_private"])
        except ValueError:
            continue
        by_type[ext].append(row)
    selected=[]
    for ext in sorted(ALLOWED_SUFFIXES):
        cohort=sorted(by_type[ext],key=lambda x:x["source_expected_sha256"])
        require(cohort,"EMPTY_PILOT_COHORT_"+ext)
        n=min(per_format,len(cohort))
        indices=[((2*i+1)*len(cohort))//(2*n) for i in range(n)]
        selected.extend(cohort[i] for i in sorted(set(indices)))
    require(len({r["candidate_id"] for r in selected})==len(selected),"PILOT_DUPLICATE")
    return selected


def fetch_one(row, dest, *, max_bytes, session):
    import requests
    expected=row["source_expected_sha256"]
    require(SHA.fullmatch(expected) and row["candidate_id"].endswith(expected),
            "EXPECTED_SHA_DRIFT")
    url,extension=pinned_raw_url(row["source_url_private"])
    out={
        "canonical_asset_id":row["canonical_asset_id"],
        "candidate_id":row["candidate_id"],
        "source_expected_sha256":expected,
        "historical_split":row["historical_split"],
        "file_type":extension,
        "status":"NOT_DOWNLOADED",
        "byte_count":0,
        "retrieved_source_sha256":None,
        "source_bytes_recovered":False,
        "current_license_revalidated":False,
        "training_authorized":False,
    }
    # Reuse prior private quarantine object only after fresh FULL SHA verification.
    # A mismatched cache is never silently overwritten or counted as recovered.
    cached=dest/(expected+extension)
    if cached.exists():
        if cached.is_file():
            digest=hashlib.sha256()
            with cached.open("rb") as stream:
                for chunk in iter(lambda:stream.read(8<<20),b""):
                    digest.update(chunk)
            if digest.hexdigest()==expected:
                out["status"]="EXACT_SOURCE_CACHED_AND_REVERIFIED"
                out["byte_count"]=cached.stat().st_size
                out["retrieved_source_sha256"]=expected
                out["source_bytes_recovered"]=True
                out["private_staged_filename"]=cached.name
                return out
        out["status"]="CACHED_ORIGINAL_SHA_DRIFT__REJECTED"
        return out
    temp=dest/(expected+".download-partial")
    try:
        with session.get(url,stream=True,timeout=(12,60),allow_redirects=False,
                         headers={"User-Agent":"RealSaS-Corpus4008-source-audit/1.0"}) as response:
            if response.status_code != 200:
                out["status"]="HTTP_STATUS_"+str(response.status_code)
                return out
            content_type=response.headers.get("Content-Type","").lower()
            if "text/html" in content_type:
                out["status"]="HTML_BODY_NOT_ORIGINAL"
                return out
            length=response.headers.get("Content-Length")
            if length and int(length)>max_bytes:
                out["status"]="ORIGINAL_TOO_LARGE_FOR_PILOT"
                return out
            h=hashlib.sha256()
            size=0
            with temp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=256*1024):
                    if not chunk: continue
                    size+=len(chunk)
                    if size>max_bytes:
                        out["status"]="ORIGINAL_TOO_LARGE_FOR_PILOT"
                        return out
                    h.update(chunk);f.write(chunk)
            out["byte_count"]=size
            out["retrieved_source_sha256"]=h.hexdigest()
            if size == 0:
                out["status"]="EMPTY_SOURCE_BYTES"
            elif h.hexdigest()!=expected:
                out["status"]=("GIT_LFS_POINTER_NOT_ORIGINAL"
                               if temp.read_bytes()[:80].startswith(
                                   b"version https://git-lfs.github.com/spec/v1")
                               else "SHA_MISMATCH__REACQUISITION_NOT_ACCEPTED")
            else:
                out["status"]="EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING"
                out["source_bytes_recovered"]=True
                final=dest/(expected+extension)
                temp.replace(final)
                out["private_staged_filename"]=final.name
    except requests.RequestException as error:
        out["status"]="TRANSPORT_"+type(error).__name__
    finally:
        if temp.exists():temp.unlink()
    return out


def run(input_file, output_dir, *, per_format, max_bytes):
    import requests
    doc=json.loads(input_file.read_text())
    require(doc.get("schema")=="RealSaS.Corpus4008MissingObjaverseRecoveryPlan.v1" and
            doc["summary"].get("missing_original_count")==3349 and
            doc["summary"].get("training_authorized") is False, "INPUT_RECOVERY_PLAN")
    require(0<max_bytes<=MAX_BYTES_DEFAULT,"MAX_BYTES_LIMIT")
    candidates=choose_pilot(doc["missing_originals_private"],per_format)
    quarantine=output_dir/"verified_quarantine_objects"
    quarantine.mkdir(parents=True,exist_ok=True)
    with requests.Session() as s:
        rows=[fetch_one(r,quarantine,max_bytes=max_bytes,session=s) for r in candidates]
    counts=Counter(r["status"] for r in rows)
    public={
        "schema":SCHEMA,"status":"PINNED_GITHUB_REACQUISITION_PILOT__NO_TRAINING",
        "candidate_count":len(rows),
        "per_filetype_candidate_counts":dict(sorted(Counter(r["file_type"] for r in rows).items())),
        "result_statuses":dict(sorted(counts.items())),
        "exact_original_bytes_recovered":sum(r["source_bytes_recovered"] for r in rows),
        "recovered_byte_count":sum(r["byte_count"] for r in rows if r["source_bytes_recovered"]),
        "current_license_revalidated_count":0,
        "training_authorized":False,
        "source_rest_pose_courted":False,
        "anatomical_camera_courted":False,
        "private_quarantine_only":True,
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"RECOVERY_PILOT_PRIVATE.json").write_text(json.dumps(
        {"summary":public,"subjects_private":rows},indent=2,sort_keys=True)+"\n")
    (output_dir/"RECOVERY_PILOT_PUBLIC_COUNTS.json").write_text(
        json.dumps(public,indent=2,sort_keys=True)+"\n")
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan-private",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--per-format",type=int,default=3)
    p.add_argument("--max-original-mib",type=int,default=64)
    a=p.parse_args()
    result=run(a.plan_private,a.output,per_format=a.per_format,
               max_bytes=a.max_original_mib*1024*1024)
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
