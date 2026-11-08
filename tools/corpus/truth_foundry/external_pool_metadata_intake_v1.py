#!/usr/bin/env python3
"""Pinned, metadata-only external-pool ingestion for Truth Foundry.

No raw 3D bytes, no heavy NPZ/PT loading, no dataset training admission.
All source IDs, license statements and SHA matches are unverified until
the separate source-byte, legal, lineage and head-specific courts run.

The default source manifest is canonical/REALSAS_TRUTH_FOUNDRY_EXTERNAL_POOL_DERIVABILITY_V1.json.
Network functions import huggingface_hub only inside live download, allowing
fixture tests to run without its installation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

SCHEMA = "RealSaS.TruthFoundryExternalMetadataIntake.v1"
CATALOG_SCHEMA = "RealSaS.ExternalPoolTruthDerivability.v1"
SOURCE_SHA = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_LIMIT = 400000
MAX_METADATA_BYTES = 96 * 1024 * 1024
CANDIDATE_COLUMNS = (
    "pool", "repo_revision", "metadata_file", "row_number", "provider_record_id",
    "provider_reported_sha256", "provider_file_identifier", "provider_license",
    "master_exact_sha_candidate_match_count", "master_asset_ids_candidate_match",
    "source_identity_court", "original_asset_rights_court",
    "train_eligible", "head_truth_admitted",
)


def require(value, code):
    if not value:
        raise ValueError("EXTERNAL_POOL_INTAKE_" + code)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_catalog(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(obj.get("schema") == CATALOG_SCHEMA, "CATALOG_SCHEMA")
    require(obj.get("training_admission") is False and obj.get("pool_admission") is False,
            "CATALOG_AUTHORITY")
    pools = obj.get("pools", [])
    require(len(pools) >= 4 and len({p["id"] for p in pools}) == len(pools),
            "POOL_IDS")
    heads = {"IRIS_GEOMETRY","IRIS_SEMANTIC","IRIS_GRAPHICS","TESSA","AXIS","MIRA"}
    for p in pools:
        require(set(p["per_head"]) == heads, "HEAD_COVERAGE:" + p["id"])
        require(p.get("primary_identity") and p.get("original_source_route"),
                "MISSING_SOURCE_ROUTE:" + p["id"])
        for info in p["per_head"].values():
            require(info.get("requires") and info.get("builder") and
                    info.get("status") in obj["head_status_enum"],
                    "UNSUPPORTED_SUPERVISION_ROUTE")
    return obj


def make_original_sha_index(source_census_path: Path | None):
    out = {}
    if not source_census_path:
        return out
    with source_census_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 4363, "LOCAL_SOURCE_ROSTER_COUNT")
    for row in rows:
        require(row["training_authorized"].lower() == "false", "SOURCE_ALREADY_AUTHORIZED")
        key = row.get("source_sha256", "")
        if SOURCE_SHA.fullmatch(key):
            out.setdefault(key, []).append(row["corpus"] + ":" + row["asset_id"])
    return out


def public_repo_info(api, repo_id):
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=False)
    rev = getattr(info, "sha", None)
    require(rev and SOURCE_SHA.fullmatch(rev) is not None or
            rev and re.fullmatch("[0-9a-f]{40}", rev) is not None,
            "REPOSITORY_REVISION_NOT_PINNED:" + repo_id)
    return info


def read_index_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        require(headers, "CSV_NO_HEADER:" + path.name)
        rows = list(reader)
    require(len(rows) <= CANDIDATE_LIMIT, "CSV_TOO_LARGE")
    return headers, rows


def original_sha_from_record(row: dict):
    for field in ("sha256", "source_sha256", "original_sha256", "fileSha256"):
        raw = str(row.get(field, "")).lower().strip()
        if SOURCE_SHA.fullmatch(raw):
            return raw
    return ""


def select_id(row: dict, row_number: int):
    for field in ("uuid", "id", "object_type", "fileIdentifier", "object_id",
                  "asset_id", "sha256", "name", "source"):
        value = str(row.get(field, "")).strip()
        if value:
            return value[:1024]
    return "UNKEYED_ROW_" + str(row_number)


def parse_indexes(pool, repo_rev, repo_dir: Path, sha_index: dict):
    rows = []
    summaries = []
    for filename in pool.get("metadata_files", []):
        if not filename.lower().endswith(".csv"):
            continue
        path = repo_dir / filename
        require(path.is_file(), "REQUIRED_METADATA_MISSING:" + pool["id"] + ":" + filename)
        headers, data = read_index_rows(path)
        matches = 0
        sha_values = set()
        names = set()
        for row_number, data_row in enumerate(data, 1):
            id_value = select_id(data_row, row_number)
            record_sha = original_sha_from_record(data_row)
            identity_matches = sha_index.get(record_sha, []) if record_sha else []
            if record_sha:
                sha_values.add(record_sha)
            if identity_matches:
                matches += 1
            names.add(id_value)
            rows.append({
                "pool":pool["id"], "repo_revision":repo_rev,
                "metadata_file":filename, "row_number":row_number,
                "provider_record_id":id_value,
                "provider_reported_sha256":record_sha,
                "provider_file_identifier":str(data_row.get("fileIdentifier", ""))[:1024],
                "provider_license":str(data_row.get("license", ""))[:256],
                "master_exact_sha_candidate_match_count":len(identity_matches),
                "master_asset_ids_candidate_match":";".join(identity_matches),
                "source_identity_court":"UNKNOWN__PROVIDER_METADATA_ONLY",
                "original_asset_rights_court":"UNKNOWN__PER_ORIGINAL_LICENSE_REQUIRED",
                "train_eligible":False, "head_truth_admitted":False,
            })
            require(len(rows) <= CANDIDATE_LIMIT, "TOO_MANY_CANDIDATES")
        summaries.append({
            "filename":filename, "rows":len(data), "unique_id_strings":len(names),
            "provider_reported_original_sha_count":len(sha_values),
            "sha_match_candidates_not_deduplicated":matches,
            "headers":headers,
        })
    return rows, summaries


def run(catalog_path: Path, out: Path, *, source_census: Path | None = None,
        api=None, downloader=None):
    cfg = load_catalog(catalog_path)
    out.mkdir(parents=True, exist_ok=True)
    sha_index = make_original_sha_index(source_census)
    if api is None or downloader is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi()
        downloader = downloader or hf_hub_download
    all_rows = []
    manifest = []
    for pool in cfg["pools"]:
        if not pool.get("repo_id"):
            continue
        info = public_repo_info(api, pool["repo_id"])
        revision = info.sha
        pool_dir = out / "provider_metadata" / pool["id"]
        pool_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for filename in pool["metadata_files"]:
            require(not Path(filename).is_absolute() and
                    ".." not in Path(filename).parts,
                    "UNSAFE_METADATA_FILENAME")
            path = downloader(repo_id=pool["repo_id"], filename=filename,
                              repo_type="dataset", revision=revision,
                              local_dir=str(pool_dir))
            path = Path(path)
            size = path.stat().st_size
            require(0 < size < MAX_METADATA_BYTES, "METADATA_FILE_SIZE:" + filename)
            files.append({"file":filename, "bytes":size, "sha256":sha256(path)})
        candidate_rows, index_summary = parse_indexes(pool, revision, pool_dir, sha_index)
        all_rows.extend(candidate_rows)
        manifest.append({
            "pool":pool["id"], "upstream_repo":pool["repo_id"],
            "upstream_pinned_revision":revision,
            "files":files, "csv_schema_observations":index_summary,
            "download_scope":"METADATA_FILES_ONLY__NO_MODEL_ORIGINAL_ASSET_BYTES",
            "source_rights_verified":False,
            "model_training_admitted":False,
        })
    with (out/"PROVIDER_METADATA_CANDIDATES_V1.csv").open("w",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=CANDIDATE_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    summary={
        "schema":SCHEMA, "pool_repos_intaken":len(manifest),
        "catalog_sha256":sha256(catalog_path),
        "local_source_census_sha256":sha256(source_census) if source_census else None,
        "provider_metadata_candidate_rows":len(all_rows),
        "possible_cross_corpus_sha_matches_unqualified":sum(int(r["master_exact_sha_candidate_match_count"])>0 for r in all_rows),
        "source_rights_qualified":0, "source_original_bytes_verified":0,
        "semantic_regions_reviewed":0, "graphics_2d_approved":0,
        "independent_head_training_admitted":0,
        "materialization_authorized":False, "training_authorized":False,
        "repos":manifest,
        "outputs_sha256":{p.name:sha256(p) for p in sorted(out.glob("*.csv"))},
    }
    (out/"SUMMARY_V1.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-census",type=Path,default=None)
    a=p.parse_args()
    print(json.dumps(run(a.catalog,a.out,source_census=a.source_census),indent=2))


if __name__=="__main__":
    main()
