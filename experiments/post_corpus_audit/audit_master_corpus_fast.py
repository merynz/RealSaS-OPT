#!/usr/bin/env python3
"""RealSaS fast post-corpus audit.

Exhaustive Drive-API metadata census for all selected assets/views, followed by
stratified deep binary/content auditing on a bounded sample via the existing
read-only deep auditor. No rendering, training, consumer export, or corpus mutation.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

from google.colab import auth
import google.auth
from googleapiclient.discovery import build

ROOT_DEFAULT = "/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3"
ASSETS_PARENT_ID = "1zMz3eg1x0yjU2wYg0lYWXK1AlgxOZLTW"
BUILD_EXPECTED = "REALSAS_MASTER_1024_V4_3_LOCAL_FIRST_FULL_PRODUCTION_20260822"
VIEW_NAMES = [f"V{i}" for i in range(8)]
REQUIRED_VIEW_FILES = [
    "cel_clean.png",
    "ink_cel.png",
    "cel_clean_512.png",
    "ink_cel_512.png",
    "raster_authority.npz",
    "camera.json",
]
MODIFIED_AFTER = "2026-08-22T23:00:00Z"


def load_slow_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("realsas_deep_auditor", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def drive_service():
    auth.authenticate_user()
    creds, _ = google.auth.default()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_all(svc, q: str):
    out = []
    token = None
    pages = 0
    while True:
        resp = svc.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken,files(id,name,mimeType,size,parents,modifiedTime)",
            pageSize=1000,
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        pages += 1
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return out, pages


def parent_hits(rows, valid_parent_ids: set[str]):
    hits = collections.defaultdict(list)
    for r in rows:
        for p in r.get("parents") or []:
            if p in valid_parent_ids:
                hits[p].append(r)
    return hits


def summarize_children(parent_ids: set[str], hits, require_nonzero_size=False):
    missing = []
    duplicates = []
    zero_size = []
    for p in parent_ids:
        rows = hits.get(p, [])
        if not rows:
            missing.append(p)
        elif len(rows) > 1:
            duplicates.append({"parent": p, "count": len(rows), "ids": [x["id"] for x in rows[:5]]})
        if require_nonzero_size:
            for r in rows:
                try:
                    if int(r.get("size") or 0) <= 0:
                        zero_size.append(r["id"])
                except Exception:
                    zero_size.append(r["id"])
    return {
        "parent_count": len(parent_ids),
        "present_parent_count": len(parent_ids) - len(missing),
        "missing_count": len(missing),
        "duplicate_parent_count": len(duplicates),
        "zero_size_count": len(zero_size),
        "missing_parent_examples": missing[:20],
        "duplicate_examples": duplicates[:20],
        "zero_size_examples": zero_size[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--deep-sample", type=int, default=64)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out_dir) if args.out_dir else root / "reports" / "post_corpus_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    slow_path = outdir / "audit_master_corpus.py"
    if not slow_path.exists():
        raise FileNotFoundError(f"missing deep auditor: {slow_path}")
    slow = load_slow_module(slow_path)

    selected_raw = slow.read_json(root / "metadata" / "CANONICAL_VARIANT_SELECTION.json")
    if isinstance(selected_raw, dict):
        selected = selected_raw.get("selected") if isinstance(selected_raw.get("selected"), list) else list(selected_raw.values())
    else:
        selected = selected_raw
    if not isinstance(selected, list):
        raise RuntimeError("unexpected selection schema")
    selected_ids = {x["canonical_asset_id"] for x in selected}
    print(f"[central] selected={len(selected_ids)}", flush=True)

    split_bad = []
    for s in selected:
        exp = slow.deterministic_split(s["canonical_asset_id"], False)
        if s.get("split") != exp:
            split_bad.append({"asset": s["canonical_asset_id"], "stored": s.get("split"), "expected": exp})

    svc = drive_service()
    folder_mime = "application/vnd.google-apps.folder"
    api_pages = 0

    rows, p = list_all(svc, f"'{ASSETS_PARENT_ID}' in parents and mimeType='{folder_mime}' and trashed=false")
    api_pages += p
    assets_by_name = collections.defaultdict(list)
    for r in rows:
        assets_by_name[r["name"]].append(r)
    missing_asset_names = sorted(selected_ids - set(assets_by_name))
    unexpected_asset_names = sorted(set(assets_by_name) - selected_ids)
    duplicate_asset_names = {k: [x["id"] for x in v] for k, v in assets_by_name.items() if len(v) > 1 and k in selected_ids}
    selected_asset_rows = [assets_by_name[n][0] for n in selected_ids if n in assets_by_name]
    asset_ids = {r["id"] for r in selected_asset_rows}
    print(f"[api] asset folders {len(asset_ids)}/{len(selected_ids)}", flush=True)

    direct = {}
    direct_specs = [
        ("RENDER_COMPLETE.json", None, False),
        ("primary_geometry.npz", None, True),
        ("renders", folder_mime, False),
    ]
    direct_hits = {}
    for name, mime, nonzero in direct_specs:
        q = f"name='{name}' and trashed=false and modifiedTime > '{MODIFIED_AFTER}'"
        if mime:
            q += f" and mimeType='{mime}'"
        rows, p = list_all(svc, q)
        api_pages += p
        hits = parent_hits(rows, asset_ids)
        direct[name] = summarize_children(asset_ids, hits, require_nonzero_size=nonzero)
        direct_hits[name] = hits
        print(f"[api] {name}: missing={direct[name]['missing_count']} dup={direct[name]['duplicate_parent_count']}", flush=True)

    render_folder_ids = {rs[0]["id"] for rs in direct_hits["renders"].values() if len(rs) == 1}

    views = {}
    all_view_folder_ids = set()
    for vn in VIEW_NAMES:
        q = f"name='{vn}' and mimeType='{folder_mime}' and trashed=false and modifiedTime > '{MODIFIED_AFTER}'"
        rows, p = list_all(svc, q)
        api_pages += p
        hits = parent_hits(rows, render_folder_ids)
        views[vn] = summarize_children(render_folder_ids, hits)
        ids = {rs[0]["id"] for rs in hits.values() if len(rs) == 1}
        all_view_folder_ids |= ids
        print(f"[api] {vn}: {len(ids)}/{len(render_folder_ids)}", flush=True)

    files = {}
    for fn in REQUIRED_VIEW_FILES:
        q = f"name='{fn}' and trashed=false and modifiedTime > '{MODIFIED_AFTER}'"
        rows, p = list_all(svc, q)
        api_pages += p
        hits = parent_hits(rows, all_view_folder_ids)
        files[fn] = summarize_children(all_view_folder_ids, hits, require_nonzero_size=True)
        print(f"[api] {fn}: missing={files[fn]['missing_count']} zero={files[fn]['zero_size_count']}", flush=True)

    fatal_counts = [len(missing_asset_names), len(duplicate_asset_names)]
    for x in direct.values():
        fatal_counts += [x["missing_count"], x["duplicate_parent_count"], x["zero_size_count"]]
    for x in views.values():
        fatal_counts += [x["missing_count"], x["duplicate_parent_count"]]
    for x in files.values():
        fatal_counts += [x["missing_count"], x["duplicate_parent_count"], x["zero_size_count"]]
    metadata_pass = not any(fatal_counts) and len(split_bad) == 0

    deep_sel = slow.choose_deep_sample(selected, args.deep_sample)
    deep = []
    for i, s in enumerate(deep_sel, 1):
        print(f"[deep] {i}/{len(deep_sel)} {s['canonical_asset_id']}", flush=True)
        deep.append(slow.deep_asset_audit(root, s))

    result = {
        "schema_version": "realsas.post_corpus_fast_audit.v1",
        "build_id": BUILD_EXPECTED,
        "root": str(root),
        "method": {
            "full_census": "Google Drive API metadata set intersection",
            "deep_content": "deterministic stratified sample via mounted bytes",
            "modified_after_query": MODIFIED_AFTER,
            "api_pages": api_pages,
            "corpus_mutation": False,
        },
        "central": {
            "selected_count": len(selected_ids),
            "split_mismatch_count": len(split_bad),
            "split_mismatch_examples": split_bad[:20],
            "source_distribution": slow.counter(selected, "source_registry_id"),
            "split_distribution": slow.counter(selected, "split"),
            "capabilities": {c: slow.cap_count(selected, c) for c in ("iris", "geppetto", "arachne")},
        },
        "metadata_census": {
            "pass": metadata_pass,
            "asset_folders": {
                "selected": len(selected_ids),
                "present": len(asset_ids),
                "missing_count": len(missing_asset_names),
                "unexpected_count": len(unexpected_asset_names),
                "duplicate_selected_name_count": len(duplicate_asset_names),
                "missing_examples": missing_asset_names[:20],
                "unexpected_examples": unexpected_asset_names[:20],
                "duplicate_examples": dict(list(duplicate_asset_names.items())[:20]),
            },
            "direct_children": direct,
            "view_folders": views,
            "required_view_files": files,
        },
        "deep_sample_policy": {"requested": args.deep_sample, "actual": len(deep_sel), "stratified": True},
        "deep": deep,
        "training_authorized_by_this_audit": False,
        "open_items": [
            "Source-level physical duplicate/fingerprint audit remains separate from metadata completeness.",
            "Evaluated-depsgraph parity for .blend subset remains a dedicated Gate-2 check.",
            "Appearance/dependency coverage remains a dedicated Gate-3 check.",
        ],
    }

    out_json = outdir / "FAST_POST_CORPUS_AUDIT_RESULT.json"
    out_md = outdir / "FAST_POST_CORPUS_AUDIT_REPORT.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(
        "# RealSaS Fast Post-Corpus Audit\n\n"
        f"- selected: **{len(selected_ids)}**\n"
        f"- asset folders present: **{len(asset_ids)}**\n"
        f"- exhaustive metadata census: **{'PASS' if metadata_pass else 'FAIL'}**\n"
        f"- split mismatches: **{len(split_bad)}**\n"
        f"- deep sample: **{len(deep)}**\n"
        f"- Drive API pages: **{api_pages}**\n\n"
        "The full census checks file/folder existence and nonzero-size metadata for every canonical view. "
        "Binary geometry/raster authority is inspected on the deterministic stratified deep sample.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": str(out_json),
        "report": str(out_md),
        "metadata_census_pass": metadata_pass,
        "selected": len(selected_ids),
        "asset_folders_present": len(asset_ids),
        "view_folders_present": len(all_view_folder_ids),
        "expected_view_folders": len(selected_ids) * 8,
        "deep_assets": len(deep),
        "api_pages": api_pages,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
