from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_SPLIT_FREEZE_SHA256 = "9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66"
EXPECTED_SPLIT_FREEZE_SCHEMA = "RealSaS.IRISControlledV1.SplitFreeze.v1"
EXPECTED_USABLE = 3930
EXPECTED_OPEN = 3248
EXPECTED_FIT = 2935
EXPECTED_TUNE = 313
CONFIRMATORY_ASSETS = 256
OPEN_SPLITS = ("FIT", "TUNE")
SEED_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationSeed.v1"


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(asset_id: str) -> str:
    return hashlib.sha256(("repr-v1:" + asset_id).encode("utf-8")).hexdigest()


def strongest_capability(record: dict) -> str:
    caps = record.get("capabilities") or {}
    if caps.get("arachne"):
        return "ARACHNE"
    if caps.get("geppetto"):
        return "GEPPETTO"
    if caps.get("iris"):
        return "IRIS"
    raise RuntimeError(f"selection record lacks IRIS capability: {record.get('canonical_asset_id')}")


def atomic_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_authorities(split_path: str | Path, selection_path: str | Path) -> tuple[dict, dict]:
    split_sha = sha256_file(split_path)
    if split_sha != EXPECTED_SPLIT_FREEZE_SHA256:
        raise RuntimeError(f"split freeze SHA mismatch expected={EXPECTED_SPLIT_FREEZE_SHA256} got={split_sha}")
    freeze = json.load(open(split_path, encoding="utf-8"))
    if freeze.get("schema") != EXPECTED_SPLIT_FREEZE_SCHEMA:
        raise RuntimeError(f"split freeze schema mismatch: {freeze.get('schema')}")
    if int(freeze.get("usable_count", -1)) != EXPECTED_USABLE or int(freeze.get("open_count", -1)) != EXPECTED_OPEN:
        raise RuntimeError("controlled population count drift")
    counts = {name: len(freeze["splits"][name]["asset_ids"]) for name in freeze["splits"]}
    if counts.get("FIT") != EXPECTED_FIT or counts.get("TUNE") != EXPECTED_TUNE:
        raise RuntimeError(f"open split count drift: {counts}")
    if sum(counts.values()) != EXPECTED_USABLE:
        raise RuntimeError(f"split membership total drift: {sum(counts.values())}")

    expected_selection_sha = freeze["source_authorities"]["CANONICAL_VARIANT_SELECTION.json"]
    selection_sha = sha256_file(selection_path)
    if selection_sha != expected_selection_sha:
        raise RuntimeError(f"canonical selection SHA mismatch expected={expected_selection_sha} got={selection_sha}")
    selection = json.load(open(selection_path, encoding="utf-8"))
    if not isinstance(selection, dict):
        raise RuntimeError("canonical selection must be an asset-id mapping")
    return freeze, selection


def open_rows(freeze: dict, selection: dict) -> list[dict]:
    rows = []
    seen = set()
    excluded = set()
    for key in ("repair_pending_assets", "shape_key_quarantine_assets", "all8_blank_assets"):
        excluded.update(freeze.get("excluded", {}).get(key, []))
    for split in OPEN_SPLITS:
        entry = freeze["splits"][split]
        if "OPEN" not in str(entry.get("access", "")):
            raise RuntimeError(f"{split} no longer marked OPEN")
        for asset_id in entry["asset_ids"]:
            if asset_id in seen:
                raise RuntimeError(f"duplicate controlled membership: {asset_id}")
            if asset_id in excluded:
                raise RuntimeError(f"excluded asset leaked into open freeze: {asset_id}")
            seen.add(asset_id)
            if asset_id not in selection:
                raise RuntimeError(f"controlled asset absent from canonical selection: {asset_id}")
            rec = selection[asset_id]
            if str(rec.get("split", "")).upper() != split:
                raise RuntimeError(f"split drift {asset_id}: freeze={split} selection={rec.get('split')}")
            if not bool((rec.get("capabilities") or {}).get("iris")):
                raise RuntimeError(f"non-IRIS asset in controlled freeze: {asset_id}")
            rows.append({
                "asset_id": asset_id,
                "split": split,
                "source_registry_id": rec["source_registry_id"],
                "capability_class": strongest_capability(rec),
                "candidate_id": rec.get("candidate_id"),
            })
    if len(rows) != EXPECTED_OPEN:
        raise RuntimeError(f"open membership expected={EXPECTED_OPEN} got={len(rows)}")
    return rows


def select_confirmatory(rows: list[dict], target: int = CONFIRMATORY_ASSETS) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        strata[(row["source_registry_id"], row["split"], row["capability_class"])].append(row)
    picked = {}
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: stable_key(r["asset_id"]))
        picked[bucket[0]["asset_id"]] = bucket[0]
    if len(picked) > target:
        raise RuntimeError(f"required strata {len(picked)} exceed panel target {target}")
    for row in sorted(rows, key=lambda r: stable_key(r["asset_id"])):
        if len(picked) >= target:
            break
        picked.setdefault(row["asset_id"], row)
    if len(picked) != target:
        raise RuntimeError(f"panel selection expected={target} got={len(picked)}")
    # Preserve selection procedure order; downstream runner uses the same stable selection rule.
    return list(picked.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the frozen 256-asset R0-R3 representation seed; optimizer=0")
    ap.add_argument("--split-freeze", required=True)
    ap.add_argument("--selection-json", required=True)
    ap.add_argument("--root", required=True, help="master corpus root; used only to bind expected source paths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    freeze, selection = verify_authorities(args.split_freeze, args.selection_json)
    rows = open_rows(freeze, selection)
    chosen = select_confirmatory(rows)
    root = Path(args.root)
    records = []
    for row in chosen:
        aid = row["asset_id"]
        records.append({
            **row,
            "master_asset_root": str(root / "master" / "assets" / aid),
            "views": 8,
            "authority_resolution": 1024,
        })
    result = {
        "schema": SEED_SCHEMA,
        "optimizer_steps": 0,
        "status": "FROZEN_CONFIRMATORY_PANEL_SEED",
        "record_count": len(records),
        "split_counts": dict(sorted(Counter(r["split"] for r in records).items())),
        "stratum_counts": dict(sorted(Counter(
            f"{r['source_registry_id']}|{r['split']}|{r['capability_class']}" for r in records
        ).items())),
        "records": records,
        "source_authorities": {
            "split_freeze_sha256": sha256_file(args.split_freeze),
            "canonical_selection_sha256": sha256_file(args.selection_json),
            "canonical_selection_expected_sha256": freeze["source_authorities"]["CANONICAL_VARIANT_SELECTION.json"],
            "training_eligibility_sha256": freeze["source_authorities"]["IRIS_TRAINING_ELIGIBILITY_V1.json"],
            "stage_b6_result_sha256": freeze["source_authorities"]["POST_CORPUS_STAGE_B6_FROZEN_REPAIR_STAGING_RESULT_V1.json"],
        },
        "selection_rule": "one witness per non-empty (provider,split,strongest-capability) stratum by SHA256('repr-v1:'+asset_id), then fill to 256 by the same stable hash; no replacement after apparatus failure",
        "sealed_splits_opened": False,
    }
    atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "record_count": result["record_count"],
        "split_counts": result["split_counts"],
        "strata": len(result["stratum_counts"]),
        "out_sha256": sha256_file(args.out),
        "sealed_splits_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
