#!/usr/bin/env python3
"""Recover the exact historical P-V5 512-family train membership.

Zero-step provenance utility. It performs no model inference or optimizer work.
Selection is derived only from the byte-pinned MASTER_VARIANTS.jsonl authority
and the preregistered per-source SHA queues.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

SOURCE_SHA = "475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913"
PROXY_SEED = "PV5_R256_FIT_SCALE_PROXY32_V1"
TRAIN_SEED = "PV5_R256_FIT_SCALE_TRAIN_V1"
PROXY_QUOTA = {
    "objaverse_animated_originals": 26,
    "quaternius_cc0": 3,
    "kaykit_cc0": 3,
}
TRAIN512_QUOTA = {
    "objaverse_animated_originals": 483,
    "quaternius_cc0": 24,
    "kaykit_cc0": 5,
}
EXPECTED_PROXY_SET_SHA = "144e5f21adcab218fa5d8226b76cc949eb3eaa3c065e0cd0a7d30a35f0ed0653"
EXPECTED_TRAIN32_SET_SHA = "ea7e99933739258f1a492326d2631d48a5b2c0018fe8181fb01428f266b3309e"
EXPECTED_TRAIN512_SET_SHA = "1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2"

def qkey(seed: str, asset_id: str) -> str:
    return hashlib.sha256(f"{seed}|{asset_id}".encode("utf-8")).hexdigest()

def set_sha(ids) -> str:
    payload = ("\n".join(sorted(ids)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def select(rows, seed, quota, excluded=frozenset()):
    by = defaultdict(list)
    for row in rows:
        if row["canonical_asset_id"] not in excluded:
            by[row["source_registry_id"]].append(row)
    out = []
    for source, n in quota.items():
        ranked = sorted(by[source], key=lambda r: qkey(seed, r["canonical_asset_id"]))
        if len(ranked) < n:
            raise RuntimeError(f"insufficient {source}: {len(ranked)} < {n}")
        out.extend(ranked[:n])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master_variants")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = Path(args.master_variants)
    raw = p.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA:
        raise SystemExit(f"source SHA mismatch: {observed}")

    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    fit = [r for r in rows if r.get("split") == "FIT"]

    proxy = select(fit, PROXY_SEED, PROXY_QUOTA)
    proxy_ids = {r["canonical_asset_id"] for r in proxy}
    train512 = select(fit, TRAIN_SEED, TRAIN512_QUOTA, proxy_ids)
    train32 = select(
        fit,
        TRAIN_SEED,
        {
            "objaverse_animated_originals": 29,
            "quaternius_cc0": 2,
            "kaykit_cc0": 1,
        },
        proxy_ids,
    )

    checks = {
        "proxy_set_sha256": set_sha(r["canonical_asset_id"] for r in proxy),
        "train32_set_sha256": set_sha(r["canonical_asset_id"] for r in train32),
        "train512_set_sha256": set_sha(r["canonical_asset_id"] for r in train512),
    }
    expected = {
        "proxy_set_sha256": EXPECTED_PROXY_SET_SHA,
        "train32_set_sha256": EXPECTED_TRAIN32_SET_SHA,
        "train512_set_sha256": EXPECTED_TRAIN512_SET_SHA,
    }
    if checks != expected:
        raise SystemExit(f"membership proof mismatch: {checks}")

    payload = {
        "schema": "RealSaS.DINO.Train512RecoveredAuthority.v1",
        "status": "RECOVERED_EQUIVALENT_SELECTION_AUTHORITY__ZERO_OPTIMIZER_STEPS",
        "source_metadata_sha256": observed,
        "proxy_seed": PROXY_SEED,
        "train_seed": TRAIN_SEED,
        "proxy_quota": PROXY_QUOTA,
        "train512_quota": TRAIN512_QUOTA,
        "proxy_asset_ids": sorted(proxy_ids),
        "train512_asset_ids": sorted(r["canonical_asset_id"] for r in train512),
        "set_hashes": checks,
        "scientific_optimizer_steps": 0,
    }
    Path(args.out).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "fit_count": len(fit),
        "proxy_count": len(proxy),
        "train512_count": len(train512),
        **checks,
    }, sort_keys=True))

if __name__ == "__main__":
    main()
