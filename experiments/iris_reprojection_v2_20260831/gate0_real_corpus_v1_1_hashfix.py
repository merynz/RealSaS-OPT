from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import gate0_real_corpus_v1 as base

SCHEMA_SUFFIX = ".hashfix1"


def set_sha_sorted_newline_v1(ids: Sequence[str]) -> str:
    """Diagnostic-only local set digest; not the historical set-hash authority."""
    return base.set_sha(ids)


def validate_declared_train512_set_hash(membership: dict) -> None:
    if membership.get("train512_set_sha256") != base.EXPECTED_TRAIN512_SET_SHA256:
        raise RuntimeError("TRAIN512_DECLARED_SET_SHA_DRIFT")


def exact_population_match(measured_ids: Sequence[str], authoritative_ids: Sequence[str]) -> bool:
    return (
        len(measured_ids) == len(authoritative_ids)
        and len(set(measured_ids)) == len(authoritative_ids)
        and set(measured_ids) == set(authoritative_ids)
    )


def load_membership(path: Path, scientific: bool) -> tuple[dict, list[str]]:
    raw_sha = base.sha256_file(path)
    o = json.loads(path.read_text(encoding="utf-8"))
    ids = list(o.get("train_order_512", []))
    if scientific:
        if raw_sha != base.EXPECTED_MEMBERSHIP_RAW_SHA256:
            raise RuntimeError(f"MEMBERSHIP_RAW_SHA_DRIFT:{raw_sha}")
        if o.get("content_sha256") != base.EXPECTED_MEMBERSHIP_CONTENT_SHA256:
            raise RuntimeError("MEMBERSHIP_CONTENT_SHA_DRIFT")
        cp = dict(o)
        field = cp.pop("content_sha256", None)
        if field != base.EXPECTED_MEMBERSHIP_CONTENT_SHA256:
            raise RuntimeError("MEMBERSHIP_CONTENT_SHA_FIELD_DRIFT")
        got = base.canonical_sha(cp)
        if got != base.EXPECTED_MEMBERSHIP_CONTENT_SHA256:
            raise RuntimeError(f"MEMBERSHIP_CONTENT_SHA_RECOMPUTE_DRIFT:{got}")
        if len(ids) != base.EXPECTED_ASSET_COUNT or len(set(ids)) != base.EXPECTED_ASSET_COUNT:
            raise RuntimeError("TRAIN512_COUNT_OR_DUPLICATE_DRIFT")
        validate_declared_train512_set_hash(o)
    if not ids:
        raise RuntimeError("EMPTY_MEMBERSHIP")
    return o, ids


def run(master_root: Path, membership_path: Path, out_path: Path, workers: int, mode: str, smoke_count: int) -> dict:
    scientific = mode == "scientific"
    membership, authoritative_ids = load_membership(membership_path, scientific=scientific)

    # Patch only the membership verifier used by the frozen V1 measurement code.
    # Geometry, sampling, thresholds and all measured operations remain V1 exact.
    old_loader = base.load_membership
    base.load_membership = load_membership
    try:
        out = base.run(master_root, membership_path, out_path, workers, mode, smoke_count)
    finally:
        base.load_membership = old_loader

    measured_ids = [r["canonical_asset_id"] for r in out.get("records", [])]
    population_match = exact_population_match(measured_ids, authoritative_ids) if scientific else None

    pop = dict(out.get("population", {}))
    old_local_digest = pop.pop("measured_set_sha256", None)
    pop["membership_declared_train512_set_sha256"] = membership.get("train512_set_sha256")
    pop["expected_historical_train512_set_sha256"] = base.EXPECTED_TRAIN512_SET_SHA256 if scientific else None
    pop["measured_set_matches_authoritative_membership"] = population_match
    pop["measured_set_sha256_sorted_newline_v1"] = old_local_digest
    pop["set_hash_semantics"] = (
        "Historical train512_set_sha256 is a sealed field of the exact membership artifact; "
        "sorted-newline digest is local diagnostic only and is not compared across schemes."
    )
    out["population"] = pop

    if scientific and population_match is not True:
        out["status"] = "FAIL_REAL_GATE0_HARD_BLOCK"

    out["schema"] = str(out.get("schema", "RealSaS.IRIS.ReprojectionV2.RealCorpusGate0Measurement.v1")) + SCHEMA_SUFFIX
    out.pop("content_sha256", None)
    out["content_sha256"] = base.canonical_sha(out)
    base.atomic_json(out_path, out)
    print(json.dumps({k: out[k] for k in ("schema", "status", "population", "aggregate", "content_sha256")}, indent=2, sort_keys=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-root", type=Path, required=True)
    ap.add_argument("--membership-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    ap.add_argument("--smoke-count", type=int, default=2)
    args = ap.parse_args()
    result = run(args.master_root, args.membership_json, args.out, args.workers, args.mode, args.smoke_count)
    if args.mode == "scientific" and not result["status"].startswith("PASS_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
