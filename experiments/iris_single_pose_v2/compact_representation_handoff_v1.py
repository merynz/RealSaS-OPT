from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def arm_row(name: str, arm: dict) -> dict:
    pooled = arm["pooled"]
    family = arm["family_tail"]
    return {
        "arm": name,
        "definition": arm["definition"],
        "queries": pooled.get("queries"),
        "top1": pooled.get("top1"),
        "top4": pooled.get("top4"),
        "top8": pooled.get("top8"),
        "physical_error": pooled.get("physical_error"),
        "reciprocal_success": pooled.get("reciprocal_success"),
        "cycle_success": pooled.get("cycle_success"),
        "ambiguous_fraction": pooled.get("ambiguous_fraction"),
        "nearest_non_equivalent_physical_gap": pooled.get("nearest_non_equivalent_physical_gap"),
        "family_top1": family.get("top1_across_families"),
        "family_top4": family.get("top4_across_families"),
        "family_top8": family.get("top8_across_families"),
        "family_error_median": family.get("asset_error_median_distribution"),
        "family_error_p90": family.get("asset_error_p90_distribution"),
        "family_error_p95": family.get("asset_error_p95_distribution"),
    }


def tail_counts(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def compact_tail_witness(row: dict) -> dict:
    keep = (
        "asset_id", "provider", "split", "capability", "component_bucket", "support_bucket",
        "pair_category", "arm", "track", "source", "target", "third", "top1", "top4", "top8",
        "physical_error", "reciprocal_success", "cycle_success", "ambiguity_set_size",
        "nearest_non_equivalent_physical_gap",
    )
    return {k: row.get(k) for k in keep}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact optimizer-zero R0-R3 evidence for continuation/handoff")
    ap.add_argument("--result", required=True)
    ap.add_argument("--hard-tail", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--witness-limit", type=int, default=100)
    args = ap.parse_args()

    result = json.load(open(args.result, encoding="utf-8"))
    if result.get("schema") != "RealSaS.RepresentationAuthority.R0R3.v1":
        raise RuntimeError(f"unexpected result schema {result.get('schema')}")
    if result.get("status") != "R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED":
        raise RuntimeError(f"compact handoff requires confirmatory measured result, got {result.get('status')}")
    if int(result.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("optimizer_steps drift")

    arms = result["arms"]
    if "R0_P_EXACT" not in arms or "R1_PN_EXACT" not in arms:
        raise RuntimeError("exact R0/R1 arms missing")
    r2_names = sorted(
        (name for name, arm in arms.items() if arm["definition"]["family"] == "R2"),
        key=lambda n: float(arms[n]["definition"]["p_sigma"]),
    )
    r3_names = sorted(
        (name for name, arm in arms.items() if arm["definition"]["family"] == "R3"),
        key=lambda n: (float(arms[n]["definition"]["p_sigma"]), float(arms[n]["definition"]["n_deg"])),
    )
    if len(r2_names) != 6 or len(r3_names) != 30:
        raise RuntimeError(f"frozen arm grid drift R2={len(r2_names)} R3={len(r3_names)}")

    tail = read_jsonl(args.hard_tail)
    limit = max(1, int(args.witness_limit))
    by_error = sorted(tail, key=lambda r: (-float(r.get("physical_error", 0.0)), str(r.get("asset_id"))))[:limit]
    finite_gap = [r for r in tail if r.get("nearest_non_equivalent_physical_gap") is not None]
    by_gap = sorted(
        finite_gap,
        key=lambda r: (float(r["nearest_non_equivalent_physical_gap"]), -float(r.get("physical_error", 0.0)), str(r.get("asset_id"))),
    )[:limit]

    out = {
        "schema": "RealSaS.RepresentationAuthority.CompactHandoff.v1",
        "optimizer_steps": 0,
        "training_authorized": False,
        "status": "MEASUREMENT_ONLY__CANONICAL_INTERPRETATION_REQUIRED",
        "source": {
            "result_path": str(args.result),
            "result_sha256": sha256_file(args.result),
            "hard_tail_path": str(args.hard_tail),
            "hard_tail_sha256": sha256_file(args.hard_tail),
        },
        "panel": {
            "asset_count": result.get("asset_count"),
            "query_slots_per_asset_max": result.get("query_slots_per_asset_max"),
            "arm_count": result.get("arm_count"),
            "mode": result.get("mode"),
            "panel_sha256": result.get("panel_sha256"),
        },
        "exact": {
            "R0_P_EXACT": arm_row("R0_P_EXACT", arms["R0_P_EXACT"]),
            "R1_PN_EXACT": arm_row("R1_PN_EXACT", arms["R1_PN_EXACT"]),
            "R0_strata": {
                "provider": arms["R0_P_EXACT"].get("by_provider"),
                "split": arms["R0_P_EXACT"].get("by_split"),
                "capability": arms["R0_P_EXACT"].get("by_capability"),
                "component_bucket": arms["R0_P_EXACT"].get("by_component_bucket"),
                "support_bucket": arms["R0_P_EXACT"].get("by_support_bucket"),
                "pair_category": arms["R0_P_EXACT"].get("by_pair_category"),
            },
            "R1_strata": {
                "provider": arms["R1_PN_EXACT"].get("by_provider"),
                "split": arms["R1_PN_EXACT"].get("by_split"),
                "capability": arms["R1_PN_EXACT"].get("by_capability"),
                "component_bucket": arms["R1_PN_EXACT"].get("by_component_bucket"),
                "support_bucket": arms["R1_PN_EXACT"].get("by_support_bucket"),
                "pair_category": arms["R1_PN_EXACT"].get("by_pair_category"),
            },
        },
        "R2_noise_envelope": [arm_row(name, arms[name]) for name in r2_names],
        "R3_noise_grid": [arm_row(name, arms[name]) for name in r3_names],
        "hard_tail": {
            "rows": len(tail),
            "by_arm": tail_counts(tail, "arm"),
            "by_provider": tail_counts(tail, "provider"),
            "by_split": tail_counts(tail, "split"),
            "by_capability": tail_counts(tail, "capability"),
            "by_pair_category": tail_counts(tail, "pair_category"),
            "by_component_bucket": tail_counts(tail, "component_bucket"),
            "by_support_bucket": tail_counts(tail, "support_bucket"),
            "worst_physical_error_witnesses": [compact_tail_witness(r) for r in by_error],
            "smallest_non_equivalent_gap_witnesses": [compact_tail_witness(r) for r in by_gap],
        },
        "structural_confusability_diagnostic": result.get("structural_confusability_diagnostic"),
        "decision_discipline": "Evidence compaction only. This artifact cannot decide P sufficiency, R4, SOI-2, checkpoint selection or training authorization.",
    }
    atomic_json(args.out, out)
    print(json.dumps({
        "status": out["status"],
        "asset_count": out["panel"]["asset_count"],
        "R2_arms": len(out["R2_noise_envelope"]),
        "R3_arms": len(out["R3_noise_grid"]),
        "hard_tail_rows": out["hard_tail"]["rows"],
        "out_sha256": sha256_file(args.out),
        "training_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
