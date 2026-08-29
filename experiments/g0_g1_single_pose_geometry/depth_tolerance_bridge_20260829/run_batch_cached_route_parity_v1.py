#!/usr/bin/env python3
"""No-outcome parity gate for the cached/vectorized structured-depth bridge route.

This runner opens no scientific tolerance outcome. It proves that the optimized
execution path preserves both distinct persistence layers:

  BASE derived matcher                  -> historical E0-B carrier (1909 on witness)
  BASE + reciprocal cycle_P <= 0.003    -> historical D2 carrier (1884 on witness)

It also checks the vectorized matcher against the scalar `derived_match_row`
contract on a nonzero structured corruption using a tiny anchor subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from bridge_grid_runtime_v1 import (
    batch_derived_rows_fixed_source,
    build_base_state,
    build_d2_mutual_p003_from_prepared,
    prepare_bridge_asset,
)
from depth_corruption_v1 import DepthCorruptionSpec
from bridge_grid_runtime_v1 import MUTUAL_P003
from surface_builder_e0_v1 import derived_match_row


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scalar_rows(P, N, G, source_view, source_half_extent, target):
    out = np.full(len(P), -1, np.int64)
    for i in range(len(P)):
        row, _ = derived_match_row(P[i], N[i], G[i], source_view, source_half_extent, target)
        out[i] = int(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--clean-e0b", required=True)
    ap.add_argument("--historical-d2-pack", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    asset_dir = Path(args.asset_dir).resolve()
    clean_path = Path(args.clean_e0b).resolve()
    pack_path = Path(args.historical_d2_pack).resolve()
    out_path = Path(args.out).resolve()

    t0 = time.perf_counter()
    prepared = prepare_bridge_asset(asset_dir, anchor_count=512)
    prepare_seconds = time.perf_counter() - t0

    zero = DepthCorruptionSpec(0.0, 0.0, "ALL8")
    t0 = time.perf_counter()
    zero_state = build_base_state(prepared, zero)
    zero_base_seconds = time.perf_counter() - t0
    base = zero_state.carrier

    with np.load(clean_path, allow_pickle=False) as z:
        clean_base_checks = {
            "P": np.array_equal(base.P, z["P"]),
            "support": np.array_equal(base.support, z["support_mask"]),
            "matched_row": np.array_equal(base.matched_row, z["matched_row"]),
            "source_view": np.array_equal(base.source_view, z["source_view"]),
            "source_row": np.array_equal(base.source_row, z["source_row"]),
        }
        base_clean_support_pairs = int(np.asarray(z["support_mask"]).sum())

    t0 = time.perf_counter()
    _state2, d2_support, d2_rows, _d2_grid, d2_X = build_d2_mutual_p003_from_prepared(prepared, zero)
    zero_d2_seconds = time.perf_counter() - t0
    with np.load(pack_path, allow_pickle=False) as z:
        d2_x_exact = np.array_equal(d2_X, z["D2_X"])
        historical_d2_support_pairs = int(np.asarray(z["D2_X"])[:, 3:11].sum())

    # Nonzero scalar-vs-vector matcher parity on a tiny cleanly re-prepared subset.
    tiny = prepare_bridge_asset(asset_dir, anchor_count=8)
    stress = DepthCorruptionSpec(0.003, 16.0, "ONE_BAD_HASHED")
    stress_state = build_base_state(tiny, stress)
    obs = stress_state.observable_views
    c = stress_state.carrier
    scalar_equal = True
    scalar_pair_count = 0
    for s in range(8):
        ids = np.flatnonzero(np.asarray(c.source_view, np.int64) == s)
        if not len(ids):
            continue
        for tv in range(8):
            if tv == s:
                continue
            P = c.P[ids]
            # Source normals/grids are reconstructed from the exact corrupted source rows.
            sr = np.asarray(c.source_row, np.int64)[ids]
            N = obs[s].N_derived[sr]
            G = obs[s].grid[sr]
            fast = batch_derived_rows_fixed_source(P, N, G, s, obs[s].half_extent, obs[tv])
            slow = _scalar_rows(P, N, G, s, obs[s].half_extent, obs[tv])
            scalar_pair_count += len(ids)
            scalar_equal = scalar_equal and np.array_equal(fast, slow)

    checks = {
        "zero_base_clean_P_bit_exact": bool(clean_base_checks["P"]),
        "zero_base_clean_support_bit_exact": bool(clean_base_checks["support"]),
        "zero_base_clean_matched_row_bit_exact": bool(clean_base_checks["matched_row"]),
        "zero_base_clean_source_view_bit_exact": bool(clean_base_checks["source_view"]),
        "zero_base_clean_source_row_bit_exact": bool(clean_base_checks["source_row"]),
        "zero_base_support_pair_count_equal": int(base.support.sum()) == base_clean_support_pairs,
        "zero_d2_X_bit_exact": bool(d2_x_exact),
        "zero_d2_support_pair_count_equal": int(d2_support.sum()) == historical_d2_support_pairs,
        "nonzero_batch_forward_matches_scalar_contract": bool(scalar_equal),
    }
    passed = all(checks.values())
    report = {
        "schema": "RealSaS.DepthToleranceBridge.BatchCachedRouteParity.v1",
        "status": "PASS" if passed else "FAIL",
        "asset_id": asset_dir.name,
        "checks": checks,
        "route_semantics": {
            "base_stage": "DERIVED_MATCH_ROW_PRE_RECIPROCAL_CYCLE",
            "historical_d2_stage": "BASE_PLUS_RECIPROCAL_CYCLE_P_LE_0P003",
            "mutual_cycle_threshold_P": MUTUAL_P003,
        },
        "support_pairs": {
            "base_bridge": int(base.support.sum()),
            "base_historical_clean": base_clean_support_pairs,
            "d2_bridge": int(d2_support.sum()),
            "d2_historical_pack": historical_d2_support_pairs,
        },
        "nonbinding_runtime_diagnostics_seconds": {
            "prepare_clean_asset_once": prepare_seconds,
            "zero_base_cell_after_prepare": zero_base_seconds,
            "zero_d2_cell_after_prepare": zero_d2_seconds,
        },
        "tiny_scalar_comparison_anchor_count": 8,
        "tiny_scalar_comparison_forward_pair_batches_rows": scalar_pair_count,
        "source_sha256": {
            "clean_e0b": sha256_file(clean_path),
            "historical_d2_pack": sha256_file(pack_path),
        },
        "firewalls": {
            "scientific_tolerance_outcomes_opened": False,
            "proxy27": "CLOSED",
            "dev32": "CLOSED",
            "training_steps": 0,
            "runtime_diagnostics_binding_to_science": False,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
