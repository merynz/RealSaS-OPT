#!/usr/bin/env python3
"""M2 identity gate for the structured predicted-depth bridge.

This runner opens no scientific tolerance cell. It proves that epsilon=0 is an
identity intervention on one already-open E0 calibration family and that the
resulting D2-style carrier is accepted by the current typed Compiler surface adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
COMPILER_DIR = REPO_ROOT / "compiler"
if str(COMPILER_DIR) not in sys.path:
    sys.path.insert(0, str(COMPILER_DIR))

from depth_corruption_v1 import DepthCorruptionSpec
from bridge_persistence_v1 import build_corrupted_persistence_carrier
from realsas_compiler_core.surface import rigging_surface_from_d2_arrays


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--clean-e0b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    asset_dir = Path(args.asset_dir).resolve()
    clean_path = Path(args.clean_e0b).resolve()
    out_path = Path(args.out).resolve()

    spec = DepthCorruptionSpec(epsilon=0.0, ell_px=0.0, asymmetry="ALL8")
    carrier = build_corrupted_persistence_carrier(asset_dir, spec)

    with np.load(clean_path, allow_pickle=False) as z:
        source_view_equal = np.array_equal(carrier.source_view, z["source_view"])
        source_row_equal = np.array_equal(carrier.source_row, z["source_row"])
        P_equal = np.array_equal(carrier.P, z["P"])
        support_equal = np.array_equal(carrier.support, z["support_mask"])
        matched_equal = np.array_equal(carrier.matched_row, z["matched_row"])
        source_grid = carrier.raster_xy[np.arange(len(carrier.P)), carrier.source_view]
        source_grid_equal = np.array_equal(source_grid, z["source_grid"])
        clean_support_pairs = int(np.asarray(z["support_mask"]).sum())

    surface = rigging_surface_from_d2_arrays(
        carrier.P,
        carrier.support,
        carrier.raster_xy,
        authority_label="E0_M2_ZERO_CORRUPTION",
        persistence_label="MUTUAL_P003",
    )

    checks = {
        "source_view_equal": bool(source_view_equal),
        "source_row_equal": bool(source_row_equal),
        "P_bit_equal": bool(P_equal),
        "support_equal": bool(support_equal),
        "matched_row_equal": bool(matched_equal),
        "source_grid_bit_equal": bool(source_grid_equal),
        "support_pair_count_equal": int(carrier.support.sum()) == clean_support_pairs,
        "typed_compiler_surface_node_count_512": len(surface.surface_nodes) == 512,
        "typed_compiler_schema_ok": surface.schema_version == "RealSaS.RiggingSurfaceIR.v1",
    }
    passed = all(checks.values())

    report = {
        "schema": "RealSaS.DepthToleranceBridge.M2ZeroCorruptionParity.v1",
        "status": "PASS" if passed else "FAIL",
        "scientific_tolerance_cells_opened": False,
        "asset_id": carrier.asset_id,
        "asset_dir_name": asset_dir.name,
        "clean_e0b_sha256": sha256_file(clean_path),
        "checks": checks,
        "support_pairs_bridge": int(carrier.support.sum()),
        "support_pairs_clean": clean_support_pairs,
        "matched_row_diff_count": 0 if matched_equal else int(np.count_nonzero(carrier.matched_row != np.load(clean_path, allow_pickle=False)["matched_row"])),
        "typed_surface_node_count": len(surface.surface_nodes),
        "typed_surface_lineage_hash": surface.geometry_lineage_hash,
        "proxy27": "CLOSED",
        "dev32": "CLOSED",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
