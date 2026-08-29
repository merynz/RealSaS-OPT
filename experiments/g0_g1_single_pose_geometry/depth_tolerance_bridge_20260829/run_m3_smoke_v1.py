#!/usr/bin/env python3
"""Run the sealed M3 non-binding apparatus smoke cells on one already-open E0 family.

This runner validates traversal only. It must not be used to infer a product tolerance.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parents[2]
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
if str(REPO_ROOT/"compiler") not in sys.path: sys.path.insert(0,str(REPO_ROOT/"compiler"))

from depth_corruption_v1 import DepthCorruptionSpec
from bridge_persistence_v1 import build_corrupted_persistence_carrier
from realsas_compiler_core.surface import rigging_surface_from_d2_arrays

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--asset-dir",required=True)
    ap.add_argument("--cells",default=str(HERE/"M3_SMOKE_CELLS_V1.json"))
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    spec_doc=json.loads(Path(a.cells).read_text(encoding="utf-8"))
    rows=[]
    for idx,cell in enumerate(spec_doc["cells"]):
        t=time.time()
        spec=DepthCorruptionSpec(**cell)
        carrier=build_corrupted_persistence_carrier(Path(a.asset_dir),spec)
        surface=rigging_surface_from_d2_arrays(carrier.P,carrier.support,carrier.raster_xy,authority_label=f"M3_SMOKE_CELL_{idx}",persistence_label="MUTUAL_P003")
        affected=[r for r in carrier.per_view_corruption if r["depth_delta_rms"]>0]
        rows.append({**cell,
            "affected_views":[r["view"] for r in affected],
            "affected_view_rms":[r["depth_delta_rms"] for r in affected],
            "affected_view_abs_p95":[r["depth_delta_abs_p95"] for r in affected],
            "max_ray_perp_residual":max(r["max_ray_perpendicular_residual"] for r in carrier.per_view_corruption),
            "support_pairs":int(carrier.support.sum()),
            "typed_surface_nodes":len(surface.surface_nodes),
            "elapsed_s":time.time()-t,
        })
    checks={
        "all_cells_completed":len(rows)==len(spec_doc["cells"]),
        "all_typed_surface_nodes_512":all(r["typed_surface_nodes"]==512 for r in rows),
        "all_affected_view_rms_match_epsilon":all(all(abs(x-r["epsilon"])<=2e-9 for x in r["affected_view_rms"]) for r in rows),
        "all_ray_perp_leakage_below_1e-6":all(r["max_ray_perp_residual"]<1e-6 for r in rows),
    }
    report={
        "schema":"RealSaS.DepthToleranceBridge.M3SmokeResult.v1",
        "status":"PASS" if all(checks.values()) else "FAIL",
        "binding_scientific_thresholds":False,
        "scientific_tolerance_claim_opened":False,
        "asset_id":Path(a.asset_dir).name,
        "cell_count":len(rows),
        "cells":rows,
        "checks":checks,
        "interpretation":"APPARATUS_SMOKE_ONLY__NO_TOLERANCE_OR_PRODUCT_CLAIM",
        "proxy27":"CLOSED","dev32":"CLOSED",
    }
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["status"]=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
