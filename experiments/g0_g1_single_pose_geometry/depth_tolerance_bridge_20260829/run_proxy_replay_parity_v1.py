#!/usr/bin/env python3
"""Reproduce the frozen D2 proxy replay parity gate.

No training. No Proxy27/DEV32 opening.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from depth_corruption_v1 import DepthCorruptionSpec
from proxy_replay_v1 import (
    aggregate_arachne,
    aggregate_geppetto,
    d2_proxy_x36_from_bridge,
    eval_arachne_d2,
    eval_geppetto_d2,
    load_d2_models,
    sha256_file,
)

IDS = [
    "asset_551ea351b43a1787d0f55536",
    "asset_0679fdef64f19a4832a6d521",
    "asset_76313e4bd82b82fcd1659c70",
    "asset_f8a40d6c5d815fe79c8b5e42",
]
PACK_SHA = {
    "asset_551ea351b43a1787d0f55536": "a4b3d5544f20a0fecefa877b74f4ad601f072d29afd0df35204c7b182d800c2c",
    "asset_0679fdef64f19a4832a6d521": "76a84883a1516839d9d5476f7c2d99ad26a29815f490d4d4c50124ecafa3db17",
    "asset_76313e4bd82b82fcd1659c70": "ddeed0eef39e13ed7831e798b8404de92083df5355bb65002f70087734283633",
    "asset_f8a40d6c5d815fe79c8b5e42": "41cda2c5623f9698e7c35bd5559b7967871dc7f5c37fff10148b220991d1ebf8",
}
HIST_CAL_SHA = "82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9"
TOL = 2e-7


def _load_pack(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _numeric_max_diff(a, b) -> float:
    out = 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            out = max(out, _numeric_max_diff(a[k], b[k]))
    elif isinstance(a, list) and isinstance(b, list):
        for x, y in zip(a, b):
            out = max(out, _numeric_max_diff(x, y))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out = abs(float(a) - float(b))
    return float(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-asset-dir", required=True)
    ap.add_argument("--pack-root", required=True)
    ap.add_argument("--geppetto", required=True)
    ap.add_argument("--arachne", required=True)
    ap.add_argument("--historical-calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    packs = {}
    for aid in IDS:
        path = Path(args.pack_root) / f"{aid}.npz"
        if sha256_file(path) != PACK_SHA[aid]:
            raise RuntimeError(f"pack SHA drift: {aid}")
        packs[aid] = _load_pack(path)

    hist_path = Path(args.historical_calibration)
    if sha256_file(hist_path) != HIST_CAL_SHA:
        raise RuntimeError("historical calibration SHA drift")
    hist = json.loads(hist_path.read_text())

    _carrier, support, _rows, X = d2_proxy_x36_from_bridge(
        args.m2_asset_dir, DepthCorruptionSpec(0.0, 0.0, "ALL8")
    )
    m2 = packs[IDS[0]]
    x_equal = np.array_equal(X, m2["D2_X"])
    support_pairs = int(support.sum())
    historical_pairs = int(np.asarray(m2["D2_support_count"]).sum())

    gm, am = load_d2_models(args.geppetto, args.arachne, args.device)
    gp, apers = [], []
    for aid in IDS:
        z = packs[aid]
        gp.append(eval_geppetto_d2(gm, z["D2_X"], z, aid, args.device))
        apers.append(eval_arachne_d2(am, z["D2_X"], z, aid, args.device))
    got_g = {"aggregate": aggregate_geppetto(gp), "per_asset": gp}
    got_a = {"aggregate": aggregate_arachne(apers), "per_asset": apers}
    hg = hist["calibration"]["geppetto"]["D2"]
    ha = hist["calibration"]["arachne"]["D2"]
    gd = _numeric_max_diff(got_g, hg)
    ad = _numeric_max_diff(got_a, ha)

    checks = {
        "m2_epsilon0_D2_X_bit_exact": bool(x_equal),
        "m2_epsilon0_d2_support_pair_count": support_pairs,
        "m2_historical_d2_support_pair_count": historical_pairs,
        "geppetto_max_abs_numeric_diff_vs_historical_calibration": gd,
        "arachne_max_abs_numeric_diff_vs_historical_calibration": ad,
        "numeric_parity_tolerance": TOL,
        "geppetto_parity_pass": bool(gd <= TOL),
        "arachne_parity_pass": bool(ad <= TOL),
    }
    passed = x_equal and support_pairs == historical_pairs and gd <= TOL and ad <= TOL
    report = {
        "schema": "RealSaS.DepthToleranceBridge.ProxyReplayParity.v1",
        "status": "PASS" if passed else "FAIL",
        "truth_capable_calibration_ids": IDS,
        "pack_sha256": PACK_SHA,
        "calibration_result_sha256": HIST_CAL_SHA,
        "checks": checks,
        "firewalls": {
            "training_steps": 0,
            "proxy27": "CLOSED_FOR_THIS_PARITY",
            "dev32": "CLOSED",
            "m3_smoke_used_for_thresholds": False,
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
