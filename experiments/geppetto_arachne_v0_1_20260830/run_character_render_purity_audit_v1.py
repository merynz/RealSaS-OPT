from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from build_c0_admission_manifests_v1 import build as build_structural_c0
from character_render_purity_audit_v1 import RenderPurityAssetInputV1, audit_asset_v1

EXPECTED_STRUCTURAL_MEMBERSHIP_SHA256 = "cd3f5d14dbbfae0996cdade124af209da478cd36c2a0b3198a9a93fda5979147"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dist(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    x = np.asarray(xs, dtype=np.float64)
    return {"count": int(len(x)), "min": float(x.min()), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "PASS"]
    def av(path: tuple[str, ...]) -> list[float]:
        out = []
        for r in ok:
            x: Any = r
            try:
                for p in path:
                    x = x[p]
                if x is not None and np.isfinite(float(x)):
                    out.append(float(x))
            except (KeyError, TypeError, ValueError):
                pass
        return out
    return {
        "asset_count": len(rows),
        "pass_count": len(ok),
        "fail_count": len(rows) - len(ok),
        "mesh_max_face_area_fraction": dist(av(("mesh", "max_face_area_fraction"))),
        "mesh_top2_face_area_fraction": dist(av(("mesh", "top2_face_area_fraction"))),
        "asset_min_view_occupancy": dist([r["aggregate"]["occupancy_fraction"]["min"] for r in ok]),
        "asset_max_view_occupancy": dist([r["aggregate"]["occupancy_fraction"]["max"] for r in ok]),
        "asset_min_frame_margin": dist([r["aggregate"]["min_frame_margin_fraction"]["min"] for r in ok]),
        "asset_max_component_count": dist([r["aggregate"]["connected_component_count"]["max"] for r in ok]),
        "asset_min_largest_component_fraction": dist([r["aggregate"]["largest_component_fraction"]["min"] for r in ok]),
        "asset_max_visible_triangle_fraction": dist([r["aggregate"]["max_visible_triangle_pixel_fraction"]["max"] for r in ok]),
        "asset_max_visible_unskinned_fraction": dist([
            r["aggregate"]["visible_fully_unskinned_pixel_fraction"]["max"]
            for r in ok if r["aggregate"]["visible_fully_unskinned_pixel_fraction"]["count"]
        ]),
        "assets_touching_frame_any_view": int(sum(r["aggregate"]["border_touch_view_count"] > 0 for r in ok)),
    }


def extreme_examples(rows: list[dict[str, Any]], n: int = 25) -> dict[str, list[dict[str, Any]]]:
    ok = [r for r in rows if r["status"] == "PASS"]
    def pack(r, key, value):
        return {"canonical_asset_id": r["canonical_asset_id"], "source_registry_id": r["source_registry_id"], key: value}
    specs = {
        "highest_mesh_face_dominance": (lambda r: r["mesh"]["max_face_area_fraction"], True, "max_face_area_fraction"),
        "highest_visible_triangle_dominance": (lambda r: r["aggregate"]["max_visible_triangle_pixel_fraction"]["max"], True, "max_visible_triangle_pixel_fraction"),
        "lowest_frame_margin": (lambda r: r["aggregate"]["min_frame_margin_fraction"]["min"], False, "min_frame_margin_fraction"),
        "highest_component_count": (lambda r: r["aggregate"]["connected_component_count"]["max"], True, "max_connected_component_count"),
        "lowest_occupancy": (lambda r: r["aggregate"]["occupancy_fraction"]["min"], False, "min_occupancy_fraction"),
        "highest_occupancy": (lambda r: r["aggregate"]["occupancy_fraction"]["max"], True, "max_occupancy_fraction"),
    }
    out = {}
    for name, (fn, reverse, key) in specs.items():
        s = sorted(ok, key=fn, reverse=reverse)[:n]
        out[name] = [pack(r, key, float(fn(r))) for r in s]
    valid_skin = [r for r in ok if r["aggregate"]["visible_fully_unskinned_pixel_fraction"]["count"]]
    s = sorted(valid_skin, key=lambda r: r["aggregate"]["visible_fully_unskinned_pixel_fraction"]["max"], reverse=True)[:n]
    out["highest_visible_unskinned_fraction"] = [pack(r, "max_visible_fully_unskinned_pixel_fraction",
        float(r["aggregate"]["visible_fully_unskinned_pixel_fraction"]["max"])) for r in s]
    return out


def report_md(result: dict[str, Any]) -> str:
    a = result["aggregate"]
    return f"""# Geppetto/Arachne V0.1 — Character/Render Purity Measurement Audit V1.1\n\nStatus: **`{result['status']}`**  \nScope complete: **{result['scope_complete']}**\n\n## Scope\n\n- structural Geppetto C0 assets measured: **{result['asset_count']}**\n- PASS: **{a['pass_count']}**\n- hard render/geometry authority failures: **{a['fail_count']}**\n- nested structural Arachne C0 assets: **{result['arachne_structural_c0_count']}**\n\n## Measurement-only boundary\n\nThis audit selects **no** character/render thresholds and excludes **no** asset for visual quality. It measures objective tails only. Image-content integrity and semantic character-vs-prop classification are one later separately frozen image gate. No consumer model outputs are read and training remains unauthorized.\n\n## Objective distributions\n\n- mesh max-face area fraction p50 / p95 / p99 / max: **{a['mesh_max_face_area_fraction']['p50']} / {a['mesh_max_face_area_fraction']['p95']} / {a['mesh_max_face_area_fraction']['p99']} / {a['mesh_max_face_area_fraction']['max']}**\n- per-asset max visible-triangle pixel fraction p50 / p95 / p99 / max: **{a['asset_max_visible_triangle_fraction']['p50']} / {a['asset_max_visible_triangle_fraction']['p95']} / {a['asset_max_visible_triangle_fraction']['p99']} / {a['asset_max_visible_triangle_fraction']['max']}**\n- assets touching frame in any view: **{a['assets_touching_frame_any_view']}**\n\n## Next gate\n\nFreeze objective render-usability thresholds and a separate semantic character-purity protocol **from this input-quality result only**, before any Geppetto/Arachne optimizer output is opened.\n"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3"))
    ap.add_argument("--projection-audit-result", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--max-assets", type=int, default=0, help="0 = complete structural C0 scope; positive = smoke only")
    ap.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    args = ap.parse_args()

    raw = args.projection_audit_result.read_bytes()
    projection = json.loads(raw)
    structural = build_structural_c0(projection)
    if structural["membership_sha256"] != EXPECTED_STRUCTURAL_MEMBERSHIP_SHA256:
        raise RuntimeError("STRUCTURAL_C0_MEMBERSHIP_SHA_DRIFT")

    arachne_ids = {x["canonical_asset_id"] for x in structural["arachne_c0_assets"]}
    assets = list(structural["geppetto_c0_assets"])
    complete = args.max_assets <= 0
    if args.max_assets > 0:
        assets = assets[:args.max_assets]

    workers = max(1, int(args.workers))
    inputs = [RenderPurityAssetInputV1(
        canonical_asset_id=x["canonical_asset_id"],
        source_registry_id=x["source_registry_id"],
        asset_dir=args.root / "master" / "assets" / x["canonical_asset_id"],
        arachne_structural_c0=x["canonical_asset_id"] in arachne_ids,
    ) for x in assets]
    rows = []
    if workers == 1:
        iterator = map(audit_asset_v1, inputs)
        for i, row in enumerate(iterator, 1):
            rows.append(row)
            if i % 25 == 0 or i == len(inputs):
                print(f"[character-render-purity] {i}/{len(inputs)} workers=1", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(audit_asset_v1, inputs, chunksize=4), 1):
                rows.append(row)
                if i % 25 == 0 or i == len(inputs):
                    print(f"[character-render-purity] {i}/{len(inputs)} workers={workers}", flush=True)

    agg = summarize(rows)
    if agg["fail_count"]:
        status = "FAIL_HARD_RENDER_AUTHORITY_CONTRACT"
    elif complete:
        status = "PASS_MEASUREMENT_COMPLETE__QUALITY_AND_SEMANTIC_FREEZE_NEXT"
    else:
        status = "SMOKE_ONLY__NOT_FREEZEABLE"
    result = {
        "schema": "RealSaS.GeppettoArachne.CharacterRenderPurityAudit.v1.1",
        "status": status,
        "scope_complete": bool(complete and agg["fail_count"] == 0),
        "root": str(args.root),
        "asset_count": len(rows),
        "arachne_structural_c0_count": int(sum(r["arachne_structural_c0"] for r in rows)),
        "structural_membership_sha256": structural["membership_sha256"],
        "projection_audit_result_sha256": hashlib.sha256(raw).hexdigest(),
        "aggregate": agg,
        "extreme_examples_measurement_only": extreme_examples(rows),
        "rows": rows,
        "authority": {
            "consumer_model_outputs_read": False,
            "closed_sets_opened": False,
            "corpus_mutation": False,
            "threshold_based_exclusion_applied": False,
            "semantic_character_classifier_applied": False,
            "image_content_decoded": False,
            "cel_clean_512_existence_verified": True,
            "training_authorized": False,
            "render_authority": "master/assets/<asset>/renders/V0..V7/raster_authority.npz; cel_clean_512.png existence only in V1.1",
            "geometry_authority": "master/assets/<asset>/primary_geometry.npz",
        },
        "next_gate": "FREEZE_OBJECTIVE_RENDER_USABILITY_AND_IMAGE_SEMANTIC_PROTOCOL",
    }

    outdir = args.output_dir or (args.root / "reports" / "geppetto_arachne_v0_1")
    outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / "CHARACTER_RENDER_PURITY_AUDIT_RESULT_V1_1.json"
    mp = outdir / "CHARACTER_RENDER_PURITY_AUDIT_REPORT_V1_1.md"
    jp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    mp.write_text(report_md(result), encoding="utf-8")
    print(json.dumps({"status": status, "result": str(jp), "report": str(mp), "aggregate": agg}, indent=2, sort_keys=True))
    if agg["fail_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
