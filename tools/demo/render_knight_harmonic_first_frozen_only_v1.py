from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    _ctx,
    run as frozen_preview_run,
)
from tools.demo.render_knight_skin_topology_repair_preview_v1 import _gif_from_sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--qualification-run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    authority_root = a.authority_root.resolve()
    source_root = authority_root / "runs" / a.source_run_id
    seal_path = source_root / "IMMUTABLE_STAGE23_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("RENDER_SOURCE_STAGE23_SEAL_MISSING")
    seal = json.loads(seal_path.read_text())
    if (
        seal.get("status") != "SEALED_IMMUTABLE_STAGE23"
        or seal.get("run_id") != a.source_run_id
        or seal.get("source_owned_visual_mesh_mode") is not False
        or seal.get("overwrite_forbidden") is not True
    ):
        raise RuntimeError("RENDER_SOURCE_STAGE23_SEAL_INVALID")

    source_ctx = _ctx(authority_root, a.source_run_id)
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            source_ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    if asset.asset_hash != seal["stage23_asset_hash"]:
        raise RuntimeError("RENDER_SOURCE_STAGE23_ASSET_DRIFT")
    if dict(asset.metadata or {}).get("source_owned_visual_mesh_mode") is True:
        raise RuntimeError("RENDER_SOURCE_MECHANICAL_CAA_REQUIRED")

    qualification_ctx = _ctx(authority_root, a.qualification_run_id)
    qualification = stage_output_payload(
        qualification_ctx,
        "24_COMPLETE_APPEARANCE_QUALIFIED",
        "RealSaS.CompleteAppearanceQualificationIR.v2",
    )
    qualification_report = dict(qualification.get("qualification_report") or {})
    if (
        qualification.get("asset_binding_hash") != asset.asset_hash
        or qualification_report.get("status") != "PASS_COMPLETE_APPEARANCE"
    ):
        raise RuntimeError("RENDER_STAGE24_PASS_BINDING_REQUIRED")

    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "frozen_renderer"

    started = perf_counter()
    frozen_preview_run(
        authority_root=authority_root,
        run_id=a.source_run_id,
        out_dir=preview_dir,
    )
    render_seconds = perf_counter() - started

    gif_rows = []
    for name, duration_ms in (("IDLE", 833), ("RUN", 208), ("SLASH", 278)):
        sheet = preview_dir / f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
        gif = out_dir / f"KNIGHT_{name}_HARMONIC_FIRST_CORRECTED_FROZEN_RENDERER_V1.gif"
        _gif_from_sheet(sheet, gif, duration_ms)
        gif_rows.append({
            "name": name,
            "gif": str(gif),
            "gif_sha256": sha256_file(gif),
            "sheet": str(sheet),
            "sheet_sha256": sha256_file(sheet),
        })

    report = {
        "schema": "RealSaS.KnightHarmonicFirstCorrectedFrozenRenderer.v1",
        "status": "PASS",
        "source_run_id": a.source_run_id,
        "qualification_run_id": a.qualification_run_id,
        "stage23_asset_hash": asset.asset_hash,
        "stage23_seal_sha256": sha256_file(seal_path),
        "stage24_qualification_hash": qualification["qualification_hash"],
        "stage24_report": qualification_report,
        "stage24_holdout_p95_rgba_l1": qualification[
            "structured_holdout_p95_rgba_l1"
        ],
        "frozen_renderer_blob": "7917be02f325ba00b5dcd2b31a786898e45f640f",
        "source_owned_visual_mesh_arap_used": False,
        "stage19_to24_recomputed": False,
        "render_seconds": render_seconds,
        "gifs": gif_rows,
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "HARMONIC_FIRST_FROZEN_RENDER_PASS="
        + json.dumps(
            {
                "stage23": asset.asset_hash,
                "stage24": qualification["qualification_hash"],
                "holdout_p95": qualification["structured_holdout_p95_rgba_l1"],
                "render_seconds": render_seconds,
                "gifs": {row["name"]: row["gif_sha256"] for row in gif_rows},
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
