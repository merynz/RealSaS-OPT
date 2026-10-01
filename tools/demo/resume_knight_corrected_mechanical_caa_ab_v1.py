from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    qualify_complete_appearance_stage,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    run as frozen_preview_run,
)


def _ctx(authority_root: Path, run_id: str) -> dict:
    root = (authority_root / "runs" / run_id).resolve()
    return {
        "repo_root": Path(".").resolve(),
        "authority_root": authority_root.resolve(),
        "run_root": root,
        "run_id": run_id,
        "run_manifest_path": root / "run_manifest.json",
        "run_manifest": json.loads((root / "run_manifest.json").read_text()),
        "ledger": json.loads((root / "ACTIVE_RUN_V2.json").read_text()),
        "stage": {"id": "24_COMPLETE_APPEARANCE_QUALIFIED"},
    }


def _stage_row(ctx: dict, stage_id: str) -> dict:
    return next(row for row in ctx["ledger"]["stages"] if row["id"] == stage_id)


def _gif_from_sheet(sheet_path: Path, out_path: Path, duration_ms: int) -> None:
    sheet = Image.open(sheet_path).convert("RGBA")
    if sheet.width % 4 != 0 or sheet.height % 2 != 0:
        raise RuntimeError(f"UNEXPECTED_FROZEN_SHEET_GEOMETRY:{sheet.size}")
    cell_w = sheet.width // 4
    row_h = sheet.height // 2
    header = 28
    cell_h = row_h - header
    frames = []
    for i in range(4):
        v0 = sheet.crop((i * cell_w, header, (i + 1) * cell_w, header + cell_h))
        y2 = row_h + header
        v2 = sheet.crop((i * cell_w, y2, (i + 1) * cell_w, y2 + cell_h))
        combined = Image.new("RGBA", (cell_w * 2, cell_h), (0, 0, 0, 0))
        combined.alpha_composite(v0, (0, 0))
        combined.alpha_composite(v2, (cell_w, 0))
        target_h = min(640, combined.height)
        target_w = max(1, round(combined.width * target_h / combined.height))
        if (target_w, target_h) != combined.size:
            combined = combined.resize((target_w, target_h), Image.Resampling.LANCZOS)
        frames.append(
            combined.convert("P", palette=Image.Palette.ADAPTIVE, colors=255)
        )
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
        optimize=False,
        transparency=0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    authority_root = a.authority_root.resolve()
    ctx = _ctx(authority_root, a.run_id)
    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )
    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    if asset.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise RuntimeError("RESUME_STAGE23_CANDIDATE_BINDING_DRIFT")
    if dict(asset.metadata or {}).get("source_owned_visual_mesh_mode") is True:
        raise RuntimeError("RESUME_MECHANICAL_CAA_EXPECTED")
    if len(asset.textures) != 8:
        raise RuntimeError("RESUME_STAGE23_TEXTURE_MATRIX_INCOMPLETE")

    t24 = perf_counter()
    ctx["stage"] = {"id": "24_COMPLETE_APPEARANCE_QUALIFIED"}
    q = qualify_complete_appearance_stage(ctx)
    stage24_seconds = perf_counter() - t24
    stage24_status = str(q.get("status") or "")
    print(
        "RESUME_STAGE24="
        + json.dumps(
            {
                "status": stage24_status,
                "seconds": stage24_seconds,
                "blockers": q.get("blockers") or [],
                "diagnostics": q.get("diagnostics") or {},
            },
            sort_keys=True,
        ),
        flush=True,
    )

    row24 = _stage_row(ctx, "24_COMPLETE_APPEARANCE_QUALIFIED")
    row24["status"] = stage24_status
    row24["blockers"] = list(q.get("blockers") or ())
    row24["outputs"] = list(q.get("outputs") or ())
    (ctx["run_root"] / "ACTIVE_RUN_V2.json").write_text(
        json.dumps(ctx["ledger"], indent=2, sort_keys=True) + "\n"
    )

    preview_root = out_dir / "frozen_renderer"
    t_render = perf_counter()
    frozen_preview_run(
        authority_root=authority_root,
        run_id=a.run_id,
        out_dir=preview_root,
    )
    render_seconds = perf_counter() - t_render

    specs = (("IDLE", 833), ("RUN", 208), ("SLASH", 278))
    gifs = []
    for name, duration in specs:
        sheet = preview_root / f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
        gif = out_dir / f"KNIGHT_{name}_CORRECTED_WEIGHT_TOPOLOGY_FROZEN_RENDERER_AB_V1.gif"
        _gif_from_sheet(sheet, gif, duration)
        gifs.append({
            "name": name,
            "path": str(gif),
            "sha256": sha256_file(gif),
            "sheet_path": str(sheet),
            "sheet_sha256": sha256_file(sheet),
        })

    report = {
        "schema": "RealSaS.KnightCorrectedMechanicalCAAFrozenRendererResume.v1",
        "status": "PASS_VISUAL_DIAGNOSTIC_RENDER",
        "run_id": a.run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "stage23_asset_hash": asset.asset_hash,
        "stage24_status": stage24_status,
        "stage24_blockers": list(q.get("blockers") or ()),
        "stage24_diagnostics": q.get("diagnostics") or {},
        "stage24_seconds": stage24_seconds,
        "render_seconds": render_seconds,
        "product_appearance_pass_claimed": stage24_status == "PASS",
        "frozen_renderer_blob": "7917be02f325ba00b5dcd2b31a786898e45f640f",
        "source_owned_visual_mesh_arap_used": False,
        "mechanical_caa_stage23_reused": True,
        "stage19_to23_recomputed": False,
        "gifs": gifs,
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "KNIGHT_FROZEN_RENDERER_RESUME_DONE="
        + json.dumps(
            {
                "stage24_status": stage24_status,
                "stage24_seconds": stage24_seconds,
                "render_seconds": render_seconds,
                "gifs": {row["name"]: row["sha256"] for row in gifs},
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
