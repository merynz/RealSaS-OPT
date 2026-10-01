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
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    appearance_domain_from_dict,
    static_mesh_qualification_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    preregister_caa_backend_stage,
    compile_caa_stage,
    seal_caa_compile_stage,
    bake_complete_appearance_stage,
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
        "stage": {"id": "RECOVERY_INIT"},
    }


def _row(ctx: dict, stage_id: str) -> dict:
    return next(row for row in ctx["ledger"]["stages"] if row["id"] == stage_id)


def _install(ctx: dict, stage_id: str, result: dict) -> float:
    status = str(result.get("status") or "")
    if status not in {"PASS", "PASS_DEMO_ONLY"}:
        raise RuntimeError(
            f"RECOVERY_STAGE_NOT_PASS::{stage_id}::{status}::{result.get('blockers')}"
        )
    row = _row(ctx, stage_id)
    row["status"] = status
    row["outputs"] = list(result.get("outputs") or ())
    row["blockers"] = []
    (ctx["run_root"] / "ACTIVE_RUN_V2.json").write_text(
        json.dumps(ctx["ledger"], indent=2, sort_keys=True) + "\n"
    )
    return status


def _gif_from_sheet(sheet_path: Path, out_path: Path, duration_ms: int) -> None:
    sheet = Image.open(sheet_path).convert("RGBA")
    if sheet.width % 4 != 0 or sheet.height % 2 != 0:
        raise RuntimeError(f"RECOVERY_SHEET_GEOMETRY_INVALID:{sheet.size}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

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
    domain = appearance_domain_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.AppearanceDomainIR.v1",
        )
    )
    static_mesh = static_mesh_qualification_from_dict(
        stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )

    dm = dict(domain.metadata or {})
    if (
        dm.get("mechanical_candidate_render_authority") is not True
        or dm.get("visual_mesh_set_binding_hash") not in {None, ""}
        or str(dm.get("domain") or "")
        != "LEGACY_CANONICAL_MECHANICAL_MESH_SURFACE_X_DISCRETE_V0_V7"
    ):
        raise RuntimeError("RECOVERY_STAGE18_NOT_MECHANICAL_APPEARANCE_DOMAIN")
    if (
        static_mesh.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
        or static_mesh.appearance_domain_binding_hash != domain.domain_hash
    ):
        raise RuntimeError("RECOVERY_STAGE19_BINDING_DRIFT")

    stages = (
        ("20_CAA_BACKEND_PREREGISTERED", preregister_caa_backend_stage),
        ("21_CAA_COMPILE", compile_caa_stage),
        ("22_CAA_COMPILE_SEALED", seal_caa_compile_stage),
        ("23_COMPLETE_APPEARANCE_ASSET_BAKED", bake_complete_appearance_stage),
    )
    timings = {}
    for stage_id, fn in stages:
        ctx["stage"] = {"id": stage_id}
        t0 = perf_counter()
        result = fn(ctx)
        seconds = perf_counter() - t0
        print(
            "RECOVERY_STAGE="
            + json.dumps(
                {
                    "stage": stage_id,
                    "status": result.get("status"),
                    "seconds": seconds,
                    "blockers": result.get("blockers") or [],
                    "performance": result.get("performance") or {},
                    "diagnostics": result.get("diagnostics") or {},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _install(ctx, stage_id, result)
        timings[stage_id] = seconds

    asset = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    if asset.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise RuntimeError("RECOVERY_STAGE23_CANDIDATE_DRIFT")
    if dict(asset.metadata or {}).get("source_owned_visual_mesh_mode") is True:
        raise RuntimeError("RECOVERY_STAGE23_STILL_SOURCE_OWNED_VISUAL")
    if len(asset.textures) != 8:
        raise RuntimeError("RECOVERY_STAGE23_TEXTURE_MATRIX_INCOMPLETE")

    preview_dir = out_dir / "frozen_renderer"
    t_render = perf_counter()
    frozen_preview_run(
        authority_root=authority_root,
        run_id=a.run_id,
        out_dir=preview_dir,
    )
    render_seconds = perf_counter() - t_render

    gifs = []
    for name, duration in (("IDLE", 833), ("RUN", 208), ("SLASH", 278)):
        sheet = preview_dir / f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
        gif = out_dir / f"KNIGHT_{name}_CORRECTED_WEIGHT_TOPOLOGY_FROZEN_RENDERER_AB_V1.gif"
        _gif_from_sheet(sheet, gif, duration)
        gifs.append(
            {
                "name": name,
                "path": str(gif),
                "sha256": sha256_file(gif),
                "sheet_path": str(sheet),
                "sheet_sha256": sha256_file(sheet),
            }
        )

    report = {
        "schema": "RealSaS.KnightCorrectedMechanicalCAARecoveryRender.v1",
        "status": "PASS_VISUAL_DIAGNOSTIC_RENDER",
        "run_id": a.run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "appearance_domain_hash": domain.domain_hash,
        "stage19_reused": True,
        "stage20_to23_recomputed": True,
        "stage24_requalified_before_render": False,
        "product_appearance_pass_claimed": False,
        "stage23_asset_hash": asset.asset_hash,
        "frozen_renderer_blob": "7917be02f325ba00b5dcd2b31a786898e45f640f",
        "source_owned_visual_mesh_arap_used": False,
        "timings_seconds": timings,
        "frozen_render_seconds": render_seconds,
        "gifs": gifs,
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "KNIGHT_CORRECTED_MECHANICAL_CAA_RECOVERY_RENDER_PASS="
        + json.dumps(
            {
                "candidate": candidate.candidate_lineage_hash,
                "skin": skin.skin_lineage_hash,
                "stage23": asset.asset_hash,
                "timings": timings,
                "render_seconds": render_seconds,
                "gifs": {row["name"]: row["sha256"] for row in gifs},
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
