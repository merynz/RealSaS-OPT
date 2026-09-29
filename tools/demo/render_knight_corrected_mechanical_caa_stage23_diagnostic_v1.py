from __future__ import annotations

import argparse
import copy
import json
import os
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


def _load_json(path: Path):
    return json.loads(path.read_text())


def _stage_row(ledger: dict, stage_id: str) -> dict:
    return next(row for row in ledger["stages"] if row["id"] == stage_id)


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
        frames.append(combined.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))
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
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--consumer-run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    repo_root = Path(".").resolve()
    authority_root = a.authority_root.resolve()
    source_root = (authority_root / "runs" / a.source_run_id).resolve()
    consumer_root = (authority_root / "runs" / a.consumer_run_id).resolve()

    if not source_root.is_dir():
        raise RuntimeError("IMMUTABLE_STAGE23_SOURCE_MISSING")
    if consumer_root.exists():
        raise RuntimeError("IMMUTABLE_CONSUMER_ALREADY_EXISTS__OVERWRITE_FORBIDDEN")

    seal_path = source_root / "IMMUTABLE_STAGE23_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("IMMUTABLE_STAGE23_SEAL_MISSING")
    seal = _load_json(seal_path)
    if (
        seal.get("status") != "SEALED_IMMUTABLE_STAGE23"
        or seal.get("run_id") != a.source_run_id
        or seal.get("overwrite_forbidden") is not True
        or seal.get("stage24_executed") is not False
        or seal.get("renderer_executed") is not False
        or seal.get("source_owned_visual_mesh_mode") is not False
    ):
        raise RuntimeError("IMMUTABLE_STAGE23_SEAL_INVALID")

    source_manifest = source_root / "run_manifest.json"
    source_ledger = source_root / "ACTIVE_RUN_V2.json"
    if sha256_file(source_manifest) != seal["run_manifest_sha256"]:
        raise RuntimeError("IMMUTABLE_SOURCE_MANIFEST_DRIFT")
    if sha256_file(source_ledger) != seal["active_ledger_sha256"]:
        raise RuntimeError("IMMUTABLE_SOURCE_LEDGER_DRIFT")

    manifest = copy.deepcopy(_load_json(source_manifest))
    ledger = copy.deepcopy(_load_json(source_ledger))
    manifest["run_id"] = a.consumer_run_id
    ledger["run_id"] = a.consumer_run_id
    ledger["execution_class"] = "DEMO_WITNESS"
    ledger["architecture_scope"] = (
        "CORRECTED_WEIGHT_TOPOLOGY__IMMUTABLE_STAGE23_CONSUMER_STAGE24_FROZEN_RENDER_V1"
    )

    consumer_root.mkdir(parents=True)
    os.symlink(source_root / "inputs", consumer_root / "inputs", target_is_directory=True)
    (consumer_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (consumer_root / "ACTIVE_RUN_V2.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )

    ctx = {
        "repo_root": repo_root,
        "authority_root": authority_root,
        "run_root": consumer_root,
        "run_id": a.consumer_run_id,
        "run_manifest_path": consumer_root / "run_manifest.json",
        "run_manifest": manifest,
        "ledger": ledger,
        "stage": {"id": "24_COMPLETE_APPEARANCE_QUALIFIED"},
    }

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
        raise RuntimeError("CONSUMER_STAGE23_CANDIDATE_BINDING_DRIFT")
    if asset.asset_hash != seal["stage23_asset_hash"]:
        raise RuntimeError("CONSUMER_STAGE23_ASSET_HASH_DRIFT")
    if dict(asset.metadata or {}).get("source_owned_visual_mesh_mode") is True:
        raise RuntimeError("CONSUMER_MECHANICAL_CAA_REQUIRED")

    t24 = perf_counter()
    q = qualify_complete_appearance_stage(ctx)
    stage24_seconds = perf_counter() - t24
    stage24_status = str(q.get("status") or "")
    print(
        "IMMUTABLE_CONSUMER_STAGE24="
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

    row24 = _stage_row(ledger, "24_COMPLETE_APPEARANCE_QUALIFIED")
    row24["status"] = stage24_status
    row24["blockers"] = list(q.get("blockers") or ())
    row24["outputs"] = list(q.get("outputs") or ())
    (consumer_root / "ACTIVE_RUN_V2.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )

    # Diagnostic causal render is allowed even when Stage24 does not qualify.
    # The report must preserve the failure and make no product-pass claim.
    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_root = out_dir / "frozen_renderer"
    t_render = perf_counter()
    frozen_preview_run(
        authority_root=authority_root,
        run_id=a.consumer_run_id,
        out_dir=preview_root,
    )
    render_seconds = perf_counter() - t_render

    gifs = []
    for name, duration in (("IDLE", 833), ("RUN", 208), ("SLASH", 278)):
        sheet = preview_root / f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
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
        "schema": "RealSaS.ImmutableMechanicalCAAStage23ConsumerFrozenRenderer.v1",
        "status": (
            "PASS_PRODUCT_QUALIFIED_VISUAL_AB"
            if stage24_status == "PASS"
            else "DIAGNOSTIC_VISUAL_AB__STAGE24_NOT_QUALIFIED"
        ),
        "source_run_id": a.source_run_id,
        "consumer_run_id": a.consumer_run_id,
        "source_stage23_seal_sha256": sha256_file(seal_path),
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "stage23_asset_hash": asset.asset_hash,
        "stage24_status": stage24_status,
        "stage24_blockers": list(q.get("blockers") or ()),
        "stage24_diagnostics": q.get("diagnostics") or {},
        "product_appearance_pass_claimed": stage24_status == "PASS",
        "stage24_seconds": stage24_seconds,
        "render_seconds": render_seconds,
        "frozen_renderer_blob": "7917be02f325ba00b5dcd2b31a786898e45f640f",
        "source_stage23_mutated": False,
        "source_owned_visual_mesh_arap_used": False,
        "stage19_to23_recomputed": False,
        "gifs": gifs,
    }
    report_path = out_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "IMMUTABLE_STAGE23_CONSUMER_RENDER_PASS="
        + json.dumps(
            {
                "source_run_id": a.source_run_id,
                "consumer_run_id": a.consumer_run_id,
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
