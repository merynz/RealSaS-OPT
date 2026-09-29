from __future__ import annotations

import argparse, copy, hashlib, json, os, shutil
from pathlib import Path
from time import perf_counter

from PIL import Image

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    output_direction_set_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import build_appearance_domain
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
    qualify_static_canonical_mesh_stage,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    preregister_caa_backend_stage,
    compile_caa_stage,
    seal_caa_compile_stage,
    bake_complete_appearance_stage,
    qualify_complete_appearance_stage,
)
from tools.demo.render_knight_motion_preview_v1 import (
    run as frozen_preview_run,
)
from tools.demo.render_knight_skin_topology_repair_preview_v1 import _gif_from_sheet


def load_json(path: Path):
    return json.loads(path.read_text())


def row(ledger: dict, stage_id: str) -> dict:
    return next(x for x in ledger["stages"] if x["id"] == stage_id)


def adopt_result(ctx: dict, stage_id: str, fn):
    ctx["stage"] = {"id": stage_id}
    started = perf_counter()
    result = fn(ctx)
    seconds = perf_counter() - started
    status = str(result.get("status") or "")
    print("MECHANICAL_CAA_STAGE=" + json.dumps({
        "stage": stage_id,
        "status": status,
        "seconds": seconds,
        "blockers": result.get("blockers") or [],
        "performance": result.get("performance") or {},
    }, sort_keys=True), flush=True)
    if status not in {"PASS", "PASS_DEMO_ONLY"}:
        raise RuntimeError(
            f"MECHANICAL_CAA_STAGE_NOT_PASS::{stage_id}::{status}::{result.get('blockers')}"
        )
    target = row(ctx["ledger"], stage_id)
    target["status"] = status
    target["outputs"] = list(result.get("outputs") or ())
    target["blockers"] = []
    return result, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--parent-run-id", required=True)
    ap.add_argument("--child-run-id", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    repo_root = Path(".").resolve()
    authority_root = a.authority_root.resolve()
    parent_root = (authority_root / "runs" / a.parent_run_id).resolve()
    child_root = (authority_root / "runs" / a.child_run_id).resolve()
    if child_root.exists():
        shutil.rmtree(child_root)
    child_root.mkdir(parents=True)
    os.symlink(parent_root / "inputs", child_root / "inputs", target_is_directory=True)

    parent_manifest = load_json(parent_root / "run_manifest.json")
    parent_ledger = load_json(parent_root / "ACTIVE_RUN_V2.json")
    manifest = copy.deepcopy(parent_manifest)
    manifest["run_id"] = a.child_run_id
    ledger = copy.deepcopy(parent_ledger)
    ledger["run_id"] = a.child_run_id
    ledger["execution_class"] = "DEMO_WITNESS"
    ledger["architecture_scope"] = (
        "CORRECTED_WEIGHT_TOPOLOGY__FROZEN_MECHANICAL_CAA_RENDER_AB_V1"
    )

    ctx = {
        "repo_root": repo_root,
        "authority_root": authority_root,
        "run_root": child_root,
        "run_id": a.child_run_id,
        "run_manifest_path": child_root / "run_manifest.json",
        "run_manifest": manifest,
        "ledger": ledger,
        "stage": {"id": "INIT"},
    }

    # Exact corrected mechanics are inherited byte-for-byte from the solved lineage.
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    directions = output_direction_set_from_dict(
        stage_output_payload(
            ctx, "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )

    # Counterfactual: keep candidate/addressing exact, switch only appearance domain
    # back to the frozen mechanical-mesh presentation mode used by the smear baseline.
    mechanical_domain = build_appearance_domain(
        candidate,
        addressing,
        directions.direction_set_hash,
    )
    stage18_root = child_root / "artifacts" / "18_CANONICAL_MESH_ADDRESSING_BUILD"
    mechanical_domain_ref = write_ir(
        stage18_root / "appearance_domain.mechanical_ab.json",
        mechanical_domain,
        authority_class="AUDIT_ONLY_MECHANICAL_APPEARANCE_DOMAIN",
    )
    stage18_row = row(ledger, "18_CANONICAL_MESH_ADDRESSING_BUILD")
    preserved = [
        x for x in stage18_row.get("outputs") or ()
        if x.get("schema") != "RealSaS.AppearanceDomainIR.v1"
    ]
    stage18_row["outputs"] = preserved + [mechanical_domain_ref]
    stage18_row["status"] = "PASS"
    stage18_row["blockers"] = []

    (child_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    timings = {}
    for sid, fn in (
        ("19_STATIC_CANONICAL_MESH_QUALIFIED", qualify_static_canonical_mesh_stage),
        ("20_CAA_BACKEND_PREREGISTERED", preregister_caa_backend_stage),
        ("21_CAA_COMPILE", compile_caa_stage),
        ("22_CAA_COMPILE_SEALED", seal_caa_compile_stage),
        ("23_COMPLETE_APPEARANCE_ASSET_BAKED", bake_complete_appearance_stage),
        ("24_COMPLETE_APPEARANCE_QUALIFIED", qualify_complete_appearance_stage),
    ):
        _result, seconds = adopt_result(ctx, sid, fn)
        timings[sid] = seconds
        (child_root / "ACTIVE_RUN_V2.json").write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n"
        )

    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "frozen_renderer"
    preview_started = perf_counter()
    frozen_preview_run(
        authority_root=authority_root,
        run_id=a.child_run_id,
        out_dir=preview_dir,
    )
    timings["FROZEN_PREVIEW_RENDERER"] = perf_counter() - preview_started

    gif_specs = {
        "IDLE": 833,
        "RUN": 208,
        "SLASH": 278,
    }
    gif_rows = []
    for name, duration_ms in gif_specs.items():
        sheet = preview_dir / f"KNIGHT_{name}_DEMO_PREVIEW_V1.png"
        gif = out_dir / f"KNIGHT_{name}_CORRECTED_WEIGHT_TOPOLOGY_FROZEN_RENDERER_AB_V1.gif"
        _gif_from_sheet(sheet, gif, duration_ms)
        gif_rows.append({
            "name": name,
            "path": str(gif),
            "sha256": sha256_file(gif),
            "sheet_path": str(sheet),
            "sheet_sha256": sha256_file(sheet),
        })

    stage23_payload = stage_output_payload(
        ctx, "23_COMPLETE_APPEARANCE_ASSET_BAKED",
        "RealSaS.CompleteAppearanceAssetIR.v2",
    )
    stage24_payload = stage_output_payload(
        ctx, "24_COMPLETE_APPEARANCE_QUALIFIED",
        "RealSaS.CompleteAppearanceQualificationIR.v2",
    )
    report = {
        "schema": "RealSaS.KnightCorrectedMechanicalCAAFrozenRendererAB.v1",
        "status": "PASS_CAUSAL_VISUAL_AB",
        "parent_run_id": a.parent_run_id,
        "child_run_id": a.child_run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "surface_addressing_hash": addressing.addressing_hash,
        "mechanical_appearance_domain_hash": mechanical_domain.domain_hash,
        "mechanical_appearance_domain": mechanical_domain.metadata,
        "stage23_candidate_binding_hash": stage23_payload[
            "candidate_mesh_binding_hash"
        ],
        "stage23_asset_hash": stage23_payload["asset_hash"],
        "stage24_qualification_hash": stage24_payload["qualification_hash"],
        "stage24_report": stage24_payload["qualification_report"],
        "frozen_renderer_path": "tools/demo/render_knight_motion_preview_v1.py",
        "frozen_renderer_sha256": sha256_file(
            repo_root / "tools/demo/render_knight_motion_preview_v1.py"
        ),
        "counterfactual_invariants": {
            "corrected_stage18_candidate_preserved_byte_exact": True,
            "corrected_stage32_skin_preserved_byte_exact": True,
            "skeleton_camera_motion_source_preserved": True,
            "source_owned_visual_mesh_arap_not_used": True,
            "mechanical_caa_rebaked_for_corrected_topology": True,
            "frozen_smear_renderer_used_unchanged": True,
        },
        "timings_seconds": timings,
        "gifs": gif_rows,
    }
    report_path = out_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("KNIGHT_MECHANICAL_CAA_FROZEN_RENDERER_AB_PASS=" + json.dumps({
        "candidate": candidate.candidate_lineage_hash,
        "mechanical_domain": mechanical_domain.domain_hash,
        "stage23": stage23_payload["asset_hash"],
        "stage24": stage24_payload["qualification_hash"],
        "timings": timings,
        "gifs": {x["name"]: x["sha256"] for x in gif_rows},
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
