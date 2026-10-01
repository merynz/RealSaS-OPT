from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
)
from compiler.realsas_compiler_core.output_presentation_v1 import (
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    build_appearance_domain,
    surface_addressing_from_dict,
)
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
)


def load_json(path: Path):
    return json.loads(path.read_text())


def stage_row(ledger: dict, stage_id: str) -> dict:
    return next(row for row in ledger["stages"] if row["id"] == stage_id)


def adopt_result(ctx: dict, stage_id: str, fn):
    ctx["stage"] = {"id": stage_id}
    started = perf_counter()
    result = fn(ctx)
    seconds = perf_counter() - started
    status = str(result.get("status") or "")
    print(
        "IMMUTABLE_STAGE23_BUILD_STAGE="
        + json.dumps(
            {
                "stage": stage_id,
                "status": status,
                "seconds": seconds,
                "blockers": result.get("blockers") or [],
                "performance": result.get("performance") or {},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if status not in {"PASS", "PASS_DEMO_ONLY"}:
        raise RuntimeError(
            f"IMMUTABLE_STAGE_NOT_PASS::{stage_id}::{status}::{result.get('blockers')}"
        )
    row = stage_row(ctx["ledger"], stage_id)
    row["status"] = status
    row["outputs"] = list(result.get("outputs") or ())
    row["blockers"] = []
    return seconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--parent-run-id", required=True)
    p.add_argument("--child-run-id", required=True)
    a = p.parse_args()

    repo_root = Path(".").resolve()
    authority_root = a.authority_root.resolve()
    parent_root = (authority_root / "runs" / a.parent_run_id).resolve()
    child_root = (authority_root / "runs" / a.child_run_id).resolve()

    if not parent_root.is_dir():
        raise RuntimeError("IMMUTABLE_PARENT_RUN_MISSING")
    if child_root.exists():
        raise RuntimeError("IMMUTABLE_CHILD_ALREADY_EXISTS__OVERWRITE_FORBIDDEN")

    child_root.mkdir(parents=True)
    os.symlink(parent_root / "inputs", child_root / "inputs", target_is_directory=True)

    manifest = copy.deepcopy(load_json(parent_root / "run_manifest.json"))
    ledger = copy.deepcopy(load_json(parent_root / "ACTIVE_RUN_V2.json"))
    manifest["run_id"] = a.child_run_id
    ledger["run_id"] = a.child_run_id
    ledger["execution_class"] = "DEMO_WITNESS"
    ledger["architecture_scope"] = (
        "CORRECTED_WEIGHT_TOPOLOGY__MECHANICAL_CAA_STAGE23_IMMUTABLE_V1"
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

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    directions = output_direction_set_from_dict(
        stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )

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
    row18 = stage_row(ledger, "18_CANONICAL_MESH_ADDRESSING_BUILD")
    row18["outputs"] = [
        ref
        for ref in row18.get("outputs") or ()
        if ref.get("schema") != "RealSaS.AppearanceDomainIR.v1"
    ] + [mechanical_domain_ref]
    row18["status"] = "PASS"
    row18["blockers"] = []

    (child_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (child_root / "ACTIVE_RUN_V2.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )

    timings = {}
    for stage_id, fn in (
        ("19_STATIC_CANONICAL_MESH_QUALIFIED", qualify_static_canonical_mesh_stage),
        ("20_CAA_BACKEND_PREREGISTERED", preregister_caa_backend_stage),
        ("21_CAA_COMPILE", compile_caa_stage),
        ("22_CAA_COMPILE_SEALED", seal_caa_compile_stage),
        ("23_COMPLETE_APPEARANCE_ASSET_BAKED", bake_complete_appearance_stage),
    ):
        timings[stage_id] = adopt_result(ctx, stage_id, fn)
        (child_root / "ACTIVE_RUN_V2.json").write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n"
        )

    compile_artifact = stage_output_payload(
        ctx,
        "21_CAA_COMPILE",
        "RealSaS.CAACompileArtifactIR.v2",
    )
    stage23 = stage_output_payload(
        ctx,
        "23_COMPLETE_APPEARANCE_ASSET_BAKED",
        "RealSaS.CompleteAppearanceAssetIR.v2",
    )
    if dict(stage23.get("metadata") or {}).get("source_owned_visual_mesh_mode") is True:
        raise RuntimeError("IMMUTABLE_STAGE23_MECHANICAL_MODE_REQUIRED")
    if stage23["candidate_mesh_binding_hash"] != candidate.candidate_lineage_hash:
        raise RuntimeError("IMMUTABLE_STAGE23_CANDIDATE_BINDING_DRIFT")

    manifest_sha = sha256_file(child_root / "run_manifest.json")
    ledger_sha = sha256_file(child_root / "ACTIVE_RUN_V2.json")
    seal = {
        "schema": "RealSaS.ImmutableMechanicalCAAStage23Seal.v1",
        "status": "SEALED_IMMUTABLE_STAGE23",
        "parent_run_id": a.parent_run_id,
        "run_id": a.child_run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "surface_addressing_hash": addressing.addressing_hash,
        "mechanical_appearance_domain_hash": mechanical_domain.domain_hash,
        "caa_compile_hash": compile_artifact["compile_hash"],
        "stage23_asset_hash": stage23["asset_hash"],
        "stage23_candidate_binding_hash": stage23["candidate_mesh_binding_hash"],
        "source_owned_visual_mesh_mode": False,
        "stage19_to23_only": True,
        "stage24_executed": False,
        "renderer_executed": False,
        "overwrite_forbidden": True,
        "run_manifest_sha256": manifest_sha,
        "active_ledger_sha256": ledger_sha,
        "timings_seconds": timings,
    }
    seal_path = child_root / "IMMUTABLE_STAGE23_SEAL.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")

    print(
        "IMMUTABLE_MECHANICAL_CAA_STAGE23_PASS="
        + json.dumps(
            {
                "run_id": a.child_run_id,
                "candidate": candidate.candidate_lineage_hash,
                "compile": compile_artifact["compile_hash"],
                "stage23": stage23["asset_hash"],
                "seal_sha256": sha256_file(seal_path),
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
