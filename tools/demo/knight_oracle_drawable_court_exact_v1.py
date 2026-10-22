from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload as canonical_stage_output_payload,
)
import tools.demo.knight_oracle_drawable_court_v1 as court


OLD_RUN_ID = "SUBJECT2_KNIGHT_CORRECTED_MECHANICAL_CAA_HARMONIC_FIRST_IMMUTABLE_V1_70CBD671"
STAGE23_ID = "23_COMPLETE_APPEARANCE_ASSET_BAKED"
STAGE23_SCHEMA = "RealSaS.CompleteAppearanceAssetIR.v2"


def run(*, authority_root: Path, run_id: str, out_dir: Path):
    authority_root = authority_root.resolve()
    if run_id != OLD_RUN_ID:
        raise RuntimeError(f"DRAWABLE_EXACT_RUN_ID_FORBIDDEN:{run_id}")
    child_root = authority_root / "runs" / run_id
    seal_path = child_root / "IMMUTABLE_STAGE23_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("DRAWABLE_EXACT_OLD_SEAL_MISSING")
    seal = json.loads(seal_path.read_text())
    if (
        seal.get("schema") != "RealSaS.ImmutableMechanicalCAAStage23Seal.v1"
        or seal.get("status") != "SEALED_IMMUTABLE_STAGE23"
        or seal.get("run_id") != run_id
        or seal.get("overwrite_forbidden") is not True
        or seal.get("source_owned_visual_mesh_mode") is not False
    ):
        raise RuntimeError("DRAWABLE_EXACT_OLD_SEAL_INVALID")

    parent_run_id = str(seal["parent_run_id"])
    parent_ctx = court._ctx(authority_root, parent_run_id)
    stage23_path = (
        child_root
        / "artifacts"
        / STAGE23_ID
        / "complete_appearance_asset.json"
    )
    if not stage23_path.is_file():
        raise RuntimeError("DRAWABLE_EXACT_STAGE23_FILE_MISSING")
    exact_stage23 = json.loads(stage23_path.read_text())
    if exact_stage23.get("schema") != STAGE23_SCHEMA:
        raise RuntimeError("DRAWABLE_EXACT_STAGE23_SCHEMA_DRIFT")
    if str(exact_stage23.get("asset_hash")) != str(seal["stage23_asset_hash"]):
        raise RuntimeError("DRAWABLE_EXACT_STAGE23_ASSET_HASH_DRIFT")
    if bool((exact_stage23.get("metadata") or {}).get("source_owned_visual_mesh_mode")):
        raise RuntimeError("DRAWABLE_EXACT_STAGE23_NOT_MECHANICAL_CAA")

    original = court.stage_output_payload

    def exact_resolver(_ctx: dict, stage_id: str, schema: str):
        if stage_id == STAGE23_ID:
            if schema != STAGE23_SCHEMA:
                raise RuntimeError(
                    f"DRAWABLE_EXACT_STAGE23_SCHEMA_REQUEST_INVALID:{schema}"
                )
            return exact_stage23
        return canonical_stage_output_payload(parent_ctx, stage_id, schema)

    court.stage_output_payload = exact_resolver
    try:
        report = court.run(
            authority_root=authority_root,
            run_id=run_id,
            out_dir=out_dir,
        )
    finally:
        court.stage_output_payload = original

    report_path = out_dir.resolve() / "REPORT.json"
    payload = json.loads(report_path.read_text())
    payload["exact_historical_binding"] = {
        "immutable_child_run_id": run_id,
        "mechanics_parent_run_id": parent_run_id,
        "stage23_path": str(stage23_path),
        "stage23_asset_hash": str(seal["stage23_asset_hash"]),
        "stage23_recompiled": False,
        "parent_mechanics_recomputed": False,
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_ORACLE_DRAWABLE_EXACT_BINDING_PASS="
        + json.dumps(payload["exact_historical_binding"], sort_keys=True),
        flush=True,
    )
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", default=OLD_RUN_ID)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(authority_root=a.authority_root, run_id=a.run_id, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
