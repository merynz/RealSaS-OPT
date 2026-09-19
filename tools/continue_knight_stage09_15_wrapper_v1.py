#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

RUN_ID = "SUBJECT2_KNIGHT_V1"
EXPECTED_LEGACY_SHA = "a1735c3b6de9bf5b12f2534a3c9175dd6934e32a3d80444c252c6442983184ed"
EXPECTED_STAGE13_POLICY_SHA = "5fb4859ca09111c908abf3041ae46e0dd3e4c29ab3a373ea9c26ebf5c9b5c59e"
EXPECTED_STAGE14_POLICY_SHA = "7e44ab9a3e371f47d125a9f05763f842ea8b48b07663e8e96f62a8b5249d115e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_ref_path(repo: Path, raw: str) -> Path:
    p = Path(os.path.expandvars(raw)).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def load_legacy(path: Path):
    if sha256_file(path) != EXPECTED_LEGACY_SHA:
        raise RuntimeError(f"LEGACY_CONTINUATION_SHA_DRIFT:{sha256_file(path)}")
    spec = importlib.util.spec_from_file_location("realsas_knight_stage09_12_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("LEGACY_CONTINUATION_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finalise_iris_fit_before_stage09(legacy, repo: Path) -> str:
    from compiler.realsas_compiler_core.preproduct_authority_v1 import (
        ModelFitPreregistrationIR,
        model_fit_preregistration_hash,
    )

    iris_dir = legacy.RUN_ROOT / "inputs" / "iris" / "STRIDE8"
    manifest_path = legacy.RUN_ROOT / "run_manifest.json"
    prereg_path = iris_dir / "IRIS_FIT_PREREG_TEMPLATE.json"
    checkpoint_path = iris_dir / "IRIS_KNIGHT_CHECKPOINT.pt"
    result_path = iris_dir / "IRIS_KNIGHT_FIT_RESULT.json"
    receipt_path = iris_dir / "IRIS_MODEL_FIT_EXECUTION_RECEIPT.json"
    external_path = iris_dir / "IRIS_KNIGHT_EXTERNAL_GPU_FIT_REPORT.json"

    prereg_template = legacy.load_json(prereg_path)
    external = legacy.load_json(external_path)
    source_ref = dict(prereg_template["model_source"])
    source_path = resolve_ref_path(repo, str(source_ref["path"]))
    if legacy.sha256_file(source_path) != legacy.EXPECTED_MODEL_SOURCE_SHA:
        raise RuntimeError("IRIS_PREREG_MODEL_SOURCE_BYTES_DRIFT")

    upstream = {
        "observation_set": legacy.EXPECTED_OBSERVATION_HASH,
        "normalization": legacy.EXPECTED_NORMALIZATION_HASH,
    }
    value = ModelFitPreregistrationIR(
        "IRIS",
        str(prereg_template["architecture_id"]),
        str(prereg_template["executor_kind"]),
        int(prereg_template.get("random_seed", 0)),
        str(source_path),
        legacy.EXPECTED_MODEL_SOURCE_SHA,
        str(prereg_path.resolve()),
        legacy.sha256_file(prereg_path),
        tuple(sorted((str(k), str(v)) for k, v in upstream.items())),
        tuple(map(str, prereg_template["expected_output_contract"])),
        dict(prereg_template.get("qualification_policy") or {}),
        "",
        metadata={
            "teacher_inference_inputs_allowed": False,
            "external_execution_is_not_product_authority": True,
            "preregistration_document_schema": "RealSaS.ModelFitPreregistrationSpec.v1",
        },
    )
    value = replace(value, preregistration_hash=model_fit_preregistration_hash(value))
    prereg_hash = value.preregistration_hash

    receipt = {
        "schema": "RealSaS.ModelFitExecutionReceipt.v1",
        "lane": "IRIS",
        "status": "PASS",
        "architecture_id": value.architecture_id,
        "preregistration_binding_hash": prereg_hash,
        "model_source_sha256": legacy.EXPECTED_MODEL_SOURCE_SHA,
        "checkpoint_sha256": legacy.EXPECTED_CHECKPOINT_SHA,
        "result_sha256": legacy.EXPECTED_RESULT_SHA,
        "teacher_inference_inputs_used": False,
        "upstream_bindings": {
            "normalization": legacy.EXPECTED_NORMALIZATION_HASH,
            "observation_set": legacy.EXPECTED_OBSERVATION_HASH,
        },
        "wall_seconds": float(external.get("wall_seconds", 0.0)),
        "timing": {
            "external_gpu_fit_wall_seconds": float(external.get("wall_seconds", 0.0))
        },
        "metadata": {
            "selected_arm": "STRIDE8",
            "external_gpu_report_sha256": legacy.sha256_file(external_path),
            "fit_bundle_sha256": legacy.EXPECTED_IRIS_BUNDLE_SHA,
            "external_execution_repo_commit": str(external.get("repo_commit", "")),
            "canonical_receipt_minted_after_stage09": True,
            "product_authority_minted": False,
        },
    }
    legacy.write_json(receipt_path, receipt)

    manifest = legacy.load_json(manifest_path)
    manifest["iris_fit"] = {
        "preregistration": {
            "path": str(prereg_path),
            "sha256": legacy.sha256_file(prereg_path),
        },
        "execution_receipt": {
            "path": str(receipt_path),
            "sha256": legacy.sha256_file(receipt_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": legacy.EXPECTED_CHECKPOINT_SHA,
        },
        "result": {
            "path": str(result_path),
            "sha256": legacy.EXPECTED_RESULT_SHA,
        },
    }
    legacy.write_json(manifest_path, manifest)
    print(f"IRIS_FIT_MANIFEST_FINAL_BEFORE_STAGE09=true preregistration_hash={prereg_hash}", flush=True)
    return prereg_hash


def run_mainline_stage(repo: Path, env: dict[str, str], stage_id: str) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "compiler.realsas_compiler_services.orchestrator.mainline",
        "execute",
        "--run-id",
        RUN_ID,
        "--from-stage",
        stage_id,
        "--to-stage",
        stage_id,
    ]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(repo), env=env)
    return result.returncode == 0


def ledger_row(repo: Path, stage_id: str) -> dict:
    ledger = load_json(repo / "canonical" / "ACTIVE_RUN_V1.json")
    return next(row for row in ledger["stages"] if row["id"] == stage_id)


def direct_failure_diagnostic(repo: Path, stage_id: str, manifest_path: Path, out_path: Path) -> None:
    try:
        plan = load_json(repo / "canonical" / "MAINLINE_EXECUTION_PLAN_V1.json")
        ledger = load_json(repo / "canonical" / "ACTIVE_RUN_V1.json")
        stage = next(row for row in plan["stages"] if row["id"] == stage_id)
        ctx = {
            "repo_root": repo,
            "authority_root": Path("/home/runner/realsas_authority"),
            "run_root": Path("/home/runner/realsas_authority") / "runs" / RUN_ID,
            "run_id": RUN_ID,
            "run_manifest_path": manifest_path,
            "run_manifest": load_json(manifest_path),
            "stage": stage,
            "ledger": ledger,
        }
        from compiler.realsas_compiler_services.orchestrator.adapters.iris_geometry_v1 import (
            build_gsa_stage,
            qualify_rest_reprojection_geometry_stage,
        )
        if stage_id == "13_REST_REPROJECTION_GEOMETRY_GATE":
            result = dict(qualify_rest_reprojection_geometry_stage(ctx) or {})
        elif stage_id == "14_GSA_BUILD":
            result = dict(build_gsa_stage(ctx) or {})
        else:
            return
        write_json(out_path, result)
        print("FAIL_DIAGNOSTIC", json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        print(f"FAIL_DIAGNOSTIC_RECOMPUTE_ERROR:{type(exc).__name__}:{exc}", flush=True)


def require_stage(repo: Path, env: dict[str, str], stage_id: str, manifest_path: Path, diagnostic_dir: Path) -> None:
    if run_mainline_stage(repo, env, stage_id):
        return
    row = ledger_row(repo, stage_id)
    print("=" * 88, flush=True)
    print(f"KNIGHT_STAGE_FAIL {stage_id}", flush=True)
    print(json.dumps({
        "status": row.get("status"),
        "attempts": row.get("attempts"),
        "blockers": row.get("blockers"),
        "diagnostics_hash": row.get("diagnostics_hash"),
    }, indent=2, sort_keys=True), flush=True)
    if stage_id in {"13_REST_REPROJECTION_GEOMETRY_GATE", "14_GSA_BUILD"}:
        direct_failure_diagnostic(
            repo,
            stage_id,
            manifest_path,
            diagnostic_dir / f"{stage_id}_FAIL_DIAGNOSTIC.json",
        )
    print("POLICY_MUTATION_FORBIDDEN_AFTER_RESULT=true", flush=True)
    print("=" * 88, flush=True)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--legacy-script", required=True, type=Path)
    ap.add_argument("--stage08-evidence", required=True, type=Path)
    ap.add_argument("--iris-bundle", required=True, type=Path)
    args = ap.parse_args()

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"PYTHON_3_11_REQUIRED:{sys.version.split()[0]}")

    repo = args.repo.expanduser().resolve()
    legacy_path = args.legacy_script.expanduser().resolve()
    stage13_policy_path = repo / "canonical" / "STAGE13_REST_REPROJECTION_POLICY_V1_20260919.json"
    stage14_policy_path = repo / "canonical" / "STAGE14_SUBSTRATE_ADEQUACY_POLICY_V1_20260919.json"
    if sha256_file(stage13_policy_path) != EXPECTED_STAGE13_POLICY_SHA:
        raise RuntimeError("STAGE13_POLICY_SHA_DRIFT")
    if sha256_file(stage14_policy_path) != EXPECTED_STAGE14_POLICY_SHA:
        raise RuntimeError("STAGE14_POLICY_SHA_DRIFT")

    legacy = load_legacy(legacy_path)
    original_run = legacy.run
    predicted = {"hash": None}

    def patched_run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        if (
            "execute" in cmd
            and "--from-stage" in cmd
            and cmd[cmd.index("--from-stage") + 1] == "09_IRIS_FIT_PREREGISTERED"
        ):
            predicted["hash"] = finalise_iris_fit_before_stage09(legacy, Path(cwd))
        original_run(cmd, cwd=Path(cwd), env=env)

    legacy.run = patched_run

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(legacy_path),
            "--repo", str(repo),
            "--stage08-evidence", str(args.stage08_evidence.expanduser().resolve()),
            "--iris-bundle", str(args.iris_bundle.expanduser().resolve()),
        ]
        rc = int(legacy.main())
    finally:
        sys.argv = old_argv
        legacy.run = original_run
    if rc != 0:
        raise RuntimeError(f"LEGACY_STAGE09_12_RETURNED:{rc}")

    prereg_ir = legacy.stage_payload(
        repo, "09_IRIS_FIT_PREREGISTERED", "model_fit_preregistration.json"
    )
    if predicted["hash"] != prereg_ir["preregistration_hash"]:
        raise RuntimeError(
            f"PREDICTED_PREREG_HASH_DRIFT:{predicted['hash']}!={prereg_ir['preregistration_hash']}"
        )

    manifest_path = legacy.RUN_ROOT / "run_manifest.json"
    stage13_policy = load_json(stage13_policy_path)
    stage14_policy = load_json(stage14_policy_path)
    manifest = load_json(manifest_path)
    manifest["geometry_gate"] = dict(stage13_policy["thresholds"])
    manifest["geometry_gate"]["policy_authority"] = {
        "path": str(stage13_policy_path.resolve()),
        "sha256": EXPECTED_STAGE13_POLICY_SHA,
        "schema": stage13_policy["schema"],
        "selected_profile": stage13_policy["selected_profile"],
    }
    manifest["gsa"] = dict(stage14_policy["gsa"])
    manifest["gsa"]["policy_authority"] = {
        "path": str(stage14_policy_path.resolve()),
        "sha256": EXPECTED_STAGE14_POLICY_SHA,
        "schema": stage14_policy["schema"],
        "selected_profile": stage14_policy["selected_profile"],
    }
    write_json(manifest_path, manifest)

    env = dict(os.environ)
    env["REALSAS_AUTHORITY_ROOT"] = str(legacy.AUTHORITY_ROOT)
    env.setdefault("PYTHONHASHSEED", "0")
    iris_dir = legacy.RUN_ROOT / "inputs" / "iris" / "STRIDE8"

    require_stage(
        repo, env, "13_REST_REPROJECTION_GEOMETRY_GATE", manifest_path, iris_dir
    )
    require_stage(repo, env, "14_GSA_BUILD", manifest_path, iris_dir)
    require_stage(repo, env, "15_RIGGING_SURFACE_QUALIFIED", manifest_path, iris_dir)

    ledger = load_json(repo / "canonical" / "ACTIVE_RUN_V1.json")
    if ledger.get("completed_count") != 15 or ledger.get("next_stage") != "16_GEPPETTO_FIT_PREREGISTERED":
        raise RuntimeError(
            f"STAGE15_PROGRESS_INVALID:{ledger.get('completed_count')}:{ledger.get('next_stage')}"
        )

    checkpoint_seal = legacy.stage_payload(
        repo, "11_IRIS_CHECKPOINT_SEALED", "model_checkpoint_seal.json"
    )
    zero = legacy.stage_payload(
        repo, "12_ZERO_SURFACE_DECODED", "signed_zero_surface_seal.json"
    )
    gate = legacy.stage_payload(
        repo, "13_REST_REPROJECTION_GEOMETRY_GATE", "rest_reprojection_geometry_gate.json"
    )
    adequacy = legacy.stage_payload(
        repo, "14_GSA_BUILD", "substrate_adequacy_report.json"
    )
    surface = legacy.stage_payload(
        repo, "14_GSA_BUILD", "rigging_surface_candidate.json"
    )
    qualification = legacy.stage_payload(
        repo, "15_RIGGING_SURFACE_QUALIFIED", "rigging_surface_qualification.json"
    )

    summary = {
        "schema": "RealSaS.KnightIRISCanonicalContinuationSummary.v3",
        "status": "PASS_STAGE15",
        "run_id": RUN_ID,
        "selected_arm": "STRIDE8",
        "progress": "15/40",
        "next_stage": "16_GEPPETTO_FIT_PREREGISTERED",
        "stage09_preregistration_hash": prereg_ir["preregistration_hash"],
        "stage11_checkpoint_seal_hash": checkpoint_seal["checkpoint_seal_hash"],
        "stage12_zero_surface_hash": zero["zero_surface_hash"],
        "stage13_policy_sha256": EXPECTED_STAGE13_POLICY_SHA,
        "stage13_geometry_gate_hash": gate["geometry_gate_hash"],
        "stage13_every_view_passed": gate["qualification_report"]["every_view_passed"],
        "stage14_policy_sha256": EXPECTED_STAGE14_POLICY_SHA,
        "stage14_adequacy_report_hash": adequacy["adequacy_report_hash"],
        "stage14_selected_target_node_cap": adequacy["selected_target_node_cap"],
        "stage14_selected_actual_node_count": adequacy["selected_actual_node_count"],
        "stage14_surface_lineage_hash": surface["geometry_lineage_hash"],
        "stage15_qualification_hash": qualification["qualification_hash"],
        "policy_mutation_after_knight_result_forbidden": True,
        "scientific_stop": "GEPPETTO_FIT_PREREGISTRATION_NEXT",
    }
    summary_path = iris_dir / "CANONICAL_CONTINUATION_STAGE15_SUMMARY.json"
    write_json(summary_path, summary)
    print("=" * 88)
    print("KNIGHT_CANONICAL_CONTINUATION PASS_STAGE15")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY", summary_path)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
