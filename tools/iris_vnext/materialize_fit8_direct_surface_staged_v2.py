#!/usr/bin/env python3
from __future__ import annotations

"""Observable staged materializer for the exact V8 Fit8 direct-surface A-arm.

Scientific policy is identical to ``materialize_fit8_direct_surface_v1``:
exact completed V8, FP32/no-autocast, and the selected undeformed
T512 -> sparse-T1024 indexed marching-tetrahedra decoder.  V2 exists only to
make the expensive runtime fail-closed *and observable*: each subject emits a
stage receipt before/after coarse query, refinement selection, fine carrier,
MT counting/build, and implicit-orientation completion.  On exception the
failure receipt is preserved instead of deleting the work directory.
"""

import argparse
import json
from pathlib import Path
import shutil
import traceback
from typing import Any

import numpy as np
import torch

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import (
    IndexedSparseTetraMeshV5,
    SparseRegularTetraPolicyV5,
    boundary_parent_cells_v5,
    build_indexed_mt_from_sparse_carrier_v5,
    count_sparse_mt_v5,
    fine_vertex_gids_v5,
    query_coarse_field_v5,
    query_fine_scalar_v5,
    select_refined_cells_v5,
)
from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.cohort_v9 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_SHA256,
    V8_GLOBAL_STEP,
    V8_SOURCE_COMMIT,
)
from models.iris.vnext.sparse_surface_v3 import (
    VNextScalarFieldQueryV3,
    VNextSparseSurfaceQueryPolicyV3,
    write_direct_iris_surface_bundle_v1,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    BATCH_SCHEMA,
    EXPECTED_PREPARED_SCHEMA,
    assert_full_sparse_policy,
    git_head,
    load_exact_v8_model,
    load_json,
    load_subject_source,
    scene_binding_sha256,
    sha256_file,
    tensor_sha256,
    value_sha256,
)

STAGE_RECEIPT_SCHEMA = "RealSaS.DirectIRISSurfaceStagedExecution.v2"
FAILURE_SCHEMA = "RealSaS.DirectIRISSurfaceMaterializationFailure.v2"
MATERIALIZER_ID = "RealSaS.Fit8DirectIRISSurfaceMaterializer.StagedObservable.v2"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".staged")
    staged.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def _stage_snapshot(
    *,
    subject_id: str,
    stage: str,
    status: str,
    execution_source_sha: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": STAGE_RECEIPT_SCHEMA,
        "materializer_id": MATERIALIZER_ID,
        "subject_id": subject_id,
        "stage": stage,
        "status": status,
        "execution_source_sha": execution_source_sha,
        "v8_checkpoint_sha256": V8_CHECKPOINT_SHA256,
        "base_cells": 512,
        "fine_cells": 1024,
        "numeric_policy": "FP32_NO_AUTOCAST",
        "teacher_truth_used": False,
        "tessa_invoked": False,
        "scientific_pass_claimed": False,
        "diagnostics": diagnostics,
    }


def _persist_stage(
    subject_root: Path,
    *,
    subject_id: str,
    stage: str,
    status: str,
    execution_source_sha: str,
    diagnostics: dict[str, Any],
) -> None:
    value = _stage_snapshot(
        subject_id=subject_id,
        stage=stage,
        status=status,
        execution_source_sha=execution_source_sha,
        diagnostics=diagnostics,
    )
    _atomic_json(subject_root / "DIRECT_IRIS_SURFACE_STAGE.json", value)
    print(
        "DIRECT_SURFACE_STAGE",
        f"subject={subject_id}",
        f"stage={stage}",
        f"status={status}",
        json.dumps(diagnostics, sort_keys=True, allow_nan=False),
        flush=True,
    )


def _staged_decode(
    model: torch.nn.Module,
    scene_planes: torch.Tensor,
    *,
    subject_id: str,
    subject_root: Path,
    execution_source_sha: str,
    sparse_policy: SparseRegularTetraPolicyV5,
    query_policy: VNextSparseSurfaceQueryPolicyV3,
    work_dir: Path,
) -> IndexedSparseTetraMeshV5:
    query = VNextScalarFieldQueryV3(model, scene_planes, policy=query_policy)
    diagnostics: dict[str, Any] = {
        "field_query_api": query.query_api,
        "model_query_chunk": int(query_policy.model_query_chunk),
    }

    _persist_stage(subject_root, subject_id=subject_id, stage="COARSE_T512_QUERY", status="STARTED", execution_source_sha=execution_source_sha, diagnostics=diagnostics)
    coarse = query_coarse_field_v5(query, policy=sparse_policy, work_dir=work_dir)
    diagnostics.update({
        "coarse_field_min": float(np.min(coarse)),
        "coarse_field_max": float(np.max(coarse)),
        "coarse_query_points": int(coarse.size),
        "coarse_zero_bracket": bool(float(np.min(coarse)) < 0.0 < float(np.max(coarse))),
    })
    _persist_stage(subject_root, subject_id=subject_id, stage="COARSE_T512_QUERY", status="PASS", execution_source_sha=execution_source_sha, diagnostics=diagnostics)

    _persist_stage(subject_root, subject_id=subject_id, stage="REFINEMENT_SELECTION", status="STARTED", execution_source_sha=execution_source_sha, diagnostics=diagnostics)
    refined = select_refined_cells_v5(coarse, policy=sparse_policy)
    diagnostics.update({
        "refined_cell_count": int(len(refined)),
        "refined_cell_fraction_of_t512": float(len(refined) / float(sparse_policy.base_cells ** 3)),
        "refine_band": float(sparse_policy.refine_band),
    })
    _persist_stage(subject_root, subject_id=subject_id, stage="REFINEMENT_SELECTION", status="PASS", execution_source_sha=execution_source_sha, diagnostics=diagnostics)

    _persist_stage(subject_root, subject_id=subject_id, stage="FINE_T1024_CARRIER", status="STARTED", execution_source_sha=execution_source_sha, diagnostics=diagnostics)
    gids = fine_vertex_gids_v5(refined, policy=sparse_policy)
    scalar = query_fine_scalar_v5(query, gids, policy=sparse_policy, work_dir=work_dir)
    diagnostics.update({
        "fine_vertex_count": int(len(gids)),
        "fine_scalar_min": float(np.min(scalar)),
        "fine_scalar_max": float(np.max(scalar)),
        "fine_zero_bracket": bool(float(np.min(scalar)) <= 0.0 <= float(np.max(scalar))),
    })
    _persist_stage(subject_root, subject_id=subject_id, stage="FINE_T1024_CARRIER", status="PASS", execution_source_sha=execution_source_sha, diagnostics=diagnostics)

    _persist_stage(subject_root, subject_id=subject_id, stage="INDEXED_MT_COUNT", status="STARTED", execution_source_sha=execution_source_sha, diagnostics=diagnostics)
    boundary = boundary_parent_cells_v5(refined, policy=sparse_policy)
    count = count_sparse_mt_v5(refined, gids, scalar, boundary, policy=sparse_policy)
    diagnostics.update({
        "boundary_parent_cell_count": int(boundary.sum()),
        "candidate_fine_tets": int(count["candidate_fine_tets"]),
        "crossing_tets": int(count["crossing_tets"]),
        "exact_faces": int(count["exact_faces"]),
        "boundary_crossing_tets": int(count["boundary_crossing_tets"]),
    })
    _persist_stage(subject_root, subject_id=subject_id, stage="INDEXED_MT_COUNT", status="PASS", execution_source_sha=execution_source_sha, diagnostics=diagnostics)

    _persist_stage(subject_root, subject_id=subject_id, stage="INDEXED_MT_BUILD_AND_NORMALS", status="STARTED", execution_source_sha=execution_source_sha, diagnostics=diagnostics)
    mesh = build_indexed_mt_from_sparse_carrier_v5(
        refined,
        gids,
        scalar,
        query_fn=query,
        policy=sparse_policy,
        work_dir=work_dir,
    )
    merged = dict(mesh.diagnostics)
    merged.update(diagnostics)
    merged.update({
        "surface_evaluation": "INDEXED_MARCHING_TETRAHEDRA",
        "representation": "SPARSE_REGULAR_TETRA_UNDEFORMED",
        "vnext_surface_bridge": VNextScalarFieldQueryV3.architecture_id,
        "vnext_field_query_api": query.query_api,
        "field_coordinate_transform": VNextScalarFieldQueryV3.coordinate_transform,
        "field_scalar_transform": VNextScalarFieldQueryV3.scalar_transform,
        "teacher_truth_used": False,
        "compiler_qualified": False,
        "scientific_pass_claimed": False,
    })
    mesh = IndexedSparseTetraMeshV5(
        vertices_normalized=mesh.vertices_normalized,
        faces=mesh.faces,
        implicit_normals=mesh.implicit_normals,
        diagnostics=merged,
        decoder_id=mesh.decoder_id,
    )
    diagnostics.update({
        "vertex_count": int(len(mesh.vertices_normalized)),
        "face_count": int(len(mesh.faces)),
        "implicit_normal_diagnostics": mesh.diagnostics.get("implicit_normal_diagnostics"),
    })
    _persist_stage(subject_root, subject_id=subject_id, stage="INDEXED_MT_BUILD_AND_NORMALS", status="PASS", execution_source_sha=execution_source_sha, diagnostics=diagnostics)
    return mesh


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source_sha = str(args.expected_source_sha).strip().lower()
    if len(expected_source_sha) != 40 or any(c not in "0123456789abcdef" for c in expected_source_sha):
        raise ValueError("DIRECT_SURFACE_EXECUTION_SOURCE_SHA_INVALID")
    actual_source_sha = git_head(repo_root)
    if actual_source_sha != expected_source_sha:
        raise RuntimeError(f"DIRECT_SURFACE_CHECKOUT_SHA_DRIFT:{actual_source_sha}:{expected_source_sha}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("DIRECT_SURFACE_FULL_RESOLUTION_REQUIRES_CUDA")

    sparse_policy = SparseRegularTetraPolicyV5()
    assert_full_sparse_policy(sparse_policy)
    query_policy = VNextSparseSurfaceQueryPolicyV3(model_query_chunk=int(args.model_query_chunk))
    query_policy.validate()

    prepared_root = Path(args.prepared_root).resolve()
    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema") != EXPECTED_PREPARED_SCHEMA:
        raise ValueError("DIRECT_SURFACE_PREPARED_MANIFEST_SCHEMA_INVALID")
    rows = list(manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("DIRECT_SURFACE_SUBJECT_ORDER_DRIFT")

    checkpoint_path = Path(args.v8_checkpoint).resolve()
    model = load_exact_v8_model(checkpoint_path, device)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("DIRECT_SURFACE_OUTPUT_DIR_MUST_BE_EMPTY")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / ".work"
    work_root.mkdir(parents=True, exist_ok=True)

    subject_results: list[dict[str, Any]] = []
    for row in rows:
        subject_id = str(row["subject_id"])
        subject_root = output_dir / "subjects" / subject_id
        subject_root.mkdir(parents=True, exist_ok=True)
        subject_work = work_root / subject_id
        if subject_work.exists():
            shutil.rmtree(subject_work)
        subject_work.mkdir(parents=True, exist_ok=True)
        stage = "SUBJECT_SETUP"
        stage_diagnostics: dict[str, Any] = {}
        try:
            shard_path = (prepared_root / str(row["shard_file"])).resolve()
            if not shard_path.is_relative_to(prepared_root):
                raise ValueError("DIRECT_SURFACE_PREPARED_SHARD_PATH_ESCAPE")
            shard_sha = str(row["shard_sha256"]).lower()
            source = load_subject_source(shard_path, shard_sha, device)

            stage = "SCENE_ENCODE"
            _persist_stage(subject_root, subject_id=subject_id, stage=stage, status="STARTED", execution_source_sha=expected_source_sha, diagnostics=stage_diagnostics)
            with torch.inference_mode():
                scene_planes = model.encode_scene(source["image_tokens"], source["token_xy"], source["camera_features"]).to(dtype=torch.float32).detach()
            if scene_planes.shape[0] != 1 or scene_planes.shape[1] != 3:
                raise ValueError("DIRECT_SURFACE_SCENE_PLANES_SHAPE_INVALID")
            if not bool(torch.isfinite(scene_planes).all().item()):
                raise ValueError("DIRECT_SURFACE_SCENE_PLANES_NONFINITE")
            scene_hash = scene_binding_sha256(
                subject_id=subject_id,
                prepared_shard_sha256=shard_sha,
                scene_planes=scene_planes,
                execution_source_sha=expected_source_sha,
            )
            stage_diagnostics.update({
                "scene_planes_sha256": tensor_sha256(scene_planes),
                "scene_binding_sha256": scene_hash,
                "scene_planes_shape": list(scene_planes.shape),
            })
            _persist_stage(subject_root, subject_id=subject_id, stage=stage, status="PASS", execution_source_sha=expected_source_sha, diagnostics=stage_diagnostics)

            stage = "STAGED_T512_T1024_DECODE"
            mesh = _staged_decode(
                model,
                scene_planes,
                subject_id=subject_id,
                subject_root=subject_root,
                execution_source_sha=expected_source_sha,
                sparse_policy=sparse_policy,
                query_policy=query_policy,
                work_dir=subject_work,
            )
            receipt = write_direct_iris_surface_bundle_v1(
                mesh,
                subject_root,
                subject_id=subject_id,
                source_commit=expected_source_sha,
                checkpoint_sha256=V8_CHECKPOINT_SHA256,
                scene_binding_sha256=scene_hash,
                sparse_policy=sparse_policy,
            )
            receipt_path = subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
            subject_results.append({
                "subject_id": subject_id,
                "prepared_shard_file": str(row["shard_file"]),
                "prepared_shard_sha256": shard_sha,
                "scene_binding_sha256": scene_hash,
                "scene_planes_sha256": tensor_sha256(scene_planes),
                "candidate_binding_hash": receipt.candidate_binding_hash,
                "stage12_npz_sha256": receipt.stage12_npz_sha256,
                "receipt_sha256": sha256_file(receipt_path),
                "vertex_count": receipt.vertex_count,
                "face_count": receipt.face_count,
                "field_query_api": mesh.diagnostics.get("vnext_field_query_api"),
                "compiler_qualified": False,
            })
            shutil.rmtree(subject_work, ignore_errors=True)
            del source, scene_planes, mesh
            torch.cuda.empty_cache()
        except Exception as exc:
            stage_receipt = subject_root / "DIRECT_IRIS_SURFACE_STAGE.json"
            if stage_receipt.exists():
                try:
                    last = load_json(stage_receipt)
                    last_stage = str(last.get("stage") or stage)
                    stage_diagnostics = dict(last.get("diagnostics") or stage_diagnostics)
                except Exception:
                    last_stage = stage
            else:
                last_stage = stage
            failure = {
                "schema": FAILURE_SCHEMA,
                "status": "FAILED__NO_SCIENTIFIC_PASS__WORKDIR_PRESERVED",
                "materializer_id": MATERIALIZER_ID,
                "execution_source_sha": expected_source_sha,
                "subject_id": subject_id,
                "failed_stage": last_stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "stage_diagnostics": stage_diagnostics,
                "work_dir": str(subject_work),
                "completed_subjects": [x["subject_id"] for x in subject_results],
                "optimizer_steps_executed": 0,
                "teacher_truth_used": False,
                "tessa_invoked": False,
                "training_authorized": False,
                "scientific_pass_claimed": False,
            }
            _atomic_json(output_dir / "FIT8_DIRECT_IRIS_SURFACE_FAILURE.json", failure)
            _persist_stage(subject_root, subject_id=subject_id, stage=last_stage, status="FAILED", execution_source_sha=expected_source_sha, diagnostics={**stage_diagnostics, "exception_type": type(exc).__name__, "exception_message": str(exc)})
            print("DIRECT_SURFACE_MATERIALIZATION_FAILED", json.dumps(failure, sort_keys=True, allow_nan=False), flush=True)
            raise

    batch = {
        "schema": BATCH_SCHEMA,
        "status": "MATERIALIZED_MODEL_SURFACES__COMPILER_COURT_PENDING",
        "materializer_id": MATERIALIZER_ID,
        "execution_source_sha": expected_source_sha,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "v8_authority": {
            "source_commit": V8_SOURCE_COMMIT,
            "checkpoint_sha256": V8_CHECKPOINT_SHA256,
            "binding_sha256": V8_BINDING_SHA256,
            "global_step": V8_GLOBAL_STEP,
            "scene_core": "SEALED",
        },
        "numeric_policy": "FP32_NO_AUTOCAST",
        "sparse_surface_policy": {
            "base_cells": sparse_policy.base_cells,
            "fine_cells": sparse_policy.fine_cells,
            "policy": "EXACT_SELECTED_T512_TO_SPARSE_T1024_V5",
            "model_query_chunk": query_policy.model_query_chunk,
        },
        "subject_order": list(SUBJECT_IDS),
        "subjects": subject_results,
        "tessa_invoked": False,
        "teacher_truth_used": False,
        "optimizer_steps_executed": 0,
        "compiler_court_executed": False,
        "training_authorized": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_INDEPENDENT_SOURCE_FIDELITY_AND_COMPILER_COURTS",
    }
    if len(subject_results) != len(SUBJECT_IDS):
        raise RuntimeError("DIRECT_SURFACE_SUBJECT_MATERIALIZATION_INCOMPLETE")
    batch["binding_sha256"] = value_sha256(batch)
    _atomic_json(output_dir / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json", batch)
    shutil.rmtree(work_root, ignore_errors=True)
    return batch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-query-chunk", type=int, default=32768)
    return p


def main() -> int:
    args = parser().parse_args()
    result = materialize(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
