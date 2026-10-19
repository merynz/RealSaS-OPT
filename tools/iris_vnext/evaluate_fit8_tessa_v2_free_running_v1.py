#!/usr/bin/env python3
from __future__ import annotations

"""Strict all-Fit8 free-running evaluation for the final locked TESSA V2 checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.structure_global_v3 import IRISVNextGlobalStructuralizationBridgeV3
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.court_v2 import _codec_config
from models.tessa.v2.experiment_v2 import load_preregistration_v1, sha256_file
from models.tessa.v2.generation_v2 import TESSAGenerationPolicyV2
from models.tessa.v2.incremental_generation_v2 import generate_tessa_candidate_incremental_v2
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject

SCHEMA = "RealSaS.TESSAV2Fit8FreeRunningEvaluation.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TESSA_V2_FREE_RUN_JSON_OBJECT_REQUIRED")
    return value


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(v)
    for corner in range(3):
        np.add.at(out, f[:, corner], fn)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    valid = norm[:, 0] > 1.0e-12
    out[valid] /= norm[valid]
    # Winding is proposal-only and not qualified. Unresolved normals remain zero.
    return out.astype(np.float32)


def _conditioning(bridge, subject: dict[str, Any], device: torch.device) -> torch.Tensor:
    def one(name: str) -> torch.Tensor:
        value = subject[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("TESSA_V2_FREE_RUN_TENSOR_MISSING:" + name)
        return value.to(device=device, dtype=torch.float32)[None]

    scene_planes = one("scene_planes")
    xyz = one("witness_xyz")
    normal = one("witness_normal")
    h = one("witness_h")
    mechanics = {
        "stretch_major": one("mechanical_stretch_major"),
        "stretch_minor": one("mechanical_stretch_minor"),
        "tangent_direction_2theta": one("mechanical_tangent_direction_2theta"),
        "bend_demand": one("mechanical_bend_demand"),
        "rigidity_probability": one("mechanical_rigidity_probability"),
    }
    scene = bridge.scene_source(scene_planes)
    surface = bridge.surface_source(
        witness_xyz=xyz,
        witness_normal=normal,
        witness_h=h,
        witness_mechanical=mechanics,
    )
    return bridge.structuralizer.encode_memory(scene_source=scene, surface_source=surface)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("TESSA_V2_FREE_RUN_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("TESSA_V2_FREE_RUN_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    prereg_path = Path(args.preregistration).resolve()
    prereg = load_preregistration_v1(prereg_path)
    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    training_root = Path(args.training_root).resolve()
    training_receipt_path = training_root / "TESSA_V2_TRAINING_RECEIPT.json"
    training_receipt = load_json(training_receipt_path)
    if training_receipt.get("schema") != "RealSaS.TESSAV2LockedTrainingRun.v1":
        raise ValueError("TESSA_V2_FREE_RUN_TRAINING_RECEIPT_SCHEMA_INVALID")
    if int(training_receipt.get("completed_optimizer_steps", -1)) != int(prereg["optimizer"]["optimizer_steps"]):
        raise ValueError("TESSA_V2_FREE_RUN_TRAINING_INCOMPLETE")
    if training_receipt.get("preregistration_sha256") != sha256_file(prereg_path):
        raise ValueError("TESSA_V2_FREE_RUN_PREREGISTRATION_DRIFT")
    if training_receipt.get("corpus_manifest_sha256") != sha256_file(corpus_path):
        raise ValueError("TESSA_V2_FREE_RUN_CORPUS_DRIFT")

    checkpoint_path = training_root / "checkpoints" / str(training_receipt["final_checkpoint"])
    if sha256_file(checkpoint_path) != training_receipt["final_checkpoint_sha256"]:
        raise ValueError("TESSA_V2_FREE_RUN_CHECKPOINT_SHA_DRIFT")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "RealSaS.TESSAV2LockedCheckpoint.v1":
        raise ValueError("TESSA_V2_FREE_RUN_CHECKPOINT_SCHEMA_INVALID")
    if int(checkpoint.get("completed_step", -1)) != int(prereg["optimizer"]["optimizer_steps"]):
        raise ValueError("TESSA_V2_FREE_RUN_FINAL_STEP_DRIFT")

    bridge = IRISVNextGlobalStructuralizationBridgeV3().to(device=device, dtype=torch.float32)
    bridge.load_state_dict(checkpoint["model"], strict=True)
    bridge.eval()
    generation = prereg["free_running"]
    policy = TESSAGenerationPolicyV2(
        maximum_new_tokens=int(generation["maximum_new_tokens"]),
        sampling=bool(generation["sampling"]),
        temperature=float(generation["temperature"]),
        top_k=int(generation["top_k"]),
    )
    if policy.sampling:
        raise RuntimeError("TESSA_V2_FREE_RUN_SAMPLING_FORBIDDEN")

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("TESSA_V2_FREE_RUN_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(corpus.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_FREE_RUN_SUBJECT_ORDER_DRIFT")

    results: list[dict[str, Any]] = []
    codec = _codec_config(bridge.tessa_config)
    for subject_index, (subject_id, row) in enumerate(zip(SUBJECT_IDS, rows, strict=True)):
        subject = load_subject(corpus_root / str(row["file"]), str(row["sha256"]))
        if subject.get("teacher_topology_in_conditioning") is not False:
            raise RuntimeError("TESSA_V2_FREE_RUN_TEACHER_CONDITIONING_FORBIDDEN")
        with torch.inference_mode():
            memory = _conditioning(bridge, subject, device)
            candidate = generate_tessa_candidate_incremental_v2(
                bridge.structuralizer,
                memory=memory,
                policy=policy,
                seed=int(prereg["optimizer"]["seed"]) + subject_index,
            )
        if not candidate.completed or candidate.stop_reason != "EOS":
            # Preserve token/court evidence, but do not fabricate a mesh file.
            result = {
                "subject_id": subject_id,
                "completed": False,
                "stop_reason": candidate.stop_reason,
                "token_count": len(candidate.token_ids),
                "topology_court": candidate.topology_court.to_dict(),
                "candidate_stage12_file": None,
                "candidate_stage12_sha256": None,
            }
        else:
            decoded = decode_tessa_asset_sequence_v1(candidate.token_ids, cfg=codec)
            # Codec outputs TESSA [-0.5,0.5]. Geometry courts use IRIS [-1,1].
            vertices_iris = (2.0 * np.asarray(decoded.vertices_normalized, dtype=np.float64)).astype(np.float32)
            faces = np.asarray(decoded.faces, dtype=np.int32)
            normals = vertex_normals(vertices_iris, faces)
            subject_root = output_root / "subjects" / subject_id
            subject_root.mkdir(parents=True, exist_ok=True)
            stage_path = subject_root / "TESSA_V2_FREE_RUNNING.stage12.npz"
            np.savez(
                stage_path,
                vertices_normalized=vertices_iris,
                faces=faces,
                implicit_normals=normals,
            )
            tokens_path = subject_root / "TESSA_V2_FREE_RUNNING.tokens.npy"
            np.save(tokens_path, np.asarray(candidate.token_ids, dtype=np.int64))
            result = {
                "subject_id": subject_id,
                "completed": True,
                "stop_reason": candidate.stop_reason,
                "token_count": len(candidate.token_ids),
                "token_log_probability_sum": candidate.token_log_probability_sum,
                "topology_court": candidate.topology_court.to_dict(),
                "candidate_stage12_file": stage_path.relative_to(output_root).as_posix(),
                "candidate_stage12_sha256": sha256_file(stage_path),
                "tokens_file": tokens_path.relative_to(output_root).as_posix(),
                "tokens_sha256": sha256_file(tokens_path),
                "coordinate_transform": "IRIS_EQUALS_2_TIMES_TESSA_CODEC_COORDINATE",
                "orientation_qualified": False,
                "compiler_qualified": False,
            }
        results.append(result)
        del subject, memory, candidate
        torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "status": "FREE_RUNNING_MEASURED__PARTIAL_AB_COURT_PENDING",
        "preregistration_sha256": sha256_file(prereg_path),
        "corpus_manifest_sha256": sha256_file(corpus_path),
        "training_receipt_sha256": sha256_file(training_receipt_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": results,
        "teacher_prefix_used": False,
        "generation_backend": "INCREMENTAL_KV_CACHE_WITH_CAUSAL_GLOBAL_BLOCK_HISTORY",
        "sampling": False,
        "scientific_pass_claimed": False,
        "tessa_mandatory": False,
        "canonical_m_authorized": False,
        "compiler_qualified": False,
        "next_action": "RUN_PREREGISTERED_PARTIAL_AB_COURT",
    }
    out = output_root / "TESSA_V2_FREE_RUNNING_RECEIPT.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", default="canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--output-root", required=True)
    result = evaluate(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
