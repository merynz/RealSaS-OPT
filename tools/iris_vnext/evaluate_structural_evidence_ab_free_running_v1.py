#!/usr/bin/env python3
from __future__ import annotations

"""Strict all-Fit8 free-running evaluation for both final TESSA V3 A/B arms."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.court_v2 import _codec_config
from models.tessa.v2.experiment_v2 import sha256_file
from models.tessa.v2.generation_v2 import TESSAGenerationPolicyV2
from models.tessa.v2.incremental_generation_v2 import generate_tessa_candidate_incremental_v2
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_A,
    ARM_B,
    ARMS,
    CHECKPOINT_SCHEMA,
    _load_presentation_subject,
    _typed_presentation,
    load_json,
    validate_preregistration,
)

SCHEMA = "RealSaS.StructuralEvidenceABFreeRunningEvaluation.v1"


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
    return out.astype(np.float32)


def _one(subject: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = subject[name]
    if not isinstance(value, torch.Tensor):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_TENSOR_MISSING:" + name)
    return value.to(device=device, dtype=torch.float32)[None]


def _conditioning(bridge: IRISVNextStructuralizationBridgeV4, *, arm_id: str,
                  subject: dict[str, Any], presentation_payload: dict[str, Any],
                  device: torch.device) -> torch.Tensor:
    scene_planes = _one(subject, "scene_planes", device)
    xyz = _one(subject, "witness_xyz", device)
    normal = _one(subject, "witness_normal", device)
    h = _one(subject, "witness_h", device)
    mechanics = {
        "stretch_major": _one(subject, "mechanical_stretch_major", device),
        "stretch_minor": _one(subject, "mechanical_stretch_minor", device),
        "tangent_direction_2theta": _one(subject, "mechanical_tangent_direction_2theta", device),
        "bend_demand": _one(subject, "mechanical_bend_demand", device),
        "rigidity_probability": _one(subject, "mechanical_rigidity_probability", device),
    }
    scene = bridge.scene_source(scene_planes)
    surface = bridge.surface_source(
        witness_xyz=xyz,
        witness_normal=normal,
        witness_h=h,
        witness_mechanical=mechanics,
    )
    presentation = None if arm_id == ARM_A else _typed_presentation(presentation_payload, device)
    typed = bridge.presentation_source(reference=surface, presentation=presentation)
    return bridge.structuralizer.encode_memory(
        scene_source=scene,
        surface_source=surface,
        presentation_source=typed.values,
        presentation_valid_mask=typed.valid_mask,
    )


def _evaluate_arm(*, arm_id: str, training_root: Path, output_root: Path,
                  prereg: dict[str, Any], prereg_sha: str, corpus: dict[str, Any],
                  corpus_root: Path, corpus_sha: str, presentations: list[dict[str, Any]],
                  presentation_sha: str, device: torch.device) -> dict[str, Any]:
    arm_training_root = training_root / arm_id
    receipt_path = arm_training_root / "TRAINING_RECEIPT.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "RealSaS.StructuralEvidenceABArmTrainingReceipt.v1" or receipt.get("arm_id") != arm_id:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_TRAINING_RECEIPT_INVALID:" + arm_id)
    protocol = prereg["training_protocol"]
    if int(receipt.get("completed_optimizer_steps", -1)) != int(protocol["optimizer_steps_per_arm"]):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_TRAINING_INCOMPLETE:" + arm_id)
    exact = {
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_LINEAGE_DRIFT:" + arm_id + ":" + key)
    checkpoint_path = arm_training_root / str(receipt["final_checkpoint"])
    if sha256_file(checkpoint_path) != receipt.get("final_checkpoint_sha256"):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKPOINT_SHA_DRIFT:" + arm_id)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("arm_id") != arm_id:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKPOINT_SCHEMA_INVALID:" + arm_id)

    bridge = IRISVNextStructuralizationBridgeV4().to(device=device, dtype=torch.float32)
    bridge.load_state_dict(checkpoint["model"], strict=True)
    bridge.eval()
    policy = TESSAGenerationPolicyV2(maximum_new_tokens=131072, sampling=False, temperature=1.0, top_k=0)
    codec = _codec_config(bridge.tessa_config)
    rows = list(corpus.get("subjects") or ())
    arm_output = output_root / arm_id
    arm_output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for subject_index, (subject_id, row, ppayload) in enumerate(zip(SUBJECT_IDS, rows, presentations, strict=True)):
        subject = load_subject(corpus_root / str(row["file"]), str(row["sha256"]))
        with torch.inference_mode():
            memory = _conditioning(bridge, arm_id=arm_id, subject=subject,
                                   presentation_payload=ppayload, device=device)
            candidate = generate_tessa_candidate_incremental_v2(
                bridge.structuralizer,
                memory=memory,
                policy=policy,
                seed=int(protocol["seed_set"][0]) + subject_index,
            )
        result: dict[str, Any] = {
            "subject_id": subject_id,
            "completed": bool(candidate.completed),
            "stop_reason": candidate.stop_reason,
            "token_count": len(candidate.token_ids),
            "topology_court": candidate.topology_court.to_dict(),
            "candidate_stage12_file": None,
            "candidate_stage12_sha256": None,
        }
        subject_root = arm_output / "subjects" / subject_id
        subject_root.mkdir(parents=True, exist_ok=True)
        tokens_path = subject_root / "FREE_RUNNING.tokens.npy"
        np.save(tokens_path, np.asarray(candidate.token_ids, dtype=np.int64))
        result["tokens_file"] = tokens_path.relative_to(output_root).as_posix()
        result["tokens_sha256"] = sha256_file(tokens_path)
        if candidate.completed and candidate.stop_reason == "EOS":
            decoded = decode_tessa_asset_sequence_v1(candidate.token_ids, cfg=codec)
            vertices_iris = (2.0 * np.asarray(decoded.vertices_normalized, dtype=np.float64)).astype(np.float32)
            faces = np.asarray(decoded.faces, dtype=np.int32)
            normals = vertex_normals(vertices_iris, faces)
            stage_path = subject_root / "FREE_RUNNING.stage12.npz"
            np.savez(stage_path, vertices_normalized=vertices_iris, faces=faces, implicit_normals=normals)
            result.update({
                "token_log_probability_sum": candidate.token_log_probability_sum,
                "candidate_stage12_file": stage_path.relative_to(output_root).as_posix(),
                "candidate_stage12_sha256": sha256_file(stage_path),
                "coordinate_transform": "IRIS_EQUALS_2_TIMES_TESSA_CODEC_COORDINATE",
                "orientation_qualified": False,
                "compiler_qualified": False,
            })
        results.append(result)
        del subject, memory, candidate
        torch.cuda.empty_cache()

    arm_receipt = {
        "schema": "RealSaS.StructuralEvidenceABArmFreeRunningEvaluation.v1",
        "arm_id": arm_id,
        "status": "FREE_RUNNING_MEASURED__INDEPENDENT_AB_COURT_PENDING",
        "training_receipt_sha256": sha256_file(receipt_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "subjects": results,
        "teacher_prefix_used": False,
        "sampling": False,
        "scientific_pass_claimed": False,
        "compiler_qualified": False,
        "canonical_m_authorized": False,
    }
    (arm_output / "FREE_RUNNING_RECEIPT.json").write_text(
        json.dumps(arm_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    del bridge
    torch.cuda.empty_cache()
    return arm_receipt


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source = str(args.expected_source_sha).lower()
    if git_head(repo_root) != expected_source:
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    prereg_path = Path(args.preregistration).resolve()
    prereg = load_json(prereg_path); validate_preregistration(prereg)
    prereg_sha = sha256_file(prereg_path)
    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path); corpus_sha = sha256_file(corpus_path)
    rows = list(corpus.get("subjects") or ())
    if corpus.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1" or tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CORPUS_INVALID")

    presentation_root = Path(args.presentation_root).resolve()
    pmanifest_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    pmanifest = load_json(pmanifest_path); presentation_sha = sha256_file(pmanifest_path)
    prows = list(pmanifest.get("subjects") or ())
    if pmanifest.get("schema") != "RealSaS.DirectIRISPresentationMemoryManifest.v1" or tuple(row.get("subject_id") for row in prows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_PRESENTATION_INVALID")
    presentations = [
        _load_presentation_subject(presentation_root / str(row["file"]), str(row["sha256"]), subject_id)
        for subject_id, row in zip(SUBJECT_IDS, prows, strict=True)
    ]

    training_root = Path(args.training_root).resolve()
    combined_training_path = training_root / "STRUCTURAL_EVIDENCE_AB_TRAINING_RECEIPT.json"
    combined_training = load_json(combined_training_path)
    if combined_training.get("schema") != "RealSaS.StructuralEvidenceABTrainingRun.v1" or tuple(combined_training.get("arm_order") or ()) != ARMS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_COMBINED_TRAINING_INVALID")

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    arm_receipts = [
        _evaluate_arm(
            arm_id=arm, training_root=training_root, output_root=output_root,
            prereg=prereg, prereg_sha=prereg_sha, corpus=corpus,
            corpus_root=corpus_root, corpus_sha=corpus_sha,
            presentations=presentations, presentation_sha=presentation_sha, device=device,
        )
        for arm in ARMS
    ]
    result = {
        "schema": SCHEMA,
        "status": "BOTH_ARMS_FREE_RUNNING_MEASURED__INDEPENDENT_COURT_REQUIRED",
        "execution_source_sha": expected_source,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "combined_training_receipt_sha256": sha256_file(combined_training_path),
        "arm_order": list(ARMS),
        "arms": arm_receipts,
        "teacher_prefix_used": False,
        "sampling": False,
        "scientific_pass_claimed": False,
        "canonical_m_authorized": False,
        "next_action": "RUN_PREREGISTERED_INDEPENDENT_STRUCTURAL_EVIDENCE_AB_COURT",
    }
    (output_root / "STRUCTURAL_EVIDENCE_AB_FREE_RUNNING_RECEIPT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--preregistration", default="canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--presentation-root", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--output-root", required=True)
    result = evaluate(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
