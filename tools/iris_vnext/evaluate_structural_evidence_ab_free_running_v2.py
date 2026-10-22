#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed all-Fit8 free-running evaluation for final TESSA V3 A/B arms.

V2 is an execution-hardening evaluator over the frozen H1 training result. It does
not change model weights, conditioning, grammar, decoding logits, sampling,
thresholds, or courts. The only semantic fixes are: (1) a grammar state with no
legal continuation is recorded as an incomplete free-running candidate instead
of crashing the entire evaluator, and (2) final training checkpoints are mmap-
loaded for evaluation so unused optimizer tensors do not create a large host-RAM
peak. Unexpected exceptions remain fatal.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.court_v2 import _codec_config, evaluate_decoded_topology_v2
from models.tessa.v2.experiment_v2 import sha256_file
from models.tessa.v2.generation_v2 import (
    TESSAGeneratedCandidateV2,
    TESSAGenerationPolicyV2,
    TESSAGrammarStateV2,
    _choose_token,
)
from models.tessa.v2.incremental_generation_v2 import (
    TESSAIncrementalStateV2,
    incremental_decode_step_v2,
)
from tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v1 import (
    _conditioning,
    vertex_normals,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head
from tools.iris_vnext.preflight_tessa_v2_a100_v1 import load_subject
from tools.iris_vnext.train_fit8_tessa_v3_structural_evidence_ab_v1 import (
    ARM_A,
    ARM_B,
    ARMS,
    CHECKPOINT_SCHEMA,
    _load_presentation_subject,
    load_json,
    validate_preregistration,
)

SCHEMA = "RealSaS.StructuralEvidenceABFreeRunningEvaluation.v2FailClosed"
ARM_SCHEMA = "RealSaS.StructuralEvidenceABArmFreeRunningEvaluation.v2FailClosed"
NO_ALLOWED_STOP = "GRAMMAR_DEAD_END__NO_ALLOWED_TOKEN"


def _phase(message: str) -> None:
    print("FREE_RUN PHASE " + message, flush=True)


@torch.no_grad()
def generate_failclosed(
    model,
    *,
    memory: torch.Tensor,
    policy: TESSAGenerationPolicyV2,
    seed: int | None,
) -> TESSAGeneratedCandidateV2:
    """Exact incremental greedy decoder with typed dead-end termination.

    This is token-identical to generate_tessa_candidate_incremental_v2 until the
    legacy implementation would raise TESSA_V2_GENERATION_NO_ALLOWED_TOKEN. At
    that state V2 records an incomplete candidate and lets the independent court
    fail it scientifically instead of turning the whole run into infrastructure
    failure.
    """

    policy.validate()
    if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[-1] != model.config.d_model:
        raise ValueError("TESSA_V2_INCREMENTAL_GENERATION_REQUIRES_SINGLE_ASSET_MEMORY")

    previous = model.training
    model.eval()
    grammar = TESSAGrammarStateV2()
    cache = TESSAIncrementalStateV2.for_model(model)
    tokens = [model.config.BOS]
    logp_sum = 0.0
    generator = None
    if seed is not None:
        generator = torch.Generator(device=memory.device)
        generator.manual_seed(int(seed))
    stop_reason = "MAXIMUM_NEW_TOKENS_REACHED"
    try:
        logits = incremental_decode_step_v2(
            model,
            memory=memory,
            token_ids=torch.tensor([model.config.BOS], device=memory.device, dtype=torch.long),
            state=cache,
        )[0]
        for _ in range(policy.maximum_new_tokens):
            allowed = grammar.allowed_token_mask(config=model.config, device=memory.device)
            if not bool(allowed.any()):
                stop_reason = NO_ALLOWED_STOP
                break
            token, logp = _choose_token(logits, allowed, policy=policy, generator=generator)
            tokens.append(token)
            logp_sum += logp
            grammar.advance(token, config=model.config)
            if grammar.complete:
                stop_reason = "EOS"
                break
            logits = incremental_decode_step_v2(
                model,
                memory=memory,
                token_ids=torch.tensor([token], device=memory.device, dtype=torch.long),
                state=cache,
            )[0]
    finally:
        model.train(previous)

    court = evaluate_decoded_topology_v2(tokens, config=model.config)
    return TESSAGeneratedCandidateV2(
        token_ids=tuple(tokens),
        topology_court=court,
        completed=bool(grammar.complete),
        stop_reason=stop_reason,
        token_log_probability_sum=float(logp_sum),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".staged")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def _evaluate_arm(
    *,
    arm_id: str,
    training_root: Path,
    output_root: Path,
    prereg: dict[str, Any],
    prereg_sha: str,
    corpus: dict[str, Any],
    corpus_root: Path,
    corpus_sha: str,
    presentations: list[dict[str, Any]],
    presentation_sha: str,
    device: torch.device,
    training_source_sha: str,
) -> dict[str, Any]:
    _phase(f"arm={arm_id} stage=receipt_validation_begin")
    arm_training_root = training_root / arm_id
    receipt_path = arm_training_root / "TRAINING_RECEIPT.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "RealSaS.StructuralEvidenceABArmTrainingReceipt.v1" or receipt.get("arm_id") != arm_id:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_TRAINING_RECEIPT_INVALID:" + arm_id)
    protocol = prereg["training_protocol"]
    if int(receipt.get("completed_optimizer_steps", -1)) != int(protocol["optimizer_steps_per_arm"]):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_TRAINING_INCOMPLETE:" + arm_id)
    exact = {
        "execution_source_sha": training_source_sha,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_LINEAGE_DRIFT:" + arm_id + ":" + key)
    _phase(f"arm={arm_id} stage=receipt_validation_pass")

    checkpoint_path = arm_training_root / str(receipt["final_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKPOINT_MISSING:" + arm_id)
    _phase(f"arm={arm_id} stage=checkpoint_sha_begin bytes={checkpoint_path.stat().st_size}")
    if sha256_file(checkpoint_path) != receipt.get("final_checkpoint_sha256"):
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKPOINT_SHA_DRIFT:" + arm_id)
    _phase(f"arm={arm_id} stage=checkpoint_sha_pass")

    # Evaluation needs only model tensors. mmap=True keeps the multi-GB optimizer
    # state lazy instead of eagerly materializing it into host RAM.
    _phase(f"arm={arm_id} stage=checkpoint_mmap_load_begin")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    _phase(f"arm={arm_id} stage=checkpoint_mmap_load_pass")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("arm_id") != arm_id:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKPOINT_SCHEMA_INVALID:" + arm_id)
    if checkpoint.get("execution_source_sha") != training_source_sha:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CHECKPOINT_TRAINING_SOURCE_DRIFT:" + arm_id)

    _phase(f"arm={arm_id} stage=model_construct_begin")
    bridge = IRISVNextStructuralizationBridgeV4().to(device=device, dtype=torch.float32)
    _phase(f"arm={arm_id} stage=model_construct_pass")
    _phase(f"arm={arm_id} stage=model_state_load_begin")
    bridge.load_state_dict(checkpoint["model"], strict=True)
    bridge.eval()
    _phase(f"arm={arm_id} stage=model_state_load_pass")
    del checkpoint
    gc.collect()
    _phase(f"arm={arm_id} stage=checkpoint_container_released")

    policy = TESSAGenerationPolicyV2(maximum_new_tokens=131072, sampling=False, temperature=1.0, top_k=0)
    codec = _codec_config(bridge.tessa_config)
    rows = list(corpus.get("subjects") or ())
    arm_output = output_root / arm_id
    arm_output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for subject_index, (subject_id, row, ppayload) in enumerate(zip(SUBJECT_IDS, rows, presentations, strict=True)):
        print(f"FREE_RUN START arm={arm_id} subject={subject_id}", flush=True)
        subject = load_subject(corpus_root / str(row["file"]), str(row["sha256"]))
        with torch.inference_mode():
            memory = _conditioning(
                bridge,
                arm_id=arm_id,
                subject=subject,
                presentation_payload=ppayload,
                device=device,
            )
            candidate = generate_failclosed(
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
            "generation_dead_end_recorded_fail_closed": candidate.stop_reason == NO_ALLOWED_STOP,
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
        _atomic_json(
            arm_output / "FREE_RUNNING_PROGRESS.json",
            {
                "schema": "RealSaS.StructuralEvidenceABFreeRunningProgress.v1",
                "arm_id": arm_id,
                "completed_subject_count": len(results),
                "subjects": results,
                "scientific_pass_claimed": False,
            },
        )
        print(
            f"FREE_RUN DONE arm={arm_id} subject={subject_id} completed={candidate.completed} "
            f"stop={candidate.stop_reason} tokens={len(candidate.token_ids)}",
            flush=True,
        )
        del subject, memory, candidate
        torch.cuda.empty_cache()

    arm_receipt = {
        "schema": ARM_SCHEMA,
        "arm_id": arm_id,
        "status": "FREE_RUNNING_MEASURED_FAIL_CLOSED__INDEPENDENT_AB_COURT_PENDING",
        "training_source_sha": training_source_sha,
        "training_receipt_sha256": sha256_file(receipt_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "subjects": results,
        "teacher_prefix_used": False,
        "sampling": False,
        "dead_end_is_success": False,
        "scientific_pass_claimed": False,
        "compiler_qualified": False,
        "canonical_m_authorized": False,
    }
    _atomic_json(arm_output / "FREE_RUNNING_RECEIPT.json", arm_receipt)
    del bridge
    gc.collect()
    torch.cuda.empty_cache()
    return arm_receipt


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    evaluator_source = str(args.expected_evaluator_sha).strip().lower()
    training_source = str(args.training_source_sha).strip().lower()
    _phase("stage=evaluator_preflight_begin")
    if git_head(repo_root) != evaluator_source:
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_EVALUATOR_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CUDA_REQUIRED")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in str(props.name).upper():
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_A100_REQUIRED:" + str(props.name))
    device = torch.device("cuda")

    prereg_path = Path(args.preregistration).resolve()
    prereg = load_json(prereg_path)
    validate_preregistration(prereg)
    prereg_sha = sha256_file(prereg_path)

    corpus_root = Path(args.corpus_root).resolve()
    corpus_path = corpus_root / "TESSA_V2_CORPUS_MANIFEST.json"
    corpus = load_json(corpus_path)
    corpus_sha = sha256_file(corpus_path)
    rows = list(corpus.get("subjects") or ())
    if corpus.get("schema") != "RealSaS.Fit8TESSAV2TrainingCorpus.v1" or tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_CORPUS_INVALID")

    presentation_root = Path(args.presentation_root).resolve()
    pmanifest_path = presentation_root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    pmanifest = load_json(pmanifest_path)
    presentation_sha = sha256_file(pmanifest_path)
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
    if combined_training.get("execution_source_sha") != training_source:
        raise ValueError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_COMBINED_TRAINING_SOURCE_DRIFT")
    _phase("stage=evaluator_preflight_pass")

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("STRUCTURAL_EVIDENCE_AB_FREE_RUN_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)

    arm_receipts = [
        _evaluate_arm(
            arm_id=arm,
            training_root=training_root,
            output_root=output_root,
            prereg=prereg,
            prereg_sha=prereg_sha,
            corpus=corpus,
            corpus_root=corpus_root,
            corpus_sha=corpus_sha,
            presentations=presentations,
            presentation_sha=presentation_sha,
            device=device,
            training_source_sha=training_source,
        )
        for arm in ARMS
    ]
    result = {
        "schema": SCHEMA,
        "status": "BOTH_ARMS_FREE_RUNNING_MEASURED_FAIL_CLOSED__INDEPENDENT_COURT_REQUIRED",
        "evaluator_source_sha": evaluator_source,
        "training_source_sha": training_source,
        "preregistration_sha256": prereg_sha,
        "corpus_manifest_sha256": corpus_sha,
        "presentation_manifest_sha256": presentation_sha,
        "combined_training_receipt_sha256": sha256_file(combined_training_path),
        "arm_order": list(ARMS),
        "arms": arm_receipts,
        "teacher_prefix_used": False,
        "sampling": False,
        "dead_end_is_success": False,
        "scientific_pass_claimed": False,
        "canonical_m_authorized": False,
        "next_action": "RUN_PREREGISTERED_INDEPENDENT_STRUCTURAL_EVIDENCE_AB_COURT",
    }
    _atomic_json(output_root / "STRUCTURAL_EVIDENCE_AB_FREE_RUNNING_RECEIPT.json", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-evaluator-sha", required=True)
    p.add_argument("--training-source-sha", required=True)
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