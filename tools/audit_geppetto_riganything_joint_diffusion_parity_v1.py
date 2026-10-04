from __future__ import annotations

"""Fail-closed source parity guard for Geppetto vs RigAnything joint diffusion."""

import argparse
import hashlib
import json
from pathlib import Path


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise RuntimeError(f"GEPPETTO_RIGANYTHING_PARITY_SIGNATURE_MISSING:{label}")


def main(args) -> None:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    frozen_bytes = args.frozen_source.read_bytes()
    hist_bytes = args.historical_source.read_bytes()
    frozen = frozen_bytes.decode("utf-8")
    hist = hist_bytes.decode("utf-8")

    expected_frozen = contract["frozen_geppetto"]["source_blob"]
    expected_hist = contract["historical_challenger"]["source_blob"]
    actual_frozen = git_blob_sha1(frozen_bytes)
    actual_hist = git_blob_sha1(hist_bytes)
    if actual_frozen != expected_frozen:
        raise RuntimeError(
            f"GEPPETTO_RIGANYTHING_FROZEN_BLOB_DRIFT:{actual_frozen}:{expected_frozen}"
        )
    if actual_hist != expected_hist:
        raise RuntimeError(
            f"GEPPETTO_RIGANYTHING_HISTORICAL_BLOB_DRIFT:{actual_hist}:{expected_hist}"
        )

    frozen_signatures = {
        "residual_target": "residual_target = teacher[step : step + 1] - coarse.detach()",
        "refined_output_adds_residual": "coarse + residual",
        "causal_parent_uses_coarse": "state,\n                coarse,\n                previous_states,",
        "recurrence_stores_coarse": "previous_positions.append(coarse)",
        "explicit_output_refiner_comment": "Diffusion is an output refinement, not a recurrent-state",
    }
    historical_signatures = {
        "absolute_sample_is_current_position":
            "current_position = self.diffusion.sample(context, generator=generator, sample_steps=sample_steps)",
        "sampled_position_to_parent":
            "logits = self._parent_logits(context, current_position, previous_tokens, previous_positions)",
        "sampled_position_to_feedback":
            "token = self._feedback_token(context, current_position, parent_position, step)",
        "recurrence_stores_sample":
            "previous_positions.append(current_position)",
        "bfs_order_augmentation":
            "def randomize_bfs_depth_order_v1(",
        "absolute_teacher_diffusion_target":
            "self.diffusion.loss(current_position, context, generator=generator)",
    }

    for label, needle in frozen_signatures.items():
        require(frozen, needle, "FROZEN_" + label.upper())
    for label, needle in historical_signatures.items():
        require(hist, needle, "HISTORICAL_" + label.upper())

    if contract["frozen_geppetto"]["parity_verdict"] != "NOT_RIGANYTHING_JOINT_DIFFUSION_PARITY":
        raise RuntimeError("GEPPETTO_RIGANYTHING_FROZEN_VERDICT_DRIFT")
    if (
        contract["historical_challenger"]["parity_verdict"]
        != "CORE_CAUSAL_MECHANISM_PARITY_PASS__EXACT_FORMULATION_PARITY_FAIL"
    ):
        raise RuntimeError("GEPPETTO_RIGANYTHING_HISTORICAL_VERDICT_DRIFT")

    result = {
        "schema": "RealSaS.GeppettoRigAnythingJointDiffusionParityGuard.v1",
        "status": "PASS",
        "contract": str(args.contract),
        "frozen_source": str(args.frozen_source),
        "historical_source": str(args.historical_source),
        "frozen_blob_sha1": actual_frozen,
        "historical_blob_sha1": actual_hist,
        "frozen_signature_count": len(frozen_signatures),
        "historical_signature_count": len(historical_signatures),
        "frozen_verdict": contract["frozen_geppetto"]["parity_verdict"],
        "historical_verdict": contract["historical_challenger"]["parity_verdict"],
        "external_reference_commit": contract["external_reference"]["commit"],
        "training_used": False,
        "product_authority_minted": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GEPPETTO_RIGANYTHING_JOINT_DIFFUSION_PARITY_GUARD=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--frozen-source", type=Path, required=True)
    ap.add_argument("--historical-source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    main(ap.parse_args())
