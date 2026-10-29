#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.iris.training.cohort_v1 import file_hash
from models.iris.v31.fixed_snake_capacity_pilot_v1 import FIXED_FOLD_ID, FIXED_HELDOUT
from models.iris.v31.fixed_snake_evidence_student_t_fixed_validation_v3 import (
    IMPLEMENTATION_REVISION_V3,
    run_fixed_snake_evidence_student_t_fixed_validation_v3,
)
from models.iris.v31.fixed_snake_evidence_student_t_pilot_v3 import TRAINER_REVISION_V3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--foundation-checkpoint-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest).resolve()
    recipe = Path(args.recipe).resolve()
    foundation_root = Path(args.foundation_checkpoint_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    result = run_fixed_snake_evidence_student_t_fixed_validation_v3(
        input_manifest_path=input_manifest,
        recipe_path=recipe,
        foundation_checkpoint_root=foundation_root,
        checkpoint_root=Path(args.checkpoint_root).resolve(),
        output_root=output,
        device=args.device,
    )

    result_path = output / "result.json"
    checkpoint_path = output / "selected_checkpoint.pt"
    audit_path = output / "MECHANICS_PARTITION_AUDIT_V3.json"
    receipt = {
        "schema": "RealSaS.IRISV31FixedSnakeEvidenceStudentTFixedValidationExecutionReceipt.v3",
        "status": "PASS",
        "trainer_revision": TRAINER_REVISION_V3,
        "implementation_revision": IMPLEMENTATION_REVISION_V3,
        "validation_row_policy": result["validation_row_policy"],
        "fold_id": FIXED_FOLD_ID,
        "heldout_subject": FIXED_HELDOUT,
        "input_manifest_sha256": file_hash(input_manifest),
        "recipe_sha256": file_hash(recipe),
        "foundation_checkpoint_root": str(foundation_root),
        "foundation_identity": result["foundation_identity"],
        "foundation_source": result["foundation_source"],
        "selected_evidence_step": result["selected_evidence_step"],
        "selection_rule": result["selection_rule"],
        "mechanics_objective": result["mechanics_objective"],
        "partition_policy": result["partition_policy"],
        "mechanics_partition_audit_sha256": file_hash(audit_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "result_sha256": file_hash(result_path),
        "heldout_used_for_gradient": False,
        "heldout_used_for_checkpoint_selection": False,
        "foundation_frozen_during_evidence": True,
        "truth_rows_deleted": False,
        "target_clipping": False,
        "subject_specific_filtering": False,
        "semantic_taxonomy_used": False,
        "pair_training_authorized": False,
        "lofo_expansion_authorized": False,
        "fit8_pass_claimed": False,
        "product_pass_claimed": False,
    }
    (output / "execution_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(
        "[STUDENT_T_FIXEDVAL_DONE] "
        f"heldout={FIXED_HELDOUT} evidence_step={result['selected_evidence_step']} "
        f"foundation={result['foundation_identity'][:16]} "
        f"checkpoint={receipt['checkpoint_sha256'][:16]} lofo_expansion=NOT_AUTHORIZED",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
