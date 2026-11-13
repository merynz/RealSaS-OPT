"""Implementation repair for the Student-t fixed-Snake pilot.

The canonical recipe already requires fixed preregistered presentation validation
rows. Legacy ``_presentation_ids(..., train=False)`` re-sampled the validation
pool as a function of step. This wrapper makes evaluation match the declared
recipe: training sampling is unchanged, while every non-heldout validation gate
uses the complete frozen validation partition in its original deterministic
order.
"""
from __future__ import annotations

import numpy as np

from models.iris.v31.fixed_snake_evidence_student_t_pilot_v3 import (
    FIXED_FOLD_ID,
    IrisV31FixedSnakeEvidenceStudentTPilotV3,
)
from models.iris.v31.lofo_trainer_v1 import _atomic_json

IMPLEMENTATION_REVISION_V3 = "IRIS_V31_FIXED_SNAKE_STUDENT_T_V3__FIXED_FULL_VALIDATION_PARTITION"


class IrisV31FixedSnakeEvidenceStudentTFixedValidationV3(
    IrisV31FixedSnakeEvidenceStudentTPilotV3
):
    """Exact recipe implementation with step-invariant validation rows."""

    def _presentation_ids(
        self,
        subject,
        family: str,
        step: int,
        *,
        train: bool,
        all_rows: bool = False,
    ):
        if not train and not all_rows:
            pool = np.asarray(subject.partitions[family][1], dtype=np.int64)
            return pool.copy() if len(pool) else None
        return super()._presentation_ids(
            subject,
            family,
            step,
            train=train,
            all_rows=all_rows,
        )

    def train(self):
        result = super().train()
        result["implementation_revision"] = IMPLEMENTATION_REVISION_V3
        result["validation_row_policy"] = "FULL_FROZEN_VALIDATION_PARTITION__STEP_INVARIANT"
        _atomic_json(self.output / "result.json", result)
        return result


def run_fixed_snake_evidence_student_t_fixed_validation_v3(
    *,
    input_manifest_path,
    recipe_path,
    foundation_checkpoint_root,
    checkpoint_root,
    output_root,
    device="cuda",
):
    import json
    from pathlib import Path

    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    return IrisV31FixedSnakeEvidenceStudentTFixedValidationV3(
        manifest=manifest,
        recipe=recipe,
        fold_id=FIXED_FOLD_ID,
        foundation_checkpoint_root=foundation_checkpoint_root,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
