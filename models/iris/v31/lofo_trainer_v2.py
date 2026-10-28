"""IRIS V3.1 LOFO trainer V2: authorized fixed-validation research execution.

V1 correctly separates train and validation presentation rows, but subsamples the
validation partition with a step-dependent seed. That makes checkpoint scores
non-comparable when a validation family has more rows than its budget. V2 changes
only that court: training sampling is inherited byte-semantically from V1, while
every preregistered train-subject validation row is evaluated at each checkpoint.
Held-out remains unopened until selection is frozen and still evaluates all valid
rows.

V2 also refuses to instantiate unless input preparation has passed the exact
student-projection observability/leakage court. Stage48 oracle candidacy by itself
is not training authorization.
"""
from __future__ import annotations

import numpy as np

from .lofo_trainer_v1 import IrisV31LofoTrainer, _indices, _require


class IrisV31LofoTrainerV2(IrisV31LofoTrainer):
    def __init__(self, *, manifest: dict, recipe: dict, fold_id: str, checkpoint_root, output_root, device="cuda"):
        qualification = dict(manifest.get("student_projection_qualification") or {})
        _require(manifest.get("student_training_authorized") is True, "IRIS_V31_LOFO_STUDENT_TRAINING_NOT_AUTHORIZED")
        _require(
            manifest.get("status") == "LOFO_INPUTS_SEALED__TRAINING_AUTHORIZED_BY_EXACT_PROJECTION_COURT",
            "IRIS_V31_LOFO_INPUT_AUTHORIZATION_STATUS",
        )
        _require(
            qualification.get("schema") == "RealSaS.IRISV31StudentProjectionQualification.v1"
            and qualification.get("status") == "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED"
            and qualification.get("teacher_inputs_in_forward") is False
            and qualification.get("privileged_ids_in_target_packet") is False
            and qualification.get("semantic_part_taxonomy_used") is False
            and qualification.get("geometric_front_order_student_objective") is False
            and qualification.get("compiler_oracle_retains_draw_order_authority") is True,
            "IRIS_V31_LOFO_STUDENT_PROJECTION_QUALIFICATION",
        )
        _require(
            int(qualification.get("pair_evidence_subject_count", 0)) >= 2
            and int(qualification.get("pair_evidence_valid_rows", 0)) > 0,
            "IRIS_V31_LOFO_PAIR_EVIDENCE_AUTHORIZATION",
        )
        super().__init__(
            manifest=manifest,
            recipe=recipe,
            fold_id=fold_id,
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            device=device,
        )

    def _presentation_ids(
        self,
        subject,
        family: str,
        step: int,
        *,
        train: bool,
        all_rows: bool = False,
    ):
        key = {
            "mechanics": "mechanical_valid",
            "appearance": "appearance_valid",
            "pair": "pair_valid",
        }[family]
        if all_rows:
            pool = np.flatnonzero(np.asarray(subject.presentation[key], dtype=np.bool_))
            return pool if len(pool) else None

        train_pool, validation_pool = subject.partitions[family]
        if not train:
            # Checkpoint selection is a court, not stochastic telemetry: compare
            # every candidate on the exact same preregistered rows.
            return validation_pool if len(validation_pool) else None

        pool = train_pool
        if not len(pool):
            return None
        budget_key = {
            "mechanics": "mechanical_queries_per_subject_step",
            "appearance": "appearance_queries_per_subject_step",
            "pair": "pair_queries_per_subject_step",
        }[family]
        budget = int(self.recipe["sampling"][budget_key])
        # Preserve V1 training sampling exactly; only validation semantics changed.
        return pool[
            _indices(
                len(pool),
                budget,
                self.seed
                + step * 8191
                + {"mechanics": 11, "appearance": 23, "pair": 37}[family],
            )
        ]


def run_fold_v2(*, input_manifest_path, recipe_path, fold_id, checkpoint_root, output_root, device="cuda"):
    import json
    from pathlib import Path

    manifest = json.loads(Path(input_manifest_path).resolve().read_text())
    recipe = json.loads(Path(recipe_path).resolve().read_text())
    _require(
        recipe.get("trainer_revision") == "IRIS_V31_LOFO_TRAINER_V2_FIXED_VALIDATION",
        "IRIS_V31_LOFO_V2_RECIPE_BINDING_REQUIRED",
    )
    return IrisV31LofoTrainerV2(
        manifest=manifest,
        recipe=recipe,
        fold_id=fold_id,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device,
    ).train()
