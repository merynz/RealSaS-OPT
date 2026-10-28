"""IRIS V3.1 LOFO trainer V2: fixed validation court across checkpoints.

V1 correctly separates train and validation presentation rows, but subsamples the
validation partition with a step-dependent seed.  That makes checkpoint scores
non-comparable when a validation family has more rows than its budget.  V2 keeps
training sampling unchanged and evaluates every preregistered train-subject
validation row at every checkpoint.  Held-out remains unopened until selection is
frozen and still evaluates all valid rows.
"""
from __future__ import annotations

import numpy as np

from .lofo_trainer_v1 import IrisV31LofoTrainer, _require


class IrisV31LofoTrainerV2(IrisV31LofoTrainer):
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
        budget = min(len(pool), int(self.recipe["sampling"][budget_key]))
        rng = np.random.default_rng(
            int(self.seed + step * 8191 + {"mechanics": 11, "appearance": 23, "pair": 37}[family])
            & ((1 << 63) - 1)
        )
        # Training rows are sampled without replacement inside an update.  Across
        # updates the stateless step seed provides deterministic resampling.
        return pool[rng.choice(len(pool), size=budget, replace=False)]


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
