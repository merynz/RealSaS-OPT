from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from models.iris.v31.lofo_trainer_v1 import _indices
from models.iris.v31.lofo_trainer_v2 import IrisV31LofoTrainerV2


def _trainer():
    trainer = IrisV31LofoTrainerV2.__new__(IrisV31LofoTrainerV2)
    trainer.seed = 26102831
    trainer.recipe = {
        "sampling": {
            "mechanical_queries_per_subject_step": 4,
            "appearance_queries_per_subject_step": 4,
            "pair_queries_per_subject_step": 4,
        }
    }
    return trainer


def _subject():
    valid = np.ones(12, dtype=np.bool_)
    return SimpleNamespace(
        presentation={
            "mechanical_valid": valid.copy(),
            "appearance_valid": valid.copy(),
            "pair_valid": valid.copy(),
        },
        partitions={
            "mechanics": (np.arange(8), np.arange(8, 12)),
            "appearance": (np.arange(8), np.arange(8, 12)),
            "pair": (np.arange(8), np.arange(8, 12)),
        },
    )


def test_v2_validation_rows_are_fixed_and_complete_across_checkpoint_steps():
    trainer, subject = _trainer(), _subject()
    a = trainer._presentation_ids(subject, "appearance", 200, train=False)
    b = trainer._presentation_ids(subject, "appearance", 400, train=False)
    np.testing.assert_array_equal(a, np.arange(8, 12))
    np.testing.assert_array_equal(b, np.arange(8, 12))


def test_v2_training_sampler_matches_v1_exact_indices_contract():
    trainer, subject = _trainer(), _subject()
    step = 17
    actual = trainer._presentation_ids(subject, "appearance", step, train=True)
    pool = subject.partitions["appearance"][0]
    expected = pool[
        _indices(
            len(pool),
            4,
            trainer.seed + step * 8191 + 23,
        )
    ]
    np.testing.assert_array_equal(actual, expected)


def test_v2_heldout_all_rows_ignores_train_validation_partition():
    trainer, subject = _trainer(), _subject()
    actual = trainer._presentation_ids(
        subject,
        "appearance",
        6000,
        train=False,
        all_rows=True,
    )
    np.testing.assert_array_equal(actual, np.arange(12))
