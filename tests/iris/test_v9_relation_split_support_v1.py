from __future__ import annotations

from pathlib import Path
import unittest

import torch

from models.iris.vnext.cohort_v9 import V9_HOLDOUT_FRACTION
from tools.iris_vnext.build_v9_relation_split_support_v1 import (
    SCHEMA,
    _binary_counts,
    _partition_pair_indices,
    _rows_from_pairs,
)


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/iris_vnext/build_v9_relation_split_support_v1.py"


class V9RelationSplitSupportTests(unittest.TestCase):
    def test_pair_split_is_deterministic_complete_and_orientation_grouped(self) -> None:
        train_a, held_a = _partition_pair_indices("Snake", 3, 1024)
        train_b, held_b = _partition_pair_indices("Snake", 3, 1024)
        self.assertTrue(torch.equal(train_a, train_b))
        self.assertTrue(torch.equal(held_a, held_b))
        self.assertEqual(len(train_a) + len(held_a), 1024)
        self.assertEqual(len(held_a), round(1024 * V9_HOLDOUT_FRACTION))
        self.assertFalse(set(train_a.tolist()).intersection(held_a.tolist()))

        train_rows = _rows_from_pairs(train_a)
        held_rows = _rows_from_pairs(held_a)
        self.assertFalse(set(train_rows.tolist()).intersection(held_rows.tolist()))
        for pair in train_a[:16].tolist():
            self.assertIn(pair * 2, train_rows.tolist())
            self.assertIn(pair * 2 + 1, train_rows.tolist())
        for pair in held_a[:16].tolist():
            self.assertIn(pair * 2, held_rows.tolist())
            self.assertIn(pair * 2 + 1, held_rows.tolist())

    def test_binary_counts_respect_validity_and_reject_soft_targets(self) -> None:
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        valid = torch.tensor([True, True, False, True])
        rows = torch.tensor([0, 1, 2, 3])
        self.assertEqual(
            _binary_counts(target, valid, rows),
            {"valid": 3, "positive": 1, "negative": 2},
        )
        with self.assertRaisesRegex(ValueError, "NONBINARY"):
            _binary_counts(torch.tensor([0.5]), torch.tensor([True]), torch.tensor([0]))

    def test_builder_is_zero_step_support_authority_only(self) -> None:
        text = BUILDER.read_text(encoding="utf-8")
        self.assertEqual(SCHEMA, "RealSaS.IRISVNextV9RelationSplitSupport.v1")
        self.assertIn("PASS__SPLIT_SUPPORT_QUALIFIED__NO_TRAINING", text)
        self.assertIn('"training_steps_executed": 0', text)
        self.assertIn('"scientific_fit_started": False', text)
        self.assertIn('"orientation_grouped": True', text)
        self.assertIn('"no_group_leakage": True', text)
        self.assertNotIn("optimizer.step", text)
        self.assertNotIn("one_step(", text)


if __name__ == "__main__":
    unittest.main()
