from __future__ import annotations

import unittest

import gate0_real_corpus_v1 as base
from gate0_real_corpus_v1_1_hashfix import (
    exact_population_match,
    set_sha_sorted_newline_v1,
    validate_declared_train512_set_hash,
)


class Gate0MembershipHashAuthorityTests(unittest.TestCase):
    def test_historical_declared_hash_is_authority_not_local_redefinition(self):
        membership = {"train512_set_sha256": base.EXPECTED_TRAIN512_SET_SHA256}
        # Arbitrary IDs deliberately produce a different local diagnostic digest.
        ids = ["asset_b", "asset_a"]
        self.assertNotEqual(set_sha_sorted_newline_v1(ids), base.EXPECTED_TRAIN512_SET_SHA256)
        validate_declared_train512_set_hash(membership)

    def test_declared_hash_drift_fails(self):
        with self.assertRaises(RuntimeError):
            validate_declared_train512_set_hash({"train512_set_sha256": "0" * 64})

    def test_exact_population_match_is_order_independent(self):
        self.assertTrue(exact_population_match(["c", "a", "b"], ["a", "b", "c"]))

    def test_exact_population_match_rejects_duplicate_or_missing(self):
        self.assertFalse(exact_population_match(["a", "a", "b"], ["a", "b", "c"]))
        self.assertFalse(exact_population_match(["a", "b"], ["a", "b", "c"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
