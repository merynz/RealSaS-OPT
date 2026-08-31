from __future__ import annotations

import unittest

from realsas_compiler_core import solver_registry as sr


class SolverRegistryExternalAuthorityTests(unittest.TestCase):
    def test_registry_import_has_no_hidden_historical_import_dependency(self):
        self.assertEqual(set(sr.SOLVER_AUTHORITY), {"CDT", "BBW", "ARAP", "XPBD"})
        self.assertEqual(sr.EXECUTABLE_SOLVERS, {})

    def test_all_heavy_solvers_are_explicitly_external_on_current_main(self):
        for solver_id, record in sr.SOLVER_AUTHORITY.items():
            self.assertEqual(
                record.current_main_status,
                "HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL",
                solver_id,
            )
            self.assertTrue(record.module_name)
            self.assertTrue(record.callable_name)
            self.assertTrue(record.promotion_rule)

    def test_historical_provenance_cannot_be_executed_without_promotion(self):
        for solver_id in sr.SOLVER_AUTHORITY:
            with self.assertRaises(sr.SolverNotPromotedError):
                sr.resolve_promoted_solver(solver_id)

    def test_unknown_solver_fails_closed(self):
        with self.assertRaises(KeyError):
            sr.resolve_promoted_solver("NOT_A_SOLVER")


if __name__ == "__main__":
    unittest.main()
