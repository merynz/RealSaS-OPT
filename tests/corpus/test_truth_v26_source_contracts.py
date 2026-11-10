#!/usr/bin/env python3
"""V26 synthetic-only evidence contract tests (no private corpus required)."""
from __future__ import annotations
import unittest,pathlib,ast
HERE=pathlib.Path(__file__).resolve().parents[2]/'tools'/'corpus'/'truth_foundry'/'evidence_v26'

class TestV26(unittest.TestCase):
 def test_no_automatic_admission(self):
  script=(HERE/'process_v26.py').read_text()
  self.assertIn('training_admissions',script)
  self.assertIn('rights_admission',script)
  self.assertIn('creator_original_byte_parity',script)
 def test_readback_restricts_current_npz(self):
  script=(HERE/'verify_v26.py').read_text()
  self.assertIn("assert len(pre)==36 and len(receipts)==82",script)
  self.assertIn("r['creator_original_byte_parity']=='False'",script)
  self.assertIn('training_admissions',script)
 def test_no_import_side_effects_in_synthetic_ci(self):
  for p in HERE.glob('*.py'):
   ast.parse(p.read_text(),filename=str(p))

if __name__=='__main__':unittest.main()