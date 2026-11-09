"""Synthetic only: V18 labels and strict non-promotion policy, not source court."""
import importlib.util, pathlib, unittest

HERE=pathlib.Path(__file__).resolve().parents[2]/'tools'/'corpus'/'truth_foundry'/'evidence_v18'

def load(name):
 spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod

class AuthorityV18Contract(unittest.TestCase):
 def test_all_corrections_are_non_geometric(self):
  v=load('correct_partverse_authority_v18')
  self.assertEqual(set(v.RECLASS),{'attribution_obligations','original_asset_provider_locator','original_asset_recovery_status'})
  for vstate in v.RECLASS.values():
   self.assertNotIn('PROCESSED_PART_GEOMETRY_ONLY',vstate)
   self.assertNotIn('TRAINING_PASS',vstate)
  self.assertEqual(len(v.V17_SHA),3)
  self.assertTrue(all(len(s)==64 for s in v.V17_SHA.values()))
 def test_metadata_receipts_are_not_scoped_numeric(self):
  g=load('source_truth_gap_audit_v18')
  states=['PROVIDER_LICENSE_CLASS_CLAIM__ATTRIBUTION_NOT_COURTED',
   'RECORDED_PROVIDER_LOCATOR_CLAIM__NOT_SOURCE_VERIFIED',
   'HISTORICAL_PROCESSED_PART_NUMERIC_RECEIPT__ORIGINAL_UNHASHED',
   'UNKNOWN_UNREVIEWED']
  for s in states[:3]:
   self.assertEqual(g.authority(s),'METADATA_OR_HISTORICAL_RECEIPT_ONLY')
  self.assertEqual(g.authority(states[-1]),'UNKNOWN')
  self.assertEqual(g.authority('OBSERVED_PROCESSED_PART_GEOMETRY_ONLY'),
   'SCOPED_NUMERIC_OR_DERIVED_OBSERVATION_NOT_ADMISSION')
 def test_unknown_authority_is_fail_closed(self):
  g=load('source_truth_gap_audit_v18')
  with self.assertRaises(ValueError):g.authority('TRAINING_PASS')
  self.assertFalse(g.gate_ok('UNKNOWN','PASS'))
  self.assertFalse(g.gate_ok('False','True'))
  self.assertTrue(g.gate_ok('True','True'))
 def test_exact_diff_guard_does_not_accept_unknown_facets(self):
  v=load('verify_authority_correction_v18')
  self.assertEqual(set(v.MAPPING),{'attribution_obligations','original_asset_provider_locator','original_asset_recovery_status'})
  self.assertFalse(any('ANATOMICAL' in x for x in v.MAPPING))

if __name__=='__main__':unittest.main()