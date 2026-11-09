"""Synthetic, no-private-Drive source-court routing contracts for V17."""
import unittest
from tools.corpus.truth_foundry.evidence_v17 import inventory_mass_observations_v17 as v

class InventoryEvidenceRouterV17(unittest.TestCase):
 def fixture(self,**kwargs):
  r={k:'' for k in ('provider','pool','source_import_observed','historical_split_hint','master_v11_holdout_protected',
   'partverse_v10_processed_numeric_court','partverse_v10_processed_parts_available','v5_source_rest_court',
   'canonical_v6_sha_verified')}
  r.update(kwargs);return r
 def test_master_fit_import_only(self):
  args={'provider':'objaverse_animated_originals','pool':'MASTER3993','source_import_observed':'True','historical_split_hint':'FIT','master_v11_holdout_protected':'False'}
  self.assertEqual(v.source_class(self.fixture(**args))[0],'MASTER')
  args['master_v11_holdout_protected']='True';self.assertEqual(v.source_class(self.fixture(**args))[0],'METADATA')
  args['master_v11_holdout_protected']='False';args['historical_split_hint']='EXTERNAL_HOLDOUT'
  self.assertEqual(v.source_class(self.fixture(**args))[0],'METADATA')
 def test_partverse_must_have_physical_processed_court(self):
  args={'provider':'PARTVERSE_XL','pool':'PARTVERSE_WAVE2_EXT','partverse_v10_processed_numeric_court':'PASS_PROCESSED_PART_GEOMETRY_NUMERIC_ONLY','partverse_v10_processed_parts_available':'True'}
  self.assertEqual(v.source_class(self.fixture(**args))[0],'PARTVERSE')
  args['partverse_v10_processed_numeric_court']='UNKNOWN'
  self.assertEqual(v.source_class(self.fixture(**args))[0],'METADATA')
 def test_canonical_and_v5_require_receipts(self):
  self.assertEqual(v.source_class(self.fixture(provider='UNIML3D',canonical_v6_sha_verified='True'))[0],'UNIML')
  self.assertEqual(v.source_class(self.fixture(provider='UNIML3D',canonical_v6_sha_verified='False'))[0],'METADATA')
  self.assertEqual(v.source_class(self.fixture(pool='CORE370',v5_source_rest_court='PASS_SOURCE_REST_INDEPENDENT_COURT'))[0],'CORE')
 def test_zero_is_real_source_value_and_missing_is_not_invented(self):
  r={'skin_rows_sampled':'200','skin_rows_bad_sum':'0','skin_rows_zero_weights':''}
  val=v.value_for(r,['skin_rows_sampled','skin_rows_bad_sum','skin_rows_zero_weights'],'MASTER', 'sample_counts_only','a'*64)
  import json
  d=json.loads(val)
  self.assertEqual(d['observed_field_values'],{'skin_rows_sampled':'200','skin_rows_bad_sum':'0'})
  self.assertFalse(d['rights_training_admission'])
  self.assertFalse(d['model_training_admission'])
  self.assertFalse(d['semantic_editable_drawable_admission'])
 def test_every_policy_is_explicit_partial_not_original_parity(self):
  for domain,mappings in [('PARTVERSE',v.PARTVERSE),('MASTER',v.MASTER),('CORE',v.CORE),('UNIML',v.UNIML),('METADATA',v.METADATA)]:
   names=[k for k,fields,scope in mappings]
   self.assertEqual(len(names),len(set(names)),domain)
   for facet,fields,scope in mappings:
    self.assertTrue(scope)
    self.assertNotIn('TRAINING_PASS',scope.upper())
    self.assertNotIn('CREATOR_ORIGINAL_VERIFIED',scope.upper())
   self.assertNotIn('PASS',v.STATES[domain])

if __name__=='__main__':unittest.main()