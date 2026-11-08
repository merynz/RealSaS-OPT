"""Source rights crosswalk must distinguish claim, provider identity and training authority."""
import unittest
from tools.corpus.truth_foundry.partverse_objaverse_original_license_crosswalk_v1 import (
 license_class,sketch_uid,correlate,HEADS)

class TestRightsCrosswalk(unittest.TestCase):
 def test_license_policy_not_automatic_train(self):
  self.assertEqual(license_class('CC0'),'CC0_OR_PUBLIC_DOMAIN_CLAIM')
  self.assertEqual(license_class('CC-BY'),'ATTRIBUTION_REVIEW')
  self.assertEqual(license_class('Creative Commons - Attribution'),'ATTRIBUTION_REVIEW')
  self.assertEqual(license_class('Creative Commons - Attribution - Share Alike'),'ATTRIBUTION_SHAREALIKE_REVIEW')
  self.assertEqual(license_class('CC BY-NC-SA 4.0'),'NONCOMMERCIAL_OR_RESTRICTED')
  self.assertEqual(license_class(None),'UNSPECIFIED')
 def test_correct_source_uid(self):
  s='a'*32
  self.assertEqual(sketch_uid('https://sketchfab.com/3d-models/'+s),s)
  self.assertEqual(sketch_uid('https://evil.example/models/'+s),'')
 def test_exact_sha_and_uid_match_is_metadata_only(self):
  uid='b'*32;sha='a'*64
  p=[{'uid':uid,'provider_original_sha_claim':sha,'provider_weak_morphology':'ANIMAL_CREATURE'}]
  m=[{'uid':uid,'sha256':sha,'fileIdentifier':'https://sketchfab.com/3d-models/'+uid,
      'license':'CC0','source':'sketchfab','fileType':'glb'}]
  r=list(correlate(p,m))[0]
  self.assertEqual(r['source_metadata_match_state'],'EXACT_SHA_AND_SOURCE_UID_METADATA_CLAIM')
  self.assertEqual(r['objaverse_license_class'],'CC0_OR_PUBLIC_DOMAIN_CLAIM')
  self.assertFalse(r['head_training_authorized'])
  self.assertFalse(r['original_source_bytes_rehashed'])
  self.assertEqual(len(HEADS),6)
 def test_conflicting_original_sha_fail_closed(self):
  uid='b'*32
  p=[{'uid':uid,'provider_original_sha_claim':'a'*64}]
  m=[{'uid':uid,'sha256':'c'*64,'fileIdentifier':'https://sketchfab.com/3d-models/'+uid,
      'license':'CC-BY','source':'sketchfab','fileType':'glb'}]
  r=list(correlate(p,m))[0]
  self.assertEqual(r['source_metadata_match_state'],'UID_MATCH_SHA_DIFFERENT')
  self.assertFalse(r['head_training_authorized'])

if __name__=='__main__':unittest.main()