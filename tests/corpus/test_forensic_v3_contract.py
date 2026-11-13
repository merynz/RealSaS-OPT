"""Fast CI fixture: forensic matrix observations never become training truth."""
import unittest
from tools.corpus.truth_foundry.forensic_matrix_v3 import initial_facets, obs, OBS_META, OBS_PROCESSED, UNKNOWN

class ForensicV3Contract(unittest.TestCase):
    def test_unknown_and_positive_evidence_are_typed_not_truth(self):
        fs=initial_facets(['source_vertex_face_count','source_face_concept_review'])
        self.assertEqual(fs['source_vertex_face_count']['observation_state'],UNKNOWN)
        self.assertTrue(obs(fs,'source_face_concept_review','11',OBS_PROCESSED))
        self.assertEqual(fs['source_face_concept_review']['observed_value'],'11')
        self.assertFalse(fs['source_face_concept_review']['training_mask_valid'])
        self.assertEqual(fs['source_face_concept_review']['independent_court_state'],'UNKNOWN')
        self.assertFalse(obs(fs,'source_vertex_face_count','',OBS_META))
        self.assertEqual(fs['source_vertex_face_count']['observation_state'],UNKNOWN)

    def test_unknown_facet_refuses(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN_FACET'):
            obs(initial_facets(['source_face_concept_review']),'fictitious_semantic_label',2,OBS_PROCESSED)

    def test_non_source_court_cannot_be_marked_pass(self):
        fs=initial_facets(['per_surface_instance_identity'])
        self.assertTrue(obs(fs,'per_surface_instance_identity','24',OBS_PROCESSED))
        self.assertEqual(fs['per_surface_instance_identity']['supervision_status'],'NOT_ADMITTED')
        self.assertIsNone(fs['per_surface_instance_identity']['court_receipt_sha256'])
        self.assertFalse(fs['per_surface_instance_identity']['training_mask_valid'])

if __name__=='__main__':unittest.main()