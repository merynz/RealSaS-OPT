import unittest
from tools.corpus.truth_foundry.parallel_external_source_triage_v1 import decide_articulation

class ParallelExternalTriageV1Test(unittest.TestCase):
    def test_categories_and_exclusions(self):
        self.assertEqual(decide_articulation({'category_label':'character, scanned data'}),'exclude_or_manual_review')
        self.assertEqual(decide_articulation({'category_label':'animal, anthropomorphic'}),'creature_animal')
        self.assertEqual(decide_articulation({'category_label':'character, mythical creature'}),'creature_animal')
        self.assertEqual(decide_articulation({'category_label':'character, anthropomorphic'}),'humanoid')
        self.assertEqual(decide_articulation({'category_label':'vehicle'}),'mechanical_vehicle')
        self.assertEqual(decide_articulation({'category_label':'weapon'}),'unknown')

if __name__=='__main__':
    unittest.main()
