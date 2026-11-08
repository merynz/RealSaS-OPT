import tempfile
from pathlib import Path
import pandas as pd
import unittest
from tools.corpus.truth_foundry.parallel_external_source_triage_v1 import (
    decide_articulation, build_uniml, RARE_NON_BIPEDAL
)

class ParallelExternalTriageV1Test(unittest.TestCase):
    def test_categories_and_exclusions(self):
        self.assertEqual(decide_articulation({'category_label':'character, scanned data'}),'exclude_or_manual_review')
        self.assertEqual(decide_articulation({'category_label':'animal, anthropomorphic'}),'creature_animal')
        self.assertEqual(decide_articulation({'category_label':'character, mythical creature'}),'creature_animal')
        self.assertEqual(decide_articulation({'category_label':'character, anthropomorphic'}),'humanoid')
        self.assertEqual(decide_articulation({'category_label':'vehicle'}),'mechanical_vehicle')
        self.assertEqual(decide_articulation({'category_label':'weapon'}),'unknown')

    def test_minor_morphology_preservation_without_train_admission(self):
        rows=[]
        for i in range(7430):
            category = ('serpentine' if i<14 else
                        'avian' if i<128 else
                        'quadrupedal' if i<486 else
                        'insectoid' if i<608 else
                        'marine' if i<663 else
                        'articulated_rigid' if i<792 else 'bipedal')
            rows.append({'dataset':'objaverse','object_type':f'asset_{i:05d}',
                         'category':category,'num_joints':8,'num_clips':2})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'skeletons.csv'
            pd.DataFrame(rows).to_csv(p,index=False)
            universe,selected=build_uniml(p)
        rare_universe=set(universe[universe.category.isin(RARE_NON_BIPEDAL)].object_type)
        rare_selected=set(selected[selected.category.isin(RARE_NON_BIPEDAL)].object_type)
        self.assertEqual(len(rare_universe),792)
        self.assertEqual(rare_universe,rare_selected)
        self.assertEqual(len(selected),2500)
        self.assertTrue((selected.training_authorized==False).all())
        self.assertTrue((selected.source_rights_court=='UNKNOWN').all())

if __name__=='__main__':
    unittest.main()
