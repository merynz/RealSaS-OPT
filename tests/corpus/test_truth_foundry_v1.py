"""Source-independent regression of Truth Foundry admission and census rules."""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from tools.corpus.truth_foundry.target_matrix_v1 import (COLUMNS, QUALIFICATION_REASSIGN, signal_master, run)
from tools.corpus.truth_foundry.physical_census_v1 import (COUNTS, bucket, run as physical_run)


class TruthFoundryRegression(unittest.TestCase):
    def test_37_head_targets_unique(self):
        self.assertEqual(len(COLUMNS), 37)
        self.assertEqual(len({(h, target) for h, target, _, _ in COLUMNS}), 37)
        self.assertTrue(all(court.endswith('_COURT') for _, _, _, court in COLUMNS))

    def test_semantic_drawable_not_class_target(self):
        for target in ('drawable_necessity_three_way','layer_source_visibility_relation'):
            self.assertTrue(QUALIFICATION_REASSIGN[target].startswith('COMPILER_GRAPHICS'))

    def test_no_imported_scene_as_semantic_truth(self):
        row={'source_import_observed':'True','source_mesh_observation':'IMPORTED_GEOMETRY_OBSERVATION'}
        self.assertEqual(signal_master(row,'SEMANTIC_REGIONS_INSTANCES'),'UNKNOWN_NO_QUALIFIED_SIGNAL')
        self.assertEqual(signal_master(row,'SOURCE_MESH_VERTICES'),'IMPORTED_OBSERVATION_ONLY')
        row['source_import_observed']='False'
        self.assertEqual(signal_master(row,'SOURCE_MESH_VERTICES'),'SOURCE_IMPORT_NOT_OBSERVED')

    def test_missing_census_has_no_invented_zero(self):
        self.assertEqual(bucket(0,[0,1,2]),'0')
        self.assertEqual(bucket(3,[0,1,2]),'>2')
        self.assertIn('mesh_disconnected_components',COUNTS)
        self.assertIn('observed_ik_constraint_count',COUNTS)

    def test_real_source_matrix_rejects_empty_or_short_roster(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for n in ('master','core','dense','families'):
                (p/(n+'.csv')).write_text('asset_id,training_authorized\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SOURCE_CENSUS'):
                run(p/'master.csv',p/'core.csv',p/'dense.csv',p/'families.csv',p/'out')

    def test_physical_census_requires_exact_master(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'master.csv').write_text('canonical_asset_id,training_authorized,materialization_authorized\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'PHYSICAL_CENSUS_MASTER_COUNT'):
                physical_run(p/'master.csv',(p/'w1',p/'w2',p/'w3',p/'w4'),p/'out')


if __name__=='__main__':
    unittest.main()