"""Fail-closed Graphics/six-head source eligibility regression."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from tools.corpus.truth_foundry.graphics_singleview_inventory_v1 import run

class GraphicsInventoryTest(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory();self.addCleanup(self.d.cleanup)
        self.p=Path(self.d.name)
    def table(self,name,rows):
        p=self.p/name
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return p
    def fixture(self,missing=0):
        base={'corpus':'MASTER3993','asset_id':'asset_a','source_sha256':'a'*64,'morphology':'UNKNOWN',
              'physical_census_loaded':'True','image_count':'2','material_count':'3','image_dependency_missing_count':str(missing),'skin_rows_sampled':'5'}
        part={'uid':'asset_part','source_sha_provider_claim':'b'*64,'weak_subject_candidate':'ANIMAL_CREATURE',
              'weak_style_candidate':'STYLIZED_CLAIM','segmented_glb_exists':'True','face_count_equal_triangles':'True'}
        art={'uuid':'art_1','candidate_family':'rare','source_sha256_provider_claim':''}
        uni={'object_type':'glb_1','category':'avian','original_asset_sha256':''}
        glb={'object_type':'glb_1','download_status':'DOWNLOADED','glb_valid':'True',
             'download_sha256':'c'*64,'glb_material_count':'1','glb_image_count':'1','glb_texture_count':'1'}
        return [self.table('source.csv',[base]),self.table('wave1.csv',[part]),
                self.table('wave2.csv',[{**part,'uid':'part_2'}]),
                self.table('art.csv',[art]),self.table('uni.csv',[uni]),self.table('glb.csv',[glb])]
    def test_source_court_not_auto_promoted(self):
        x=run(self.fixture(),self.p/'out',expected=5)
        self.assertEqual(x['pixel_source_recovery_candidates_not_approved'],2)
        self.assertEqual(x['approved_stylized_2d_art_target_count'],0)
        self.assertEqual(x['teacher_front_back_exact_multiview_pairs_pass'],0)
        with (self.p/'out'/'GRAPHICS_SINGLEVIEW_SOURCE_COVERAGE_V1.csv').open() as f:
            r=list(csv.DictReader(f))
        self.assertEqual(len(r),5)
        self.assertEqual(r[0]['graphics_priority_bucket'],'P0_PIXEL_RECOVERY_CANDIDATE')
        self.assertEqual(r[-1]['graphics_priority_bucket'],'P0_CANONICAL_PIXEL_INSPECTION_CANDIDATE')
        self.assertTrue(all(a['approved_stylized_2d_art_pass']=='False' and
                            a['teacher_target_camera_leakage_court']=='UNKNOWN' for a in r))
    def test_importer_missing_texture_demotes(self):
        x=run(self.fixture(missing=2),self.p/'out',expected=5)
        self.assertEqual(x['pixel_source_recovery_candidates_not_approved'],1)
        self.assertEqual(x['graphics_priority_buckets']['P1_RELINK_TEXTURE_DEPENDENCIES'],1)
    def test_no_fake_extra_sources(self):
        with self.assertRaisesRegex(ValueError,'EXPECTED_SOURCE_COUNT'):
            run(self.fixture(),self.p/'out',expected=6)
    def test_schema_contract_teacher_student_separated(self):
        p=Path(__file__).resolve().parents[2]/'canonical/REALSAS_GRAPHICS_SINGLEVIEW_TEACHER_TARGET_V1.json'
        if not p.exists(): self.skipTest('Contract not mounted in miniature fixture')
        x=json.loads(p.read_text())
        self.assertEqual(x['input_modes']['source_teacher_views']['count'],8)
        self.assertEqual(x['input_modes']['predicted_multiview_conditioning']['count'],6)
        self.assertEqual(x['input_modes']['user_real_observation']['count'],1)
        self.assertTrue(x['input_modes']['source_teacher_views']['never_expose_to_student'])
        self.assertTrue(all(a['training_label_authorized'] is False for a in x['facets']))
if __name__=='__main__':unittest.main()