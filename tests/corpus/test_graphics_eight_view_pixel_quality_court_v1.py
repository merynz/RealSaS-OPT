import csv
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.corpus.truth_foundry.graphics_eight_view_pixel_quality_court_v1 import run, file_sha


class QualityCourtTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.uid = 'testsource'
        manifest = self.root / 'SIX_SOURCE_SELECTED_MANIFEST_V1.csv'
        with manifest.open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=['object_type','morphology_provider_claim','glb_sha256','training_authorized'])
            w.writeheader();w.writerow({'object_type':self.uid,'morphology_provider_claim':'avian','glb_sha256':'a'*64,'training_authorized':'False'})
        self.receipts = []
        dst=self.root/self.uid
        dst.mkdir()
        for deg in range(0,360,45):
            frame=Image.new('RGBA',(256,256),(0,0,0,0))
            for x in range(40+deg//45,150+deg//45):
                for y in range(50,175):frame.putpixel((x,y),(200,100,100,255))
            nat=dst/f'orbit_{deg:03d}_native.png'; sty=dst/f'orbit_{deg:03d}_posterized_candidate.png'
            frame.save(nat);frame.save(sty)
            self.receipts.append({'object_type':self.uid,'glb_sha256':'a'*64,'camera_azimuth_deg_relative':deg,
             'camera_matrix_world_rowmajor':[[float(i==j)+deg*.0001 for j in range(4)] for i in range(4)],
             'camera_source_frame':'CANONICAL_GLB_BLENDER_IMPORT__ANATOMICAL_FRONT_UNKNOWN',
             'native_rgba_path':str(nat.relative_to(self.root)),'native_rgba_sha256':file_sha(nat),
             'style_candidate_path':str(sty.relative_to(self.root)),'style_candidate_sha256':file_sha(sty),
             'source_pixel_original_authority':'NOT_VERIFIED','source_rights_court':'NOT_RUN',
             'style_art_court':'PROCEDURAL_2D_CANDIDATE__NOT_APPROVED','training_authorized':False})
        self.write()

    def write(self):
        with (self.root/'RENDER_RECEIPTS_V1.jsonl').open('w') as f:
            for x in self.receipts:f.write(json.dumps(x)+'\n')

    def test_measurements_are_real_and_do_not_admit(self):
        s=run(self.root,1)
        self.assertEqual(s['source_relative_view_receipts_validated'],8)
        self.assertEqual(s['source_relative_pair_0_180_measured'],1)
        self.assertEqual(s['alpha_same_native_vs_stylized_views'],8)
        self.assertTrue(s['structural_render_receipts_qa_pass'])
        self.assertFalse(s['training_authorized'])
        self.assertEqual(s['approved_stylized_2d_art_targets'],0)

    def test_missing_angle_fails(self):
        self.receipts.pop()
        self.write()
        with self.assertRaisesRegex(ValueError,'EXPECTED_EIGHT'):
            run(self.root,1)

    def test_duplicate_camera_fails(self):
        self.receipts[1]['camera_matrix_world_rowmajor']=self.receipts[0]['camera_matrix_world_rowmajor']
        self.write()
        with self.assertRaisesRegex(ValueError,'DEGENERATE_CAMERA_MATRICES'):
            run(self.root,1)

    def test_source_sha_corrupt_fails(self):
        self.receipts[0]['native_rgba_sha256']='f'*64
        self.write()
        with self.assertRaisesRegex(ValueError,'NATIVE_PIXEL_SHA_MISMATCH'):
            run(self.root,1)

    def test_stylization_alpha_corrupt_fails(self):
        p=self.root/self.uid/'orbit_000_posterized_candidate.png'
        frame=Image.open(p).convert('RGBA')
        frame.putpixel((60,60),(100,0,0,0))
        frame.save(p)
        self.receipts[0]['style_candidate_sha256']=file_sha(p)
        self.write()
        with self.assertRaisesRegex(ValueError,'STYLIZATION_ALTERS_ALPHA'):
            run(self.root,1)


if __name__=='__main__':unittest.main()