"""Eight-view source teacher pilot must not invent anatomy or approved 2D artwork."""
import csv
import tempfile
import unittest
from pathlib import Path
from tools.corpus.truth_foundry.graphics_8view_source_teacher_pilot_v1 import (
  CATS,ANGLES,choose_sources,parse_selection)

class EightViewPilotTest(unittest.TestCase):
 def test_exact_eight_camera_azimuths(self):
  self.assertEqual(ANGLES,(0,45,90,135,180,225,270,315))
 def test_six_rare_textured_morphology_selections_and_no_admission(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);manifest=d/'manifest.csv';selected=d/'selected.csv'
   rows=[]
   for n in range(96):
    cat=CATS[n%6]
    rows.append({'object_type':f'asset_{n:03d}','category_provider_claim':cat,
      'download_status':'DOWNLOADED' if n<84 else 'ERROR',
      'glb_valid':'True','glb_image_count':'2','glb_texture_count':'1',
      'glb_triangle_indices_count':str(600+n),
      'download_sha256':str(n%10)*64,'downloaded_bytes':'1024'})
   with manifest.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   result=choose_sources(manifest,selected)
   self.assertEqual(result['selected'],6)
   picks=parse_selection(selected)
   self.assertEqual({x['morphology_provider_claim'] for x in picks},set(CATS))
   self.assertTrue(all(x['training_authorized']=='False' and x['anatomical_front_axis']=='UNKNOWN' for x in picks))
 def test_selection_fails_when_one_rare_group_missing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'in.csv';q=Path(d)/'out.csv'
   rows=[{'object_type':str(i),'category_provider_claim':'avian','download_status':'DOWNLOADED',
          'glb_valid':'True','glb_image_count':'1','glb_texture_count':'1',
          'glb_triangle_indices_count':'200','download_sha256':'a'*64,'downloaded_bytes':'1024'} for i in range(96)]
   with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   with self.assertRaisesRegex(ValueError,'MISSING_TEXTURED_RARE_MORPHOLOGY'):
    choose_sources(p,q)

if __name__=='__main__':unittest.main()