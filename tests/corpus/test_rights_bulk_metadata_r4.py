import csv,tempfile,unittest
from pathlib import Path
from rights_bulk_metadata_r4 import github_key,sketchfab_uid,license_class,run

class T(unittest.TestCase):
 def test_keys(self):
  a='https://github.com/ABC/DEf/blob/0123456789abcdef0123456789abcdef01234567/Assets/Animated%20Character.fbx'
  b='https://raw.githubusercontent.com/abc/def/0123456789abcdef0123456789abcdef01234567/Assets/Animated%20Character.fbx'
  self.assertEqual(github_key(a),github_key(b));self.assertEqual(github_key('https://github.com/A/B/blob/main/f.fbx'),'')
  self.assertEqual(sketchfab_uid('https://sketchfab.com/3d-models/hedgehog-0123456789abcdef0123456789abcdef'),'0123456789abcdef0123456789abcdef')
 def test_license(self):
  self.assertEqual(license_class('CC Attribution-NonCommercial'),'EXCLUDE_NC_POLICY')
  self.assertEqual(license_class('CC Attribution'),'CANDIDATE_CC_BY')
  self.assertEqual(license_class('CC Attribution-ShareAlike'),'REVIEW_SHAREALIKE')
  self.assertEqual(license_class('CC0'),'CANDIDATE_CC0')
  self.assertEqual(license_class('MIT'),'REPO_LICENSE_CLAIM_MIT')
  self.assertEqual(license_class('CC BY-ND'),'REVIEW_NODERIVATIVES')
 def test_fixture(self):
  def wr(p,rows):
   with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'source_fixture.csv';sf=root/'sf.csv';gh=root/'gh.csv';out=root/'out';uid='abcd'*8;sha='a'*64;pin='1'*40
   r={'source_key':'GH:1','source_type':'GITHUB_PINNED','original_sha':sha,'original_sha_verified':'True','source_url':f'https://github.com/aa/bb/blob/{pin}/mesh.fbx','source_target':'aa/bb@'+pin,'asset_id':'1','provider':'ARTICULATION_XL2'}
   r2={**r,'source_key':'UNIML:2','source_type':'HF_DERIVED_NOT_ORIGINAL','source_url':'https://huggingface.co/xx','original_sha':'b'*64,'asset_id':uid,'provider':'UNIML3D'}
   r3={**r,'source_key':'UNIML:3','source_type':'HF_DERIVED_NOT_ORIGINAL','source_url':'https://huggingface.co/xx','asset_id':'6'*24+'_fbx','provider':'UNIML3D'}
   r4={**r,'source_key':'GH:badpath','original_sha':sha,'source_url':f'https://github.com/aa/bb/blob/{pin}/wrong.fbx'}
   wr(src,[r,r2,r3,r4]);wr(gh,[{'fileIdentifier':f'https://github.com/aa/bb/blob/{pin}/mesh.fbx','sha256':sha,'license':'MIT','source':'github'}]);wr(sf,[{'fileIdentifier':f'https://sketchfab.com/3d-models/{uid}','sha256':'c'*64,'license':'CC Attribution','source':'sketchfab'}])
   s=run(src,sf,gh,out,sha_pins=False)
   self.assertEqual(s['sources'],4);self.assertEqual(s['source_rows_with_license_claim'],2)
   with (out/'R29_R4_UPSTREAM_LICENSE_METADATA_CROSSWALK.csv').open() as f:rows={r['source_key']:r for r in csv.DictReader(f)}
   self.assertEqual(rows['GH:1']['match_strength'],'EXACT_ORIGINAL_SHA_AND_COMMIT_PATH')
   self.assertEqual(rows['GH:badpath']['match_strength'],'NONE')
   self.assertEqual(rows['UNIML:2']['match_strength'],'SOURCE_UID_ONLY_DERIVED_FILE_NOT_IDENTICAL')
   self.assertIn('REQUIRES_ORIGINAL_IDENTIFIER_CROSSWALK',rows['UNIML:3']['match_reason'])
   self.assertEqual({r['commercial_training_authorized'] for r in rows.values()},{'False'})

if __name__=='__main__':unittest.main()
