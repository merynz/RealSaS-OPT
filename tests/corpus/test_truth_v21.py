"""No-private-bytes source identity and authority regression tests for V21."""
import csv,gzip,io,os,pathlib,tempfile,unittest,zipfile,importlib.util
P=pathlib.Path(__file__).resolve().parents[2]/'tools'/'corpus'/'truth_foundry'/'evidence_v21'/'court_new_sources_v21.py'
spec=importlib.util.spec_from_file_location('v21court',P)
v21=importlib.util.module_from_spec(spec);spec.loader.exec_module(v21)

class SourceIdentityAndAdmissionGates(unittest.TestCase):
 def write_zip(self,path,rows):
  hdr=['relative_folder_under_variants','source_key','candidate_id','historical_recovery_sha256','creator_original_byte_parity','rights_training_admission']
  dst=io.StringIO(); w=csv.DictWriter(dst,fieldnames=hdr);w.writeheader();w.writerows(rows)
  with zipfile.ZipFile(path,'w') as z:z.writestr('MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz',gzip.compress(dst.getvalue().encode()))
 def sample(self,n=726):
  return [{'relative_folder_under_variants':'objaverse_animated_originals_'+f'{i:064x}',
   'source_key':f'source_{i}','candidate_id':f'candidate_{i}','historical_recovery_sha256':'a'*64,
   'creator_original_byte_parity':'False','rights_training_admission':'False'}for i in range(n)]
 def test_exact_allowlist_726_and_no_rights(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'plan.zip'; self.write_zip(p,self.sample())
   old=v21.V17;v21.V17=p
   try:self.assertEqual(len(v21.load_plan()),726)
   finally:v21.V17=old
 def test_725_plan_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'plan.zip';self.write_zip(p,self.sample(725))
   old=v21.V17;v21.V17=p
   try:
    with self.assertRaises(AssertionError):v21.load_plan()
   finally:v21.V17=old
 def test_promoted_original_rights_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'plan.zip';rows=self.sample();rows[0]['rights_training_admission']='True';self.write_zip(p,rows)
   old=v21.V17;v21.V17=p
   try:
    with self.assertRaises(AssertionError):v21.load_plan()
   finally:v21.V17=old
 def test_duplicate_metadata_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'m.txt';name='objaverse_animated_originals_'+'1'*64
   line='META'+(' '*8)+name+(' '*8)+'folderid'+(' '*8)+'1'+(' '*8)+'1'+(' '*8)+'npzid'+(' '*8)+'123'+(' '*8)+'jsonid'+(' '*8)+'40'+(' '*8)+'5'+(' '*8)+'\n'
   p.write_text(line+line)
   old=v21.V19;v21.V19=p
   try:
    with self.assertRaises(RuntimeError):v21.load_meta()
   finally:v21.V19=old

if __name__=='__main__':unittest.main()