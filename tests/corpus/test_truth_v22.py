"""No user source reads. Source eligibility and nonpromotion regression for V22."""
import csv, gzip, io, pathlib, tempfile, unittest, zipfile, importlib.util
MODULE=pathlib.Path(__file__).resolve().with_name('process_v22.py')
if not MODULE.exists():
 MODULE=pathlib.Path(__file__).resolve().parents[2]/'tools'/'corpus'/'truth_foundry'/'evidence_v22'/'process_v22.py'
spec=importlib.util.spec_from_file_location('v22court',MODULE)
v22=importlib.util.module_from_spec(spec);spec.loader.exec_module(v22)

class TestSourceGates(unittest.TestCase):
 def fixture(self,path,count=726,rights=False):
  rows=[]
  for i in range(count):
   rows.append({'relative_folder_under_variants':'objaverse_animated_originals_'+f'{i:064x}','creator_original_byte_parity':'False','rights_training_admission':'True' if rights and i==0 else 'False'})
  f=io.StringIO();w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
  with zipfile.ZipFile(path,'w') as z:z.writestr('MASTER726_SAFE_RECOVERY_PLAN_V17.csv.gz',gzip.compress(f.getvalue().encode()))
 def test_allowlist_good(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'plan.zip';self.fixture(p)
   self.assertEqual(len(v22.load_allowlist(p)),726)
 def test_allowlist_missing_record(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'plan.zip';self.fixture(p,725)
   with self.assertRaises(ValueError):v22.load_allowlist(p)
 def test_no_rights_promotion(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'plan.zip';self.fixture(p,rights=True)
   with self.assertRaises(ValueError):v22.load_allowlist(p)
 def test_manifest_no_unapproved_and_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'manifest.csv'
   key='objaverse_animated_originals_'+'0'*64
   with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['source_key','npz_id','npz_size','structure_id','structure_size']);w.writeheader()
    w.writerow({'source_key':key,'npz_id':'f','npz_size':1,'structure_id':'g','structure_size':2})
    w.writerow({'source_key':key,'npz_id':'f','npz_size':1,'structure_id':'g','structure_size':2})
   with self.assertRaises(ValueError):v22.read_manifest(p,{key:{}},set())
   with self.assertRaises(ValueError):v22.read_manifest(p,{},set())
   with self.assertRaises(ValueError):v22.read_manifest(p,{key:{}},{key})
 def test_refuse_empty_manifest(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'manifest.csv';p.write_text('source_key,npz_id,npz_size,structure_id,structure_size\n')
   with self.assertRaises(ValueError):v22.read_manifest(p,{},set())
if __name__=='__main__':unittest.main()