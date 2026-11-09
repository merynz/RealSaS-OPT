"""Synthetic V16 contract tests. Not physical source training admission."""
import csv,gzip,hashlib,importlib.util,json,pathlib,tempfile,unittest
from unittest import mock

from tools.corpus.truth_foundry.evidence_v16 import mounted_recovery_v16 as v16

class MountedV16Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=pathlib.Path(self.tmp.name)
  self.queue=self.root/'queue.csv';self.inventory=self.root/'v11.csv.gz'
  with self.queue.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['source_key','candidate_id']);w.writeheader()
   for i in range(1527):w.writerow({'source_key':f'MASTER3993:asset_{i:024x}', 'candidate_id':f'quaternius_cc0:{i:024x}'})
  with gzip.open(self.inventory,'wt',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['source_key','pool','master_v11_source_recovery_status']);w.writeheader()
   for i in range(1527):
    state=(v16.ELIGIBLE if i<726 else (v16.HOLDOUT if i<861 else 'LEGACY_PATH_MATCH_UNREHASHED'))
    w.writerow({'source_key':f'MASTER3993:asset_{i:024x}','pool':'MASTER3993','master_v11_source_recovery_status':state})
  self.fake_lock={'queue':v16.sha(self.queue),'inventory':v16.sha(self.inventory)}

 def test_holdout_firewall_and_exact_726(self):
  with mock.patch.object(v16,'INPUT_LOCKS',self.fake_lock):
   groups=[]
   for shard in range(8):
    subset,count=v16.select(self.queue,self.inventory,shard,8)
    self.assertEqual(count[v16.ELIGIBLE],726)
    self.assertEqual(count[v16.HOLDOUT],135)
    groups.extend(subset)
  self.assertEqual(len(groups),726)
  self.assertEqual(len({x['source_key'] for x in groups}),726)
  self.assertTrue(all(int(x['candidate_id'].split(':')[1],16)<726 for x in groups))

 def test_sha_drift_fails_closed(self):
  with mock.patch.object(v16,'INPUT_LOCKS',self.fake_lock):
   self.queue.write_text(self.queue.read_text()+'DRIFT\n')
   with self.assertRaisesRegex(ValueError,'SHA_LOCK_DRIFT'):
    v16.select(self.queue,self.inventory,0,8)

 def test_path_escape_rejected(self):
  with self.queue.open() as f:rows=list(csv.DictReader(f))
  rows[0]['candidate_id']='../../evil/normalized.npz'
  with self.queue.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['source_key','candidate_id']);w.writeheader();w.writerows(rows)
  with mock.patch.object(v16,'INPUT_LOCKS',{'queue':v16.sha(self.queue),'inventory':v16.sha(self.inventory)}):
   with self.assertRaisesRegex(ValueError,'FORBIDDEN_SOURCE_ID'):
    v16.select(self.queue,self.inventory,0,8)

 def test_preflight_does_not_open_source_root(self):
  missing_root=self.root/'do-not-open-any-data'
  with mock.patch.object(v16,'INPUT_LOCKS',self.fake_lock):
   code=v16.main(['--queue',str(self.queue),'--inventory',str(self.inventory),
    '--variants-root',str(missing_root),'--output',str(self.root/'out'),
    '--shard-index','0','--shard-count','8','--preflight-only'])
  self.assertEqual(code,0)
  self.assertFalse(missing_root.exists())
  plan=json.loads((self.root/'out'/'PLAN_V16.json').read_text())
  self.assertEqual(plan['holdouts_excluded'],135)
  self.assertEqual(plan['training_admissions'],0)

if __name__=='__main__':unittest.main()