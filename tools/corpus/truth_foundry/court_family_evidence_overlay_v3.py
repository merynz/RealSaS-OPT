"""Independent Core370/Master3993 provenance overlay recounter."""
import csv,gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def check(p=Path('/mnt/data/truth_foundry_forensic_family_v3')):
 s=json.load(open(p/'SUMMARY_FAMILY_V3.json'))
 assert s['family_evidence_rows']==57089 and s['original_historical_head_routes']==25438
 assert all(sha(p/name)==value for name,value in s['outputs_sha256'].items())
 counts=Counter();seen=set();states=Counter()
 with gzip.open(p/'SOURCE_FAMILY_EVIDENCE_57089_V3.csv.gz','rt',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   key=(row['source_key'],row['family']);assert key not in seen;seen.add(key)
   counts[row['corpus']]+=1
   assert row['training_authorized']=='False'
   assert row['current_independent_court_pass']=='False'
   if row['corpus']=='CORE370':states[row['machine_evidence_state']]+=1
 assert counts=={'MASTER3993':51909,'CORE370':5180}
 with gzip.open(p/'SOURCE_HEAD_HISTORICAL_ROUTES_25438_V3.csv.gz','rt',encoding='utf-8') as f:
  route=Counter()
  for row in csv.DictReader(f):
   route[row['corpus']]+=1
   assert row['training_authorized']=='False'
   assert row['current_head_target_court']=='UNKNOWN'
 assert route=={'MASTER3993':23958,'CORE370':1480}
 return {'families':dict(counts),'historic_head_routes':dict(route),'core_historical_machine_states':dict(states)}
if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/truth_foundry_forensic_family_v3')),indent=2))