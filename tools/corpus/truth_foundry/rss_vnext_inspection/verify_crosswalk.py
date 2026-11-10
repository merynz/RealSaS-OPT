#!/usr/bin/env python3
"""Independent rerun of immutable source hashes and 16,397×264/47 receipts."""
from __future__ import annotations
import csv,gzip,hashlib,json,pathlib,sys,collections

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def run(root):
 audit=root/'DEEP_DERIVATIVE_SUBJECT_AUDIT_V1.jsonl'
 entries=[json.loads(l) for l in audit.read_text().splitlines()]
 assert len(entries)==138 and len({x['source_key_fragment'] for x in entries})==138
 kcount=collections.Counter()
 for x in entries:
  k=x['source_key_fragment'];v=x['court_version'];p=root.parent/f'realsas_{v.lower()}'
  src=p/'sources'/(k+'.npz');struct=p/'structures'/(k+'.json')
  assert src.is_file() and struct.is_file(),f'MISSING_{k}'
  assert sha(src)==x['derivative_npz_sha256'],f'NPZ_SHA_DRIFT_{k}'
  assert sha(struct)==x['derivative_structure_sha256'],f'STRUCT_SHA_DRIFT_{k}'
  assert x['original_asset_byte_parity']=='NOT_TESTED'
  assert x['supervision_admission']=='NOT_COURTED'
  kcount[v]+=1
 assert dict(kcount)=={'V24':20,'V25':36,'V26':82}
 matrix=root.parent/'realsas_v26'/'FACET_STATES_EFFECTIVE_V26_16397x264.csv.gz'
 universe=root.parent/'realsas_v26'/'SOURCE_UNIVERSE_V26_16397.csv.gz'
 with gzip.open(matrix,'rt') as f:
  rdr=csv.DictReader(f)
  assert len(rdr.fieldnames)==265
  ids=set();states=collections.Counter()
  for row in rdr:
   k=row['source_key'];assert k not in ids;ids.add(k)
   states.update(v for c,v in row.items() if c!='source_key')
 assert len(ids)==16397 and sum(states.values())==16397*264
 with gzip.open(universe,'rt') as f:
  ids2={r['source_key'] for r in csv.DictReader(f)}
 assert ids2==ids
 cat=json.loads((root/'TARGET_CATALOG_PROVISIONAL_V1.json').read_text())
 assert cat['provisional_target_count']==47 and cat['training_targets']==43 and cat['compiler_qualification_targets']==4
 assert all(t['source_facet_states_are_not_training_admission'] for t in cat['targets'])
 with gzip.open(root/'TARGET_EVIDENCE_SCREEN_16397x47_V1.csv.gz','rt') as f:
  rdr=csv.DictReader(f);assert len(rdr.fieldnames)==49
  ids3=[]
  for row in rdr:
   ids3.append(row['source_key']);assert all(v!='PASS' for k,v in row.items() if k not in ('source_key','provider'))
 assert len(ids3)==16397 and set(ids3)==ids
 with (root/'FACET_COVERAGE_264_V1.csv').open() as f:assert len(list(csv.DictReader(f)))==264
 with (root/'SUPERVISION_TARGET_EVIDENCE_COVERAGE_V1.csv').open() as f:assert len(list(csv.DictReader(f)))==47
 summary=json.loads((root/'TARGET_COVERAGE_SUMMARY_V1.json').read_text())
 for gate in ['character_vs_vehicle_prop_environment','source_rights_independent_pass']:
  assert summary['hard_gates'][gate]['UNKNOWN_UNREVIEWED']==16397
 result={'schema':'RealSaS.RSSvNext.CrosswalkIndependentReadback.v1','result':'PASS',
         'verified_local_derivative_npz_and_structure_sha256_pairs':len(entries),
         'verified_source_matrix_keys':len(ids),'verified_matrix_cells':sum(states.values()),
         'verified_facets':264,'verified_provisional_targets':47,
         'no_false_training_admissions':True,'source_original_and_license_passes_from_this_readback':0,
         'source_sha256':sha(matrix),'target_catalog_sha256':sha(root/'TARGET_CATALOG_PROVISIONAL_V1.json'),
         'deep_audit_sha256':sha(audit),'target_screen_sha256':sha(root/'TARGET_EVIDENCE_SCREEN_16397x47_V1.csv.gz')}
 (root/'INDEPENDENT_READBACK_VNEXT_V1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result))

if __name__=='__main__':run(pathlib.Path(sys.argv[1]))