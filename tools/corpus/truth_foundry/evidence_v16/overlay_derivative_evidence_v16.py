#!/usr/bin/env python3
"""Build an independently verifiable, source-locked V16 facet overlay.

V11 inputs are bytewise immutable. Exactly 100 independent historical normalized
DERIVATIVE observations may replace UNKNOWN_UNREVIEWED; no admission promotion.
Never treats recovered derivative as creator-original mesh or training truth.
"""
import argparse, csv, gzip, hashlib, io, json, os, pathlib, tempfile, collections

BASE=pathlib.Path('.')
ROOT=pathlib.Path('.')
N=16397
M=264
LOCKS={'inventory':'6436a6696a55c7643818914b6ebab5a5728d4eae23409d67e6e6d210c2b17ae4',
       'states':'88d8bc9cad2c0003002e58df80c19db8d72fc2d2b4ece0b67031fa8f2d7318c0',
       'values':'bbd9c9ae67b4acc4700dd0bb1feb0df10ad0b4c1745911c1a1ea6089bf39f075',
       'queue':'18234396bf86fa0e9c156c6e3eddb73aa3963870459491a479ffb56974349ab7',
       'delta':'c910975ff98e43faf4f8d8c814d029edd483c9477ad327a63fef7747056106ad',
       'pilot':'5bd3d7f14f821abffc513bc628d57490b74aba06075a62e9abbfff98f7f561b5'}
STATE='OBSERVED_HISTORICAL_NORMALIZED_DERIVATIVE_ONLY'


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for data in iter(lambda:f.read(2<<20),b''):h.update(data)
 return h.hexdigest()


def gzip_writer(path):
 fd,temp=tempfile.mkstemp(prefix='.pending-v16-',dir=str(path.parent))
 raw=os.fdopen(fd,'wb')
 z=gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0)
 txt=io.TextIOWrapper(z,encoding='utf-8',newline='')
 return temp,raw,z,txt


def main(argv=None):
 global BASE,ROOT
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--base-dir',type=pathlib.Path,required=True)
 parser.add_argument('--output-dir',type=pathlib.Path,required=True)
 a=parser.parse_args(argv)
 BASE=a.base_dir.resolve();ROOT=a.output_dir.resolve()
 assert BASE.is_dir() and ROOT.is_dir(), 'V16_INPUT_DIR_MISSING'
 paths={'inventory':BASE/'SOURCE_INVENTORY_DEEP_16397_V11.csv.gz',
        'states':BASE/'FACET_STATES_EFFECTIVE_V11_16397x264.csv.gz',
        'values':BASE/'FACET_VALUES_EFFECTIVE_V11_16397x264.csv.gz',
        'queue':BASE/'MASTER1527_MISSING_FULL_IMPORT_RECEIPT_QUEUE.csv',
        'delta':ROOT/'PHYSICAL_FACET_DELTA_V16.csv',
        'pilot':ROOT/'PHYSICAL_DERIVATIVE_PILOT_V16.jsonl'}
 for kind,path in paths.items():
  if sha(path)!=LOCKS[kind]:raise ValueError('SOURCE_OR_EVIDENCE_SHA_DRIFT:'+kind)
 physical={}
 for text in paths['pilot'].read_text().splitlines():
  rec=json.loads(text);key=rec['source_key'];assert key not in physical
  assert rec['independent_structural_array_verifier']=='PASS_NUMERIC_ARRAY_INVARIANTS_ONLY'
  assert not rec['training_admission'] and not rec['original_byte_parity_admission']
  physical[key]=rec
 assert len(physical)==10
 eligible={}
 with gzip.open(paths['inventory'],'rt',newline='',encoding='utf-8') as f:
  for rec in csv.DictReader(f):
   key=rec['source_key']
   if key in physical:
    assert rec['pool']=='MASTER3993' and rec['master_v11_source_recovery_status']=='SHA_VERIFIED_RECOVERY_RECEIPT'
    eligible[key]=rec['candidate_id']
 assert len(eligible)==10
 delta={};facet_counts=collections.Counter()
 with paths['delta'].open(newline='',encoding='utf-8') as f:
  records=list(csv.DictReader(f))
 for r in records:
  key,facet=r['source_key'],r['facet'];p=physical[key]
  assert r['source_npz_sha256']==p['normalized_npz_sha256']
  assert r['source_structure_sha256']==p['source_structure_sha256']
  assert r['observation_state']==STATE
  assert r['authority']=='DERIVED_ARRAY_COURT_ONLY_NO_CREATOR_PARITY'
  assert r['admission']=='NO_TRAINING_ADMISSION'
  value=json.loads(r['value_json'])
  assert isinstance(value,dict) and (key,facet) not in delta
  delta[(key,facet)]=json.dumps(value,sort_keys=True,separators=(',',':'))
  facet_counts[facet]+=1
 assert len(delta)==100 and len(facet_counts)==10 and all(v==10 for v in facet_counts.values())
 outputs={'states':ROOT/'FACET_STATES_EFFECTIVE_V16_16397x264.csv.gz',
          'values':ROOT/'FACET_VALUES_EFFECTIVE_V16_16397x264.csv.gz'}
 temporary={}
 try:
  state_tmp,sraw,sz,stxt=gzip_writer(outputs['states']);temporary['states']=state_tmp
  value_tmp,vraw,vz,vtxt=gzip_writer(outputs['values']);temporary['values']=value_tmp
  with gzip.open(paths['states'],'rt',newline='',encoding='utf-8') as sf, gzip.open(paths['values'],'rt',newline='',encoding='utf-8') as vf:
   sr,vr=csv.DictReader(sf),csv.DictReader(vf)
   assert sr.fieldnames==vr.fieldnames and len(sr.fieldnames)==M+1 and sr.fieldnames[0]=='source_key'
   assert all(facet in sr.fieldnames for facet in facet_counts)
   sw,vw=csv.DictWriter(stxt,fieldnames=sr.fieldnames),csv.DictWriter(vtxt,fieldnames=sr.fieldnames)
   sw.writeheader();vw.writeheader()
   seen=set();modified=0;rows=0
   for st,va in zip(sr,vr):
    key=st['source_key'];assert key==va['source_key'] and key not in seen
    seen.add(key);rows+=1
    if key in physical:
     for facet in facet_counts:
      v=delta.get((key,facet))
      if v is None:continue
      if st[facet]!='UNKNOWN_UNREVIEWED' or va[facet]:
       raise ValueError('NONUNKNOWN_OR_EXISTING_FACET_COLLISION:'+key+':'+facet)
      st[facet]=STATE;va[facet]=v;modified+=1
    sw.writerow(st);vw.writerow(va)
   assert rows==N and modified==100 and len(seen)==N
   assert next(sr,None) is None and next(vr,None) is None
  for obj in [stxt,vtxt]:obj.close()
  # gzip.GzipFile does not close its supplied raw fileobj: flush the OS-backed
  # file BEFORE rename/SHA to prevent a false content digest in the manifest.
  for raw in (sraw,vraw):
   raw.flush();os.fsync(raw.fileno());raw.close()
  for kind,tmp in temporary.items():os.replace(tmp,outputs[kind])
 except BaseException:
  for obj in [locals().get('stxt'),locals().get('vtxt')]:
   try:obj and obj.close()
   except Exception:pass
  for tmp in temporary.values():
   if os.path.exists(tmp):os.unlink(tmp)
  raise
 summary={'schema':'RealSaS.TruthFoundryV16EffectiveFacetOverlay.v1',
  'base_sha256':{k:LOCKS[k] for k in ('inventory','states','values','queue')},
  'delta_sha256':LOCKS['delta'],'physical_pilot_sha256':LOCKS['pilot'],
  'source_rows':N,'facet_columns':M,'facet_cells':N*M,
  'qualified_derivative_only_sources':len(physical), 'new_observed_derivative_cells':modified,
  'preexisting_cells_overwritten':0,'protected_holdouts_touched':0,
  'creator_original_parity_admissions':0,'rights_admissions':0,'training_admissions':0,
  'tessa_animation_ready_admissions':0,'semantic_admissions':0,'graphics_art_admissions':0,
  'output_sha256':{kind:sha(path) for kind,path in outputs.items()},
  'observed_facet_names':dict(facet_counts)}
 (ROOT/'SUMMARY_EFFECTIVE_V16.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'rows':rows,'facets':M,'new_derivative_evidence_cells':modified,'shas':summary['output_sha256']},indent=2))

if __name__=='__main__':main()