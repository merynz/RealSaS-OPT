#!/usr/bin/env python3
"""Independent exact-cell V17 -> V18 authority downgrade readback."""
import argparse,collections,csv,gzip,hashlib,json,pathlib
MAPPING={
'attribution_obligations':'PROVIDER_LICENSE_CLASS_CLAIM__ATTRIBUTION_NOT_COURTED',
'original_asset_provider_locator':'RECORDED_PROVIDER_LOCATOR_CLAIM__NOT_SOURCE_VERIFIED',
'original_asset_recovery_status':'HISTORICAL_PROCESSED_PART_NUMERIC_RECEIPT__ORIGINAL_UNHASHED',
}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def run(v17,v18):
 report=json.loads((v18/'AUTHORITY_CORRECTION_SUMMARY_V18.json').read_text())
 names=[v17/'FACET_STATES_EFFECTIVE_V17_16397x264.csv.gz',v17/'FACET_VALUES_EFFECTIVE_V17_16397x264.csv.gz',v18/'FACET_STATES_EFFECTIVE_V18_16397x264.csv.gz',v18/'FACET_VALUES_EFFECTIVE_V18_16397x264.csv.gz']
 for name,target in zip(names,[report['v17_inputs_sha256']['states'],report['v17_inputs_sha256']['values'],report['v18_outputs_sha256']['states'],report['v18_outputs_sha256']['values']]):
  if sha(name)!=target:raise ValueError('SOURCE_OR_RESULT_SHA_MISMATCH:'+name.name)
 delta={}
 with gzip.open(v18/'AUTHORITY_CORRECTIONS_V18.csv.gz','rt',newline='') as f:
  for d in csv.DictReader(f):
   pair=(d['source_key'],d['facet'])
   if pair in delta or d['facet'] not in MAPPING or d['new_state']!=MAPPING[d['facet']]:raise ValueError('DUPLICATE_OR_INVALID_CORRECTION')
   delta[pair]=d
 if len(delta)!=21084:raise ValueError('CORRECTION_COUNT')
 visited=set();rows=0;unchanged=0;changes=collections.Counter()
 with gzip.open(names[0],'rt',newline='') as fa,gzip.open(names[1],'rt',newline='') as fb,\
      gzip.open(names[2],'rt',newline='') as fc,gzip.open(names[3],'rt',newline='') as fd:
  a,b,c,d=(csv.DictReader(x) for x in (fa,fb,fc,fd))
  if not (a.fieldnames==b.fieldnames==c.fieldnames==d.fieldnames) or len(a.fieldnames)!=265:raise ValueError('HEADERS')
  for i,j,k,l in zip(a,b,c,d):
   key=i['source_key'];rows+=1
   if key!=j['source_key'] or key!=k['source_key'] or key!=l['source_key']:raise ValueError('ROW_ID')
   for facet in a.fieldnames[1:]:
    if j[facet]!=l[facet]:raise ValueError('VALUE_CHANGED:'+key+':'+facet)
    entry=delta.get((key,facet))
    if entry:
     if i[facet]!=entry['old_state'] or k[facet]!=entry['new_state'] or i[facet]!='OBSERVED_PROCESSED_PART_GEOMETRY_ONLY':raise ValueError('UNAUTHORIZED_STATE_CHANGE:'+key+':'+facet)
     if (key,facet) in visited:raise ValueError('DUP')
     visited.add((key,facet));changes[facet]+=1
    else:
     if i[facet]!=k[facet]:raise ValueError('UNAUTHORIZED_STATE_CHANGE:'+key+':'+facet)
     unchanged+=1
  if any(next(x,None) is not None for x in (a,b,c,d)):raise ValueError('TRAILING_ROWS')
 if rows!=16397 or len(visited)!=21084 or unchanged!=16397*264-21084:raise ValueError('INCOMPLETE')
 result={'status':'PASS_EXACT_AUTHORITY_CORRECTION_ONLY','rows':rows,'facet_columns':264,'corrections':len(visited),
  'unmodified_states':unchanged,'unchanged_values':rows*264,'counts_by_facet':dict(changes),
  'new_numeric_source_truth':0,'original_source_rehashes':0,'training_admissions':0,'source_sha256':report['v17_inputs_sha256'],
  'v18_state_sha256':report['v18_outputs_sha256']['states']}
 (v18/'INDEPENDENT_READBACK_V18.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--v17',required=True,type=pathlib.Path);p.add_argument('--v18',required=True,type=pathlib.Path)
 args=p.parse_args();print(json.dumps(run(args.v17,args.v18),indent=2))