#!/usr/bin/env python3
"""R2.9 R4: verified-source bulk upstream license *claim* crosswalk.

The output is a data-evidence supplement, NEVER commercial-ML training approval.
Original GitHub/Sketchfab object metadata must agree on source identity. A repo
licence is NOT necessarily a licence for an embedded third-party FBX/GLB.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

SOURCE_SHA = 'd43858a3e710fb44391711dbf8a1d082891a7c03f2ab2452cd88f24c287957c3'
GH_SHA = 'f14bf4daceac97072eee5721489ad160089b9fc1514e6a9e096574c3e4276da6'
SF_SHA = '6462ef2ecf39069e210f6df1416e8a28ca931039e3c7fe6aa3dca593115c64e3'
HEX64 = re.compile(r'^[0-9a-f]{64}$')
HEX32 = re.compile(r'^[0-9a-f]{32}$')
UNIML_GH = re.compile(r'^[0-9a-f]{24}_(?:fbx|glb|gltf)$')
SCHEMA='RealSaS.R29.R4.UpstreamMetadataClaims.v1'


def sha256(path:Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()


def github_key(url):
 """Canonical repo/sha/path, never discard the path or accept a different commit."""
 try:
  u=urlsplit(str(url).strip()); host=u.netloc.lower(); p=unquote(u.path).strip('/').split('/')
  if host in ('github.com','www.github.com') and len(p)>=5 and p[2]=='blob' and re.fullmatch(r'[0-9a-f]{40}',p[3],re.I):
   return '/'.join([p[0].lower(),p[1].lower(),p[3].lower(),*p[4:]])
  if host=='raw.githubusercontent.com' and len(p)>=4 and re.fullmatch(r'[0-9a-f]{40}',p[2],re.I):
   return '/'.join([p[0].lower(),p[1].lower(),p[2].lower(),*p[3:]])
  if host=='api.github.com' and len(p)>=7 and p[0]=='repos' and p[3]=='contents':
   return ''
 except Exception: pass
 return ''


def sketchfab_uid(url):
 s=str(url).strip()
 if HEX32.fullmatch(s.lower()): return s.lower()
 try:
  u=urlsplit(s)
  if u.netloc.lower() not in ('sketchfab.com','www.sketchfab.com'):return ''
  for bit in reversed(u.path.strip('/').split('/')):
   if HEX32.fullmatch(bit.lower()):return bit.lower()
   m=re.search(r'([0-9a-f]{32})$',bit,re.I)
   if m:return m.group(1).lower()
 except Exception: pass
 return ''


def license_class(raw):
 s=re.sub(r'[^a-z0-9]+',' ',str(raw or '').lower()).strip()
 if not s or s in ('nan','unknown','none','null','no license','unlicensed','other'):return 'UNSPECIFIED'
 if 'noai' in s.replace(' ','') or 'no ai' in s:return 'EXPLICIT_NOAI'
 if 'noncommercial' in s.replace(' ','') or re.search(r'\bnc\b',s):return 'EXCLUDE_NC_POLICY'
 if 'noderivatives' in s.replace(' ','') or re.search(r'\bnd\b',s):return 'REVIEW_NODERIVATIVES'
 if 'sharealike' in s.replace(' ','') or re.search(r'\bsa\b',s):return 'REVIEW_SHAREALIKE'
 if re.search(r'\bcc0\b',s) or 'creative commons zero' in s or 'public domain' in s:return 'CANDIDATE_CC0'
 if s in ('cc by','cc by 4 0','cc by 3 0','cc attribution','creative commons attribution','cc attribution 4 0','cc attribution 3 0') or s.startswith('creative commons attribution '):return 'CANDIDATE_CC_BY'
 if 'mit'==s or s.startswith('mit license'):return 'REPO_LICENSE_CLAIM_MIT'
 if 'apache' in s or 'bsd' in s or 'gpl' in s or 'lgpl' in s:return 'REVIEW_OTHER_SOURCE_LICENSE'
 return 'REVIEW_UNRECOGNIZED_LICENSE'


def csv_rows(path):
 with path.open(newline='',encoding='utf-8-sig') as f:yield from csv.DictReader(f)


def parquet_rows(path, columns):
 import pyarrow.parquet as pq
 p=pq.ParquetFile(str(path))
 schema=set(p.schema_arrow.names)
 if not {'sha256','license'}.issubset(schema):raise ValueError('UNEXPECTED_SOURCE_SCHEMA_'+str(sorted(schema)))
 id_col=next((x for x in ('fileIdentifier','githubUrl') if x in schema),'')
 if not id_col:raise ValueError('MISSING_SOURCE_IDENTIFIER')
 wanted=[c for c in [id_col,'sha256','license','source','fileType'] if c in schema]
 for batch in p.iter_batches(columns=wanted,batch_size=20000):
  d=batch.to_pydict()
  for i in range(batch.num_rows):
   yield {'fileIdentifier':str(d[id_col][i] or ''),'sha256':str(d['sha256'][i] or '').lower(),
          'license':str(d['license'][i] or ''),'source':str(d.get('source',[''])[i] or '') if 'source' in d else '',
          'fileType':str(d.get('fileType',[''])[i] or '') if 'fileType' in d else ''}


def read_metadata(path,columns):
 if path.suffix=='.parquet':yield from parquet_rows(path,columns)
 elif path.suffix=='.csv':yield from csv_rows(path)
 else:raise ValueError('NEED_PARQUET_OR_CSV')


def source_rows(path):
 rows=list(csv_rows(path))
 if len({r['source_key'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_SOURCE_KEY')
 if len(rows)!=20847 and not path.name.endswith('_fixture.csv'):raise ValueError('NOT_FULL_20847_CORPUS')
 return rows


def build_index(records,require_source,sha_targets,uid_targets):
 by_sha=defaultdict(list);by_uid=defaultdict(list);n=0;found=0
 for rec in records:
  n+=1; source=rec.get('source','').lower()
  if source and source!=require_source:continue
  sh=rec.get('sha256','').lower();uid=sketchfab_uid(rec.get('fileIdentifier','')) if require_source=='sketchfab' else ''
  if sh not in sha_targets and uid not in uid_targets:continue
  found+=1
  if sh in sha_targets:by_sha[sh].append(rec)
  if uid in uid_targets:by_uid[uid].append(rec)
 return by_sha,by_uid,n,found


def evidence_for_row(r,gh_by_sha,sf_by_sha,sf_by_uid):
 typ=r['source_type']; sh=r['original_sha'].lower(); uid=''; matches=[];strength='NONE';reason='';origin=''
 if typ=='GITHUB_PINNED' and HEX64.fullmatch(sh):
  key=github_key(r['source_url']);cands=gh_by_sha.get(sh,[])
  matches=[m for m in cands if key and github_key(m['fileIdentifier'])==key]
  origin='github';strength='EXACT_ORIGINAL_SHA_AND_COMMIT_PATH' if len(matches)==1 else 'NONE'
  reason='MATCHED_SHA_AND_COMMIT_PATH' if len(matches)==1 else ('SOURCE_SHA_MATCH_DIFFERENT_PATH' if cands else 'NO_GITHUB_METADATA_SHA')
 elif typ=='SKETCHFAB' or (r['provider']=='UNIML3D' and HEX32.fullmatch(r['asset_id'])):
  uid=sketchfab_uid(r['source_target']) or (r['asset_id'].lower() if r['provider']=='UNIML3D' else '')
  if typ=='SKETCHFAB' and HEX64.fullmatch(sh):
   cands=sf_by_sha.get(sh,[]);matches=[m for m in cands if sketchfab_uid(m['fileIdentifier'])==uid];
   strength='EXACT_ORIGINAL_SHA_AND_SOURCE_UID' if len(matches)==1 else 'NONE'
   reason='MATCHED_SHA_AND_UID' if len(matches)==1 else 'NO_SHA_UID_MATCH'
  else:
   matches=sf_by_uid.get(uid,[]) if uid else []
   strength='SOURCE_UID_ONLY_DERIVED_FILE_NOT_IDENTICAL' if len(matches)==1 else 'NONE'
   reason='UNIML_DERIVED_SOURCE_UID_MATCH' if len(matches)==1 else 'NO_SOURCE_UID_MATCH'
  origin='sketchfab'
 elif r['provider']=='UNIML3D' and UNIML_GH.fullmatch(r['asset_id']):
  origin='github';reason='UNIML_24HEX_OBJECT_ID_REQUIRES_ORIGINAL_IDENTIFIER_CROSSWALK'
 else:reason='NOT_APPLICABLE_TO_THIS_METADATA_JOIN'
 if len(matches)>1:
  strengths={(m['sha256'],m['fileIdentifier'],m['license']) for m in matches}
  if len(strengths)==1:matches=[matches[0]]
  else:strength='NONE';reason='AMBIGUOUS_CONFLICTING_SOURCE_METADATA'
 if len(matches)==1 and strength!='NONE':
  m=matches[0]; raw=m['license']; classified=license_class(raw)
  if not raw or classified=='UNSPECIFIED':reason='METADATA_MATCH_BUT_LICENSE_UNSPECIFIED'
 else:m={};raw='';classified='UNSPECIFIED'
 return dict(source_key=r['source_key'],provider=r['provider'],upstream_provider=origin,
             match_strength=strength,match_reason=reason,license_raw=raw,license_class=classified,
             original_asset_sha=r['original_sha'],original_asset_sha_verified=r['original_sha_verified'],
             upstream_sha=m.get('sha256',''),upstream_fileIdentifier=m.get('fileIdentifier',''),
             upstream_source=m.get('source',''),evidence_scope='ORIGINAL_SOURCE_METADATA_ONLY',
             commercial_training_authorized='False',asset_specific_license_verified='False',
             noai_resolved='False',source_rights_evidence_url='https://huggingface.co/datasets/allenai/objaverse-xl',
             notes='CLAIM_ONLY_DO_NOT_INFER_ASSET_LICENSE_FROM_REPO_OR_DERIVED_LICENCE')


def run(src,sketchfab,github,out,sha_pins=True):
 src=Path(src);sf=Path(sketchfab);gh=Path(github);out=Path(out)
 if sha_pins:
  for p,sha in [(src,SOURCE_SHA),(sf,SF_SHA),(gh,GH_SHA)]:
   actual=sha256(p)
   if actual!=sha:raise ValueError('UPSTREAM_PIN_MISMATCH:'+str(p)+':'+actual)
 rows=source_rows(src)
 ghshas={r['original_sha'].lower() for r in rows if r['source_type']=='GITHUB_PINNED' and HEX64.fullmatch(r['original_sha'].lower())}
 sfshas={r['original_sha'].lower() for r in rows if r['source_type']=='SKETCHFAB' and HEX64.fullmatch(r['original_sha'].lower())}
 uids={r['source_target'].lower() for r in rows if r['source_type']=='SKETCHFAB' and HEX32.fullmatch(r['source_target'].lower())}
 uids|={r['asset_id'].lower() for r in rows if r['provider']=='UNIML3D' and HEX32.fullmatch(r['asset_id'].lower())}
 g_sha,_,g_total,g_found=build_index(read_metadata(gh,[]),'github',ghshas,set())
 s_sha,s_uid,s_total,s_found=build_index(read_metadata(sf,[]),'sketchfab',sfshas,uids)
 out.mkdir(parents=True,exist_ok=True)
 target=out/'R29_R4_UPSTREAM_LICENSE_METADATA_CROSSWALK.csv'
 counts=Counter(); per_provider=defaultdict(Counter);usable=0
 with target.open('w',newline='',encoding='utf-8') as f:
  fields=None; w=None
  for r in rows:
   d=evidence_for_row(r,g_sha,s_sha,s_uid)
   if fields is None:
    fields=list(d);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   w.writerow(d);counts[d['match_strength']]+=1;per_provider[d['provider']][d['license_class']]+=1
   if d['match_strength']!='NONE' and d['license_class']!='UNSPECIFIED':usable+=1
 summary={'schema':SCHEMA,'sources':len(rows),'metadata_provider_rows_scanned':{'github':g_total,'sketchfab':s_total},
  'metadata_targets_found':{'github':g_found,'sketchfab':s_found},'match_strengths':dict(counts),
  'source_rows_with_license_claim':usable,'licences_by_provider':{k:dict(v) for k,v in per_provider.items()},
  'pinned_source_sha256':{x.name:sha256(x) for x in [src,sf,gh]},
  'outputs_sha256':{target.name:sha256(target)},
  'training_admitted':0,'license_scope_caveat':'Upstream metadata claim is not asset-level training permission. License obligations, NoAI, original-source identity and time remain separate.'}
 (out/'R29_R4_CLOSEOUT.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 return summary


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ledger',required=True);p.add_argument('--sketchfab',required=True);p.add_argument('--github',required=True);p.add_argument('--out',required=True)
 args=p.parse_args();print(json.dumps(run(args.ledger,args.sketchfab,args.github,args.out),indent=2))
