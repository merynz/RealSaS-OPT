from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

PILOT64_N=64
M256_FIT=208
M256_TUNE=48


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def hkey(aid,tag): return hashlib.sha256(f'{tag}|{aid}'.encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--full-seed',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    s=json.load(open(a.full_seed)); open_rows=[r for r in s['records'] if r['split'] in ('FIT','TUNE')]
    pilot64=sorted(open_rows,key=lambda r:hashlib.sha256(r['asset_id'].encode()).hexdigest())[:PILOT64_N]
    ex={r['asset_id'] for r in pilot64}
    fit=[r for r in open_rows if r['split']=='FIT' and r['asset_id'] not in ex]
    tune=[r for r in open_rows if r['split']=='TUNE' and r['asset_id'] not in ex]
    fit=sorted(fit,key=lambda r:hkey(r['asset_id'],'M256_FIT_20260823'))[:M256_FIT]
    tune=sorted(tune,key=lambda r:hkey(r['asset_id'],'M256_TUNE_20260823'))[:M256_TUNE]
    rows=sorted(fit+tune,key=lambda r:r['asset_id'])
    if len(fit)!=M256_FIT or len(tune)!=M256_TUNE or len(rows)!=256: raise RuntimeError('M256 selection count failure')
    out={'schema':'RealSaS.IRISControlledV1.M256Seed.v1','date':'2026-08-23','record_count':256,
         'split_counts':{'FIT':M256_FIT,'TUNE':M256_TUNE},'records':rows,
         'source_full_seed':str(Path(a.full_seed).resolve()),'source_full_seed_sha256':sha(a.full_seed),
         'pilot64_excluded_count':len(ex),'pilot64_excluded_asset_ids':sorted(ex),
         'selection':{'FIT':'first 208 by SHA256(M256_FIT_20260823|asset_id) after excluding pilot64',
                      'TUNE':'first 48 by SHA256(M256_TUNE_20260823|asset_id) after excluding pilot64'},
         'sealed_splits_opened':False}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':a.out,'sha256':sha(a.out),'FIT':len(fit),'TUNE':len(tune),'pilot64_overlap':len(ex & {r['asset_id'] for r in rows})},indent=2))
if __name__=='__main__': main()
