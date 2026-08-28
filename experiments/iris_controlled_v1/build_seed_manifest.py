from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--split-freeze',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    root=Path(a.root); sf=json.load(open(a.split_freeze)); selp=root/'metadata'/'CANONICAL_VARIANT_SELECTION.json'
    got=sha(selp); expected=sf['source_authorities']['CANONICAL_VARIANT_SELECTION.json']
    if got!=expected: raise RuntimeError(f'canonical selection SHA mismatch expected={expected} got={got}')
    sel=json.load(open(selp)); rec=[]
    for split,entry in sf['splits'].items():
        assets=entry['asset_ids'] if isinstance(entry,dict) else entry
        access=(entry.get('access') if isinstance(entry,dict) else None) or ('TRAIN_OPEN' if split in ('FIT','TUNE') else 'SEALED')
        for aid in assets:
            if aid not in sel: raise RuntimeError(f'{aid} absent from canonical selection')
            v=sel[aid]
            if v.get('split')!=split: raise RuntimeError(f'{aid} split drift selection={v.get("split")} freeze={split}')
            rec.append({'asset_id':aid,'split':split,'access':access,
                        'candidate_id':v['candidate_id'],'source_registry_id':v['source_registry_id'],
                        'master_asset_root':f'master/assets/{aid}','primary_geometry':f'master/assets/{aid}/primary_geometry.npz',
                        'render_root':f'master/assets/{aid}/renders','views':8,'native_resolution':1024,'input_source_resolution':512,
                        'input_styles':['cel_clean_512.png','ink_cel_512.png'],'camera':'camera.json','raster_authority':'raster_authority.npz'})
    rec.sort(key=lambda x:x['asset_id'])
    out={'schema':'RealSaS.IRISControlledV1.ManifestSeed.v1','date':'2026-08-23','root':str(root),'record_count':len(rec),'records':rec,
         'source_authorities':sf['source_authorities'],'note':'Generated deterministically from frozen split membership + canonical selection; no random resplit.'}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'out':a.out,'records':len(rec),'sha256':sha(a.out)},indent=2))
if __name__=='__main__': main()
