#!/usr/bin/env python3
"""E0 geometry-stage runner over a frozen membership split. No optimizer exists in this file."""
from pathlib import Path
import argparse, json, hashlib, numpy as np
from surface_builder_e0_v1 import build_e0_asset, load_asset_authority, deterministic_full_mesh_surface, full_vs_observable_distribution_metrics


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,help='RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--membership',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--split',choices=['calibration_anchor8','qualification_proxy32'],required=True)
    ap.add_argument('--anchors',type=int,default=512)
    a=ap.parse_args()
    root=Path(a.root); mpath=Path(a.membership); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    m=json.load(open(mpath,encoding='utf-8'))
    if m.get('status')!='FROZEN_FROM_PREEXISTING_FIT_SCALE_MEMBERSHIP__NO_POST_RESULT_SELECTION': raise RuntimeError('E0 membership drift')
    if any(v is not True for v in m.get('sealed',{}).values()): raise RuntimeError('sealed split firewall drift')
    rows=m[a.split]
    ids=rows if rows and isinstance(rows[0],str) else [x['asset_id'] for x in rows]
    if a.split=='calibration_anchor8' and len(ids)!=8: raise RuntimeError('anchor8 drift')
    if a.split=='qualification_proxy32' and len(ids)!=32: raise RuntimeError('proxy32 drift')
    per=[]
    for i,aid in enumerate(ids,1):
        ad=root/'master'/'assets'/aid
        od=out/aid
        print(f'[E0] {i}/{len(ids)} {aid}',flush=True)
        od.mkdir(parents=True,exist_ok=True)
        arm_a,arm_b,metric=build_e0_asset(ad,anchor_count=a.anchors)
        geom,_auth=load_asset_authority(ad)
        arm_0=deterministic_full_mesh_surface(geom,aid,count=a.anchors)
        dist=full_vs_observable_distribution_metrics(geom,aid,arm_a['P'])
        np.savez_compressed(od/'E0_0_FULL_MESH_CEILING.npz',**arm_0)
        np.savez_compressed(od/'E0_A_ORACLE_SURFACE.npz',**arm_a)
        np.savez_compressed(od/'E0_B_DERIVED_SURFACE.npz',**arm_b)
        (od/'E0_PERSISTENCE_EVAL.json').write_text(json.dumps(metric,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        (od/'E0_FULL_VS_OBSERVABLE_DISTRIBUTION.json').write_text(json.dumps(dist,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        metric['full_vs_observable_distribution']=dist
        per.append(metric)
    summary={
      'schema':'RealSaS.E0.ObservableGeometrySplitSummary.v1','status':'GEOMETRY_STAGE_COMPLETE__DOWNSTREAM_SUFFICIENCY_NOT_YET_RUN',
      'split':a.split,'asset_count':len(ids),'membership_sha256':sha(mpath),'scientific_optimizer_steps':0,
      'camera_json_consumed':False,'teacher_identity_used_by_e0_b':False,'geometry_arms':['E0-0_FULL_MESH_CANONICAL','E0-a_OBSERVABLE_ORACLE','E0-b_OBSERVABLE_DETERMINISTIC'],
      'macro_precision':sum(x['precision'] for x in per)/len(per),'macro_recall':sum(x['recall'] for x in per)/len(per),
      'worst_asset_precision':min(x['precision'] for x in per),'worst_asset_recall':min(x['recall'] for x in per),
      'per_asset':per,
    }
    (out/'E0_SPLIT_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_asset'},indent=2,sort_keys=True))

if __name__=='__main__': main()
