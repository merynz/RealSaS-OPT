#!/usr/bin/env python3
"""Calibration-8-only E0 geometry runner. No qualification path exists in this file."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from surface_builder_e0_v1 import (
    build_e0_asset,
    deterministic_full_mesh_surface,
    full_vs_observable_distribution_metrics,
    load_asset_authority,
)

CALIBRATION_IDS = (
    'asset_551ea351b43a1787d0f55536',
    'asset_36fb02305846592b1ecdf3d4',
    'asset_0679fdef64f19a4832a6d521',
    'asset_76313e4bd82b82fcd1659c70',
    'asset_6f086a5b1a66378ffe04d7e4',
    'asset_425122d500ecf5767404f9c0',
    'asset_5a19f8c5254be7bf30c504f5',
    'asset_f8a40d6c5d815fe79c8b5e42',
)
SOURCE_MEMBERSHIP_SHA256='53856330a24f5c04b17db69bf059c6b242f0320c7e3c3c92caf1d24117c6d419'
SOURCE_BRANCH_HEAD='8c6d339ed7f59d5225ce9a224f281df3e39ff5e0'


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def _assert_asset(ad: Path):
    req=[ad/'primary_geometry.npz']
    req.extend(ad/'renders'/f'V{v}'/'raster_authority.npz' for v in range(8))
    missing=[str(p) for p in req if not p.is_file()]
    if missing:
        raise FileNotFoundError({'asset':ad.name,'missing':missing})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--anchors',type=int,default=512)
    a=ap.parse_args()
    if a.anchors != 512:
        raise RuntimeError('calibration point budget is frozen at 512')
    root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if not root.is_dir(): raise FileNotFoundError(root)
    per=[]
    for i,aid in enumerate(CALIBRATION_IDS,1):
        ad=root/'master'/'assets'/aid
        _assert_asset(ad)
        od=out/aid; od.mkdir(parents=True,exist_ok=True)
        print(f'[E0-CAL8] {i}/8 {aid}',flush=True)
        arm_a,arm_b,metric=build_e0_asset(ad,anchor_count=512)
        geom,_=load_asset_authority(ad)
        arm_0=deterministic_full_mesh_surface(geom,aid,count=512)
        dist=full_vs_observable_distribution_metrics(geom,aid,arm_a['P'])
        np.savez_compressed(od/'E0_0_FULL_MESH_CANONICAL.npz',**arm_0)
        np.savez_compressed(od/'E0_A_OBSERVABLE_ORACLE.npz',**arm_a)
        np.savez_compressed(od/'E0_B_OBSERVABLE_DETERMINISTIC.npz',**arm_b)
        (od/'E0_PERSISTENCE_EVAL.json').write_text(json.dumps(metric,indent=2,sort_keys=True)+'\n')
        (od/'E0_FULL_VS_OBSERVABLE_DISTRIBUTION.json').write_text(json.dumps(dist,indent=2,sort_keys=True)+'\n')
        per.append({**metric,'full_vs_observable_distribution':dist})
    summary={
      'schema':'RealSaS.E0.Calibration8Geometry.v1',
      'status':'CALIBRATION8_GEOMETRY_COMPLETE__NO_NONINFERIORITY_MARGIN_FROZEN_YET',
      'asset_ids':list(CALIBRATION_IDS),
      'asset_count':8,
      'anchors_per_arm':512,
      'geometry_arms':['E0-0_FULL_MESH_CANONICAL','E0-a_OBSERVABLE_ORACLE','E0-b_OBSERVABLE_DETERMINISTIC'],
      'source_membership_sha256':SOURCE_MEMBERSHIP_SHA256,
      'source_branch_head':SOURCE_BRANCH_HEAD,
      'scientific_optimizer_steps':0,
      'qualification_executable_present':False,
      'camera_json_consumed':False,
      'teacher_identity_used_by_e0_b':False,
      'macro_precision':float(np.mean([x['precision'] for x in per])),
      'macro_recall':float(np.mean([x['recall'] for x in per])),
      'worst_asset_precision':float(np.min([x['precision'] for x in per])),
      'worst_asset_recall':float(np.min([x['recall'] for x in per])),
      'full_to_observable_nn_p95_family_median':float(np.median([x['full_vs_observable_distribution']['full_to_observable_nn_p95'] for x in per])),
      'full_to_observable_nn_p95_family_max':float(np.max([x['full_vs_observable_distribution']['full_to_observable_nn_p95'] for x in per])),
      'per_asset':per,
    }
    sp=out/'E0_CALIBRATION8_GEOMETRY_SUMMARY.json'
    sp.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_asset'},indent=2,sort_keys=True))
    print('SUMMARY_SHA256',_sha(sp))

if __name__=='__main__': main()
