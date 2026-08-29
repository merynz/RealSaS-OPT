#!/usr/bin/env python3
"""Fail-closed RealSaS M4 structured-depth runner.

Default mode is preflight only. Scientific outcomes require both the explicit
--open-scientific-outcomes flag and a matching pre-open execution seal.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys, types as pytypes
from pathlib import Path
import numpy as np

BASE_COMMIT="eb18e8a0e12d813419e6e8274c29f784c560daaf"
SOURCE_SHA="f044587a12b51a8ae141bb6a3b37c67b7120f9e391e0f0d2afebad939a2da7c8"
PREREG_SHA="c23166ff45a9c12fa9978f448a01e0501c5a1972aa5b46fd06bcef3d1305f0e8"
RUNTIME_SHA="4ebf3636438aa14b5f832d51da3e9222086b948b7480164dcd3c50d3ad4ca9dd"
SURFACE_SHA="e81e2eb25145238d7dcc42b493ba94825ac202dd5eed31335537c33372e1b589"
G_SHA="f8c6146fc3ad81146ced01805b9be454ad194b86db3a9ab3d72d7b9eb3747b65"
A_SHA="72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745"
PROJECT_BLOBS={
 "e0_geometry.py":"2c6bc9b0c1588d1760d20173f7ef4f89f718a11f",
 "surface_builder_e0_v1.py":"496a4981a149f9a6971b56d455d960781cf8e1d5",
 "bridge_persistence_v1.py":"8a873dcdbce97a6b6030dd31bba20c63ed6351bd",
 "depth_corruption_v1.py":"6b6a3aeb873ed23e65e8de5356c72b9dbe161ee2",
 "proxy_replay_v1.py":"5ed5fe4cf27d527990e0f21c900e88de1d77b4cd",
 "compiler/hashing.py":"8beb27031a3ed9d073edc07de785a8626d4f1e8d",
 "compiler/types.py":"928f5a89d415d281018c4e30f8e738d57fa57905",
 "compiler/surface.py":"d5f94eb93a80c75874603edb15e8fad570958a80",
}
TARGET_SHA={
 "asset_551ea351b43a1787d0f55536":"a4b3d5544f20a0fecefa877b74f4ad601f072d29afd0df35204c7b182d800c2c",
 "asset_0679fdef64f19a4832a6d521":"76a84883a1516839d9d5476f7c2d99ad26a29815f490d4d4c50124ecafa3db17",
 "asset_76313e4bd82b82fcd1659c70":"ddeed0eef39e13ed7831e798b8404de92083df5355bb65002f70087734283633",
 "asset_f8a40d6c5d815fe79c8b5e42":"41cda2c5623f9698e7c35bd5559b7967871dc7f5c37fff10148b220991d1ebf8",
}
ASSETS=(
 "asset_551ea351b43a1787d0f55536","asset_36fb02305846592b1ecdf3d4",
 "asset_0679fdef64f19a4832a6d521","asset_76313e4bd82b82fcd1659c70",
 "asset_6f086a5b1a66378ffe04d7e4","asset_425122d500ecf5767404f9c0",
 "asset_5a19f8c5254be7bf30c504f5","asset_f8a40d6c5d815fe79c8b5e42")
PROXY=(ASSETS[0],ASSETS[2],ASSETS[3],ASSETS[7])
RAY_TOL=1e-6

def h256(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def blob(p):
 b=Path(p).read_bytes(); return hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
def req(v,e,msg):
 if v!=e: raise RuntimeError(f"{msg}: {v} != {e}")
def req_file(p,e,msg,fn=h256):
 p=Path(p)
 if not p.is_file(): raise RuntimeError(f"missing {msg}: {p}")
 req(fn(p),e,msg)
def root_from(p):
 for q in [p,*p.parents]:
  if (q/"CURRENT_STATE.md").is_file() and (q/"compiler").is_dir(): return q
 raise RuntimeError("cannot infer repo root; pass --repo-root")
def paths(root):
 b=root/"experiments/g0_g1_single_pose_geometry/depth_tolerance_bridge_20260829"
 e=root/"experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827"
 c=root/"compiler/realsas_compiler_core"
 return dict(bridge=b,prereg=b/"M4_SURFACE_GRID_PREREG_V1.json",runtime=b/"bridge_grid_runtime_v1.py",
  persistence=b/"bridge_persistence_v1.py",depth=b/"depth_corruption_v1.py",proxy=b/"proxy_replay_v1.py",
  e0=e/"e0_geometry.py",builder=e/"surface_builder_e0_v1.py",surface=c/"surface.py",hashing=c/"hashing.py",types=c/"types.py")
def grid(pr):
 g=pr["grid"]; out=[dict(g["baseline"])]; tag=g["asymmetry_cell_tag"]
 for eps in g["nonzero_epsilon_order"]:
  for ell in g["ell_px_order"]:
   for a in g["asymmetry_order"]:
    out.append(dict(cell_id=f"E{round(float(eps)*10000):04d}_L{int(float(ell)):03d}_{tag[a]}",epsilon=float(eps),ell_px=float(ell),asymmetry=a))
 if len(out)!=65 or len({x['cell_id'] for x in out})!=65 or out[0]["cell_id"]!="E0000_L000_ALL8": raise RuntimeError("grid drift")
 return out
def validate_prereg(p):
 req_file(p,PREREG_SHA,"prereg"); pr=json.loads(Path(p).read_text())
 req(pr.get("schema"),"RealSaS.DepthToleranceBridge.M4SurfaceGridPrereg.v1","prereg schema")
 req(tuple(pr["population"]["surface_compiler_asset_ids"]),ASSETS,"surface population")
 req(tuple(pr["population"]["frozen_proxy_eligible_asset_ids"]),PROXY,"proxy population")
 f=pr["firewalls"]
 if f.get("scientific_outcomes_opened_at_prereg") is not False or f.get("proxy27")!="CLOSED" or f.get("dev32")!="CLOSED" or int(f.get("training_steps",-1))!=0: raise RuntimeError("prereg firewall drift")
 return pr,grid(pr)
def source_path(root,aid,role,view=None):
 if role=='primary_geometry': return root/aid/'primary_geometry.npz'
 p=root/aid/'renders'/view/'raster_authority.npz'
 return p if p.is_file() else root/aid/f'{view}_raster_authority.npz'
def validate_source(root,mp):
 req_file(mp,SOURCE_SHA,"raw source manifest"); m=json.loads(Path(mp).read_text())
 req(m.get("schema"),"RealSaS.M4.RawSourceManifest.v2","source schema"); req(tuple(m.get("asset_ids",())),ASSETS,"source population")
 if m.get("file_count")!=72 or m.get("asset_count")!=8 or m.get("views_per_asset")!=8: raise RuntimeError("source count drift")
 i=m.get("integrity",{})
 if not all(i.get(k) for k in ("all_expected_files_present","zip_test_pass","npz_load_pass")): raise RuntimeError("source integrity not closed")
 for aid in ASSETS:
  a=m["assets"][aid]; r=a["primary_geometry"]; p=source_path(root,aid,"primary_geometry")
  if not p.is_file() or p.stat().st_size!=int(r["size_bytes"]) or h256(p)!=r["sha256"]: raise RuntimeError(f"source byte drift: {aid}/primary_geometry.npz")
  for v in [f"V{x}" for x in range(8)]:
   r=a["rasters"][v]; p=source_path(root,aid,"raster",v)
   if not p.is_file() or p.stat().st_size!=int(r["size_bytes"]) or h256(p)!=r["sha256"]: raise RuntimeError(f"source byte drift: {aid}/{v}/raster_authority.npz")
 return m
def validate_static(root,src,manifest,gck,ack):
 p=paths(root); pr,cells=validate_prereg(p["prereg"]); validate_source(src,manifest)
 req_file(p["runtime"],RUNTIME_SHA,"grid runtime"); req_file(p["surface"],SURFACE_SHA,"Compiler surface")
 req_file(gck,G_SHA,"Geppetto D2"); req_file(ack,A_SHA,"Arachne D2")
 for path,key in ((p['e0'],'e0_geometry.py'),(p['builder'],'surface_builder_e0_v1.py'),(p['persistence'],'bridge_persistence_v1.py'),(p['depth'],'depth_corruption_v1.py'),(p['proxy'],'proxy_replay_v1.py'),(p['hashing'],'compiler/hashing.py'),(p['types'],'compiler/types.py'),(p['surface'],'compiler/surface.py')): req_file(path,PROJECT_BLOBS[key],key,blob)
 return p,pr,cells
def validate_targets(target):
 target=Path(target)
 for aid,sha in TARGET_SHA.items():
  p=target/f"{aid}.npz"; req_file(p,sha,f"proxy target {aid}")
  with np.load(p,allow_pickle=False) as z:
   if "asset_id" not in z.files or str(np.asarray(z["asset_id"]).item())!=aid: raise RuntimeError(f"proxy target identity drift: {aid}")
 return target
def load_surface_adapter(root):
 c=paths(root); pkg="_realsas_m4_compiler_exact"
 for k in list(sys.modules):
  if k==pkg or k.startswith(pkg+"."): del sys.modules[k]
 m=pytypes.ModuleType(pkg); m.__path__=[str(c['surface'].parent)]; sys.modules[pkg]=m
 for name,path in (("types",c['types']),("hashing",c['hashing']),("surface",c['surface'])):
  spec=importlib.util.spec_from_file_location(f"{pkg}.{name}",path)
  if spec is None or spec.loader is None: raise RuntimeError(f"cannot load Compiler adapter module: {path}")
  mod=importlib.util.module_from_spec(spec); sys.modules[f"{pkg}.{name}"]=mod; spec.loader.exec_module(mod)
 return sys.modules[f"{pkg}.surface"].rigging_surface_from_d2_arrays
def imports(root):
 p=paths(root)
 for d in (p['bridge'],p['e0'].parent):
  if str(d) not in sys.path: sys.path.insert(0,str(d))
def prepare_zero(root,src):
 imports(root); from bridge_grid_runtime_v1 import prepare_bridge_asset
 out=[]
 for aid in ASSETS:
  x=prepare_bridge_asset(src/aid,anchor_count=512); vv=sorted(set(map(int,np.asarray(x.source_view))))
  if x.asset_id!=aid or int(x.anchor_count)!=512 or vv!=list(range(8)): raise RuntimeError(f"zero prepare drift: {aid}")
  out.append(dict(asset_id=aid,anchor_count=512,source_views=vv,pass_=True))
 return out
def corruption_ok(reps):
 for r in reps:
  nums=[v for k,v in r.items() if k!='view' and isinstance(v,(int,float))]
  if not all(math.isfinite(float(v)) for v in nums) or float(r.get('max_ray_perpendicular_residual',math.inf))>RAY_TOL: raise RuntimeError("corruption invariant failure")
def stats(s): s=np.asarray(s,bool); return int(s.sum()),float(s.mean())
def pack(p):
 with np.load(p,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}
def ratio(x,z): return float(x/z) if z>0 else (1.0 if x<=z else math.inf)
def checks(cur,zero,m):
 g,a=cur['geppetto'],cur['arachne']; g0,a0=zero['geppetto'],zero['arachne']
 vals={
  'arachne.ce_ratio_to_zero':ratio(a['ce'],a0['ce']),
  'arachne.influence_disp_mean_ratio_to_zero':ratio(a['influence_disp_mean'],a0['influence_disp_mean']),
  'geppetto.family_p95_ratio_to_zero':ratio(g['family_p95'],g0['family_p95']),
  'geppetto.joint_mean_ratio_to_zero':ratio(g['joint_mean'],g0['joint_mean']),
  'geppetto.pck_005_delta_from_zero':float(g['pck_005']-g0['pck_005']),
  'geppetto.pck_008_delta_from_zero':float(g['pck_008']-g0['pck_008'])}
 out={}
 for k,v in vals.items():
  lim=m[k]; ok=math.isfinite(v) and (v<=lim['max'] if 'max' in lim else v>=lim['min'])
  out[k]={'value':v,**lim,'pass':bool(ok)}
 return out
def run_grid(root,src,target,gck,ack,pr,cells):
 imports(root)
 from bridge_grid_runtime_v1 import prepare_bridge_asset,build_d2_mutual_p003_from_prepared
 from depth_corruption_v1 import DepthCorruptionSpec,affected_views
 from proxy_replay_v1 import load_d2_models,eval_geppetto_d2,eval_arachne_d2,aggregate_geppetto,aggregate_arachne
 rigging_surface_from_d2_arrays=load_surface_adapter(root)
 prep={a:prepare_bridge_asset(src/a,anchor_count=512) for a in ASSETS}; gm,am=load_d2_models(gck,ack,device='cpu')
 tgt={a:pack(target/f"{a}.npz") for a in PROXY}; results=[]; zero=None; margins=pr['decision_operator']['proxy_checks']
 for cell in cells:
  spec=DepthCorruptionSpec(float(cell['epsilon']),float(cell['ell_px']),cell['asymmetry']); per=[]; gg=[]; aa=[]; surface_fail=False; proxy_unavailable=False
  for aid in ASSETS:
   r=dict(asset_id=aid,cell_id=cell['cell_id'],epsilon=float(spec.epsilon),ell_px=float(spec.ell_px),asymmetry=spec.asymmetry,affected_views=list(affected_views(aid,spec.asymmetry)) if spec.epsilon else [])
   try:
    st,d2,_rows,dgrid,X=build_d2_mutual_p003_from_prepared(prep[aid],spec); corruption_ok(st.carrier.per_view_corruption)
    if not np.all(np.isfinite(st.carrier.P)) or not np.all(np.isfinite(X)): raise RuntimeError('nonfinite bridge output')
    bp,bf=stats(st.carrier.support); dp,df=stats(d2); surf=rigging_surface_from_d2_arrays(st.carrier.P,d2,dgrid,authority_label=f"M4:{cell['cell_id']}:{aid}",persistence_label='D2_MUTUAL_P003')
    if surf.schema_version!='RealSaS.RiggingSurfaceIR.v1' or len(surf.surface_nodes)!=512: raise RuntimeError('typed surface mismatch')
    r.update(base_support_pairs=bp,d2_support_pairs=dp,base_support_fraction=bf,d2_support_fraction=df,typed_surface_ok=True,typed_surface_node_count=512,typed_surface_lineage_hash=surf.geometry_lineage_hash,corruption_diagnostics=list(st.carrier.per_view_corruption))
   except Exception as e:
    surface_fail=True; r.update(typed_surface_ok=False,error=f"{type(e).__name__}: {e}"); per.append(r); continue
   if aid in PROXY:
    try:
     gr=eval_geppetto_d2(gm,X,tgt[aid],aid,device='cpu'); ar=eval_arachne_d2(am,X,tgt[aid],aid,device='cpu'); r['proxy']={'geppetto':gr,'arachne':ar}; gg.append(gr); aa.append(ar)
    except Exception as e:
     proxy_unavailable=True; r['proxy_error']=f"{type(e).__name__}: {e}"
   per.append(r)
  base={'support_pairs_sum':sum(x.get('base_support_pairs',0) for x in per),'support_fraction_mean':float(np.mean([x.get('base_support_fraction',0) for x in per]))}
  d2a={'support_pairs_sum':sum(x.get('d2_support_pairs',0) for x in per),'support_fraction_mean':float(np.mean([x.get('d2_support_fraction',0) for x in per]))}
  pagg=None; ck={}
  if surface_fail: label='SURFACE_ROUTE_FAIL'; sr='FAIL'
  elif proxy_unavailable or len(gg)!=4 or len(aa)!=4: label='SURFACE_ROUTE_PASS__PROXY_UNAVAILABLE'; sr='PASS'
  else:
   sr='PASS'; pagg={'geppetto':aggregate_geppetto(gg),'arachne':aggregate_arachne(aa)}
   if zero is None:
    if cell['cell_id']!='E0000_L000_ALL8': raise RuntimeError('baseline not first')
    zero=pagg
   ck=checks(pagg,zero,margins); label='SURFACE_ROUTE_PASS__PROXY_PASS' if all(v['pass'] for v in ck.values()) else 'SURFACE_ROUTE_PASS__PROXY_FAIL'
  results.append(dict(cell_id=cell['cell_id'],surface_route_status=sr,base_aggregate=base,d2_aggregate=d2a,proxy_aggregate=pagg,proxy_checks=ck,cell_label=label,per_asset=per))
 return {'schema':'RealSaS.DepthToleranceBridge.M4SurfaceGridResult.v1','prereg_sha256':PREREG_SHA,'source_manifest_sha256':SOURCE_SHA,'baseline':results[0],'cells':results,'firewalls':{'proxy27':'CLOSED','dev32':'CLOSED','training_steps':0,'product_safe_claim_authorized':False}}
def args():
 p=argparse.ArgumentParser(); p.add_argument('--repo-root',type=Path); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--source-manifest',type=Path,required=True); p.add_argument('--geppetto-checkpoint',type=Path,required=True); p.add_argument('--arachne-checkpoint',type=Path,required=True); p.add_argument('--proxy-target-root',type=Path); p.add_argument('--execution-seal',type=Path); p.add_argument('--output',type=Path); p.add_argument('--open-scientific-outcomes',action='store_true'); return p.parse_args()
def main():
 a=args(); here=Path(__file__).resolve(); root=a.repo_root.resolve() if a.repo_root else root_from(here.parent); src=a.source_root.resolve(); p,pr,cells=validate_static(root,src,a.source_manifest.resolve(),a.geppetto_checkpoint.resolve(),a.arachne_checkpoint.resolve()); runner=h256(here)
 pre={'schema':'RealSaS.DepthToleranceBridge.M4RunnerPreflight.v1','status':'PASS__SCIENTIFIC_OUTCOMES_STILL_CLOSED','runner_sha256':runner,'expected_base_commit':BASE_COMMIT,'source_manifest_sha256':SOURCE_SHA,'prereg_sha256':PREREG_SHA,'grid_runtime_sha256':RUNTIME_SHA,'compiler_surface_sha256':SURFACE_SHA,'geppetto_d2_sha256':G_SHA,'arachne_d2_sha256':A_SHA,'cell_count':65,'asset_count':8,'project_dependency_git_blobs':PROJECT_BLOBS,'proxy_target_sha256':TARGET_SHA,'compiler_import_mode':'ISOLATED_EXACT_SURFACE_TYPES_HASHING__NO_PACKAGE_INIT_SIDE_EFFECTS','scientific_outcome_opened':False,'proxy27':'CLOSED','dev32':'CLOSED','training_steps':0}
 if not a.open_scientific_outcomes:
  pre['prepare_zero_only']=prepare_zero(root,src); out=pre
 else:
  if not a.execution_seal: raise RuntimeError('--open-scientific-outcomes requires --execution-seal')
  seal=json.loads(a.execution_seal.read_text()); req(seal.get('schema'),'RealSaS.DepthToleranceBridge.M4ExecutionSeal.v1','seal schema'); req(seal.get('status'),'SEALED_BEFORE_M4_SCIENTIFIC_OUTCOMES','seal status')
  if seal.get('scientific_outcome_opened') is not False: raise RuntimeError('seal is not pre-open')
  for k,v in {'runner_sha256':runner,'source_manifest_sha256':SOURCE_SHA,'prereg_sha256':PREREG_SHA,'grid_runtime_sha256':RUNTIME_SHA,'compiler_surface_sha256':SURFACE_SHA,'geppetto_d2_sha256':G_SHA,'arachne_d2_sha256':A_SHA}.items(): req(seal.get(k),v,f'seal {k}')
  req(seal.get('project_dependency_git_blobs'),PROJECT_BLOBS,'seal project dependency blobs'); req(seal.get('proxy_target_sha256'),TARGET_SHA,'seal proxy target bytes')
  req(int(seal.get('cell_count',-1)),65,'seal cell count'); req(tuple(seal.get('asset_ids',())),ASSETS,'seal assets')
  if not a.proxy_target_root: raise RuntimeError('scientific grid requires --proxy-target-root')
  target=validate_targets(a.proxy_target_root.resolve())
  out=run_grid(root,src,target,a.geppetto_checkpoint.resolve(),a.arachne_checkpoint.resolve(),pr,cells); out['runner_sha256']=runner; out['execution_seal_sha256']=h256(a.execution_seal)
 text=json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'
 if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text)
 print(text,end='')
if __name__=='__main__': main()
