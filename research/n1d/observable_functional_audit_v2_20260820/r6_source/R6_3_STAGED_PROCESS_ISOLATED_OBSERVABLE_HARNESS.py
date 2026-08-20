from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

R6_1_SOURCE_ROOT = Path('/mnt/data/observable_functional_audit_v2_src_r6_1')
OBS_SRC = R6_1_SOURCE_ROOT / 'src'
FROZEN = OBS_SRC / 'frozen_deps'
EXPECTED_SOURCE_MEMBERS = {
    'src/observable_phase.py': '6eea236647e04a5566e93733f35c927639cc6407d4f62db80793edfa7e97dadf',
    'src/frozen_deps/realsas_n1d_hybrid_v11_frozen_runner.py': 'ddfd3c989138bb256c2c466e80b914bf89fef1796520d41acc734af0ff577407',
    'src/frozen_deps/v8_base_frozen.py': '44fe83a56a588d4f0cb771351497090ca302b173a7c70619ff701fb0ed28b1f0',
    'src/frozen_deps/v5_seed_geometry_frozen.py': '6185e0fe7b47fedd19c845aecc95d1b8c80a45c17fa683c9f5caedff4bd7d372',
}
EXPECTED_CHECKPOINT_SHA256 = '0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18'


def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def assert_source():
    for rel, exp in EXPECTED_SOURCE_MEMBERS.items():
        p=R6_1_SOURCE_ROOT/rel
        got=sha256_file(p)
        if got != exp: raise RuntimeError(f'R6.1 source member SHA mismatch {rel}: {got}')


def import_observable():
    assert_source()
    sys.path.insert(0,str(OBS_SRC))
    import observable_phase as o
    return o


def raster_paths(root: Path, family: int, episode: str):
    A=sorted((root/str(family)/'A').glob('*.png'))
    B=sorted((root/str(family)/episode/'B').glob('*.png'))
    if len(A)!=8 or len(B)!=8: raise RuntimeError((family,episode,len(A),len(B)))
    return A,B


def current_raster_hashes(root: Path, family: int, episode: str, o):
    A,B=raster_paths(root,family,episode)
    return {str(p):o.sha256_file(p) for p in A+B}


def v8_mode(root: Path, family: int, episode: str, work: Path):
    o=import_observable(); work.mkdir(parents=True,exist_ok=True); stage=work/'baseline_stage1'; stage.mkdir(parents=True,exist_ok=True)
    A,B=raster_paths(root,family,episode)
    subprocess.run([sys.executable,str(FROZEN/'v8_base_frozen.py'),'--family',str(family),'--episode',episode,'--root',str(root),'--out',str(stage)],check=True,stdout=subprocess.DEVNULL)
    generated=stage/f'{family}_{episode}.npz'; base=stage/f'{family}_{episode}_v8.npz'
    if not generated.is_file(): raise RuntimeError('V8 output missing')
    shutil.move(generated,base)
    gj=generated.with_suffix('.json')
    if gj.exists(): shutil.move(gj,stage/f'{family}_{episode}_v8.json')
    meta={'schema':'RealSaS.N1D.ObservableFunctionalAuditV2.R6_3V8Stage.v1','family':family,'episode':episode,'v8_prediction_sha256':sha256_file(base),'checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,'raster_sha256':current_raster_hashes(root,family,episode,o),'truth_access':'NONE'}
    (stage/f'{family}_{episode}_v8_stage.json').write_text(json.dumps(meta,indent=2,sort_keys=True))
    print(json.dumps(meta,sort_keys=True))


def gate_mode(root: Path, family: int, episode: str, work: Path):
    o=import_observable(); r=o.runner; stage=work/'baseline_stage1'; A,B=raster_paths(root,family,episode)
    base=stage/f'{family}_{episode}_v8.npz'; vm=stage/f'{family}_{episode}_v8_stage.json'
    if not base.is_file() or not vm.is_file(): raise RuntimeError('V8 stage incomplete')
    vmeta=json.loads(vm.read_text())
    if vmeta.get('truth_access')!='NONE' or vmeta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA256: raise RuntimeError('V8 stage guard')
    if sha256_file(base)!=vmeta.get('v8_prediction_sha256'): raise RuntimeError('V8 stage SHA mismatch')
    if current_raster_hashes(root,family,episode,o)!=vmeta.get('raster_sha256'): raise RuntimeError('V8 stage raster mismatch')
    z=np.load(base);P=np.array(z['P_A'],np.float32);V=np.array(z['V_A'],np.uint8);XY=np.array(z['XY_A'],np.float32);PB0=np.array(z['P_B'],np.float32)
    cur,camp,_=r.model_current(r.v8,A,B,P,V,XY);D=PB0-P;a=np.linalg.norm(D,axis=1);b=np.linalg.norm(cur,axis=1);valid=(a>1e-6)&(b>1e-6);co=np.sum(D*cur,axis=1)/np.maximum(a*b,1e-9);w=np.maximum(camp,1e-4);signed=float(np.average(co[valid],weights=w[valid])) if valid.any() else 0.;absolute=float(np.average(np.abs(co[valid]),weights=w[valid])) if valid.any() else 0.;switch=bool(signed<r.SIGNED_T and absolute>=r.ABS_T)
    meta={'schema':'RealSaS.N1D.ObservableFunctionalAuditV2.R6_3GateStage.v1','family':family,'episode':episode,'v8_prediction_sha256':sha256_file(base),'gate':{'signed':signed,'abs':absolute,'signed_threshold':r.SIGNED_T,'abs_threshold':r.ABS_T,'switch':switch},'checkpoint_sha256':r.v8.CHECKPOINT_EXPECTED,'raster_sha256':current_raster_hashes(root,family,episode,o),'truth_access':'NONE'}
    (stage/f'{family}_{episode}_gate.json').write_text(json.dumps(meta,indent=2,sort_keys=True,default=float))
    print(json.dumps(meta,sort_keys=True,default=float))

def finalize_baseline_mode(root: Path, family: int, episode: str, work: Path):
    o=import_observable(); r=o.runner; stage=work/'baseline_stage1'; cache=work/'baseline_cache'; cache.mkdir(parents=True,exist_ok=True)
    A,B=raster_paths(root,family,episode)
    base=stage/f'{family}_{episode}_v8.npz'; sj=stage/f'{family}_{episode}_gate.json'
    if not base.is_file() or not sj.is_file(): raise RuntimeError('gate stage incomplete')
    smeta=json.loads(sj.read_text())
    if smeta.get('truth_access')!='NONE': raise RuntimeError('gate stage truth access')
    if smeta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA256: raise RuntimeError('gate stage checkpoint mismatch')
    if sha256_file(base)!=smeta.get('v8_prediction_sha256'): raise RuntimeError('gate stage v8 SHA mismatch')
    if current_raster_hashes(root,family,episode,o)!=smeta.get('raster_sha256'): raise RuntimeError('gate stage raster SHA mismatch')
    gate=smeta['gate']; signed=float(gate['signed']); absolute=float(gate['abs']); switch=bool(gate['switch'])
    # Recompute switch from frozen thresholds; stage cache cannot choose route by itself.
    expected_switch=bool(signed<r.SIGNED_T and absolute>=r.ABS_T)
    if switch!=expected_switch: raise RuntimeError('gate stage switch mismatch')
    out=cache/f'{family}_{episode}_baseline.npz'
    if not switch:
        shutil.copy2(base,out); route='V8_BASE'; extra={'source_v8_prediction_sha256':r.sha(base)}
    else:
        pred,par,u,wseed=r.seed_predict(A,B); np.savez_compressed(out,**pred); route='SEED_BASIN_WSEED'; extra={'seed_solver_uobs':u,'par':par.tolist(),'w_nontrivial':int((wseed>.05).sum())}
    meta={'schema':'RealSaS.N1D.HybridV11.FrozenPrediction.v1','family':family,'episode':episode,'route':route,'gate':{'signed':signed,'abs':absolute,'signed_threshold':r.SIGNED_T,'abs_threshold':r.ABS_T},'checkpoint_sha256':r.v8.CHECKPOINT_EXPECTED,'prediction_sha256':r.sha(out),'raster_sha256':{str(p):r.sha(p) for p in A+B},'truth_access':'NONE',**extra}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2,sort_keys=True,default=float))
    if meta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA256 or meta.get('truth_access')!='NONE': raise RuntimeError('final baseline guard')
    if sha256_file(out)!=meta.get('prediction_sha256'): raise RuntimeError('final baseline SHA mismatch')
    print(json.dumps(meta,sort_keys=True,default=float))


def state_mode(root: Path, family: int, episode: str, work: Path):
    # Semantically identical to frozen R6.2 state mode.
    o=import_observable(); cache=work/'baseline_cache'; out=work/'observable'; out.mkdir(parents=True,exist_ok=True)
    src=cache/f'{family}_{episode}_baseline.npz'; meta_path=src.with_suffix('.json')
    if not src.is_file() or not meta_path.is_file(): raise RuntimeError('baseline cache incomplete')
    cached_meta=json.loads(meta_path.read_text())
    if cached_meta.get('truth_access')!='NONE': raise RuntimeError('cached baseline truth access')
    if cached_meta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA256: raise RuntimeError('cached checkpoint mismatch')
    if sha256_file(src)!=cached_meta.get('prediction_sha256'): raise RuntimeError('cached baseline prediction SHA mismatch')
    if current_raster_hashes(root,family,episode,o)!=cached_meta.get('raster_sha256'): raise RuntimeError('cached raster SHA mismatch')
    orig=o.runner.run
    def cached_runner(r,f,e,dst):
        if Path(r).resolve()!=root.resolve() or int(f)!=family or str(e)!=episode: raise RuntimeError('cached runner request mismatch')
        dst=Path(dst)
        if src.resolve()!=dst.resolve(): shutil.copy2(src,dst)
        dst.with_suffix('.json').write_text(json.dumps(cached_meta,indent=2,sort_keys=True))
        return dict(cached_meta)
    o.runner.run=cached_runner
    try: smeta=o.run_family(root,family,episode,out)
    finally: o.runner.run=orig
    if smeta.get('truth_access')!='NONE': raise RuntimeError('observable state truth access')
    if smeta.get('baseline_prediction_sha256')!=cached_meta.get('prediction_sha256'): raise RuntimeError('state baseline SHA mismatch')
    print(json.dumps(smeta,sort_keys=True))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['v8','gate','baseline-finalize','state'],required=True); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--family',type=int,required=True); ap.add_argument('--episode',default='e01'); ap.add_argument('--work',type=Path,required=True); a=ap.parse_args()
    if a.mode=='v8': v8_mode(a.root,a.family,a.episode,a.work)
    elif a.mode=='gate': gate_mode(a.root,a.family,a.episode,a.work)
    elif a.mode=='baseline-finalize': finalize_baseline_mode(a.root,a.family,a.episode,a.work)
    else: state_mode(a.root,a.family,a.episode,a.work)
if __name__=='__main__': main()
