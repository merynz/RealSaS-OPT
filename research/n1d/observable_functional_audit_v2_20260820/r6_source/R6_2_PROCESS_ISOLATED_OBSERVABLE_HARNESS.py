from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path

R6_1_SOURCE_ROOT = Path('/mnt/data/observable_functional_audit_v2_src_r6_1')
OBS_SRC = R6_1_SOURCE_ROOT / 'src'
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


def current_raster_hashes(root: Path, family: int, episode: str, o):
    A=sorted((root/str(family)/'A').glob('*.png'))
    B=sorted((root/str(family)/episode/'B').glob('*.png'))
    if len(A)!=8 or len(B)!=8: raise RuntimeError((family,episode,len(A),len(B)))
    return {str(p):o.sha256_file(p) for p in A+B}


def baseline_mode(root: Path, family: int, episode: str, cache: Path):
    o=import_observable(); cache.mkdir(parents=True,exist_ok=True)
    npz=cache/f'{family}_{episode}_baseline.npz'
    meta=o.runner.run(root,family,episode,npz)
    if meta.get('truth_access')!='NONE': raise RuntimeError('baseline truth access')
    if meta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA256: raise RuntimeError('checkpoint mismatch')
    if sha256_file(npz)!=meta.get('prediction_sha256'): raise RuntimeError('baseline prediction SHA mismatch')
    if current_raster_hashes(root,family,episode,o)!=meta.get('raster_sha256'): raise RuntimeError('baseline raster SHA mismatch')
    print(json.dumps(meta,sort_keys=True))


def state_mode(root: Path, family: int, episode: str, cache: Path, out: Path):
    o=import_observable(); out.mkdir(parents=True,exist_ok=True)
    src=cache/f'{family}_{episode}_baseline.npz'; meta_path=src.with_suffix('.json')
    if not src.is_file() or not meta_path.is_file(): raise RuntimeError('baseline cache incomplete')
    cached_meta=json.loads(meta_path.read_text())
    if cached_meta.get('truth_access')!='NONE': raise RuntimeError('cached baseline truth access')
    if cached_meta.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT_SHA256: raise RuntimeError('cached checkpoint mismatch')
    if sha256_file(src)!=cached_meta.get('prediction_sha256'): raise RuntimeError('cached baseline prediction SHA mismatch')
    if current_raster_hashes(root,family,episode,o)!=cached_meta.get('raster_sha256'): raise RuntimeError('cached raster SHA mismatch')
    orig=o.runner.run
    def cached_runner(r,f,e,dst):
        if Path(r).resolve()!=root.resolve() or int(f)!=family or str(e)!=episode:
            raise RuntimeError('cached runner request mismatch')
        dst=Path(dst)
        if src.resolve()!=dst.resolve(): shutil.copy2(src,dst)
        dst.with_suffix('.json').write_text(json.dumps(cached_meta,indent=2,sort_keys=True))
        return dict(cached_meta)
    o.runner.run=cached_runner
    try:
        smeta=o.run_family(root,family,episode,out)
    finally:
        o.runner.run=orig
    if smeta.get('truth_access')!='NONE': raise RuntimeError('observable state truth access')
    if smeta.get('baseline_prediction_sha256')!=cached_meta.get('prediction_sha256'):
        raise RuntimeError('state baseline SHA mismatch')
    print(json.dumps(smeta,sort_keys=True))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['baseline','state'],required=True)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--family',type=int,required=True)
    ap.add_argument('--episode',default='e01')
    ap.add_argument('--work',type=Path,required=True)
    a=ap.parse_args(); cache=a.work/'baseline_cache'; out=a.work/'observable'
    if a.mode=='baseline': baseline_mode(a.root,a.family,a.episode,cache)
    else: state_mode(a.root,a.family,a.episode,cache,out)

if __name__=='__main__': main()
