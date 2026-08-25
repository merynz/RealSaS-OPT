from __future__ import annotations
import argparse, ast, importlib, json, sys
from pathlib import Path

LOCAL_PREFIXES=('model','coords','geometry','dataset','evaluate','prepare','pv5_','train_','run_','stage_')
REQUIRED={
    'train_pv5_r256_8x2_v2.py',
    'train_pv5_r256_8x2_v2_exact_cont_v1.py',
    'pv5_r256_8x2_v2_gpu_preflight.py',
    'pv5_r256_8x2_v2_exact_cont_cpu_preflight.py',
    'run_pv5_r256_8x2_v2_exact_cont_v1.py',
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--package-dir',default='.'); ap.add_argument('--out',required=True); a=ap.parse_args()
    root=Path(a.package_dir).resolve()
    py=sorted(root.glob('*.py'))
    names={p.stem for p in py}
    missing_required=sorted(x for x in REQUIRED if not (root/x).is_file())
    if missing_required: raise RuntimeError(f'missing required package files: {missing_required}')
    local_edges={}; missing_imports=[]
    for p in py:
        src=p.read_text(encoding='utf-8'); compile(src,str(p),'exec'); tree=ast.parse(src)
        edges=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                mods=[x.name for x in n.names]
            elif isinstance(n,ast.ImportFrom) and n.level==0 and n.module:
                mods=[n.module]
            else:
                continue
            for mod in mods:
                base=mod.split('.')[0]
                if base.startswith(LOCAL_PREFIXES):
                    edges.append(base)
                    if base not in names: missing_imports.append({'source':p.name,'module':base})
        local_edges[p.name]=sorted(set(edges))
    if missing_imports: raise RuntimeError(f'local dependency closure failure: {missing_imports}')
    sys.path.insert(0,str(root)); imported=[]
    for p in py:
        importlib.import_module(p.stem); imported.append(p.stem)
    rep={
        'schema':'RealSaS.PV5R256EightByTwoV2ExactContinuationPackagePreflight.v1',
        'status':'PASS',
        'python_files':len(py),
        'required_files_present':sorted(REQUIRED),
        'local_dependency_edges':local_edges,
        'imported_modules':imported,
        'scientific_optimizer_steps':0,
    }
    Path(a.out).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(rep,indent=2,sort_keys=True))

if __name__=='__main__': main()
