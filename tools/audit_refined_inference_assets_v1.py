"""Inspect the sealed model inputs/checkpoints before a fresh inference attempt."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    report={'python':sys.executable,'lanes':{},'inference_executed':False}
    torch=None
    if importlib.util.find_spec('torch'):
        import torch
        report['torch']={'version':torch.__version__,'cuda_available':torch.cuda.is_available()}
        if torch.cuda.is_available():report['torch']['gpu']=torch.cuda.get_device_name(0)
    for lane,pre,exe in [('geppetto','26_GEPPETTO_FIT_PREREGISTERED','27_GEPPETTO_FIT'),('arachne','30_ARACHNE_FIT_PREREGISTERED','31_ARACHNE_FIT')]:
        row={};report['lanes'][lane]=row
        for name,stage,file in [('prereg',pre,'model_fit_preregistration.json'),('execution',exe,'model_fit_execution.json')]:
            path=args.run_root/'artifacts'/stage/file
            if not path.is_file():row[name]={'missing':str(path)};continue
            payload=json.loads(path.read_text());row[name]=payload;shutil.copyfile(path,out/(lane+'_'+name+'.json'))
        prereg=row.get('prereg',{});execution=row.get('execution',{})
        source=Path(prereg.get('model_source_path','/nonexistent'))
        if source.is_file():
            digest=sha(source);row['source_verified']=digest==prereg.get('model_source_sha256')
            if row['source_verified']:shutil.copyfile(source,out/(lane+'_sealed_source.py'))
        checkpoint=Path(execution.get('checkpoint_path','/nonexistent'))
        row['checkpoint_exists']=checkpoint.is_file()
        if checkpoint.is_file():
            row['checkpoint_bytes']=checkpoint.stat().st_size;row['checkpoint_verified']=sha(checkpoint)==execution.get('checkpoint_sha256')
            if torch is not None and row['checkpoint_verified']:
                try:
                    data=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
                    def describe(x,depth=0):
                        if isinstance(x,torch.Tensor):return {'tensor_shape':list(x.shape),'dtype':str(x.dtype)}
                        if isinstance(x,dict):return {str(k):describe(v,depth+1) for k,v in x.items()}
                        if isinstance(x,(list,tuple)):return [describe(v,depth+1) for v in x[:50]]
                        if x is None or isinstance(x,(str,int,float,bool)):return x
                        return {'type':type(x).__name__}
                    row['checkpoint_structure']=describe(data);del data
                except Exception as exc:row['checkpoint_inspection_error']=type(exc).__name__+':'+str(exc)
    (out/'REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'lanes':{k:{f:v.get(f) for f in ['checkpoint_exists','checkpoint_verified','checkpoint_bytes','checkpoint_inspection_error','source_verified']} for k,v in report['lanes'].items()},'torch':report.get('torch')}),flush=True)


if __name__=='__main__':main()
