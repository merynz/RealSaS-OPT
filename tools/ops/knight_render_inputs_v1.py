"""Read and measure pinned render inputs; no pipeline execution or qualification."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
SOURCE_REPORT = Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json")

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def stage_row(ledger,sid):
    rows=[r for r in ledger['stages'] if r['id']==sid]
    if len(rows)!=1: raise RuntimeError(f'STAGE_ID_CARDINALITY::{sid}::{len(rows)}')
    return rows[0]

def stage_payload(ledger,sid,schema):
    outputs=[o for o in stage_row(ledger,sid).get('outputs',[]) if o.get('schema')==schema]
    if len(outputs)!=1: raise RuntimeError(f'STAGE_SCHEMA_CARDINALITY::{sid}::{schema}::{len(outputs)}')
    out=outputs[0]
    p=Path(out['path'])
    if not p.is_file() or sha256(p)!=str(out['sha256']): raise RuntimeError(f'STAGE_BYTES_DRIFT::{sid}::{schema}')
    return json.loads(p.read_text()),p

def mechanical_faces(candidate):
    ids=[str(v.candidate_vertex_id) for v in candidate.vertices]
    idx={x:i for i,x in enumerate(ids)}
    if len(idx)!=len(ids): raise RuntimeError('DUPLICATE_CARRIER_VERTEX_ID')
    return np.asarray([[idx[str(x)] for x in f] for f in candidate.faces],dtype=np.int64)

def motion_payloads(input_root:Path):
    found={}
    for p in input_root.rglob('*.json'):
        try: x=json.loads(p.read_text())
        except Exception: continue
        if x.get('schema')!='RealSaS.MotionSourceClip.v2': continue
        cid=str(x.get('clip_id') or '')
        if cid in {'demo_idle_v1','demo_run_v1','demo_slash_v1'}:
            if cid in found:
                raise RuntimeError('MOTION_CLIP_ID_AMBIGUOUS::' + cid)
            found[cid]=(x,p)
    if set(found)!={'demo_idle_v1','demo_run_v1','demo_slash_v1'}:
        raise RuntimeError(f'MOTION_SET_INCOMPLETE::{sorted(found)}')
    return found

def visual_metrics(rest,posed,faces):
    r=np.asarray(rest,float)[np.asarray(faces,np.int64)]
    p=np.asarray(posed,float)[np.asarray(faces,np.int64)]
    def area(t):
        a=t[:,1]-t[:,0]; b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=area(r); pa=area(p)
    rl=np.stack((np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)),axis=1)
    pl=np.stack((np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)),axis=1)
    q=pl/np.maximum(rl,1e-9)
    return {'flipped_triangles':int(np.count_nonzero(ra*pa<0)),'p95_edge_ratio':float(np.quantile(q,.95)),'p99_edge_ratio':float(np.quantile(q,.99)),'max_edge_ratio':float(np.max(q))}

def compose_grid(images,labels):
    if len(images)!=8: raise ValueError('need 8 views')
    w,h=images[0].size
    canvas=Image.new('RGBA',(w*4,h*2),(22,22,22,255)); draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)):
        x=(i%4)*w; y=(i//4)*h
        canvas.alpha_composite(im,(x,y)); draw.text((x+6,y+6),label,fill=(255,255,255,255),stroke_width=1,stroke_fill=(0,0,0,255))
    return canvas
