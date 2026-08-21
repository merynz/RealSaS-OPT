from __future__ import annotations
import importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np

EVAL=Path(__file__).resolve().parents[1]/'src/rank2_g2_truth_evaluator.py'
spec=importlib.util.spec_from_file_location('g2_truth_test',EVAL);evr=importlib.util.module_from_spec(spec);sys.modules[spec.name]=evr;spec.loader.exec_module(evr)

with tempfile.TemporaryDirectory() as td0:
    td=Path(td0);science=td/'science';science.mkdir()
    # Observable raster authority: two exact carriers. Serialized state P_A below is deliberately wrong.
    (science/'observable_phase.py').write_text('''\nimport numpy as np\nclass M:\n def project_points(self,P,v): return np.asarray(P,np.float32)[:,:2]\ndef _model_context(root,family,episode,route):\n m=M(); PA=np.array([[10.,20.,0.],[30.,40.,0.]],np.float32); VA=np.ones((8,2),np.uint8); out=np.array([0,1],np.int32); xy=np.stack([m.project_points(PA,v) for v in range(8)])\n return m,[],[],PA,VA,None,out,None,None,None,{},xy\n''')
    (science/'evaluation_phase.py').write_text('''\nimport numpy as np\ndef _truth(p):\n z=np.load(p,allow_pickle=False); return {k:z[k] for k in z.files}\ndef _map64(PA,t): return np.arange(len(PA),dtype=np.int64),np.zeros(len(PA),np.float32)\ndef _subset_truth(t,ids):\n out={}\n for k,v in t.items():\n  a=np.asarray(v)\n  if k=='V_B': out[k]=a[ids].T\n  elif a.ndim>0 and len(a)>=len(ids): out[k]=a[ids]\n  else: out[k]=v\n return out\ndef local_scale(P,k): return np.ones(len(P),np.float32)\n''')
    state=td/'1_e01_OBSERVABLE_STATE.npz';np.savez_compressed(state,P_A=np.full((2,3),999.,np.float32))
    state.with_suffix('.json').write_text(json.dumps({'truth_access':'NONE','route':'fake'}))
    PA=np.array([[10.,20.,0.],[30.,40.,0.]],np.float32);VA=np.ones((8,2),np.uint8);XY=np.stack([PA[:,:2] for _ in range(8)]).astype(np.float32)
    coords=np.empty((2,8,4,2),np.float32);scores=np.empty((2,8,4),np.float32);valid=np.ones((2,8,4),np.uint8)
    for i in range(2):
      for v in range(8):
       for k in range(4): coords[i,v,k]=XY[v,i]+np.array([k,0]);scores[i,v,k]=1-.1*k
    evidence=td/'e.npz';np.savez_compressed(evidence,coords=coords,scores=scores,valid=valid,V_A=VA,XY_A=XY,output_idx=np.array([0,1],np.int32))
    TB=PA.copy();TV=np.ones((2,8),np.uint8)
    side=td/'truth.npz';np.savez_compressed(side,P_A=PA,P_B=TB,V_B=TV)
    def selection(path,is_g2):
      views=[]
      for v in range(8):
       rows={};assigned={}
       for i in range(2):
        sites=[coords[i,v,k].astype(int).tolist() for k in range(4)];assigned[str(i)]=sites[0]
        rows[str(i)]={'sites':sites,'G_rank':[0.,1/3,2/3,1.]}
        if is_g2: rows[str(i)]['G_reciprocal_cycle_rank']=[0.,1/3,2/3,1.]
       views.append({'view':v,'visible_n':2,'K':2,'assigned':assigned,'abstained':[],'rows':rows})
      path.write_text(json.dumps({'family':1,'truth_access':'NONE','views':views}))
    g1=td/'g1.json';g2=td/'g2.json';selection(g1,False);selection(g2,True)
    r=evr.evaluate_family(td,science,state,evidence,g1,g2,side,1,'e01')
    assert r['mapping_reliable_n']==2
    assert len(r['rows'])==16
    assert all(x['g1_hit'] and x['g2_hit'] for x in r['rows'])
print('PASS: evaluator ignores deliberately corrupted serialized P_A and uses raster _model_context authority')
