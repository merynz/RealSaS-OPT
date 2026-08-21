from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve()
G2=HERE.parents[1]/'src/rank2_g2_reciprocal_cycle_proposal.py'
N1D=HERE.parents[3]
G1=N1D/'observable_geometry_closure_20260820/rank2_g1_geometry_grounded_proposal.py'
V5=N1D/'rank2_global_relational_world_20260820/rank2_r_v5_full_pairwise_proposal.py'

def load(name,p):
    s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m
    assert s.loader is not None
    s.loader.exec_module(m);return m

v5=load('v5_test',V5);g1=load('g1_test',G1);g2=load('g2_test',G2)
assert g2.sha256_file(V5)==g2.EXPECTED_V5_SHA256

rng=np.random.default_rng(20260821)
n=7
coords=np.full((n,8,4,2),np.nan,np.float32);scores=np.full((n,8,4),np.nan,np.float32);valid=np.ones((n,8,4),np.uint8)
VA=np.ones((8,n),np.uint8)
XYA=np.empty((8,n,2),np.float32)
cycle=np.empty((n,8,4),np.float32);rxy=np.empty((n,8,4,2),np.float32);rscore=np.empty((n,8,4),np.float32)
for i in range(n):
    P=np.array([-.35+.1*i,-.2+.05*i,-.15+.04*i])
    for v in range(8):
        qa=g2.project_points_np(P[None],v)[0];XYA[v,i]=qa
        for k in range(4):
            q=qa+np.array([k*2.0,(k%2)*1.5])
            coords[i,v,k]=q;scores[i,v,k]=1.0-.1*k-.001*i
            cycle[i,v,k]=float(k+0.01*i);rxy[i,v,k]=qa+np.array([cycle[i,v,k],0.]);rscore[i,v,k]=.9-.1*k

for view in range(8):
    p0=v5.build_view_problem(777,coords,scores,valid,VA,XYA,view)
    p1=g1.build_view_problem(777,coords,scores,valid,VA,XYA,view)
    p2=g2.build_view_problem(v5,777,coords,scores,valid,VA,XYA,cycle,rxy,rscore,view)
    assert p0.carriers==p1.carriers==p2.carriers
    assert p0.sites==p1.sites==p2.sites
    assert p0.feasible==p1.feasible==p2.feasible
    assert p0.pair_r.keys()==p1.pair_r.keys()==p2.pair_r.keys()
    for key in p0.pair_r:
        assert p0.pair_r[key].keys()==p1.pair_r[key].keys()==p2.pair_r[key].keys()
        for pair in p0.pair_r[key]:
            assert p0.pair_r[key][pair]==p1.pair_r[key][pair]==p2.pair_r[key][pair]
    for i in p2.carriers:
        r1=p1.rows_serial[str(i)];r2=p2.rows_serial[str(i)]
        assert r1['sites']==r2['sites']
        assert np.array_equal(np.asarray(r1['G_descriptor_rank']),np.asarray(r2['G_descriptor_rank']))
        assert np.array_equal(np.asarray(r1['G_geometry_raw_reproj_px']),np.asarray(r2['G_geometry_raw_reproj_px']))
        assert np.array_equal(np.asarray(r1['G_geometry_rank']),np.asarray(r2['G_geometry_rank']))
        d=np.asarray(r2['G_descriptor_rank']);gm=np.asarray(r2['G_geometry_rank']);c=np.asarray(r2['G_reciprocal_cycle_rank']);f=np.asarray(r2['G_rank'])
        assert np.allclose(f,(d+gm+c)/3.,rtol=0,atol=1e-15)
        assert np.all((f>=0)&(f<=1))
        assert r2['G_weights']=={'descriptor':1/3,'geometry':1/3,'reciprocal_cycle':1/3}
print('PASS: exact G1 geometry + exact V5 R/support preserved; only equal-third reciprocal-cycle G added')
