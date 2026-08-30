from __future__ import annotations
from dataclasses import dataclass, field
import sys, types
import numpy as np
import torch
from contracts_v0_1 import VolumeState, stack_feature_vectors
from teacher_projection_v1 import project_skeleton_teacher_v1, bfs_training_order_v1, control_skin_column_map_v1
from substrate_adapter_v1 import surface_tokens_from_rigging_surface_v1, sample_interior_tokens_v1, path_state_features_v1, point_control_geometry_v1
from geppetto_g0_1 import GeppettoG01, GeppettoG01Config

@dataclass(frozen=True)
class Node:
    surface_id:str; P:tuple[float,float,float]; support_views:tuple[int,...]
    derived_normal:tuple[float,float,float]|None=None; metadata:dict=field(default_factory=dict)
@dataclass(frozen=True)
class Surface:
    surface_nodes:tuple[Node,...]; geometry_lineage_hash:str='SURF_TEST'
@dataclass(frozen=True)
class Vol:
    bounds_min:tuple; bounds_max:tuple; possible_interior:np.ndarray; strict_hull:np.ndarray
    certain_outside:np.ndarray; supported_surface:np.ndarray; high_confidence_interior:np.ndarray; unknown_concavity:np.ndarray
    lineage_sha256:str='VOL_TEST'
@dataclass(frozen=True)
class Interior:
    medialness:np.ndarray; local_thickness:np.ndarray; branch_likelihood:np.ndarray; lineage_sha256:str='INT_TEST'

def volume():
    sh=(7,7,7); poss=np.ones(sh,bool); out=np.zeros(sh,bool); out[3,3,3]=1; poss[3,3,3]=0; strict=poss.copy()
    surf=np.zeros(sh,bool); surf[0,:,:]=1; high=np.zeros(sh,bool); high[1:3,1:3,1:3]=1; unk=poss&~surf&~high
    med=np.zeros(sh,np.float32); med[2,2,2]=1; thick=np.ones(sh,np.float32)*.1; br=np.zeros(sh,np.float32); br[2,2,2]=.75
    return Vol((-1,-1,-1),(1,1,1),poss,strict,out,surf,high,unk),Interior(med,thick,br)

def test_foundation():
    H=np.asarray([[0,0,0],[0,0,.5],[0,0,1],[1,0,1]],np.float32); T=H+np.asarray([0,0,.4],np.float32)
    pr=project_skeleton_teacher_v1(H,T,np.asarray([-1,0,1,0]),np.asarray([1,0,1,1],bool))
    assert pr.bfs_control_ids==('TC:00000','TC:00002','TC:00003'); assert pr.control_by_id()['TC:00002'].parent_control_id=='TC:00000'
    assert control_skin_column_map_v1(pr)=={'TC:00000':0,'TC:00002':2,'TC:00003':3}; assert set(bfs_training_order_v1(pr,sibling_seed=5))==set(pr.bfs_control_ids)
    s=Surface((Node('a',(0,0,0),(0,2,4),(0,0,2)),Node('b',(1,0,0),(1,),None)))
    tok=surface_tokens_from_rigging_surface_v1(s,geometry_uncertainty_by_surface_id={'a':.02}); assert stack_feature_vectors(tok).shape==(2,27)
    assert tok[0].normal_valid and tok[0].geometry_uncertainty_valid and not tok[1].geometry_uncertainty_valid
    v,i=volume(); ints=sample_interior_tokens_v1(v,i,count=12,stream_id='TEST'); assert ints and all(t.volume_state not in {VolumeState.CERTAIN_OUTSIDE,VolumeState.SUPPORTED_SURFACE_BAND} for t in ints)
    f=path_state_features_v1((-1,-1,-1),(1,1,1),v,samples=43); assert f.certain_outside_fraction>0 and abs(sum(f.length_by_state)-f.total_length)<1e-6
    assert np.allclose(point_control_geometry_v1((1,0,0),(0,0,0),parent=None,normal=(1,0,0)),[1,0,0,0])

def inputs(B=2,N=17):
    g=torch.Generator().manual_seed(20260830); x=torch.randn(B,N,27,generator=g); x[...,:3]=torch.randn(B,N,3,generator=g)+torch.arange(N)[None,:,None]*.07
    m=torch.ones(B,N,dtype=torch.bool); m[1,-3:]=0; x[1,-3:]=0; t=torch.randn(B,6,3,generator=g); return x,m,t

def install_stub():
    from dataclasses import make_dataclass
    Joint=make_dataclass('Joint',[('proposal_id',str),('position',tuple),('root_score',float),('confidence',float),('support_surface_ids',tuple,field(default=())),('metadata',dict,field(default_factory=dict))],frozen=True)
    Edge=make_dataclass('Edge',[('edge_id',str),('parent_proposal_id',str),('child_proposal_id',str),('score',float),('confidence',float,field(default=1.)),('hard_required',bool,field(default=False)),('hard_forbidden',bool,field(default=False)),('reason',str,field(default='')),('metadata',dict,field(default_factory=dict))],frozen=True)
    Prop=make_dataclass('Prop',[('joints',tuple),('edges',tuple),('surface_binding_hash',str),('model_provenance',str,field(default='')),('schema_version',str,field(default='RealSaS.SkeletonProposalIR.v1')),('metadata',dict,field(default_factory=dict))],frozen=True)
    pkg=types.ModuleType('realsas_compiler_core'); mod=types.ModuleType('realsas_compiler_core.types'); mod.SkeletonProposalJoint=Joint; mod.SkeletonProposalEdge=Edge; mod.SkeletonProposalIR=Prop
    sys.modules['realsas_compiler_core']=pkg; sys.modules['realsas_compiler_core.types']=mod

def test_geppetto():
    torch.manual_seed(9); model=GeppettoG01(GeppettoG01Config(model_dim=40,knn_k=6,encoder_layers=2,decoder_layers=1,max_controls=8)).eval(); x,m,t=inputs()
    y=model(x,m,teacher_positions=t); assert y['positions'].shape==(2,6,3) and y['parent_logits'].shape==(2,6,6); assert torch.all(torch.diagonal(y['parent_logits'],dim1=1,dim2=2)<-1e8)
    perm=torch.tensor([5,0,9,2,11,1,15,3,7,4,13,6,16,10,8,12,14]); z=model(x[:,perm],m[:,perm],teacher_positions=t)
    for k in ('positions','position_log_sigma','exist_logits','root_logits','parent_logits'): assert torch.allclose(y[k],z[k],atol=2e-5,rtol=2e-5),k
    install_stub(); p=model.proposal_from_output(y,surface_binding_hash='SURFACE_SHA',active_count=4); assert len(p.joints)==4 and len(p.edges)==12 and all(not j.proposal_id.startswith('J:') for j in p.joints)

def main():
    test_foundation(); print('PASS foundation')
    test_geppetto(); print('PASS geppetto')
    print('GEPPETTO_ARACHNE_V0_1_SLICE_PASS')
if __name__=='__main__': main()
