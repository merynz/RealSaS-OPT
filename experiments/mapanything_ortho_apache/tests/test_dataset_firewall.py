import numpy as np
from pathlib import Path
from realsas_mapanything_ortho.dataset import load_target_npz


def test_target_copy_firewall(tmp_path:Path):
    n=4;p=tmp_path/'t.npz';np.savez(p,family_id=np.array(7),P_A=np.zeros((n,3),np.float32),N_A=np.tile([0,0,1],(n,1)).astype(np.float32),XY_A=np.zeros((8,n,2),np.float32),V_A=np.ones((8,n),np.uint8),direct_obs_A=np.ones((n,),np.uint8),joint_id=np.arange(n),skin_weight=np.ones((n,2),np.float32),P_B=np.ones((n,3),np.float32));t=load_target_npz(p);assert set(t)=={'family_id','P','N','XY01','V','direct_obs','xy_source_mode'};assert t['family_id']==7


def test_pixel256_xy_conversion(tmp_path:Path):
    n=2;xy=np.zeros((8,n,2),np.float32);xy[...,0]=255;xy[...,1]=127.5;p=tmp_path/'t.npz';np.savez(p,family_id=np.array(1),P_A=np.zeros((n,3),np.float32),N_A=np.tile([0,0,1],(n,1)).astype(np.float32),XY_A=xy,V_A=np.ones((8,n),np.uint8),direct_obs_A=np.ones((n,),np.uint8));t=load_target_npz(p,256);assert t['xy_source_mode']=='pixel256_center';assert abs(float(t['XY01'][0,0,0])-(255.5/256.0))<1e-6
