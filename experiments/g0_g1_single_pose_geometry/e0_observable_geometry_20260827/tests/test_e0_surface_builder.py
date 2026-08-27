import importlib.util,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); P=ROOT/'surface_builder_e0_v1.py'
spec=importlib.util.spec_from_file_location('e0',P); e0=importlib.util.module_from_spec(spec); sys.modules['e0']=e0; spec.loader.exec_module(e0)

def test_camera_basis_contract():
    c0=e0.camera_for_view(0); c2=e0.camera_for_view(2)
    assert np.allclose(c0['right'],[1,0,0],atol=1e-7); assert np.allclose(c0['forward'],[0,1,0],atol=1e-7)
    assert np.allclose(c2['right'],[0,-1,0],atol=1e-7); assert np.allclose(c2['forward'],[1,0,0],atol=1e-7)

def test_pixel_grid_roundtrip():
    p=np.array([[0,0],[511,511],[1023,1023]],np.float32); g=e0.pixel_center_to_grid(p,1024); q=e0.grid_to_pixel_center(g,1024); assert np.max(np.abs(p-q))<1e-4

def test_derived_normal_uses_only_visible_p():
    R=9; pix=np.arange(R*R,dtype=np.int64); yy,xx=np.divmod(pix,R); P=np.stack([xx,yy,np.zeros_like(xx)],axis=1).astype(np.float32)*.01
    n,valid=e0.derive_view_local_normals(pix,P,R,stride_px=1); assert valid[4*R+4]; assert abs(abs(float(n[4*R+4,2]))-1.0)<1e-6

def test_e0_b_api_has_no_teacher_identity_fields():
    names=e0.derived_match_row.__code__.co_varnames[:e0.derived_match_row.__code__.co_argcount]
    assert 'triangle_id' not in names and 'barycentric_uv' not in names and 'faces' not in names

def test_derived_match_on_consistent_observable_plane():
    R=33; cam0=e0.camera_for_view(0); cam1=e0.camera_for_view(1); xs=np.linspace(-.08,.08,9); zs=np.linspace(-.08,.08,9); pts=np.array([[x,.1,z] for z in zs for x in xs],np.float32); views=[]
    for v,cam in [(0,cam0),(1,cam1)]:
        g=e0.project_grid(pts,cam); xy=np.floor((g+1)*.5*R).astype(int); ok=(xy[:,0]>=0)&(xy[:,0]<R)&(xy[:,1]>=0)&(xy[:,1]<R); pp=pts[ok]; xy=xy[ok]; pix=(xy[:,1]*R+xy[:,0]).astype(np.int64); order=np.argsort(pix); pix=pix[order]; pp=pp[order]; _,ix=np.unique(pix,return_index=True); ix=np.sort(ix); pix=pix[ix]; pp=pp[ix]; n=np.tile(np.array([0,1,0],np.float32),(len(pp),1)); valid=np.ones(len(pp),bool); views.append(e0.ObservableView(v,R,pix,pp,n,valid))
    src,tgt=views; i=len(src.P)//2; row,info=e0.derived_match_row(src.P[i],src.N_derived[i],src.grid[i],0,tgt,radius_px=5,max_common_frame_error=.03,max_reciprocal_error_px=5); assert row>=0,info; assert np.linalg.norm(tgt.P[row]-src.P[i])<.03
