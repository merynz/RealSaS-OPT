import importlib.util, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); P=ROOT/'surface_builder_e0_v1.py'
spec=importlib.util.spec_from_file_location('e0',P); e0=importlib.util.module_from_spec(spec); sys.modules['e0']=e0; spec.loader.exec_module(e0)


def test_camera_basis_contract():
    c0=e0.camera_for_view(0); c2=e0.camera_for_view(2)
    assert np.allclose(c0['right'],[1,0,0],atol=1e-7)
    assert np.allclose(c0['forward'],[0,1,0],atol=1e-7)
    assert np.allclose(c2['right'],[0,-1,0],atol=1e-7)
    assert np.allclose(c2['forward'],[1,0,0],atol=1e-7)


def test_pixel_grid_roundtrip():
    p=np.array([[0,0],[511,511],[1023,1023]],np.float32)
    g=e0.pixel_center_to_grid(p,1024)
    q=e0.grid_to_pixel_center(g,1024)
    assert np.max(np.abs(p-q))<1e-4


def test_derived_normal_uses_only_visible_p():
    R=9
    pix=np.arange(R*R,dtype=np.int64)
    yy,xx=np.divmod(pix,R)
    P=np.stack([xx,yy,np.zeros_like(xx)],axis=1).astype(np.float32)*0.01
    n,valid=e0.derive_view_local_normals(pix,P,R,stride_px=1)
    assert valid[4*R+4]
    assert abs(abs(float(n[4*R+4,2]))-1.0)<1e-6


def test_e0_b_api_has_no_teacher_identity_fields():
    names=e0.derived_match_row.__code__.co_varnames[:e0.derived_match_row.__code__.co_argcount]
    assert 'triangle_id' not in names and 'barycentric_uv' not in names and 'faces' not in names


def test_derived_match_on_consistent_observable_plane():
    R=33
    cam0=e0.camera_for_view(0); cam1=e0.camera_for_view(1)
    xs=np.linspace(-0.08,0.08,9); zs=np.linspace(-0.08,0.08,9)
    pts=np.array([[x,0.1,z] for z in zs for x in xs],np.float32)
    views=[]
    for v,cam in [(0,cam0),(1,cam1)]:
        g=e0.project_grid(pts,cam); xy=np.floor((g+1)*0.5*R).astype(int)
        ok=(xy[:,0]>=0)&(xy[:,0]<R)&(xy[:,1]>=0)&(xy[:,1]<R)
        pp=pts[ok]; xy=xy[ok]; pix=(xy[:,1]*R+xy[:,0]).astype(np.int64)
        order=np.argsort(pix); pix=pix[order]; pp=pp[order]
        _,ix=np.unique(pix,return_index=True); ix=np.sort(ix); pix=pix[ix]; pp=pp[ix]
        n=np.tile(np.array([0,1,0],np.float32),(len(pp),1)); valid=np.ones(len(pp),bool)
        views.append(e0.ObservableView(v,R,pix,pp,n,valid))
    src=views[0]; tgt=views[1]
    i=len(src.P)//2; ap=src.P[i]; an=src.N_derived[i]; ag=src.grid[i]
    row,info=e0.derived_match_row(ap,an,ag,0,src.half_extent,tgt,radius_px=5,max_common_frame_error=.03,max_reciprocal_error_px=5)
    assert row>=0,info
    assert np.linalg.norm(tgt.P[row]-ap)<.03


def test_e0_a_oracle_requires_teacher_triangle_identity():
    R=33
    p=np.array([[0.0,0.1,0.0]],np.float32)
    cam=e0.camera_for_view(1)
    g=e0.project_grid(p,cam)
    xy=np.floor((g+1)*.5*R).astype(int)
    pix=np.array([xy[0,1]*R+xy[0,0]],np.int64)
    n=np.array([[0,1,0]],np.float32); valid=np.array([True])
    obs=e0.ObservableView(1,R,pix,p.copy(),n,valid)
    same=e0.AuthorityView(obs,np.array([7],np.int64),np.array([[.2,.3]],np.float32))
    wrong=e0.AuthorityView(obs,np.array([8],np.int64),np.array([[.2,.3]],np.float32))
    row,_,_,_=e0.oracle_match_row(p[0],7,np.array([.2,.3],np.float32),same,radius_px=1,max_reprojection_error_px=2,max_surface_error=.01)
    assert row==0
    row,_,_,_=e0.oracle_match_row(p[0],7,np.array([.2,.3],np.float32),wrong,radius_px=1,max_reprojection_error_px=2,max_surface_error=.01)
    assert row==-1


def test_full_mesh_ceiling_is_area_sampled_and_deterministic():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    F=np.array([[0,1,2],[0,1,3]],np.int64)
    geom={'vertices':V,'faces':F}
    a=e0.deterministic_full_mesh_surface(geom,'asset_test',count=64)
    b=e0.deterministic_full_mesh_surface(geom,'asset_test',count=64)
    assert np.array_equal(a['P'],b['P'])
    assert a['P'].shape==(64,3) and a['N_geometric'].shape==(64,3)
    assert np.allclose(np.linalg.norm(a['N_geometric'],axis=1),1,atol=1e-6)


def test_full_vs_observable_detects_missing_surface_region():
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    F=np.array([[0,1,2],[0,1,3]],np.int64)
    geom={'vertices':V,'faces':F}
    obs=np.array([[0,0,0],[.5,0,0],[.25,.25,0],[0,.5,0]],np.float32)
    m=e0.full_vs_observable_distribution_metrics(geom,'asset_gap',obs,dense_reference_count=512)
    assert m['full_to_observable_nn_p95']>0.05
    assert m['consumer_native_normalization_applied'] is False


def test_recover_nondefault_half_extent_from_observable():
    R=1024; h=0.6172158837; view=0
    xs=np.linspace(-0.45,0.45,64,dtype=np.float32)
    zs=np.linspace(-0.35,0.35,64,dtype=np.float32)
    P=np.array([[x,0.1,z] for z in zs for x in xs],np.float32)
    g=e0.project_grid(P,e0.camera_for_view(view,h))
    xy=np.floor((g+1.0)*0.5*R).astype(np.int64)
    ok=(xy[:,0]>=0)&(xy[:,0]<R)&(xy[:,1]>=0)&(xy[:,1]<R)
    P=P[ok]; xy=xy[ok]
    pix=xy[:,1]*R+xy[:,0]
    order=np.argsort(pix,kind='stable'); pix=pix[order]; P=P[order]
    _,idx=np.unique(pix,return_index=True); idx=np.sort(idx); pix=pix[idx]; P=P[idx]
    # Make P exactly consistent with pixel-center authority in x/z so recovery has no raster quantization bias.
    grid=e0.pixel_center_to_grid(np.stack([pix%R,pix//R],axis=1),R)
    P[:,0]=grid[:,0]*h; P[:,2]=-grid[:,1]*h
    got,stats=e0.estimate_half_extent_from_observable(view,pix,P,R)
    assert abs(got-h)<1e-5
    assert stats['reprojection_px_p95']<1e-3
