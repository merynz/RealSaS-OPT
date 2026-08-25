from __future__ import annotations
import json,shutil,tempfile
from pathlib import Path
import numpy as np
import prepare_cache as pc

def main():
 root=Path(tempfile.mkdtemp(prefix='irisv2_learner_cache_visibility_'));old=pc.choose_visible_correspondence
 try:
  stage=root/'stage';aid='asset_partial_visibility_synth';ar=stage/'assets'/aid;(ar/'renders').mkdir(parents=True)
  vertices=np.asarray([[-.4,-.4,0.],[.4,-.4,0.],[0.,.4,0.]],np.float32);faces=np.asarray([[0,1,2]],np.int32);np.savez(ar/'primary_geometry.npz',vertices=vertices,faces=faces)
  pix=np.arange(100,164,dtype=np.int64);u=np.linspace(.05,.40,len(pix),dtype=np.float32);uv=np.stack([u,u[::-1]],axis=1);uv*=np.minimum(1.,.8/np.maximum(uv.sum(1,keepdims=True),1e-6))
  for v in range(8):
   vd=ar/'renders'/f'V{v}';vd.mkdir(parents=True);np.savez(vd/'raster_authority.npz',resolution=np.asarray([1024],np.int32),pixel_linear_index=pix,triangle_id=np.zeros(len(pix),np.int32),barycentric_uv=uv);(vd/'camera.json').write_text(json.dumps({'yaw_deg':float(v*45),'right':[1.,0.,0.],'up':[0.,1.,0.],'half_extent':.5}),encoding='utf-8')
  (ar/'STAGE.json').write_text(json.dumps({'schema':'RealSaS.IRISSinglePoseV2.Stage.v2','asset_id':aid,'input_resolution':256,'authority_resolution':1024,'physical_firewall':True,'geometry_fields':['vertices','faces']}),encoding='utf-8')
  call={'view':0,'saw_partial':False}
  def partial(P,cam,ra,vertices_,faces_,vn,radius_px,max_surface_error):
   T=len(P);v=call['view'];call['view']+=1;idx=np.arange(T);ok=((idx+v)%3)!=0;call['saw_partial']|=bool(ok.any() and (~ok).any());g=np.zeros((T,2),np.float32);err=np.full(T,np.inf,np.float32);err[ok]=np.float32(.001+.00001*v);row=np.zeros(T,np.int64);return ok,g,err,row
  pc.choose_visible_correspondence=partial;out=root/'cache';meta=pc.build_asset(stage,{'asset_id':aid,'split':'FIT'},out,geom_samples=16,anchors_per_view=16,max_tracks=128,radius_px=3,max_surface_error=.003)
  assert call['saw_partial']
  with np.load(meta['truth_path'],allow_pickle=False) as z:
   vis=z['track_visible'].astype(bool);serr=z['track_surface_error'].astype(np.float32);assert vis.shape==serr.shape;assert vis.any() and (~vis).any();assert np.isfinite(serr[vis]).all();assert np.isinf(serr[~vis]).all()
  assert set(meta['input_dependencies']['builder_sha256'])=={'prepare_cache.py','geometry.py','coords.py'}
  print(json.dumps({'status':'PASS','genuine_partial_visibility':True,'visibility_error_shape_equal':True,'visible_error_finite':True,'hidden_error_inf':True,'coords_bound_to_cache_fingerprint':True,'optimizer_steps':0},indent=2))
 finally:
  pc.choose_visible_correspondence=old;shutil.rmtree(root,ignore_errors=True)
if __name__=='__main__':main()
