from __future__ import annotations
import json,tempfile,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from stage_source import stage_asset
from prepare_cache import build_asset

def make_synthetic_master(root:Path):
 aid="asset_synth";ar=root/"master"/"assets"/aid;(ar/"renders").mkdir(parents=True,exist_ok=True);vertices=np.asarray([[-.4,-.4,0.],[.4,-.4,0.],[0.,.4,0.]],np.float32);faces=np.asarray([[0,1,2]],np.int32);np.savez(ar/"primary_geometry.npz",vertices=vertices,faces=faces,parents=np.asarray([-1,0]),skin=np.ones((3,2),np.float32),bone_heads=np.zeros((2,3),np.float32));rng=np.random.default_rng(123);uv=[]
 while len(uv)<600:
  a,b=rng.random(2)
  if a+b<=1:uv.append((a,b))
 uv=np.asarray(uv,np.float32);w=np.stack([uv[:,0],uv[:,1],1-uv[:,0]-uv[:,1]],-1);p=(vertices[faces[0]][None]*w[:,:,None]).sum(1);res=1024;gx=p[:,0]/.5;gy=-p[:,1]/.5;x=np.floor((gx+1)*.5*res).astype(np.int64);y=np.floor((gy+1)*.5*res).astype(np.int64);pix=y*res+x;order=np.argsort(pix);pix=pix[order];uv=uv[order];_,first=np.unique(pix,return_index=True);pix=pix[first];uv=uv[first]
 for v in range(8):
  vd=ar/"renders"/f"V{v}";vd.mkdir(parents=True,exist_ok=True);np.savez(vd/"raster_authority.npz",resolution=np.asarray([res],np.int32),pixel_linear_index=pix,triangle_id=np.zeros(len(pix),np.int32),barycentric_uv=uv);(vd/"camera.json").write_text(json.dumps({"yaw_deg":float(v*45),"right":[1,0,0],"up":[0,1,0],"half_extent":.5}),encoding="utf-8")
  for style in ("cel_clean","ink_cel"):Image.new("RGBA",(1024,1024),(255,255,255,255)).save(vd/f"{style}.png");Image.new("RGBA",(512,512),(255,255,255,255)).save(vd/f"{style}_512.png")
 return aid

def main():
 root=Path(tempfile.mkdtemp(prefix="irisv2_data_preflight_"))
 try:
  master=root/"masterroot";stage=root/"stage";cache=root/"cache";aid=make_synthetic_master(master);meta=stage_asset(master,stage,aid,256)
  with np.load(stage/"assets"/aid/"primary_geometry.npz",allow_pickle=False) as z:assert set(z.files)=={"vertices","faces"},z.files
  cm=build_asset(stage,{"asset_id":aid,"split":"FIT","asset_dir":str(stage/"assets"/aid)},cache,geom_samples=64,anchors_per_view=64,max_tracks=128,radius_px=3,max_surface_error=.004)
  with np.load(cm["truth_path"],allow_pickle=False) as z:assert int(z["geom_mask"].sum())>0;assert len(z["track_p"])>0;assert int(z["track_visible"].sum())>=2
  print(json.dumps({"status":"PASS","physical_firewall":meta["physical_firewall"],"track_count":cm["track_count"],"geom_valid":cm["geom_valid"]},indent=2))
 finally:shutil.rmtree(root,ignore_errors=True)
if __name__=="__main__":main()
