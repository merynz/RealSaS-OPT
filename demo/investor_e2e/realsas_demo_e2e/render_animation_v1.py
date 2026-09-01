from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np


def _norm_to_px(points: np.ndarray,w:int,h:int)->np.ndarray:
    out=np.empty_like(points,dtype=np.float32);out[:,0]=(points[:,0]+1.0)*0.5*w-0.5;out[:,1]=(points[:,1]+1.0)*0.5*h-0.5;return out


def _warp_triangle(src:np.ndarray,dst:np.ndarray,t_src:np.ndarray,t_dst:np.ndarray):
    rs=cv2.boundingRect(np.float32([t_src]));rd=cv2.boundingRect(np.float32([t_dst]));xs,ys,ws,hs=rs;xd,yd,wd,hd=rd
    if ws<=0 or hs<=0 or wd<=0 or hd<=0:return
    src_crop=src[ys:ys+hs,xs:xs+ws]
    if src_crop.size==0:return
    ts=np.float32([[p[0]-xs,p[1]-ys] for p in t_src]);td=np.float32([[p[0]-xd,p[1]-yd] for p in t_dst])
    M=cv2.getAffineTransform(ts,td);warped=cv2.warpAffine(src_crop,M,(wd,hd),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    mask=np.zeros((hd,wd),np.float32);cv2.fillConvexPoly(mask,np.int32(np.round(td)),1.0,lineType=cv2.LINE_AA)
    y0=max(0,yd);x0=max(0,xd);y1=min(dst.shape[0],yd+hd);x1=min(dst.shape[1],xd+wd)
    if y1<=y0 or x1<=x0:return
    wy0=y0-yd;wx0=x0-xd;wy1=wy0+(y1-y0);wx1=wx0+(x1-x0)
    patch=warped[wy0:wy1,wx0:wx1].astype(np.float32)/255.0;m=mask[wy0:wy1,wx0:wx1,None]
    a=patch[...,3:4]*m;dst_patch=dst[y0:y1,x0:x1].astype(np.float32)/255.0;oa=dst_patch[...,3:4]
    out_a=a+oa*(1-a);rgb=(patch[...,:3]*a+dst_patch[...,:3]*oa*(1-a))/np.maximum(out_a,1e-8)
    out=np.concatenate([rgb,out_a],axis=-1);dst[y0:y1,x0:x1]=np.clip(out*255,0,255).astype(np.uint8)


def render_textured_runtime_animation(
    *,image_only_npz:str|Path,runtime_request_json:str|Path,runtime_frames_json:str|Path,out_dir:str|Path,
    output_size:int=768,fps:int=24,
)->dict:
    data=np.load(image_only_npz,allow_pickle=False);rgba=np.asarray(data["rgba"])[0]
    if rgba.dtype!=np.uint8:rgba=np.clip(rgba*255.0,0,255).astype(np.uint8)
    req=json.loads(Path(runtime_request_json).read_text());run=json.loads(Path(runtime_frames_json).read_text())
    rest=np.asarray(req["mesh"]["vertices_xy"],np.float32);tris=np.asarray(req["mesh"]["triangles"],np.int32);src_pts=_norm_to_px(rest,rgba.shape[1],rgba.shape[0])
    out=Path(out_dir);frames_dir=out/"frames";frames_dir.mkdir(parents=True,exist_ok=True)
    video_path=out/"DEMO_MOTION_PREVIEW.mp4";writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(output_size,output_size))
    rendered=[]
    for fi,row in enumerate(run["frames"]):
        dest_norm=np.asarray(row["vertices_xy"],np.float32);dest_pts=_norm_to_px(dest_norm,output_size,output_size);canvas=np.zeros((output_size,output_size,4),np.uint8)
        # Sort by source triangle centroid y for deterministic painter order. One subject layer only.
        for tri in tris[np.argsort(src_pts[tris].mean(axis=1)[:,1])]:
            _warp_triangle(rgba,canvas,src_pts[tri],dest_pts[tri])
        fn=frames_dir/f"F{fi:04d}.png";cv2.imwrite(str(fn),cv2.cvtColor(canvas,cv2.COLOR_RGBA2BGRA));writer.write(cv2.cvtColor(canvas[...,:3],cv2.COLOR_RGB2BGR));rendered.append(str(fn))
    writer.release()
    report={"schema":"RealSaS.DemoTexturedRuntimeRender.v1","status":"PASS","frame_count":len(rendered),"fps":fps,"video":str(video_path),"authority":"VISUALIZATION_ONLY__RUNTIME_VERTICES_UNMODIFIED"}
    (out/"RENDER_REPORT_V1.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");return report
