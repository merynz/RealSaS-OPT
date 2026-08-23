from __future__ import annotations

import argparse, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ALLOWED_RESOLUTIONS=(256,512,1024);VIEWS=8

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)

def _stage_image(src_view:Path,dst:Path,style:str,input_resolution:int):
    native=src_view/f"{style}.png"
    if input_resolution==1024:
        source=native
        with Image.open(source) as im:
            if im.size!=(1024,1024) or im.mode!="RGBA":raise RuntimeError(f"bad native image {source}: {im.mode} {im.size}")
            im.save(dst)
        return {"source":source.name,"transform":"identity","source_resolution":1024}
    deriv512=src_view/f"{style}_512.png"
    if input_resolution==512:
        with Image.open(native) as n:
            if n.size!=(1024,1024) or n.mode!="RGBA":raise RuntimeError(f"bad native authority {native}")
        with Image.open(deriv512) as im:
            if im.size!=(512,512) or im.mode!="RGBA":raise RuntimeError(f"bad 512 derivative {deriv512}: {im.mode} {im.size}")
            im.save(dst)
        return {"source":deriv512.name,"transform":"canonical_512_derivative","source_resolution":512}
    with Image.open(deriv512) as im:
        if im.size!=(512,512) or im.mode!="RGBA":raise RuntimeError(f"bad 512 derivative {deriv512}: {im.mode} {im.size}")
        im=im.resize((256,256),resample=Image.Resampling.BILINEAR);im.save(dst)
    return {"source":deriv512.name,"transform":"PIL_RGBA_BILINEAR_512_to_256","source_resolution":512}

def stage_asset(master_root:Path,out_root:Path,asset_id:str,input_resolution:int):
    if input_resolution not in ALLOWED_RESOLUTIONS:raise ValueError(input_resolution)
    src=master_root/"master"/"assets"/asset_id;final=out_root/"assets"/asset_id;marker=final/"STAGE.json"
    if marker.exists():
        m=json.load(open(marker,encoding="utf-8"))
        if m.get("asset_id")==asset_id and m.get("input_resolution")==input_resolution and m.get("physical_firewall") is True:return m
    final.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=asset_id+".partial.",dir=final.parent))
    try:
        with np.load(src/"primary_geometry.npz",allow_pickle=False) as z:vertices=np.asarray(z["vertices"],np.float32);faces=np.asarray(z["faces"],np.int32)
        np.savez(tmp/"primary_geometry.npz",vertices=vertices,faces=faces);views=[]
        for v in range(VIEWS):
            sv=src/"renders"/f"V{v}";dv=tmp/"renders"/f"V{v}";dv.mkdir(parents=True,exist_ok=True);shutil.copyfile(sv/"raster_authority.npz",dv/"raster_authority.npz");shutil.copyfile(sv/"camera.json",dv/"camera.json");cam=json.load(open(dv/"camera.json",encoding="utf-8"));styles={}
            for style in ("cel_clean","ink_cel"):styles[style]=_stage_image(sv,dv/f"{style}_input.png",style,input_resolution)
            views.append({"view":v,"yaw_deg":float(cam["yaw_deg"]),"styles":styles})
        yaws=np.asarray([x["yaw_deg"] for x in views],np.float32)
        if not np.allclose(yaws,np.arange(8,dtype=np.float32)*45.,atol=1e-4):raise RuntimeError(f"noncanonical yaws {yaws.tolist()}")
        with np.load(tmp/"primary_geometry.npz",allow_pickle=False) as z:
            if set(z.files)!={"vertices","faces"}:raise RuntimeError(f"firewall failed: {z.files}")
        meta={"schema":"RealSaS.IRISSinglePoseV2.Stage.v1","asset_id":asset_id,"input_resolution":input_resolution,"authority_resolution":1024,"physical_firewall":True,"geometry_fields":["vertices","faces"],"views":views};atomic_json(tmp/"STAGE.json",meta)
        if final.exists():shutil.rmtree(final)
        os.replace(tmp,final);return meta
    finally:
        if tmp.exists():shutil.rmtree(tmp,ignore_errors=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--seed-manifest",required=True);ap.add_argument("--out",required=True);ap.add_argument("--input-resolution",type=int,choices=ALLOWED_RESOLUTIONS,required=True);ap.add_argument("--splits",default="FIT,TUNE");ap.add_argument("--limit",type=int,default=0);a=ap.parse_args();root=Path(a.root);out=Path(a.out);seed=json.load(open(a.seed_manifest,encoding="utf-8"));splits={x.strip().upper() for x in a.splits.split(",") if x.strip()}
    if splits&{"CAL","DEV","EXTERNAL_HOLDOUT"}:raise RuntimeError("sealed split requested")
    rows=[r for r in seed["records"] if r["split"] in splits];rows=rows[:a.limit] if a.limit else rows
    for i,r in enumerate(rows,1):
        stage_asset(root,out,r["asset_id"],a.input_resolution)
        if i%5==0 or i==len(rows):print(f"[stage-v2] {i}/{len(rows)}",flush=True)
    atomic_json(out/"STAGE_MANIFEST.json",{"schema":"RealSaS.IRISSinglePoseV2.StageManifest.v1","input_resolution":a.input_resolution,"authority_resolution":1024,"records":[{"asset_id":r["asset_id"],"split":r["split"],"asset_dir":str(out/"assets"/r["asset_id"])} for r in rows]})
if __name__=="__main__":main()
