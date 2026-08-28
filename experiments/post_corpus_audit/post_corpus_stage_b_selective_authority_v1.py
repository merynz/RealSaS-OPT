#!/usr/bin/env python3
"""RealSaS Stage-B selective Blender authority/appearance audit.
Requires Stage-A sample manifests. Read-only source inspection; no rerender/training.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, subprocess, tarfile
from pathlib import Path
import requests

ROOT_DEFAULT="/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3"
BLENDER_VERSION="5.2.0"
RELEASE_DIR="Blender5.2"
PROBE_NAME="realsas_blender_source_probe_v1.py"

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def ensure_blender():
    home=Path(f"/content/blender-{BLENDER_VERSION}-linux-x64"); bin=home/"blender"
    if bin.exists(): return bin
    work=Path("/content/realsas_blender_bootstrap"); work.mkdir(exist_ok=True)
    arc=work/f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"; sha=work/f"blender-{BLENDER_VERSION}.sha256"; bases=[f"https://mirror.blender.org/release/{RELEASE_DIR}",f"https://download.blender.org/release/{RELEASE_DIR}"]
    def get(urls,dst,minbytes=1):
        errs=[]
        for u in urls:
            try:
                with requests.get(u,stream=True,timeout=(30,600),allow_redirects=True) as r:
                    r.raise_for_status(); tmp=dst.with_suffix(dst.suffix+".part")
                    with open(tmp,"wb") as f:
                        for c in r.iter_content(8<<20):
                            if c: f.write(c)
                    if tmp.stat().st_size<minbytes: raise RuntimeError("too small")
                    os.replace(tmp,dst); return u
            except Exception as e: errs.append(str(e))
        raise RuntimeError("download failed: "+" | ".join(errs[-4:]))
    get([f"{b}/{sha.name}" for b in bases],sha,64); expected=None
    for line in sha.read_text(errors="ignore").splitlines():
        parts=line.split()
        if len(parts)>=2 and Path(parts[-1].lstrip("*")).name==arc.name and re.fullmatch(r"[0-9a-fA-F]{64}",parts[0]): expected=parts[0].lower(); break
    if not expected: raise RuntimeError("could not resolve Blender checksum")
    if not arc.exists() or sha256_file(arc)!=expected:
        get([f"{b}/{arc.name}" for b in bases],arc,100_000_000); got=sha256_file(arc)
        if got!=expected: raise RuntimeError(f"Blender SHA mismatch {got} != {expected}")
    with tarfile.open(arc,"r:xz") as tf: tf.extractall("/content")
    if not bin.exists(): raise RuntimeError("Blender binary missing")
    bin.chmod(bin.stat().st_mode|0o111); ver=subprocess.check_output([str(bin),"--version"],text=True,timeout=60)
    if f"Blender {BLENDER_VERSION}" not in ver: raise RuntimeError("wrong Blender version: "+ver.splitlines()[0])
    return bin

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=ROOT_DEFAULT); ap.add_argument("--workers",type=int,default=2); ap.add_argument("--timeout",type=int,default=240); args=ap.parse_args()
    root=Path(args.root); outdir=root/"reports"/"post_corpus_audit"; local=Path("/content/realsas_stageb_probe"); local.mkdir(exist_ok=True)
    blend_manifest=outdir/"BLEND_EVALUATED_MESH_AUDIT_SAMPLE_V1.json"; app_manifest=outdir/"APPEARANCE_RECOVERABILITY_SAMPLE_V1.json"
    if not blend_manifest.exists() or not app_manifest.exists(): raise RuntimeError("Run post_corpus_stage_a_closure_v2.py first")
    rows=json.loads(blend_manifest.read_text())+json.loads(app_manifest.read_text()); uniq={x["asset"]:x for x in rows if x.get("raw_path")}; rows=list(uniq.values()); print(f"[stageB] unique sources={len(rows)}",flush=True)
    blender=ensure_blender(); probe=outdir/PROBE_NAME
    if not probe.exists(): raise RuntimeError(f"missing probe {probe}")
    def run_one(x):
        aid=x["asset"]; src=root/x["raw_path"]; out=local/f"{aid}.json"
        if not src.exists(): return {"asset":aid,"raw_path":x["raw_path"],"driver_error":"raw source missing on mounted Drive"}
        try:
            cp=subprocess.run([str(blender),"-b","--factory-startup","--python",str(probe),"--",str(src),str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout)
            if cp.returncode!=0 or not out.exists(): return {"asset":aid,"raw_path":x["raw_path"],"driver_error":f"blender rc={cp.returncode}","stdout_tail":cp.stdout[-3000:]}
            obj=json.loads(out.read_text()); obj["asset"]=aid; obj["raw_path"]=x["raw_path"]; return obj
        except Exception as e: return {"asset":aid,"raw_path":x["raw_path"],"driver_error":f"{type(e).__name__}: {e}"}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futs=[ex.submit(run_one,x) for x in rows]
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            results.append(f.result())
            if i%10==0 or i==len(futs): print(f"[stageB] {i}/{len(futs)}",flush=True)
    ok=[x for x in results if not x.get("driver_error") and not x.get("errors")]; blend=[x for x in ok if x.get("extension")==".blend"]
    def n(cond,seq=ok): return sum(bool(cond(x)) for x in seq)
    normal_meds=[float(x["normal_authority_summary"]["median_of_object_medians_deg"]) for x in ok if x.get("normal_authority_summary")]
    import numpy as np
    summary={"schema":"RealSaS.PostCorpus.StageBSelectiveAuthority.v1","sample_requested":len(rows),"sample_ok":len(ok),"driver_or_probe_failures":len(results)-len(ok),"blend_sample_ok":len(blend),"blend_assets_with_evaluated_geometry_diff":n(lambda x:x.get("evaluated_diff_object_count",0)>0,blend),"blend_assets_with_builder_fan_vs_blender_triangulation_mismatch":n(lambda x:x.get("triangulation_mismatch_object_count",0)>0,blend),"blend_assets_with_modifiers":n(lambda x:x.get("modifier_object_count",0)>0,blend),"blend_assets_with_shape_keys":n(lambda x:x.get("shape_key_object_count",0)>0,blend),"appearance_assets_with_materials":n(lambda x:(x.get("appearance") or {}).get("material_count",0)>0,ok),"appearance_assets_with_image_nodes":n(lambda x:(x.get("appearance") or {}).get("image_node_reference_count",0)>0,ok),"appearance_assets_with_packed_images":n(lambda x:(x.get("appearance") or {}).get("packed_image_count",0)>0,ok),"appearance_assets_with_missing_external_images":n(lambda x:(x.get("appearance") or {}).get("external_image_missing_count",0)>0,ok),"normal_builder_vs_corner_object_median_deg":{"count":len(normal_meds),"median":float(np.median(normal_meds)) if normal_meds else None,"p90":float(np.quantile(normal_meds,.9)) if normal_meds else None},"results":results,"corpus_mutation":False}
    j=outdir/"POST_CORPUS_STAGE_B_SELECTIVE_AUTHORITY_RESULT_V1.json"; j.write_text(json.dumps(summary,indent=2),encoding="utf-8"); md=outdir/"POST_CORPUS_STAGE_B_SELECTIVE_AUTHORITY_REPORT_V1.md"; md.write_text("\n".join(["# Stage-B Selective Authority / Appearance Audit","",f"- sample ok: **{len(ok)}/{len(rows)}**",f"- blend sample ok: **{len(blend)}**",f"- evaluated geometry diff: **{summary['blend_assets_with_evaluated_geometry_diff']}**",f"- fan-vs-Blender triangulation mismatch: **{summary['blend_assets_with_builder_fan_vs_blender_triangulation_mismatch']}**",f"- materials present: **{summary['appearance_assets_with_materials']}**",f"- image nodes present: **{summary['appearance_assets_with_image_nodes']}**",f"- packed images present: **{summary['appearance_assets_with_packed_images']}**",f"- missing external images: **{summary['appearance_assets_with_missing_external_images']}**",""])+"\n",encoding="utf-8"); print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2),flush=True)

if __name__=="__main__": main()
