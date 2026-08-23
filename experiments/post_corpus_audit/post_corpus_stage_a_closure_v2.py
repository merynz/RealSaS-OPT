#!/usr/bin/env python3
"""RealSaS post-corpus Stage-A closure.

Read-only with respect to corpus evidence. Closes/characterizes remaining
Gate-0/1/2/3 metadata/apparatus questions using Google Drive API pagination,
and downloads only small JSONs / suspicious tiny raster authorities.

Does NOT render, train, mutate corpus assets, resume consumer exports, or open
sealed model panels.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, io, json
from pathlib import Path
import numpy as np
import requests
from google.colab import auth
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build

ROOT_DEFAULT="/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3"
ASSETS_PARENT_ID="1zMz3eg1x0yjU2wYg0lYWXK1AlgxOZLTW"
VARIANTS_PARENT_ID="1mZX14XTspqLSAc5j4AzpuJGTzGIgIQwH"
SOURCE_PARENT_IDS={
    "objaverse_animated_originals":"1jWijYGy5pzNn2oEAR2XcZpn4cWn9w1d3",
    "quaternius_cc0":"1MCzscvA24tUdSiwolyWm8G4mykaJw-ny",
    "kaykit_cc0":"1n5KQ5bkbLgm6eA2eROlkn0v9hosKLNu0",
    "khronos_gltf_qa":"1htgo579RT_EKi4CeZ2r4au0_78GMSmc5",
}
RENDER_AFTER="2026-08-23T00:00:00Z"
FOLDER_MIME="application/vnd.google-apps.folder"

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def auth_drive():
    auth.authenticate_user()
    creds,_=google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    if not creds.valid: creds.refresh(GoogleAuthRequest())
    svc=build("drive","v3",credentials=creds,cache_discovery=False)
    return svc,creds

def list_all(svc,q,fields="id,name,mimeType,size,parents,modifiedTime",label=None):
    out=[]; token=None; pages=0
    while True:
        resp=svc.files().list(q=q,spaces="drive",fields=f"nextPageToken,files({fields})",pageSize=1000,pageToken=token,supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
        pages+=1; out.extend(resp.get("files",[])); token=resp.get("nextPageToken")
        if not token: break
    if label: print(f"[api] {label}: rows={len(out)} pages={pages}",flush=True)
    return out,pages

def hits_by_parent(rows, valid):
    d=collections.defaultdict(list)
    for r in rows:
        for p in r.get("parents") or []:
            if p in valid: d[p].append(r)
    return d

def exactly_one_map(hits, parent_ids):
    missing=[]; dup=[]; one={}
    for p in parent_ids:
        rs=hits.get(p,[])
        if len(rs)==1: one[p]=rs[0]
        elif len(rs)==0: missing.append(p)
        else: dup.append((p,[x["id"] for x in rs]))
    return one,missing,dup

def media_get(file_id, token, timeout=90):
    u=f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true"
    r=requests.get(u,headers={"Authorization":f"Bearer {token}"},timeout=timeout); r.raise_for_status(); return r.content

def parallel_json(rows, token, workers=24):
    def one(r):
        try: return r,json.loads(media_get(r["id"],token).decode("utf-8")),None
        except Exception as e: return r,None,f"{type(e).__name__}: {e}"
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(one,r) for r in rows]
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            out.append(f.result())
            if i%500==0 or i==len(futs): print(f"[download-json] {i}/{len(futs)}",flush=True)
    return out

def ext_class(ext):
    e=ext.lower()
    if e==".glb": return "GLB_CONTAINER__EMBEDDED_APPEARANCE_POSSIBLE"
    if e==".gltf": return "GLTF__EXTERNAL_DEPENDENCY_HIGH"
    if e==".blend": return "BLEND__PACKED_OR_EXTERNAL_UNKNOWN__EVALUATED_MESH_RISK"
    if e==".fbx": return "FBX__EMBEDDED_OR_EXTERNAL_UNKNOWN"
    if e==".obj": return "OBJ__MTL_TEXTURE_EXTERNAL_RISK"
    if e in {".usd",".usda",".usdc",".usdz"}: return "USD__DEPENDENCY_STATE_REQUIRES_INSPECTION"
    if e==".dae": return "DAE__EXTERNAL_TEXTURE_RISK"
    return "OTHER__INSPECT"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=ROOT_DEFAULT); ap.add_argument("--tiny-authority-bytes",type=int,default=8192); ap.add_argument("--workers",type=int,default=24); args=ap.parse_args()
    root=Path(args.root); outdir=root/"reports"/"post_corpus_audit"; outdir.mkdir(parents=True,exist_ok=True)
    selraw=read_json(root/"metadata"/"CANONICAL_VARIANT_SELECTION.json")
    selected=selraw["selected"] if isinstance(selraw,dict) and isinstance(selraw.get("selected"),list) else (list(selraw.values()) if isinstance(selraw,dict) else selraw)
    selected_by_aid={x["canonical_asset_id"]:x for x in selected}; selected_ids=set(selected_by_aid); print(f"[central] selected={len(selected_ids)}",flush=True)
    svc,creds=auth_drive(); token=creds.token; pages=0
    asset_rows,p=list_all(svc,f"'{ASSETS_PARENT_ID}' in parents and mimeType='{FOLDER_MIME}' and trashed=false",label="asset folders"); pages+=p
    by_name=collections.defaultdict(list)
    for r in asset_rows: by_name[r["name"]].append(r)
    asset_row_by_aid={a:by_name[a][0] for a in selected_ids if len(by_name.get(a,[]))==1}; asset_folder_ids={r["id"] for r in asset_row_by_aid.values()}; asset_folder_to_aid={r["id"]:aid for aid,r in asset_row_by_aid.items()}
    geom_rows,p=list_all(svc,"name='primary_geometry.npz' and trashed=false",label="all primary_geometry"); pages+=p
    geom_hits=hits_by_parent(geom_rows,asset_folder_ids); geom_one,geom_missing,geom_dup=exactly_one_map(geom_hits,asset_folder_ids); geom_zero=[r for r in geom_one.values() if int(r.get("size") or 0)<=0]
    print(f"[gate1] primary_geometry present={len(geom_one)}/{len(asset_folder_ids)} missing={len(geom_missing)} dup={len(geom_dup)} zero={len(geom_zero)}",flush=True)
    renders_rows,p=list_all(svc,f"name='renders' and mimeType='{FOLDER_MIME}' and trashed=false and modifiedTime > '{RENDER_AFTER}'",label="renders"); pages+=p
    render_hits=hits_by_parent(renders_rows,asset_folder_ids); render_one,render_missing,render_dup=exactly_one_map(render_hits,asset_folder_ids); render_ids=set(r["id"] for r in render_one.values()); render_to_aid={r["id"]:asset_folder_to_aid[parent] for parent,r in render_one.items()}
    view_to_asset={}; view_meta={}
    for vi in range(8):
        vn=f"V{vi}"; rows,p=list_all(svc,f"name='{vn}' and mimeType='{FOLDER_MIME}' and trashed=false and modifiedTime > '{RENDER_AFTER}'"); pages+=p; h=hits_by_parent(rows,render_ids); one,mis,dup=exactly_one_map(h,render_ids); view_meta[vn]={"present":len(one),"missing":len(mis),"duplicates":len(dup)}
        for render_parent,r in one.items(): view_to_asset[r["id"]]=(render_to_aid[render_parent],vi)
    view_ids=set(view_to_asset); print(f"[gate1] view folders={len(view_ids)}/{len(selected_ids)*8}",flush=True)
    auth_rows,p=list_all(svc,f"name='raster_authority.npz' and trashed=false and modifiedTime > '{RENDER_AFTER}'",label="raster authorities"); pages+=p
    auth_hits=hits_by_parent(auth_rows,view_ids); auth_one,auth_missing,auth_dup=exactly_one_map(auth_hits,view_ids); tiny=[(parent,r) for parent,r in auth_one.items() if int(r.get("size") or 0)<=args.tiny_authority_bytes]
    print(f"[gate1] authority present={len(auth_one)}/{len(view_ids)}; tiny<={args.tiny_authority_bytes}B: {len(tiny)}",flush=True)
    def inspect_npz(item):
        parent,r=item; aid,vi=view_to_asset[parent]
        try:
            with np.load(io.BytesIO(media_get(r["id"],token)),allow_pickle=False) as z: n=int(len(z["pixel_linear_index"]))
            return {"asset":aid,"view":vi,"file_id":r["id"],"size":int(r.get("size") or 0),"authority_rows":n,"error":None}
        except Exception as e: return {"asset":aid,"view":vi,"file_id":r["id"],"size":int(r.get("size") or 0),"authority_rows":None,"error":f"{type(e).__name__}: {e}"}
    tiny_results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers,24)) as ex:
        futs=[ex.submit(inspect_npz,x) for x in tiny]
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            tiny_results.append(f.result())
            if i%100==0 or i==len(futs): print(f"[tiny-authority] {i}/{len(futs)}",flush=True)
    per_asset={aid:{"zero_views":[],"tiny_nonzero_views":[],"inspection_errors":[]} for aid in selected_ids}
    for x in tiny_results:
        if x["error"]: per_asset[x["asset"]]["inspection_errors"].append(x)
        elif x["authority_rows"]==0: per_asset[x["asset"]]["zero_views"].append(x["view"])
        else: per_asset[x["asset"]]["tiny_nonzero_views"].append({"view":x["view"],"rows":x["authority_rows"],"size":x["size"]})
    all8_blank=[aid for aid,x in per_asset.items() if len(x["zero_views"])==8]; any_blank=[aid for aid,x in per_asset.items() if x["zero_views"]]; print(f"[gate1] exact zero-authority: any-view assets={len(any_blank)} all-8 assets={len(all8_blank)}",flush=True)
    variant_rows,p=list_all(svc,f"'{VARIANTS_PARENT_ID}' in parents and mimeType='{FOLDER_MIME}' and trashed=false",label="variant folders"); pages+=p
    variant_by_name=collections.defaultdict(list)
    for r in variant_rows: variant_by_name[r["name"]].append(r)
    selected_variant_names={Path(str(x["variant_dir"])).name:x["canonical_asset_id"] for x in selected}; variant_folder_to_aid={}; variant_missing=[]
    for name,aid in selected_variant_names.items():
        rs=variant_by_name.get(name,[])
        if len(rs)==1: variant_folder_to_aid[rs[0]["id"]]=aid
        else: variant_missing.append({"asset":aid,"variant":name,"count":len(rs)})
    variant_ids=set(variant_folder_to_aid); small_json={}; json_errors={}
    for fn in ["admission.json","technical_audit.json"]:
        rows,p=list_all(svc,f"name='{fn}' and trashed=false",label=fn); pages+=p; h=hits_by_parent(rows,variant_ids); one,mis,dup=exactly_one_map(h,variant_ids); downloaded=parallel_json(list(one.values()),token,args.workers); d={}; errs=[]
        for meta,obj,err in downloaded:
            parent=(meta.get("parents") or [None])[0]; aid=variant_folder_to_aid.get(parent)
            if aid is None: continue
            if err: errs.append({"asset":aid,"error":err})
            else: d[aid]=obj
        small_json[fn]=d; json_errors[fn]={"missing":len(mis),"duplicates":len(dup),"download_errors":errs[:50],"download_error_count":len(errs)}; print(f"[gate2/3] {fn}: parsed={len(d)}/{len(selected_ids)} missing={len(mis)} dlerr={len(errs)}",flush=True)
    admissions=small_json["admission.json"]; tech=small_json["technical_audit.json"]
    ext_counts=collections.Counter(); appearance_class_counts=collections.Counter(); sha_to_assets=collections.defaultdict(list); expected_source_basenames=collections.defaultdict(dict); source_meta=[]
    for aid,a in admissions.items():
        sid=a.get("source_identity") or {}; raw=str(sid.get("preserved_raw_source_path") or ""); ext=Path(raw).suffix.lower(); ext_counts[ext or "<none>"]+=1; appearance_class_counts[ext_class(ext)]+=1; sha=str(sid.get("source_sha256") or "")
        if sha: sha_to_assets[sha].append(aid)
        provider=a.get("source_registry_id") or selected_by_aid[aid].get("source_registry_id")
        if raw: expected_source_basenames[str(provider)][Path(raw).name]=aid
        source_meta.append({"asset":aid,"provider":provider,"raw_path":raw,"ext":ext,"appearance_class":ext_class(ext),"source_sha256":sha,"source_size_bytes":sid.get("source_size_bytes")})
    dup_sha={sha:aids for sha,aids in sha_to_assets.items() if len(aids)>1}; raw_missing=[]; raw_counts={}
    for provider,expected in expected_source_basenames.items():
        parent=SOURCE_PARENT_IDS.get(provider)
        if not parent: raw_counts[provider]={"expected":len(expected),"checked":0,"missing":None,"reason":"no_source_parent_id_configured"}; continue
        rows,p=list_all(svc,f"'{parent}' in parents and trashed=false",label=f"raw {provider}"); pages+=p; names={r["name"] for r in rows}; miss=sorted(set(expected)-names); raw_counts[provider]={"expected":len(expected),"present":len(expected)-len(miss),"missing":len(miss)}
        for n in miss[:1000]: raw_missing.append({"provider":provider,"name":n,"asset":expected[n]})
    tech_summary=collections.Counter(); nondeg=[]; skin_zero=[]
    for aid,t in tech.items():
        for k in ["iris_capable","geppetto_capable","arachne_capable","geometry_pass","rig_pass","skin_pass","canonicalization_pass"]: tech_summary[k+"_true"]+=int(bool(t.get(k)))
        gm=t.get("geometry_metrics") or {}
        if "nondegenerate_fraction" in gm: nondeg.append((aid,float(gm["nondegenerate_fraction"])))
        sm=t.get("skin_metrics") or {}
        if "zero_row_fraction" in sm: skin_zero.append((aid,float(sm["zero_row_fraction"])))
    nondeg_sorted=sorted(nondeg,key=lambda x:x[1]); skin_zero_sorted=sorted(skin_zero,key=lambda x:x[1],reverse=True)
    blend_assets=[x for x in source_meta if x["ext"]==".blend"]; glb_assets=[x for x in source_meta if x["ext"]==".glb"]; fbx_assets=[x for x in source_meta if x["ext"]==".fbx"]; gltf_assets=[x for x in source_meta if x["ext"]==".gltf"]
    import hashlib
    def stable_pick(rows,n,prefix): return sorted(rows,key=lambda x:hashlib.sha256((prefix+x["asset"]).encode()).hexdigest())[:min(n,len(rows))]
    blend_sample=stable_pick(blend_assets,64,"blend-audit:"); appearance_sample=[]
    for ext_rows,label in [(glb_assets,"glb"),(blend_assets,"blend"),(fbx_assets,"fbx"),(gltf_assets,"gltf")]: appearance_sample.extend(stable_pick(ext_rows,24,"appearance-"+label+":"))
    result={"schema":"RealSaS.PostCorpus.StageAClosure.v2","build_id":"REALSAS_MASTER_1024_V4_3_LOCAL_FIRST_FULL_PRODUCTION_20260822","selected":len(selected_ids),"api_pages":pages,"corpus_mutation":False,
      "gate1":{"primary_geometry":{"present":len(geom_one),"expected":len(asset_folder_ids),"missing":len(geom_missing),"duplicates":len(geom_dup),"zero_size":len(geom_zero),"missing_asset_examples":[asset_folder_to_aid.get(x,x) for x in geom_missing[:50]]},"render_folders":{"present":len(render_one),"expected":len(asset_folder_ids),"missing":len(render_missing),"duplicates":len(render_dup)},"view_folders":{"present":len(view_ids),"expected":len(selected_ids)*8,"per_view":view_meta},"raster_authority":{"present":len(auth_one),"expected":len(view_ids),"missing":len(auth_missing),"duplicates":len(auth_dup),"tiny_threshold_bytes":args.tiny_authority_bytes,"tiny_files_inspected":len(tiny_results),"inspection_errors":sum(x["error"] is not None for x in tiny_results)},"observable_coverage":{"assets_with_any_zero_view":len(any_blank),"assets_blank_all8":len(all8_blank),"blank_all8_assets":sorted(all8_blank),"any_zero_view_examples":[{"asset":a,**per_asset[a]} for a in sorted(any_blank)[:100]]}},
      "gate2":{"technical_json_coverage":json_errors["technical_audit.json"],"technical_true_counts":dict(tech_summary),"nondegenerate_fraction":{"count":len(nondeg_sorted),"min_examples":nondeg_sorted[:50]},"skin_zero_row_fraction":{"count":len(skin_zero_sorted),"max_examples":skin_zero_sorted[:50]},"source_extension_distribution":dict(ext_counts),"blend_evaluated_mesh_risk_count":len(blend_assets),"blend_audit_sample":blend_sample},
      "gate3":{"admission_json_coverage":json_errors["admission.json"],"appearance_recoverability_by_container_class":dict(appearance_class_counts),"raw_source_existence_by_provider":raw_counts,"raw_source_missing_count":len(raw_missing),"raw_source_missing_examples":raw_missing[:100],"duplicate_source_sha_group_count":len(dup_sha),"duplicate_source_sha_asset_count":sum(len(v) for v in dup_sha.values()),"duplicate_source_sha_examples":list(dup_sha.items())[:50],"appearance_inspection_sample":appearance_sample,"note":"Container class is not proof that textures/materials are actually embedded or complete; sample/content inspection is still required before B render."},
      "next_required":{"gate2_blend_eval":"Run evaluated-depsgraph/custom-normal/triangulation audit on blend_audit_sample; expand only if failure prevalence is material.","gate3_appearance":"Inspect appearance_inspection_sample for materials/images/packed-vs-external dependencies; then decide B-pass coverage.","gate4":"Run exact representation authority study only after Gate2/3 target/observation authority is frozen."}}
    (outdir/"POST_CORPUS_STAGE_A_CLOSURE_V2.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); (outdir/"BLEND_EVALUATED_MESH_AUDIT_SAMPLE_V1.json").write_text(json.dumps(blend_sample,indent=2),encoding="utf-8"); (outdir/"APPEARANCE_RECOVERABILITY_SAMPLE_V1.json").write_text(json.dumps(appearance_sample,indent=2),encoding="utf-8")
    (outdir/"POST_CORPUS_STAGE_A_CLOSURE_V2.md").write_text("\n".join(["# Post-Corpus Stage-A Closure V2","",f"- selected: **{len(selected_ids)}**",f"- primary geometry: **{len(geom_one)}/{len(asset_folder_ids)}**",f"- views: **{len(view_ids)}/{len(selected_ids)*8}**",f"- raster authorities: **{len(auth_one)}/{len(view_ids)}**",f"- assets blank in all 8 views: **{len(all8_blank)}**",f"- admission JSON parsed: **{len(admissions)}/{len(selected_ids)}**",f"- technical JSON parsed: **{len(tech)}/{len(selected_ids)}**",f"- `.blend` evaluated-mesh risk subset: **{len(blend_assets)}**",f"- duplicate source-SHA groups: **{len(dup_sha)}**",f"- raw-source missing: **{len(raw_missing)}**","","This run is read-only with respect to corpus evidence. It produces only compact reports/sample manifests."])+"\n",encoding="utf-8")
    print(json.dumps({"report":str(outdir/"POST_CORPUS_STAGE_A_CLOSURE_V2.json"),"primary_geometry":f"{len(geom_one)}/{len(asset_folder_ids)}","views":f"{len(view_ids)}/{len(selected_ids)*8}","blank_all8":len(all8_blank),"blend_risk_count":len(blend_assets),"raw_missing":len(raw_missing),"duplicate_sha_groups":len(dup_sha),"api_pages":pages},indent=2),flush=True)

if __name__=="__main__": main()
