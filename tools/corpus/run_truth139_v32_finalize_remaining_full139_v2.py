#!/usr/bin/env python3
"""Finalize the remaining Truth139 V3.2 pretraining truth with explicit courts.

Prerequisites are sealed Base139, V2 Extension and Response 139/139 receipts. The
runner materializes Exact3D+court, RGB+court, Mechanical TESSA teacher+court and
Stage48 source-equivalent renderer witness, then seals a leakage-component split
and exact TRAIN-internal validation row indices. No optimizer executes here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.corpus.build_truth139_v32_exact_geometry_v1 import sha256

BASE_PASS="PASS_FULL139_CANONICAL_V2_COMPOSITE_MATERIALIZATION_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
V2_PASS="PASS_FULL139_V2_EXTENSION_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
RESPONSE_PASS="PASS_FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
GEOM_PASS="PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
GEOM_COURT_PASS="PASS_V32_EXACT_GEOMETRY_REBINDS_SOURCE_AND_INDEPENDENT_SAMPLES"
RGB_PASS="PASS_V32_RGB37_PATCH_EVIDENCE_MATERIALIZED"
RGB_COURT_PASS="PASS_V32_RGB37_PATCH_EVIDENCE_REBINDS_EXACT_SOURCE_PIXELS"
TESSA_PASS="PASS_V32_MECHANICALLY_QUALIFIED_TESSA_TEACHER"
TESSA_COURT_PASS="PASS_V32_MECHANICAL_TESSA_TEACHER_REBINDS_EXACT_SOURCE_AND_DECLARED_COURTS"
STAGE48_PASS="PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"
FINAL_PASS="PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY"

def utc():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def load(path,code):
    p=Path(path).resolve()
    if not p.is_file():raise FileNotFoundError(code+":"+str(p))
    x=json.loads(p.read_text())
    if not isinstance(x,dict):raise ValueError(code+"_NOT_OBJECT")
    return x
def atomic(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial");tmp.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n");tmp.replace(path)
def complete_json(path,status,sid=None):
    try:
        x=load(path,"V32_FINAL_OUTPUT")
        return x.get("status")==status and (sid is None or x.get("subject_id")==sid)
    except Exception:return False
def packet_root(base,rescue,sid):
    found=[]
    for root,lineage in ((base,"PREPARED_BASE"),(rescue,"PREPARED_RESCUE")):
        for p in sorted(root.glob(f"{sid}/objects/*/packet/{sid}")):
            if p.is_dir():found.append((p.resolve(),lineage))
    if len(found)!=1:raise RuntimeError(f"V32_FINAL_PACKET_CARDINALITY:{sid}:{len(found)}")
    return found[0]
def run(cmd,log=None):
    cmd=[str(x) for x in cmd];print("[RUN]"," ".join(cmd),flush=True);env={**os.environ,"PYTHONPATH":str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH","")};p=subprocess.run(cmd,cwd=str(ROOT),env=env,text=True,capture_output=True);text=(p.stdout or "")+("\n[STDERR]\n"+p.stderr if p.stderr else "")
    if log:Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text(text)
    if p.stdout:print(p.stdout,end="",flush=True)
    if p.returncode:raise RuntimeError(f"COMMAND_FAILED:{p.returncode}\n"+text[-20000:])
def verified_receipt(path,status,count=139):
    x=load(path,"V32_FINAL_PREREQUISITE")
    if x.get("status")!=status or int(x.get("completed_subject_count",-1))!=count:raise ValueError("V32_FINAL_PREREQUISITE_NOT_CLOSED:"+status)
    return x

def skeleton_signature(snapshot):
    rows=sorted((str(r.get("source_bone_name") or ""),str(r.get("source_parent_bone_name") or ""),bool(r.get("use_deform",True))) for r in snapshot.get("bones") or [])
    return "SKEL:"+digest(rows)
def topology_signature(path):
    with np.load(path,allow_pickle=False) as z:f=np.asarray(z["faces"],dtype=np.int64);vc=np.asarray(z["vertex_component_id"],dtype=np.int32)
    return "TOPO:"+hashlib.sha256(f.tobytes()+vc.tobytes()).hexdigest()
def material_signature(row):
    source=dict(row.get("source") or {});token={"pack_id":source.get("pack_id"),"creator":source.get("creator")};return "MATPACK:"+digest(token)

class UF:
    def __init__(self,n):self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)
def build_split(rows,lineages,target_train=.82,seed=261030139):
    uf=UF(len(rows));owner={};fields=("source_sha256","skeleton_hierarchy_signature","topology_lineage_signature","material_lineage_signature","source_family_signature")
    for i,row in enumerate(rows):
        lin=lineages[row["subject_id"]]
        for field in fields:
            vals=lin.get(field);vals=vals if isinstance(vals,list) else [vals]
            for token in vals:
                if not token:continue
                key=(field,str(token));prior=owner.get(key)
                if prior is None:owner[key]=i
                else:uf.union(i,prior)
    groups={}
    for i,row in enumerate(rows):groups.setdefault(uf.find(i),[]).append(row)
    comps=[]
    for members in groups.values():
        ids=sorted(r["subject_id"] for r in members);eligible=all(not bool(r.get("development_or_regression_tainted")) for r in members);cid="lc_"+digest(ids)[:20];comps.append({"component_id":cid,"subject_ids":ids,"subject_count":len(ids),"unseen_eligible":eligible})
    target=len(rows)*(1-target_train);eligible=[c for c in comps if c["unseen_eligible"]];ordered=sorted(eligible,key=lambda c:hashlib.sha256(f"{seed}|{c['component_id']}".encode()).hexdigest());reachable={0:()}
    for i,c in enumerate(ordered):
        for n,chosen in list(reachable.items()):reachable.setdefault(n+c["subject_count"],chosen+(i,))
    candidates=[n for n in reachable if 0<n<len(rows)]
    if not candidates:raise RuntimeError("V32_FINAL_SPLIT_UNSPLITTABLE")
    best=min(candidates,key=lambda n:(abs(n-target),n));unseen_components={ordered[i]["component_id"] for i in reachable[best]};unseen=sorted(s for c in comps if c["component_id"] in unseen_components for s in c["subject_ids"]);train=sorted(set(r["subject_id"] for r in rows)-set(unseen));component_rows=[{**c,"split":"UNSEEN" if c["component_id"] in unseen_components else "TRAIN"} for c in sorted(comps,key=lambda x:x["component_id"])]
    out={"schema":"RealSaS.LineageSafeSubjectSplit.v2","split_unit":"LEAKAGE_CONNECTED_COMPONENT","required_lineage_fields":["source_sha256","skeleton_hierarchy_signature","topology_lineage_signature","material_lineage_signature"],"conservative_extra_lineage_field":"source_family_signature","target_train_fraction":target_train,"actual_train_fraction":len(train)/len(rows),"seed":seed,"train_subjects":train,"unseen_subjects":unseen,"train_subject_count":len(train),"unseen_subject_count":len(unseen),"components":component_rows,"development_connected_components_forced_out_of_unseen":True,"unseen_bytes_opened_for_fit":False,"checkpoint_selection_may_read_unseen":False,"optimizer_may_read_unseen":False,"training_authorized":False};out["split_identity_sha256"]=digest(out);return out

def stable_indices(count,sid,lane,fraction=.10):
    if count<=0:return []
    take=max(1,min(count-1 if count>1 else 1,int(round(count*fraction))))
    ranked=sorted(range(count),key=lambda i:hashlib.sha256(f"261030139|{sid}|{lane}|{i}".encode()).hexdigest())
    return sorted(ranked[:take])
def validation_rows(train,subjects,response_bind):
    result={}
    for sid in train:
        binding=subjects[sid];gm=load(binding["geometry"]["path"],"VAL_GEOM");rgbm=load(binding["rgb"]["path"],"VAL_RGB");tm=load(binding["tessa_teacher"]["path"],"VAL_TESSA");s48=load(binding["stage48"]["path"],"VAL_STAGE48");chain=load(response_bind[sid]["chain_path"],"VAL_RESPONSE_CHAIN")
        with np.load(gm["outputs"]["distance_sign"]["path"],allow_pickle=False) as z:geom_n=len(z["query_xyz_canonical"])
        with np.load(chain["outputs"]["response_arrays"]["path"],allow_pickle=False) as z:response_n=len(z["surface_nonrigid_response"])
        with np.load(rgbm["output"]["path"],allow_pickle=False) as z:rgb_n=int(np.prod(z["rgb_patch_mean"].shape[:3]))
        with np.load(tm["teacher_mesh"]["path"],allow_pickle=False) as z:tessa_n=len(z["faces"])
        with np.load(s48["arrays"]["path"],allow_pickle=False) as z:stage48_n=len(z["composite_pair_face_a"]) if "composite_pair_face_a" in z.files else 0
        lanes={"geometry_query":(geom_n,gm["outputs"]["distance_sign"]["sha256"]),"response_surface_face":(response_n,chain["outputs"]["response_arrays"]["sha256"]),"rgb_patch":(rgb_n,rgbm["output"]["sha256"]),"tessa_face":(tessa_n,tm["teacher_mesh"]["sha256"]),"stage48_composite_pair":(stage48_n,s48["arrays"]["sha256"])}
        result[sid]={lane:{"parent_sha256":parent,"row_count":n,"selected_indices":stable_indices(n,sid,lane)} for lane,(n,parent) in lanes.items()}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--work-orders",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--prepared-rescue-root",type=Path,required=True);p.add_argument("--base-receipt",type=Path,required=True);p.add_argument("--v2-extension-receipt",type=Path,required=True);p.add_argument("--response-receipt",type=Path,required=True);p.add_argument("--blender",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--continue-on-failure",action="store_true");a=p.parse_args()
    work=load(a.work_orders,"V32_FINAL_WORK_ORDERS");rows=list(work.get("subjects") or []);base=verified_receipt(a.base_receipt,BASE_PASS);v2=verified_receipt(a.v2_extension_receipt,V2_PASS);response=verified_receipt(a.response_receipt,RESPONSE_PASS)
    if len(rows)!=139 or int(work.get("subject_count",-1))!=139:raise ValueError("V32_FINAL_DENOMINATOR")
    response_bind=dict(response.get("subject_output_bindings") or {})
    if len(response_bind)!=139:raise ValueError("V32_FINAL_RESPONSE_BINDING_COUNT")
    code_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),text=True).strip();base_root=a.prepared_root.resolve();rescue_root=a.prepared_rescue_root.resolve();out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    packet_map={}
    for row in rows:
        sid=str(row["subject_id"]);packet,lineage=packet_root(base_root,rescue_root,sid);packet_map[sid]=(packet,lineage);b=response_bind.get(sid) or {};chain=Path(str(b.get("chain_path") or "")).resolve()
        if not chain.is_file() or sha256(chain)!=b.get("chain_sha256"):raise ValueError("V32_FINAL_RESPONSE_CHAIN_PREFLIGHT:"+sid)
    failures={};completed={};lineages={}
    for index,row in enumerate(rows,1):
        sid=str(row["subject_id"]);packet,lineage=packet_map[sid];subject_out=out/"subjects"/sid;subject_out.mkdir(parents=True,exist_ok=True);print(f"\n[V32-FINAL {index:03d}/139] {sid} packet={lineage}",flush=True)
        try:
            chain_path=Path(response_bind[sid]["chain_path"]).resolve()
            geom=subject_out/"geometry"/"V32_EXACT_GEOMETRY_MANIFEST.json";geom_court=subject_out/"geometry"/"V32_EXACT_GEOMETRY_COURT.json"
            if not complete_json(geom,GEOM_PASS,sid):run([sys.executable,ROOT/"tools/corpus/build_truth139_v32_exact_geometry_v1.py","--subject-id",sid,"--geometry-npz",packet/"REST_SURFACE_GEOMETRY.npz","--source-scene-snapshot",packet/"SOURCE_SCENE_SNAPSHOT.json","--camera-bundle",packet/"camera_bundle.json","--output-dir",subject_out/"geometry"],subject_out/"geometry.producer.log")
            if not complete_json(geom_court,GEOM_COURT_PASS,sid):run([sys.executable,ROOT/"tools/corpus/validate_truth139_v32_exact_geometry_v1.py","--geometry-manifest",geom,"--output",geom_court],subject_out/"geometry.court.log")
            rgb=subject_out/"rgb"/"V32_RGB_PATCH_EVIDENCE_MANIFEST.json";rgb_court=subject_out/"rgb"/"V32_RGB_PATCH_EVIDENCE_COURT.json"
            if not complete_json(rgb,RGB_PASS,sid):run([sys.executable,ROOT/"tools/corpus/build_truth139_v32_rgb_patch_cache_v1.py","--subject-id",sid,"--packet-root",packet,"--output-dir",subject_out/"rgb"],subject_out/"rgb.producer.log")
            if not complete_json(rgb_court,RGB_COURT_PASS,sid):run([sys.executable,ROOT/"tools/corpus/validate_truth139_v32_rgb_patch_cache_v1.py","--rgb-manifest",rgb,"--output",rgb_court],subject_out/"rgb.court.log")
            tessa=subject_out/"tessa_teacher"/"V32_MECHANICAL_TESSA_TEACHER_MANIFEST.json";tessa_court=subject_out/"tessa_teacher"/"V32_MECHANICAL_TESSA_TEACHER_COURT.json"
            if not complete_json(tessa,TESSA_PASS,sid):run([sys.executable,ROOT/"tools/training/build_truth139_v32_mechanical_tessa_teacher_v1.py","--subject-id",sid,"--geometry-manifest",geom,"--response-chain",chain_path,"--camera-bundle",packet/"camera_bundle.json","--output-dir",subject_out/"tessa_teacher"],subject_out/"tessa.producer.log")
            if not complete_json(tessa_court,TESSA_COURT_PASS,sid):run([sys.executable,ROOT/"tools/training/validate_truth139_v32_mechanical_tessa_teacher_v1.py","--teacher-manifest",tessa,"--output",tessa_court],subject_out/"tessa.court.log")
            stage48=subject_out/"stage48"/"STAGE48_RENDERER_WITNESS_MANIFEST.json"
            if not complete_json(stage48,STAGE48_PASS,sid):run([sys.executable,ROOT/"tools/corpus/run_truth139_stage48_witness_subject_v1.py","--subject-id",sid,"--response-chain",chain_path,"--blender",a.blender,"--output-dir",subject_out/"stage48"],subject_out/"stage48.runner.log")
            gm=load(geom,"V32_FINAL_GEOM");snapshot=load(packet/"SOURCE_SCENE_SNAPSHOT.json","V32_FINAL_SNAPSHOT");topo=Path(gm["outputs"]["topology"]["path"]);lineages[sid]={"source_sha256":str(row["source"]["sha256"]),"skeleton_hierarchy_signature":skeleton_signature(snapshot),"topology_lineage_signature":topology_signature(topo),"material_lineage_signature":material_signature(row),"source_family_signature":list(row["source"].get("source_family_signature") or [])}
            completed[sid]={"packet_lineage":lineage,"geometry":{"path":str(geom),"sha256":sha256(geom)},"geometry_court":{"path":str(geom_court),"sha256":sha256(geom_court)},"rgb":{"path":str(rgb),"sha256":sha256(rgb)},"rgb_court":{"path":str(rgb_court),"sha256":sha256(rgb_court)},"tessa_teacher":{"path":str(tessa),"sha256":sha256(tessa)},"tessa_court":{"path":str(tessa_court),"sha256":sha256(tessa_court)},"stage48":{"path":str(stage48),"sha256":sha256(stage48)}};print("[PASS-V32-FINAL]",sid,flush=True)
        except Exception as exc:
            failures[sid]=repr(exc);print("[FAIL-V32-FINAL]",sid,repr(exc),flush=True)
            if not a.continue_on_failure:break
        atomic(out/"V32_FINALIZE_PROGRESS.json",{"schema":"RealSaS.Truth139V32FinalizeProgress.v2","code_commit_sha":code_sha,"completed_subject_count":len(completed),"failed_subject_count":len(failures),"failures":failures,"updated_at_utc":utc()})
    if len(completed)!=139 or failures:
        receipt={"schema":"RealSaS.Truth139V32FinalizeReceipt.v2","status":"V32_REMAINING_TRUTHS_COMPLETED_WITH_FAILURES","code_commit_sha":code_sha,"completed_subject_count":len(completed),"failed_subject_count":len(failures),"failures":failures,"subjects":completed,"training_performed":False};atomic(out/"V32_FINALIZE_RECEIPT.json",receipt);raise RuntimeError(f"V32_FINAL_INCOMPLETE:{len(completed)}/139")
    split=build_split(rows,lineages);atomic(out/"LINEAGE_SAFE_TRAIN_UNSEEN_SPLIT_V2.json",split);val_rows=validation_rows(split["train_subjects"],completed,response_bind);val={"schema":"RealSaS.Truth139TrainInternalValidationRows.v2","split_identity_sha256":split["split_identity_sha256"],"policy":"EXACT_HASH_RANKED_ROW_INDICES_WITHIN_TRAIN_ONLY","fraction":.10,"subjects":val_rows,"unseen_subjects_included":False,"unseen_subject_ids":[]};val["validation_identity_sha256"]=digest(val);atomic(out/"TRAIN_INTERNAL_VALIDATION_ROWS_V2.json",val)
    ledger={"schema":"RealSaS.Truth139V32PretrainTruthLedger.v2","status":FINAL_PASS,"code_commit_sha":code_sha,"subject_count":139,"prior_truth_receipts":{"base":{"path":str(a.base_receipt.resolve()),"sha256":sha256(a.base_receipt),"status":base["status"]},"v2_extension":{"path":str(a.v2_extension_receipt.resolve()),"sha256":sha256(a.v2_extension_receipt),"status":v2["status"]},"response":{"path":str(a.response_receipt.resolve()),"sha256":sha256(a.response_receipt),"status":response["status"]}},"subjects":completed,"lineage":lineages,"split":{"path":str(out/"LINEAGE_SAFE_TRAIN_UNSEEN_SPLIT_V2.json"),"sha256":sha256(out/"LINEAGE_SAFE_TRAIN_UNSEEN_SPLIT_V2.json"),"identity_sha256":split["split_identity_sha256"]},"train_internal_validation":{"path":str(out/"TRAIN_INTERNAL_VALIDATION_ROWS_V2.json"),"sha256":sha256(out/"TRAIN_INTERNAL_VALIDATION_ROWS_V2.json"),"identity_sha256":val["validation_identity_sha256"]},"training_performed":False,"training_authorized":True,"qualification_minted":False,"compiler_phase_e_watertight_court_pending_training":True,"completed_at_utc":utc()};ledger["ledger_identity_sha256"]=digest(ledger);atomic(out/"V32_TRAINING_READINESS_RECEIPT.json",ledger)
    print(json.dumps({"status":FINAL_PASS,"subject_count":139,"train_subject_count":split["train_subject_count"],"unseen_subject_count":split["unseen_subject_count"],"ledger_identity_sha256":ledger["ledger_identity_sha256"],"receipt_path":str(out/"V32_TRAINING_READINESS_RECEIPT.json")},sort_keys=True,indent=2))
if __name__=="__main__":main()
