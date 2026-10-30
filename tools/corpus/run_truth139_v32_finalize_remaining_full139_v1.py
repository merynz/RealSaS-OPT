#!/usr/bin/env python3
"""Finalize all Truth139 V3.2 truth families after Response has closed 139/139.

Per subject: Stage48 renderer witness -> exact 3D geometry + independent court ->
patch-aligned RGB evidence -> mechanically-qualified TESSA teacher. Then seal a
leakage-connected-component TRAIN/UNSEEN split and fixed TRAIN-internal validation
row seeds. Every stage is resumable and no model optimizer is run here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.corpus.build_truth139_v32_exact_geometry_v1 import sha256

RESPONSE_PASS="PASS_FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
GEOM_PASS="PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
GEOM_COURT_PASS="PASS_V32_EXACT_GEOMETRY_REBINDS_SOURCE_AND_INDEPENDENT_SAMPLES"
RGB_PASS="PASS_V32_RGB37_PATCH_EVIDENCE_MATERIALIZED"
STAGE48_PASS="PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"
TESSA_PASS="PASS_V32_MECHANICALLY_QUALIFIED_TESSA_TEACHER"
FINAL_PASS="PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY"

def utc():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def load(p,code):
    p=Path(p).resolve()
    if not p.is_file():raise FileNotFoundError(code+":"+str(p))
    x=json.loads(p.read_text())
    if not isinstance(x,dict):raise ValueError(code+"_NOT_OBJECT")
    return x
def atomic(path,v):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial");tmp.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n");tmp.replace(path)
def packet_root(base,rescue,sid):
    found=[]
    for root,lineage in ((base,"PREPARED_BASE"),(rescue,"PREPARED_RESCUE")):
        for p in sorted(root.glob(f"{sid}/objects/*/packet/{sid}")):
            if p.is_dir():found.append((p.resolve(),lineage))
    if len(found)!=1:raise RuntimeError(f"V32_FINAL_PACKET_CARDINALITY:{sid}:{len(found)}")
    return found[0]
def run(cmd,log=None):
    cmd=[str(x) for x in cmd];print("[RUN]"," ".join(cmd),flush=True);env={**os.environ,"PYTHONPATH":str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH","")};p=subprocess.run(cmd,cwd=str(ROOT),env=env,text=True,capture_output=True)
    text=(p.stdout or "")+("\n[STDERR]\n"+p.stderr if p.stderr else "")
    if log:log.parent.mkdir(parents=True,exist_ok=True);log.write_text(text)
    if p.stdout:print(p.stdout,end="",flush=True)
    if p.returncode:raise RuntimeError(f"COMMAND_FAILED:{p.returncode}\n"+text[-20000:])

def complete_json(path,status):
    try:return load(path,"V32_FINAL_OUTPUT").get("status")==status
    except Exception:return False

def skeleton_signature(snapshot):
    bones=[]
    for row in snapshot.get("bones") or []:bones.append((str(row.get("source_bone_name") or ""),str(row.get("source_parent_bone_name") or ""),bool(row.get("use_deform",True))))
    return "SKEL:"+digest(sorted(bones))
def topology_signature(topology_path):
    with np.load(topology_path,allow_pickle=False) as z:f=np.asarray(z["faces"],dtype=np.int64);vc=np.asarray(z["vertex_component_id"],dtype=np.int32)
    return "TOPO:"+hashlib.sha256(f.tobytes()+vc.tobytes()).hexdigest()
def material_signature(row):
    source=dict(row.get("source") or {});return "MATPACK:"+hashlib.sha256(str(source.get("pack_id") or source.get("creator") or "").encode()).hexdigest()

class UF:
    def __init__(self,n):self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def build_split(rows,lineages,target_train=0.82,seed=261030139):
    uf=UF(len(rows));owner={}
    fields=("source_sha256","skeleton_hierarchy_signature","topology_lineage_signature","material_lineage_signature","source_family_signature")
    for i,row in enumerate(rows):
        lin=lineages[row["subject_id"]]
        for field in fields:
            vals=lin.get(field);vals=vals if isinstance(vals,list) else [vals]
            for token in vals:
                if not token:continue
                key=(field,str(token));prior=owner.get(key);owner.setdefault(key,i)
                if prior is not None:uf.union(i,prior)
    groups={}
    for i,row in enumerate(rows):groups.setdefault(uf.find(i),[]).append(row)
    comps=[]
    for members in groups.values():
        ids=sorted(r["subject_id"] for r in members);eligible=all(not bool(r.get("development_or_regression_tainted")) for r in members);cid="lc_"+digest(ids)[:20];comps.append({"component_id":cid,"subject_ids":ids,"subject_count":len(ids),"unseen_eligible":eligible})
    total=len(rows);target=total*(1-target_train);eligible=[c for c in comps if c["unseen_eligible"]]
    ordered=sorted(eligible,key=lambda c:hashlib.sha256(f"{seed}|{c['component_id']}".encode()).hexdigest());reachable={0:()}
    for i,c in enumerate(ordered):
        for n,chosen in list(reachable.items()):reachable.setdefault(n+c["subject_count"],chosen+(i,))
    candidates=[n for n in reachable if 0<n<total]
    if not candidates:raise RuntimeError("V32_FINAL_SPLIT_UNSPLITTABLE")
    n=min(candidates,key=lambda x:(abs(x-target),x));unseen_comp={ordered[i]["component_id"] for i in reachable[n]}
    unseen=sorted(s for c in comps if c["component_id"] in unseen_comp for s in c["subject_ids"]);train=sorted(set(r["subject_id"] for r in rows)-set(unseen))
    comp_rows=[{**c,"split":"UNSEEN" if c["component_id"] in unseen_comp else "TRAIN"} for c in sorted(comps,key=lambda x:x["component_id"])]
    out={"schema":"RealSaS.LineageSafeSubjectSplit.v2","split_unit":"LEAKAGE_CONNECTED_COMPONENT","target_train_fraction":target_train,"actual_train_fraction":len(train)/total,"seed":seed,"train_subjects":train,"unseen_subjects":unseen,"train_subject_count":len(train),"unseen_subject_count":len(unseen),"components":comp_rows,"unseen_bytes_opened_for_fit":False,"checkpoint_selection_may_read_unseen":False,"optimizer_may_read_unseen":False}
    out["split_identity_sha256"]=digest(out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--work-orders",type=Path,required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--prepared-rescue-root",type=Path,required=True);p.add_argument("--response-receipt",type=Path,required=True);p.add_argument("--blender",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--continue-on-failure",action="store_true");a=p.parse_args()
    work=load(a.work_orders,"V32_FINAL_WORK_ORDERS");rows=list(work.get("subjects") or []);response=load(a.response_receipt,"V32_FINAL_RESPONSE_RECEIPT")
    if len(rows)!=139 or int(work.get("subject_count",-1))!=139:raise ValueError("V32_FINAL_DENOMINATOR")
    if response.get("status")!=RESPONSE_PASS or int(response.get("completed_subject_count",-1))!=139:raise ValueError("V32_FINAL_RESPONSE_NOT_139_PASS")
    response_bind=dict(response.get("subject_output_bindings") or {});out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True);failures={};completed={};lineages={}
    for index,row in enumerate(rows,1):
        sid=str(row["subject_id"]);packet,lineage=packet_root(a.prepared_root.resolve(),a.prepared_rescue_root.resolve(),sid);subject_out=out/"subjects"/sid;subject_out.mkdir(parents=True,exist_ok=True);print(f"\n[V32-FINAL {index:03d}/139] {sid} packet={lineage}",flush=True)
        try:
            chain_path=Path(response_bind[sid]["chain_path"]).resolve()
            if not chain_path.is_file() or sha256(chain_path)!=response_bind[sid]["chain_sha256"]:raise ValueError("V32_FINAL_RESPONSE_CHAIN_DRIFT:"+sid)
            stage48=subject_out/"stage48"/"STAGE48_RENDERER_WITNESS_MANIFEST.json"
            if not complete_json(stage48,STAGE48_PASS):run([sys.executable,ROOT/"tools/corpus/run_truth139_stage48_witness_subject_v1.py","--subject-id",sid,"--response-chain",chain_path,"--blender",a.blender,"--output-dir",subject_out/"stage48"],subject_out/"stage48.runner.log")
            geom=subject_out/"geometry"/"V32_EXACT_GEOMETRY_MANIFEST.json";court=subject_out/"geometry"/"V32_EXACT_GEOMETRY_COURT.json"
            if not complete_json(geom,GEOM_PASS):run([sys.executable,ROOT/"tools/corpus/build_truth139_v32_exact_geometry_v1.py","--subject-id",sid,"--geometry-npz",packet/"REST_SURFACE_GEOMETRY.npz","--source-scene-snapshot",packet/"SOURCE_SCENE_SNAPSHOT.json","--camera-bundle",packet/"camera_bundle.json","--output-dir",subject_out/"geometry"],subject_out/"geometry.producer.log")
            if not complete_json(court,GEOM_COURT_PASS):run([sys.executable,ROOT/"tools/corpus/validate_truth139_v32_exact_geometry_v1.py","--geometry-manifest",geom,"--output",court],subject_out/"geometry.court.log")
            rgb=subject_out/"rgb"/"V32_RGB_PATCH_EVIDENCE_MANIFEST.json"
            if not complete_json(rgb,RGB_PASS):run([sys.executable,ROOT/"tools/corpus/build_truth139_v32_rgb_patch_cache_v1.py","--subject-id",sid,"--packet-root",packet,"--output-dir",subject_out/"rgb"],subject_out/"rgb.runner.log")
            tessa=subject_out/"tessa_teacher"/"V32_MECHANICAL_TESSA_TEACHER_MANIFEST.json"
            if not complete_json(tessa,TESSA_PASS):run([sys.executable,ROOT/"tools/training/build_truth139_v32_mechanical_tessa_teacher_v1.py","--subject-id",sid,"--geometry-manifest",geom,"--response-chain",chain_path,"--camera-bundle",packet/"camera_bundle.json","--output-dir",subject_out/"tessa_teacher"],subject_out/"tessa.runner.log")
            gm=load(geom,"V32_FINAL_GEOM");snapshot=load(packet/"SOURCE_SCENE_SNAPSHOT.json","V32_FINAL_SNAPSHOT");topo=Path(gm["outputs"]["topology"]["path"])
            lineages[sid]={"source_sha256":str(row["source"]["sha256"]),"skeleton_hierarchy_signature":skeleton_signature(snapshot),"topology_lineage_signature":topology_signature(topo),"material_lineage_signature":material_signature(row),"source_family_signature":list(row["source"].get("source_family_signature") or [])}
            completed[sid]={"packet_lineage":lineage,"stage48":{"path":str(stage48),"sha256":sha256(stage48)},"geometry":{"path":str(geom),"sha256":sha256(geom)},"geometry_court":{"path":str(court),"sha256":sha256(court)},"rgb":{"path":str(rgb),"sha256":sha256(rgb)},"tessa_teacher":{"path":str(tessa),"sha256":sha256(tessa)}};print("[PASS-V32-FINAL]",sid,flush=True)
        except Exception as e:
            failures[sid]=repr(e);print("[FAIL-V32-FINAL]",sid,repr(e),flush=True)
            if not a.continue_on_failure:break
        atomic(out/"V32_FINALIZE_PROGRESS.json",{"schema":"RealSaS.Truth139V32FinalizeProgress.v1","completed_subject_count":len(completed),"failed_subject_count":len(failures),"failures":failures,"updated_at_utc":utc()})
    if len(completed)!=139 or failures:
        receipt={"schema":"RealSaS.Truth139V32FinalizeReceipt.v1","status":"V32_REMAINING_TRUTHS_COMPLETED_WITH_FAILURES","completed_subject_count":len(completed),"failed_subject_count":len(failures),"failures":failures,"subjects":completed,"training_performed":False};atomic(out/"V32_FINALIZE_RECEIPT.json",receipt);raise RuntimeError(f"V32_FINAL_INCOMPLETE:{len(completed)}/139")
    split=build_split(rows,lineages);atomic(out/"LINEAGE_SAFE_TRAIN_UNSEEN_SPLIT_V2.json",split)
    val={"schema":"RealSaS.Truth139TrainInternalValidationRows.v1","split_identity_sha256":split["split_identity_sha256"],"policy":"FIXED_HASH_ROW_SELECTION_WITHIN_TRAIN_ONLY","fraction":0.10,"subjects":{sid:{"seed":int(hashlib.sha256((sid+"|train_internal_val").encode()).hexdigest()[:8],16),"fraction":0.10} for sid in split["train_subjects"]},"unseen_subjects_included":False};val["validation_identity_sha256"]=digest(val);atomic(out/"TRAIN_INTERNAL_VALIDATION_ROWS_V1.json",val)
    ledger={"schema":"RealSaS.Truth139V32PretrainTruthLedger.v1","status":FINAL_PASS,"subject_count":139,"response_receipt":{"path":str(a.response_receipt.resolve()),"sha256":sha256(a.response_receipt)},"subjects":completed,"lineage":lineages,"split":{"path":str(out/"LINEAGE_SAFE_TRAIN_UNSEEN_SPLIT_V2.json"),"sha256":sha256(out/"LINEAGE_SAFE_TRAIN_UNSEEN_SPLIT_V2.json"),"identity_sha256":split["split_identity_sha256"]},"train_internal_validation":{"path":str(out/"TRAIN_INTERNAL_VALIDATION_ROWS_V1.json"),"sha256":sha256(out/"TRAIN_INTERNAL_VALIDATION_ROWS_V1.json"),"identity_sha256":val["validation_identity_sha256"]},"training_performed":False,"training_authorized":True,"qualification_minted":False,"compiler_phase_e_watertight_court_pending_training":True,"completed_at_utc":utc()};ledger["ledger_identity_sha256"]=digest(ledger);atomic(out/"V32_TRAINING_READINESS_RECEIPT.json",ledger)
    print(json.dumps({"status":FINAL_PASS,"subject_count":139,"train_subject_count":split["train_subject_count"],"unseen_subject_count":split["unseen_subject_count"],"ledger_identity_sha256":ledger["ledger_identity_sha256"],"receipt_path":str(out/"V32_TRAINING_READINESS_RECEIPT.json")},sort_keys=True,indent=2))
if __name__=="__main__":main()
