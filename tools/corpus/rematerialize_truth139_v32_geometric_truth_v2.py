#!/usr/bin/env python3
"""Repair Truth139 V3.2 topology/sign-derived truth without re-running Blender.

Inputs are the already-closed V3.2 readiness receipt and Full139 response receipt.
The 112/27 split and TRAIN-internal validation rows are preserved byte-for-byte.

Recomputed per subject:
  * geometric topology + signed-distance V2 + independent solid-angle court,
  * response pair adjacency V2 + independent pair court (Stage47 reused),
  * TESSA teacher V3 + independent rebind court (Stage47 reused).

Inherited byte-for-byte: source packets/observations, Stage47, Stage48, RGB, split
and validation-row policy. No model training or checkpoint selection occurs.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

GEOM_STATUS="PASS_V32_GEOMETRIC_TOPOLOGY_SDF_V2__INDEPENDENT_COURT_REQUIRED"
GEOM_COURT_STATUS="PASS_V32_GEOMETRIC_TOPOLOGY_SDF_V2_INDEPENDENT_COURT"
RESPONSE_STATUS="PASS_IRIS_RESPONSE_EVIDENCE_V2__INDEPENDENT_COURT_REQUIRED"
RESPONSE_COURT_STATUS="PASS_IRIS_RESPONSE_EVIDENCE_V2_INDEPENDENT_NUMERIC_COURT"
TESSA_STATUS="PASS_V32_GEOMETRIC_TOPOLOGY_MECHANICALLY_QUALIFIED_TESSA_TEACHER"
TESSA_COURT_STATUS="PASS_V32_GEOMETRIC_TOPOLOGY_TESSA_TEACHER_INDEPENDENT_COURT"
FINAL_STATUS="PASS_TRUTH139_V32_GEOMETRIC_TOPOLOGY_SDF_REPAIR_READY"


def utc(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()

def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def load(path,code):
    p=Path(path).resolve()
    if not p.is_file(): raise FileNotFoundError(code+":"+str(p))
    value=json.loads(p.read_text())
    if not isinstance(value,dict): raise ValueError(code+"_NOT_OBJECT")
    return value

def ref(path):
    p=Path(path).resolve()
    if not p.is_file(): raise FileNotFoundError("TRUTH139_REPAIR_REF:"+str(p))
    return {"path":str(p),"sha256":sha256(p)}

def atomic(path,value):
    p=Path(path).resolve(); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+".partial"); tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"); tmp.replace(p)

def run(command,log_path):
    command=[str(x) for x in command]; print("[RUN]"," ".join(command),flush=True)
    env={**os.environ,"PYTHONPATH":str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH","")}
    process=subprocess.run(command,cwd=str(ROOT),env=env,text=True,capture_output=True)
    text=(process.stdout or "")+("\n[STDERR]\n"+process.stderr if process.stderr else "")
    Path(log_path).parent.mkdir(parents=True,exist_ok=True); Path(log_path).write_text(text)
    if process.stdout: print(process.stdout,end="",flush=True)
    if process.returncode: raise RuntimeError("TRUTH139_REPAIR_COMMAND_FAILED:"+str(process.returncode)+"\n"+text[-20000:])

def complete(path,status,schema=None):
    try:
        value=load(path,"TRUTH139_REPAIR_EXISTING")
        return value.get("status")==status and (schema is None or value.get("schema")==schema)
    except Exception: return False

def verify_bound_ref(row,code):
    p=Path(str((row or {}).get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str((row or {}).get("sha256") or ""): raise ValueError(code)
    return p


def main():
    p=argparse.ArgumentParser(); p.add_argument("--prior-readiness",type=Path,required=True); p.add_argument("--response-receipt",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--continue-on-failure",action="store_true"); a=p.parse_args()
    prior=load(a.prior_readiness,"TRUTH139_REPAIR_PRIOR_READINESS"); response=load(a.response_receipt,"TRUTH139_REPAIR_RESPONSE_RECEIPT")
    if prior.get("status")!="PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY" or int(prior.get("subject_count",-1))!=139: raise ValueError("TRUTH139_REPAIR_PRIOR_NOT_READY")
    if response.get("status")!="PASS_FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE" or int(response.get("completed_subject_count",-1))!=139: raise ValueError("TRUTH139_REPAIR_RESPONSE_NOT_139")
    split_path=verify_bound_ref(prior["split"],"TRUTH139_REPAIR_SPLIT_DRIFT"); validation_path=verify_bound_ref(prior["train_internal_validation"],"TRUTH139_REPAIR_VALIDATION_DRIFT"); split=load(split_path,"TRUTH139_REPAIR_SPLIT")
    response_bind=dict(response.get("subject_output_bindings") or {}); subjects=dict(prior.get("subjects") or {})
    if len(subjects)!=139 or set(subjects)!=set(response_bind): raise ValueError("TRUTH139_REPAIR_SUBJECT_SET_DRIFT")
    out=a.output_root.resolve(); out.mkdir(parents=True,exist_ok=True); repaired={}; audit_rows=[]; failures={}

    for index,sid in enumerate(sorted(subjects),1):
        print(f"\n[GEOMETRIC-REPAIR {index:03d}/139] {sid}",flush=True); subject_out=out/"subjects"/sid; subject_out.mkdir(parents=True,exist_ok=True)
        try:
            binding=response_bind[sid]; chain_path=Path(binding["chain_path"]).resolve()
            if not chain_path.is_file() or sha256(chain_path)!=binding["chain_sha256"]: raise ValueError("TRUTH139_REPAIR_CHAIN_DRIFT:"+sid)
            chain=load(chain_path,"TRUTH139_REPAIR_CHAIN"); packet_root=Path((binding.get("prepared_packet_lineage") or {}).get("packet_root") or "").resolve()
            required_packet=(packet_root/"REST_SURFACE_GEOMETRY.npz",packet_root/"SOURCE_SCENE_SNAPSHOT.json",packet_root/"camera_bundle.json")
            if not all(x.is_file() for x in required_packet): raise FileNotFoundError("TRUTH139_REPAIR_PACKET_MISSING:"+sid)
            teacher_scene=verify_bound_ref(chain["outputs"]["teacher_scene"],"TRUTH139_REPAIR_TEACHER_SCENE_DRIFT")

            geom_dir=subject_out/"geometry_v2"; geom=geom_dir/"V32_EXACT_GEOMETRY_MANIFEST_V2.json"; geom_court=geom_dir/"V32_EXACT_GEOMETRY_COURT_V2.json"
            if not complete(geom,GEOM_STATUS,"RealSaS.Truth139V32ExactGeometry.v2"): run([sys.executable,ROOT/"tools/corpus/build_truth139_v32_exact_geometry_v3.py","--subject-id",sid,"--geometry-npz",packet_root/"REST_SURFACE_GEOMETRY.npz","--source-scene-snapshot",packet_root/"SOURCE_SCENE_SNAPSHOT.json","--camera-bundle",packet_root/"camera_bundle.json","--output-dir",geom_dir],subject_out/"geometry_v2.producer.log")
            if not complete(geom_court,GEOM_COURT_STATUS,"RealSaS.Truth139V32ExactGeometryCourt.v2"): run([sys.executable,ROOT/"tools/corpus/validate_truth139_v32_exact_geometry_v2.py","--geometry-manifest",geom,"--output",geom_court],subject_out/"geometry_v2.court.log")

            response_dir=subject_out/"response_v2"; response_manifest=response_dir/"IRIS_RESPONSE_EVIDENCE_MANIFEST_V2.json"; response_court=response_dir/"IRIS_RESPONSE_EVIDENCE_COURT_V2.json"
            if not complete(response_manifest,RESPONSE_STATUS,"RealSaS.Truth139IrisResponseEvidence.v2"): run([sys.executable,ROOT/"tools/corpus/build_truth139_response_evidence_v2.py","--subject-id",sid,"--teacher-scene",teacher_scene,"--output-dir",response_dir],subject_out/"response_v2.producer.log")
            if not complete(response_court,RESPONSE_COURT_STATUS,"RealSaS.Truth139IrisResponseEvidenceCourt.v2"): run([sys.executable,ROOT/"tools/corpus/validate_truth139_response_evidence_v2.py","--response-manifest",response_manifest,"--output",response_court],subject_out/"response_v2.court.log")

            tessa_dir=subject_out/"tessa_v3"; tessa=tessa_dir/"V32_MECHANICAL_TESSA_TEACHER_MANIFEST_V3.json"; tessa_court=tessa_dir/"V32_MECHANICAL_TESSA_TEACHER_COURT_V3.json"
            if not complete(tessa,TESSA_STATUS,"RealSaS.Truth139V32MechanicalTESSATeacher.v3"): run([sys.executable,ROOT/"tools/training/build_truth139_v32_mechanical_tessa_teacher_v3.py","--subject-id",sid,"--geometry-manifest",geom,"--response-chain",chain_path,"--camera-bundle",packet_root/"camera_bundle.json","--output-dir",tessa_dir],subject_out/"tessa_v3.producer.log")
            if not complete(tessa_court,TESSA_COURT_STATUS,"RealSaS.Truth139V32MechanicalTESSATeacherCourt.v3"): run([sys.executable,ROOT/"tools/training/validate_truth139_v32_mechanical_tessa_teacher_v3.py","--teacher-manifest",tessa,"--output",tessa_court],subject_out/"tessa_v3.court.log")

            gm=load(geom,"TRUTH139_REPAIR_GEOM"); gc=load(geom_court,"TRUTH139_REPAIR_GEOM_COURT"); rm=load(response_manifest,"TRUTH139_REPAIR_RESPONSE"); tm=load(tessa,"TRUTH139_REPAIR_TESSA"); old_subject=subjects[sid]
            old_geom=load(verify_bound_ref(old_subject["geometry"],"TRUTH139_REPAIR_OLD_GEOM"),"TRUTH139_REPAIR_OLD_GEOM_JSON"); old_tessa=load(verify_bound_ref(old_subject["tessa_teacher"],"TRUTH139_REPAIR_OLD_TESSA"),"TRUTH139_REPAIR_OLD_TESSA_JSON"); old_response_manifest=load(verify_bound_ref(chain["outputs"]["response_manifest"],"TRUTH139_REPAIR_OLD_RESPONSE"),"TRUTH139_REPAIR_OLD_RESPONSE_JSON")
            if not gm["validity"]["global_sign_topology_valid"]: raise ValueError("TRUTH139_REPAIR_GEOMETRIC_TOPOLOGY_NOT_SIGN_VALID:"+sid)
            repaired[sid]={"packet_lineage":old_subject.get("packet_lineage"),"stage48":old_subject["stage48"],"rgb":old_subject["rgb"],**({"rgb_court":old_subject["rgb_court"]} if "rgb_court" in old_subject else {}),"geometry_v2":ref(geom),"geometry_court_v2":ref(geom_court),"response_v2":ref(response_manifest),"response_court_v2":ref(response_court),"tessa_teacher_v3":ref(tessa),"tessa_court_v3":ref(tessa_court),"source_response_chain":ref(chain_path)}
            audit_rows.append({"subject_id":sid,"raw_vertex_count":int(gm["counts"]["raw_vertex_count"]),"topology_vertex_count":int(gm["counts"]["topology_vertex_count"]),"exact_duplicate_collapse_count":int(gm["counts"]["exact_duplicate_collapse_count"]),"old_component_count":int((old_geom.get("counts") or {}).get("component_count",-1)),"new_component_count":int(gm["counts"]["component_count"]),"old_boundary_edge_count":int((old_geom.get("counts") or {}).get("boundary_edge_count",-1)),"new_boundary_edge_count":int(gm["counts"]["boundary_edge_count"]),"sign_valid_fraction":float(gc["sign"]["nearest_component_valid_fraction"]),"old_response_pair_count":int((old_response_manifest.get("counts") or {}).get("response_pair_count",-1)),"new_response_pair_count":int(rm["counts"]["response_pair_count"]),"old_tessa_selected_kind":str((old_tessa.get("selected") or {}).get("kind")),"new_tessa_selected_kind":str((tm.get("selected") or {}).get("kind")),"new_tessa_selected_face_count":int((tm.get("selected") or {}).get("face_count",-1))})
            print("[PASS-GEOMETRIC-REPAIR]",sid,flush=True)
        except Exception as exc:
            failures[sid]=repr(exc); print("[FAIL-GEOMETRIC-REPAIR]",sid,repr(exc),flush=True)
            if not a.continue_on_failure: break
        atomic(out/"GEOMETRIC_TRUTH_REPAIR_PROGRESS.json",{"schema":"RealSaS.Truth139GeometricTruthRepairProgress.v1","completed_subject_count":len(repaired),"failed_subject_count":len(failures),"failures":failures,"updated_at_utc":utc()})

    audit={"schema":"RealSaS.Truth139GeometricTopologyAudit.v1","subject_count":len(audit_rows),"all_subjects_closed_sign_valid":bool(len(audit_rows)==139 and not failures and all(row["new_boundary_edge_count"]==0 and row["sign_valid_fraction"]>=.995 for row in audit_rows)),"totals":{"raw_vertex_count":sum(row["raw_vertex_count"] for row in audit_rows),"topology_vertex_count":sum(row["topology_vertex_count"] for row in audit_rows),"exact_duplicate_collapse_count":sum(row["exact_duplicate_collapse_count"] for row in audit_rows),"old_boundary_edge_count":sum(max(0,row["old_boundary_edge_count"]) for row in audit_rows),"new_boundary_edge_count":sum(row["new_boundary_edge_count"] for row in audit_rows),"old_response_pair_count":sum(max(0,row["old_response_pair_count"]) for row in audit_rows),"new_response_pair_count":sum(row["new_response_pair_count"] for row in audit_rows)},"subjects":audit_rows}
    audit["audit_identity_sha256"]=digest(audit); atomic(out/"GEOMETRIC_TOPOLOGY_AUDIT_V1.json",audit)
    if len(repaired)!=139 or failures or not audit["all_subjects_closed_sign_valid"]:
        receipt={"schema":"RealSaS.Truth139GeometricTruthRepairReceipt.v1","status":"GEOMETRIC_TRUTH_REPAIR_INCOMPLETE","completed_subject_count":len(repaired),"failed_subject_count":len(failures),"failures":failures,"audit":ref(out/"GEOMETRIC_TOPOLOGY_AUDIT_V1.json"),"training_performed":False}; atomic(out/"GEOMETRIC_TRUTH_REPAIR_RECEIPT.json",receipt); raise RuntimeError(f"TRUTH139_GEOMETRIC_REPAIR_INCOMPLETE:{len(repaired)}/139 failures={len(failures)}")
    receipt={"schema":"RealSaS.Truth139GeometricTruthRepairReceipt.v1","status":FINAL_STATUS,"subject_count":139,"subjects":repaired,"prior_readiness":ref(a.prior_readiness),"prior_response_receipt":ref(a.response_receipt),"split_preserved_byte_for_byte":True,"split":prior["split"],"train_internal_validation_preserved_byte_for_byte":True,"train_internal_validation":prior["train_internal_validation"],"train_subject_count":int(split["train_subject_count"]),"unseen_subject_count":int(split["unseen_subject_count"]),"unseen_truth_rematerialized":True,"unseen_used_for_gradient":False,"unseen_used_for_checkpoint_selection":False,"audit":ref(out/"GEOMETRIC_TOPOLOGY_AUDIT_V1.json"),"training_performed":False,"qualification_minted":False,"completed_at_utc":utc()}
    receipt["receipt_identity_sha256"]=digest(receipt); atomic(out/"GEOMETRIC_TRUTH_REPAIR_RECEIPT.json",receipt)
    readiness={"schema":"RealSaS.Truth139V32GeometricRepairTrainingReadiness.v1","status":FINAL_STATUS,"subject_count":139,"subjects":repaired,"split":prior["split"],"train_internal_validation":prior["train_internal_validation"],"geometric_truth_repair_receipt":ref(out/"GEOMETRIC_TRUTH_REPAIR_RECEIPT.json"),"audit":ref(out/"GEOMETRIC_TOPOLOGY_AUDIT_V1.json"),"prior_readiness":ref(a.prior_readiness),"training_authorized":True,"training_performed":False,"qualification_minted":False,"unseen_optimizer_access":False,"unseen_checkpoint_selection_access":False}
    readiness["ledger_identity_sha256"]=digest(readiness); atomic(out/"V32_GEOMETRIC_REPAIR_TRAINING_READINESS_RECEIPT.json",readiness)
    print(json.dumps({"status":FINAL_STATUS,"subject_count":139,"train_subject_count":split["train_subject_count"],"unseen_subject_count":split["unseen_subject_count"],"audit_identity_sha256":audit["audit_identity_sha256"],"ledger_identity_sha256":readiness["ledger_identity_sha256"],"readiness_path":str(out/"V32_GEOMETRIC_REPAIR_TRAINING_READINESS_RECEIPT.json")},sort_keys=True,indent=2))

if __name__=="__main__": main()
