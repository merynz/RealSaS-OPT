#!/usr/bin/env python3
"""Run the Stage48 renderer-intervention witness for one Truth139 response chain.

Consumes the already closed Stage47/Response chain. It does not recompute Response.
Two packed-scene measurements and two fresh raw-source reference replays establish
renderer/source equivalence before compositing/order evidence is materialized.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.iris.truth.presentation_qualification_v1 import qualify_subject_presentation_v1
from models.iris.truth.presentation_response_v1 import build_measured_compositing_arrays_v1, build_pose_conditioned_order_arrays_v1

SCHEMA="RealSaS.Truth139Stage48RendererWitness.v1"
STATUS="PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"
RESPONSE_STATUS="PASS_TRUTH139_IRIS_RESPONSE_EVIDENCE_CHAIN__DERIVED_TRUTH_ADMISSION_SEPARATE"

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""):h.update(c)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def ref(p):
    p=Path(p).resolve()
    if not p.is_file():raise FileNotFoundError(p)
    return {"path":str(p),"sha256":sha256(p)}
def load_ref(r,code):
    p=Path(str(r.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=r.get("sha256"):raise ValueError(code)
    return p
def payload_ref(scene,name,field):
    rows=[r for r in scene.get(field) or [] if Path(str(r.get("path") or "")).name==name]
    if len(rows)!=1:raise ValueError("STAGE48_PAYLOAD_CARDINALITY:"+name)
    return load_ref(rows[0],"STAGE48_PAYLOAD_DRIFT:"+name)
def run_blender(blender,program,args,log,maximum,blend=None):
    cmd=[str(blender),"-b"]+([str(blend)] if blend else ["--factory-startup"])+["--python-exit-code","2","--python",str(program),"--",*map(str,args)]
    log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with log.open("w") as f:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT)
        while True:
            try:rc=p.wait(timeout=20);break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-start;print(f"[STAGE48] elapsed_s={elapsed:.1f}",flush=True)
                if elapsed>=maximum:p.terminate();raise RuntimeError("STAGE48_TIME_BUDGET")
    if rc:
        raise RuntimeError("STAGE48_BLENDER_FAILED\n"+log.read_text(errors="replace")[-16000:])
def save_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)

def build(subject_id,response_chain_path:Path,blender:Path,output_dir:Path,maximum_seconds=1800.0,maximum_rays=8,maximum_hits=4,cycles_samples=256):
    chain=json.loads(response_chain_path.resolve().read_text())
    if chain.get("status")!=RESPONSE_STATUS or chain.get("subject_id")!=subject_id:raise ValueError("STAGE48_RESPONSE_CHAIN_NOT_PASS")
    teacher_path=load_ref(chain["outputs"]["teacher_scene"],"STAGE48_TEACHER_DRIFT");scene=json.loads(teacher_path.read_text())
    if scene.get("schema")!="RealSaS.TeacherSceneIR.v2" or scene.get("rest_contract")!="EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY":raise ValueError("STAGE48_TEACHER_SCOPE")
    geometry=payload_ref(scene,"geometry_and_response.npz","geometry_payloads");appearance=payload_ref(scene,"appearance.json","appearance_payloads");blend=payload_ref(scene,"source_scene.blend","appearance_payloads")
    source=load_ref(chain["inputs"]["source"],"STAGE48_SOURCE_DRIFT");candidate=load_ref(chain["inputs"]["source_rest_candidate"],"STAGE48_CANDIDATE_DRIFT");camera=load_ref(chain["inputs"]["camera_bundle"],"STAGE48_CAMERA_DRIFT")
    output_dir=output_dir.resolve();output_dir.mkdir(parents=True,exist_ok=True)
    packed=[];refs=[]
    for replay in ("A","B"):
        mp=output_dir/f"PACKED_{replay}.json"
        if not mp.exists():run_blender(blender,ROOT/"tools/truth_foundry/blender_presentation_ray_measure_v1.py",["--subject-id",subject_id,"--teacher-scene",teacher_path,"--geometry-npz",geometry,"--candidate",candidate,"--camera-bundle",camera,"--expected-blend-sha256",sha256(blend),"--output",mp,"--maximum-rays",maximum_rays,"--maximum-hits",maximum_hits,"--cycles-samples",cycles_samples,"--large-footprint-fraction",0.0625,"--small-footprint-fraction",0.03125,"--convergence-tolerance",0.005,"--representation-tolerance",0.005,"--measurement-tolerance",0.0075],output_dir/f"packed_{replay}.blender.log",maximum_seconds,blend=blend)
        rp=output_dir/f"SOURCE_REFERENCE_{replay}.json"
        if not rp.exists():run_blender(blender,ROOT/"tools/truth_foundry/blender_source_ray_appearance_reference_v1.py",["--source",source,"--expected-source-sha256",sha256(source),"--subject-id",subject_id,"--candidate",candidate,"--camera-bundle",camera,"--appearance",appearance,"--geometry-npz",geometry,"--packed-measurement",mp,"--output",rp],output_dir/f"reference_{replay}.blender.log",maximum_seconds)
        packed.append(json.loads(mp.read_text()));refs.append(json.loads(rp.read_text()))
    qualification=qualify_subject_presentation_v1(packed[0],packed[1],refs[0],refs[1],appearance_tolerance=0.005,convergence_tolerance=0.005)
    if qualification.get("status")!="PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE":raise ValueError("STAGE48_SOURCE_EQUIVALENCE_FAILED")
    with np.load(geometry,allow_pickle=False) as z:
        rest=np.asarray(z["rest_vertices_source_world"],dtype=np.float64);faces=np.asarray(z["faces"],dtype=np.int64);isolated=np.asarray(z["isolated_position"],dtype=np.float64)
    cameras=json.loads(camera.read_text()).get("cameras") or []
    arrays={**build_pose_conditioned_order_arrays_v1(rest,faces,isolated,cameras,maximum_actions=16,maximum_rays_per_view=32),**build_measured_compositing_arrays_v1(packed[0],contribution_floor=1e-6)}
    arrays_path=output_dir/"STAGE48_RENDERER_WITNESS_ARRAYS.npz";save_npz(arrays_path,arrays)
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"response_chain":ref(response_chain_path),"teacher_scene":ref(teacher_path),"parents":{"geometry":ref(geometry),"appearance":ref(appearance),"source":ref(source),"camera":ref(camera)},"replays":{"packed_a":ref(output_dir/"PACKED_A.json"),"packed_b":ref(output_dir/"PACKED_B.json"),"source_reference_a":ref(output_dir/"SOURCE_REFERENCE_A.json"),"source_reference_b":ref(output_dir/"SOURCE_REFERENCE_B.json")},"renderer_subject_qualification":qualification,"arrays":ref(arrays_path),"counts":{"pose_order_pair_count":int(len(arrays.get("order_pair_face_a",[]))),"measured_composite_hit_count":int(len(arrays.get("composite_hit_face",[]))),"measured_composite_pair_count":int(len(arrays.get("composite_pair_face_a",[])))},"response_recomputed":False,"training_performed":False,"qualification_minted":False}
    manifest["stage48_identity_sha256"]=digest(manifest);mp=output_dir/"STAGE48_RENDERER_WITNESS_MANIFEST.json";mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"subject_id":subject_id,"stage48_identity_sha256":manifest["stage48_identity_sha256"]},sort_keys=True));return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--subject-id",required=True);p.add_argument("--response-chain",type=Path,required=True);p.add_argument("--blender",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--maximum-seconds",type=float,default=1800);a=p.parse_args();build(a.subject_id,a.response_chain,a.blender.resolve(),a.output_dir,a.maximum_seconds)
if __name__=="__main__":main()
