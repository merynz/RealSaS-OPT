#!/usr/bin/env python3
"""Run canonical Stage48 renderer-intervention witness for one Truth139 subject.

Consumes an already closed Stage47/Response chain and never recomputes Response.
The historical renderer adapter requires an Attempt-bound run_manifest whenever
non-PASS rays are retained; this standalone runner therefore recreates the exact
Attempt layout:

  <output>/attempt/run_manifest.json
  <output>/attempt/artifacts/48_PRESENTATION_RESPONSE_TRUTH/PACKED_*.json

The preregistered policy is loaded byte-for-byte from
canonical/PRESENTATION_TRUTH_OPERATOR_REQUEST_V1.json: 32 measured rays,
minimum 16 PASS rays, maximum 6 hits, 256 Cycles samples.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.iris.truth.presentation_qualification_v1 import qualify_subject_presentation_v1
from models.iris.truth.presentation_response_v1 import build_measured_compositing_arrays_v1, build_pose_conditioned_order_arrays_v1

SCHEMA="RealSaS.Truth139Stage48RendererWitness.v1"
STATUS="PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"
RESPONSE_STATUS="PASS_TRUTH139_IRIS_RESPONSE_EVIDENCE_CHAIN__DERIVED_TRUTH_ADMISSION_SEPARATE"
OPERATOR_REQUEST=ROOT/"canonical/PRESENTATION_TRUTH_OPERATOR_REQUEST_V1.json"
STAGE_ID="48_PRESENTATION_RESPONSE_TRUTH"

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
def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial");tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n");tmp.replace(path)
def run_blender(blender,program,args,log,maximum,blend=None):
    cmd=[str(blender),"-b"]+([str(blend)] if blend else ["--factory-startup"])+["--python-exit-code","2","--python",str(program),"--",*map(str,args)]
    log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with log.open("w") as f:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT)
        while True:
            try:rc=p.wait(timeout=20);break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-start;print(f"[STAGE48] elapsed_s={elapsed:.1f}",flush=True)
                if elapsed>=maximum:
                    p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise RuntimeError("STAGE48_TIME_BUDGET")
    if rc:raise RuntimeError("STAGE48_BLENDER_FAILED\n"+log.read_text(errors="replace")[-16000:])
def save_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)
def canonical_policy():
    request=json.loads(OPERATOR_REQUEST.read_text());cfg=dict(request.get("presentation_truth") or {})
    required={"mode":"OBSERVATION_REST_PRESENTATION_RESPONSE_TRUTH_V1","measured_rays_per_subject":32,"minimum_pass_rays_per_subject":16,"maximum_hits_per_ray":6,"cycles_samples":256}
    for key,value in required.items():
        if cfg.get(key)!=value:raise ValueError(f"STAGE48_CANONICAL_POLICY_DRIFT:{key}:{cfg.get(key)}:{value}")
    if request.get("training_allowed") is not False:raise ValueError("STAGE48_OPERATOR_REQUEST_TRAINING_AUTHORITY")
    return request,cfg
def make_attempt_manifest(attempt_root,subject_id,response_chain,cfg):
    manifest={"schema":"RealSaS.Truth139StandaloneStage48AttemptManifest.v1","subject_id":subject_id,"stage_id":STAGE_ID,"presentation_truth":cfg,"operator_request":{"path":str(OPERATOR_REQUEST.resolve()),"sha256":sha256(OPERATOR_REQUEST)},"response_chain":ref(response_chain),"training_allowed":False,"product_pass_authority":False,"standalone_attempt_equivalent_to_platform_layout":True}
    manifest["attempt_identity_sha256"]=digest(manifest);atomic_json(attempt_root/"run_manifest.json",manifest);return manifest

def build(subject_id,response_chain_path:Path,blender:Path,output_dir:Path):
    chain=json.loads(response_chain_path.resolve().read_text())
    if chain.get("status")!=RESPONSE_STATUS or chain.get("subject_id")!=subject_id:raise ValueError("STAGE48_RESPONSE_CHAIN_NOT_PASS")
    teacher_path=load_ref(chain["outputs"]["teacher_scene"],"STAGE48_TEACHER_DRIFT");scene=json.loads(teacher_path.read_text())
    if scene.get("schema")!="RealSaS.TeacherSceneIR.v2" or scene.get("rest_contract")!="EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY":raise ValueError("STAGE48_TEACHER_SCOPE")
    if (scene.get("observation_rest_binding") or {}).get("status")!="PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY":raise ValueError("STAGE48_TEACHER_REST_BINDING")
    geometry=payload_ref(scene,"geometry_and_response.npz","geometry_payloads");appearance=payload_ref(scene,"appearance.json","appearance_payloads");blend=payload_ref(scene,"source_scene.blend","appearance_payloads")
    source=load_ref(chain["inputs"]["source"],"STAGE48_SOURCE_DRIFT");candidate=load_ref(chain["inputs"]["source_rest_candidate"],"STAGE48_CANDIDATE_DRIFT");camera=load_ref(chain["inputs"]["camera_bundle"],"STAGE48_CAMERA_DRIFT")
    _,cfg=canonical_policy();maximum=float(cfg["maximum_seconds_per_subject"]);maximum_rays=int(cfg["measured_rays_per_subject"]);minimum_pass=int(cfg["minimum_pass_rays_per_subject"]);maximum_hits=int(cfg["maximum_hits_per_ray"]);cycles_samples=int(cfg["cycles_samples"])
    output_dir=output_dir.resolve();output_dir.mkdir(parents=True,exist_ok=True);attempt_root=output_dir/"attempt";artifact_root=attempt_root/"artifacts"/STAGE_ID;artifact_root.mkdir(parents=True,exist_ok=True);make_attempt_manifest(attempt_root,subject_id,response_chain_path,cfg)
    packed=[];refs=[]
    for replay in ("A","B"):
        mp=artifact_root/f"PACKED_{replay}.json"
        if not mp.exists():run_blender(blender,ROOT/"tools/truth_foundry/blender_presentation_ray_measure_v1.py",["--subject-id",subject_id,"--teacher-scene",teacher_path,"--geometry-npz",geometry,"--candidate",candidate,"--camera-bundle",camera,"--expected-blend-sha256",sha256(blend),"--output",mp,"--maximum-rays",maximum_rays,"--maximum-hits",maximum_hits,"--cycles-samples",cycles_samples,"--large-footprint-fraction",float(cfg["large_footprint_fraction"]),"--small-footprint-fraction",float(cfg["small_footprint_fraction"]),"--convergence-tolerance",float(cfg["convergence_tolerance"]),"--representation-tolerance",float(cfg["representation_tolerance"]),"--measurement-tolerance",float(cfg["measurement_tolerance"])],output_dir/f"packed_{replay}.blender.log",maximum,blend=blend)
        packed_value=json.loads(mp.read_text());passed=int(packed_value.get("pass_ray_count",-1));selected=int(packed_value.get("selected_ray_count",-1))
        if not (0<selected<=maximum_rays and passed>=minimum_pass):raise ValueError(f"STAGE48_PASS_DOMAIN_NOT_MET:{replay}:selected={selected}:pass={passed}:minimum={minimum_pass}")
        rp=artifact_root/f"SOURCE_REFERENCE_{replay}.json"
        if not rp.exists():run_blender(blender,ROOT/"tools/truth_foundry/blender_source_ray_appearance_reference_v1.py",["--source",source,"--expected-source-sha256",sha256(source),"--subject-id",subject_id,"--candidate",candidate,"--camera-bundle",camera,"--appearance",appearance,"--geometry-npz",geometry,"--packed-measurement",mp,"--output",rp],output_dir/f"reference_{replay}.blender.log",maximum)
        packed.append(packed_value);refs.append(json.loads(rp.read_text()))
    qualification=qualify_subject_presentation_v1(packed[0],packed[1],refs[0],refs[1],appearance_tolerance=float(cfg["source_equivalence_tolerance"]),convergence_tolerance=float(cfg["convergence_tolerance"]))
    if qualification.get("status")!="PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE":raise ValueError("STAGE48_SOURCE_EQUIVALENCE_FAILED")
    with np.load(geometry,allow_pickle=False) as z:rest=np.asarray(z["rest_vertices_source_world"],dtype=np.float64);faces=np.asarray(z["faces"],dtype=np.int64);isolated=np.asarray(z["isolated_position"],dtype=np.float64)
    cameras=json.loads(camera.read_text()).get("cameras") or []
    arrays={**build_pose_conditioned_order_arrays_v1(rest,faces,isolated,cameras,maximum_actions=int(cfg["pose_conditioned_action_budget"]),maximum_rays_per_view=int(cfg["geometric_rays_per_view"])),**build_measured_compositing_arrays_v1(packed[0],contribution_floor=float(cfg["student_observable_contribution_floor"]))}
    arrays_path=output_dir/"STAGE48_RENDERER_WITNESS_ARRAYS.npz";save_npz(arrays_path,arrays)
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"response_chain":ref(response_chain_path),"teacher_scene":ref(teacher_path),"operator_request":ref(OPERATOR_REQUEST),"attempt_manifest":ref(attempt_root/"run_manifest.json"),"parents":{"geometry":ref(geometry),"appearance":ref(appearance),"source":ref(source),"camera":ref(camera)},"policy":{"measured_rays_per_subject":maximum_rays,"minimum_pass_rays_per_subject":minimum_pass,"maximum_hits_per_ray":maximum_hits,"cycles_samples":cycles_samples,"pose_conditioned_action_budget":int(cfg["pose_conditioned_action_budget"]),"geometric_rays_per_view":int(cfg["geometric_rays_per_view"]),"student_observable_contribution_floor":float(cfg["student_observable_contribution_floor"])},"replays":{"packed_a":ref(artifact_root/"PACKED_A.json"),"packed_b":ref(artifact_root/"PACKED_B.json"),"source_reference_a":ref(artifact_root/"SOURCE_REFERENCE_A.json"),"source_reference_b":ref(artifact_root/"SOURCE_REFERENCE_B.json")},"pass_domain":{"packed_a_pass_ray_count":int(packed[0]["pass_ray_count"]),"packed_b_pass_ray_count":int(packed[1]["pass_ray_count"]),"minimum_required":minimum_pass,"failed_rays_retained":bool(int(packed[0].get("fail_ray_count",0)) or int(packed[1].get("fail_ray_count",0)))},"renderer_subject_qualification":qualification,"arrays":ref(arrays_path),"counts":{"pose_order_pair_count":int(len(arrays.get("order_pair_face_a",[]))),"measured_composite_hit_count":int(len(arrays.get("composite_hit_face",[]))),"measured_composite_pair_count":int(len(arrays.get("composite_pair_face_a",[])))},"response_recomputed":False,"training_performed":False,"qualification_minted":False}
    manifest["stage48_identity_sha256"]=digest(manifest);mp=output_dir/"STAGE48_RENDERER_WITNESS_MANIFEST.json";atomic_json(mp,manifest);print(json.dumps({"status":STATUS,"subject_id":subject_id,"pass_rays_a":manifest["pass_domain"]["packed_a_pass_ray_count"],"pass_rays_b":manifest["pass_domain"]["packed_b_pass_ray_count"],"stage48_identity_sha256":manifest["stage48_identity_sha256"]},sort_keys=True));return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--subject-id",required=True);p.add_argument("--response-chain",type=Path,required=True);p.add_argument("--blender",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();build(a.subject_id,a.response_chain,a.blender.resolve(),a.output_dir)
if __name__=="__main__":main()
