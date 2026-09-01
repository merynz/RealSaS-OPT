from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os

import numpy as np
import torch

from realsas_compiler_core.mesh_binding import bind_identity_mesh_skin, qualify_identity_subset_mesh
from realsas_compiler_core.product import assemble_product_v2, bind_proof, project_runtime_package
from realsas_compiler_core.rig import qualify_skeleton
from realsas_compiler_core.skin import qualify_skin
from realsas_compiler_core.surface import build_surface_from_persistence

from .arachne_demo_v1 import ArachneDemoConfig, ArachneDemoV1
from .checkpoint_v1 import load_checkpoint
from .geppetto_demo_v1 import GeppettoDemoConfig, GeppettoDemoV1
from .historical_runtime_bridge_v1 import run_historical_reference_runtime
from .inference_firewall_v1 import ImageOnlyInferenceRequest, assert_image_only_request, write_firewall_receipt
from .iris_demo_v1 import IrisDemoConfig, IrisDemoV1, build_observation_evidence_and_persistence
from .mesh_demo_v1 import build_view_local_identity_mesh_candidate
from .runtime_projection_v1 import build_generic_articulation_sweep, build_historical_runtime_request


def _load_model(path: str|Path, cls, cfg_cls, component: str, device: torch.device):
    payload=torch.load(path,map_location="cpu",weights_only=False);meta=dict(payload["metadata"]);cfg=cfg_cls(**dict(meta["config"]))
    model=cls(cfg).to(device);load_checkpoint(path,model,expected_component=component);model.eval();return model,cfg,meta


def _load_image_only(path: str|Path):
    d=np.load(path,allow_pickle=False)
    allowed={"rgba","yaw_deg"}
    extra=set(d.files)-allowed
    if extra:raise RuntimeError(f"IMAGE_ONLY_INPUT_CONTAINS_UNDECLARED_ARRAYS:{sorted(extra)}")
    rgba=np.asarray(d["rgba"])
    if rgba.ndim!=4 or rgba.shape[0]!=8 or rgba.shape[-1]!=4:raise ValueError("rgba must be [8,H,W,4]")
    if rgba.dtype==np.uint8:rgba=rgba.astype(np.float32)/255.0
    else:rgba=rgba.astype(np.float32)
    yaw=np.asarray(d["yaw_deg"],np.float32)
    if yaw.shape!=(8,):raise ValueError("yaw_deg must be [8]")
    return torch.from_numpy(np.transpose(rgba,(0,3,1,2))[None]),torch.from_numpy(yaw)


def run_final_image_only_inference(request: ImageOnlyInferenceRequest, *, device: str="auto") -> dict:
    assert_image_only_request(request)
    out=Path(request.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_firewall_receipt(request,out/"INFERENCE_FIREWALL_RECEIPT_V1.json")
    dev=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device))
    images,yaw=_load_image_only(request.images_path);cameras=json.loads(Path(request.cameras_path).read_text(encoding="utf-8"))
    if not isinstance(cameras,list) or len(cameras)!=8:raise ValueError("cameras.json must contain 8 cameras")

    iris,icfg,_=_load_model(request.iris_checkpoint,IrisDemoV1,IrisDemoConfig,"IRIS",dev)
    geppetto,gcfg,_=_load_model(request.geppetto_checkpoint,GeppettoDemoV1,GeppettoDemoConfig,"GEPPETTO",dev)
    arachne,acfg,_=_load_model(request.arachne_checkpoint,ArachneDemoV1,ArachneDemoConfig,"ARACHNE",dev)

    with torch.no_grad():raw=iris(images.to(dev),yaw.to(dev))
    evidence,groups=build_observation_evidence_and_persistence(raw,images,cameras,cfg=icfg)
    surface=build_surface_from_persistence(evidence,groups)
    if not surface.surface_nodes:raise RuntimeError("FINAL_INFERENCE_EMPTY_SURFACE")

    skeleton=qualify_skeleton(surface,geppetto.propose(surface,device=dev))
    skin=qualify_skin(surface,skeleton,arachne.propose(surface,skeleton,device=dev),max_influences=acfg.top_k)

    view_index=0
    camera_contract=cameras[view_index]
    candidate=build_view_local_identity_mesh_candidate(surface,view_index=view_index,camera_contract=camera_contract)
    mesh=qualify_identity_subset_mesh(surface,candidate)
    mesh_skin=bind_identity_mesh_skin(surface,skeleton,skin,mesh)
    product=assemble_product_v2(surface,skeleton,skin,mesh,mesh_skin,editable_metadata={"demo_image_only":True,"generalization_claim":False})

    motion=build_generic_articulation_sweep(skeleton,camera_contract,frame_count=48)
    runtime_req=build_historical_runtime_request(
        skeleton=skeleton,mesh=mesh,mesh_skin=mesh_skin,camera=camera_contract,frames=motion,
        raster_size=(icfg.model_resolution,icfg.model_resolution),arap_enabled=True,
    )
    source_zip=os.environ.get("REALSAS_V05_SOURCE_ZIP","")
    if not source_zip:raise RuntimeError("REALSAS_V05_SOURCE_ZIP environment variable is required")
    worker=Path(__file__).with_name("historical_runtime_worker_v1.py")
    runtime=run_historical_reference_runtime(runtime_req,source_zip=source_zip,worker_script=worker)

    frames=runtime.output["frames"]
    finite=all(np.isfinite(np.asarray(f["vertices_xy"],dtype=float)).all() for f in frames)
    no_mutation=runtime.output.get("compiler_artifacts_mutated") is False
    proof_pass=bool(finite and no_mutation and len(frames)==48)
    proof=bind_proof(product,probe_plan={"schema":"RealSaS.DemoMotionProbe.v1","frame_count":48,"historical_evaluator":runtime.output.get("historical_evaluator_semantic_version")},measurements={"finite_frames":finite,"frame_count":len(frames),"compiler_artifacts_mutated":not no_mutation},passed=proof_pass,proof_payload={"demo_scope":"SINGLE_SPECIMEN_ARCHITECTURE_CLOSURE","generalization_claim":False})
    runtime_pkg=project_runtime_package(product,proof,manifest={"schema":"RealSaS.InvestorDemoRuntimeManifest.v1","historical_source_zip_sha256":runtime.source_zip_sha256},runtime_payload_ref=str(out/"HISTORICAL_RUNTIME_FRAMES_V1.json"))

    (out/"HISTORICAL_RUNTIME_FRAMES_V1.json").write_text(json.dumps(runtime.output,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"CANONICAL_PRODUCT_V2.json").write_text(json.dumps(product.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"PROOF_FRAME_V1.json").write_text(json.dumps(proof.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"RUNTIME_PACKAGE_IR_V1.json").write_text(json.dumps(runtime_pkg.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report={
        "schema":"RealSaS.SingleSpecimenImageOnlyE2EClosure.v1","status":"PASS" if proof_pass else "FAIL",
        "generalization_claim":False,"image_only_final_inference":True,"manual_output_injection":False,
        "device":str(dev),"surface_node_count":len(surface.surface_nodes),"joint_count":len(skeleton.joints),
        "skin_row_count":len(skin.rows),"mesh_vertex_count":len(mesh.vertices),"mesh_face_count":len(mesh.faces),
        "runtime_frame_count":len(frames),"historical_runtime_source_sha256":runtime.source_zip_sha256,
        "product_state_hash":product.product_state_hash,"proof_passed":proof.passed,
    }
    (out/"E2E_CLOSURE_V1.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report
