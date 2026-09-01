from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from realsas_compiler_core.rig import qualify_skeleton
from realsas_compiler_core.skin import qualify_skin
from realsas_compiler_core.surface import build_surface_from_persistence

from .arachne_demo_v1 import ArachneDemoConfig, ArachneDemoV1, canonical_joint_order
from .checkpoint_v1 import save_checkpoint
from .fit_contracts_v1 import (
    arachne_fit_loss, evaluate_arachne_weights,
    evaluate_geppetto_fit, evaluate_iris_fit,
    geppetto_fit_loss, iris_fit_loss,
)
from .geppetto_demo_v1 import GeppettoDemoConfig, GeppettoDemoV1
from .iris_demo_v1 import IrisDemoConfig, IrisDemoV1, build_observation_evidence_and_persistence
from .specimen_bundle_v1 import DemoFitBundle, load_fit_bundle

SURFACE_TO_SOURCE_P95_MAX_FRAC = 0.02


def _device(requested: str) -> torch.device:
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def fit_iris(bundle: DemoFitBundle, out: Path, device: torch.device, *, max_steps: int=3000):
    cfg=IrisDemoConfig(model_resolution=int(bundle.teacher_depth.shape[-1]))
    model=IrisDemoV1(cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    images=bundle.rgba.to(device);yaw=bundle.yaw_deg.to(device);td=bundle.teacher_depth.to(device);ts=bundle.teacher_support.to(device)
    verdict=None;stable=0
    for step in range(1,max_steps+1):
        model.train();raw=model(images,yaw);loss,_=iris_fit_loss(raw.depth,raw.support_logits,raw.log_sigma,td,ts)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        if step%25==0 or step==max_steps:
            model.eval();raw=model(images,yaw);verdict=evaluate_iris_fit(raw.depth,raw.support_logits,td,ts,support_threshold=cfg.support_threshold)
            stable=stable+1 if verdict.passed else 0
            print(f"IRIS step={step} loss={float(loss):.6g} pass={verdict.passed} metrics={verdict.metrics}",flush=True)
            if stable>=3:break
    if verdict is None or not verdict.passed:raise RuntimeError(f"IRIS_FIT_FAIL:{None if verdict is None else verdict.blockers}")
    ck=out/"iris_demo_v1.pt"
    meta=save_checkpoint(ck,component="IRIS",model=model,config=cfg,training_observation_sha256=bundle.observation_sha256,training_truth_sha256=bundle.truth_sha256,fit_verdict={"passed":True,"metrics":verdict.metrics},source_revision="demo/investor-single-specimen-e2e")
    model.eval();raw=model(images,yaw)
    evidence,groups=build_observation_evidence_and_persistence(raw,bundle.rgba,bundle.cameras,cfg=cfg)
    surface=build_surface_from_persistence(evidence,groups)
    if not surface.surface_nodes:raise RuntimeError("IRIS_FIT_PASS_BUT_EMPTY_SURFACE")
    return model,cfg,surface,verdict,meta


def _qualified_eval(skeleton,bundle:DemoFitBundle):
    pos=torch.tensor([j.position for j in skeleton.joints],dtype=torch.float32)
    idx={j.canonical_joint_id:i for i,j in enumerate(skeleton.joints)}
    parent=torch.tensor([-1 if j.parent_canonical_id is None else idx[j.parent_canonical_id] for j in skeleton.joints],dtype=torch.long)
    root=idx[skeleton.root_id]
    pmin=bundle.source_vertices.amin(dim=0);pmax=bundle.source_vertices.amax(dim=0)
    return evaluate_geppetto_fit(pos,parent,root,bundle.joint_positions,bundle.parent_index,bundle.root_index,subject_bbox_min=pmin,subject_bbox_max=pmax)


def fit_geppetto(bundle:DemoFitBundle,surface,out:Path,device:torch.device,*,max_steps:int=3500):
    if len(bundle.joint_positions)>64:raise RuntimeError("GEPPETTO_TEACHER_EXCEEDS_64_QUERY_DEMO_CAP")
    cfg=GeppettoDemoConfig(max_joints=64)
    model=GeppettoDemoV1(cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-6)
    sp=torch.tensor([[n.P for n in surface.surface_nodes]],dtype=torch.float32,device=device)
    tp=bundle.joint_positions.to(device);parent=bundle.parent_index.to(device)
    verdict=None;qualified=None;stable=0
    for step in range(1,max_steps+1):
        model.train();raw=model(sp);loss,_=geppetto_fit_loss(raw,sp[0],tp,parent,bundle.root_index)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        if step%25==0 or step==max_steps:
            try:
                proposal=model.propose(surface,device=device);qualified=qualify_skeleton(surface,proposal);verdict=_qualified_eval(qualified,bundle)
            except Exception as exc:
                verdict=None;qualified=None;print(f"GEPPETTO step={step} compile_wait={type(exc).__name__}:{exc}",flush=True)
            if verdict is not None:
                stable=stable+1 if verdict.passed else 0
                print(f"GEPPETTO step={step} loss={float(loss):.6g} pass={verdict.passed} metrics={verdict.metrics}",flush=True)
                if stable>=3:break
    if verdict is None or not verdict.passed or qualified is None:raise RuntimeError(f"GEPPETTO_FIT_FAIL:{None if verdict is None else verdict.blockers}")
    ck=out/"geppetto_demo_v1.pt"
    meta=save_checkpoint(ck,component="GEPPETTO",model=model,config=cfg,training_observation_sha256=bundle.observation_sha256,training_truth_sha256=bundle.truth_sha256,fit_verdict={"passed":True,"metrics":verdict.metrics},source_revision="demo/investor-single-specimen-e2e")
    return model,cfg,qualified,verdict,meta


def _teacher_weights_on_surface(bundle:DemoFitBundle,surface,skeleton):
    source=bundle.source_vertices.cpu().numpy();points=np.asarray([n.P for n in surface.surface_nodes],dtype=np.float64)
    tree=cKDTree(source);dist,nearest=tree.query(points,k=1)
    diag=float(np.linalg.norm(source.max(axis=0)-source.min(axis=0))) or 1.0
    p95=float(np.quantile(dist/diag,0.95))
    if p95>SURFACE_TO_SOURCE_P95_MAX_FRAC:raise RuntimeError(f"ARACHNE_SURFACE_TRUTH_QUERY_TOO_FAR:p95/D={p95}")
    qpos=np.asarray([j.position for j in skeleton.joints],dtype=np.float64);tpos=bundle.joint_positions.cpu().numpy()
    row,col=linear_sum_assignment(np.linalg.norm(qpos[:,None,:]-tpos[None,:,:],axis=-1))
    if len(row)!=len(tpos) or len(qpos)!=len(tpos):raise RuntimeError("ARACHNE_REQUIRES_EXACT_JOINT_COUNT_MATCH")
    teacher_col_for_q={int(q):int(t) for q,t in zip(row,col)}
    order=canonical_joint_order(skeleton);qid={j.canonical_joint_id:i for i,j in enumerate(skeleton.joints)}
    source_w=bundle.source_vertex_weights.cpu().numpy()[nearest]
    target=np.stack([source_w[:,teacher_col_for_q[qid[jid]]] for jid in order],axis=1).astype(np.float32)
    target/=np.maximum(target.sum(axis=1,keepdims=True),1e-12)
    return torch.from_numpy(target),{"nearest_surface_to_source_p95_over_D":p95,"nearest_surface_to_source_max_over_D":float(np.max(dist/diag))}


def _arachne_inputs(surface,skeleton,device):
    order=canonical_joint_order(skeleton);by={j.canonical_joint_id:j for j in skeleton.joints}
    sp=torch.tensor([n.P for n in surface.surface_nodes],dtype=torch.float32,device=device)
    jp=torch.tensor([by[j].position for j in order],dtype=torch.float32,device=device)
    pp=torch.tensor([by[by[j].parent_canonical_id].position if by[j].parent_canonical_id else by[j].position for j in order],dtype=torch.float32,device=device)
    root=torch.tensor([1.0 if j==skeleton.root_id else 0.0 for j in order],dtype=torch.float32,device=device)
    return sp,jp,pp,root


def fit_arachne(bundle:DemoFitBundle,surface,skeleton,out:Path,device:torch.device,*,max_steps:int=3000):
    cfg=ArachneDemoConfig();model=ArachneDemoV1(cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-6)
    teacher,audit=_teacher_weights_on_surface(bundle,surface,skeleton);teacher=teacher.to(device)
    sp,jp,pp,root=_arachne_inputs(surface,skeleton,device)
    verdict=None;stable=0
    for step in range(1,max_steps+1):
        model.train();logits=model(sp,jp,pp,root);loss,_=arachne_fit_loss(logits,teacher)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        if step%25==0 or step==max_steps:
            pred=torch.softmax(model(sp,jp,pp,root),dim=-1);verdict=evaluate_arachne_weights(pred,teacher)
            stable=stable+1 if verdict.passed else 0
            print(f"ARACHNE step={step} loss={float(loss):.6g} pass={verdict.passed} metrics={verdict.metrics}",flush=True)
            if stable>=3:break
    if verdict is None or not verdict.passed:raise RuntimeError(f"ARACHNE_FIT_FAIL:{None if verdict is None else verdict.blockers}")
    proposal=model.propose(surface,skeleton,device=device);qualified=qualify_skin(surface,skeleton,proposal,max_influences=cfg.top_k)
    ck=out/"arachne_demo_v1.pt"
    metrics={**verdict.metrics,**audit}
    meta=save_checkpoint(ck,component="ARACHNE",model=model,config=cfg,training_observation_sha256=bundle.observation_sha256,training_truth_sha256=bundle.truth_sha256,fit_verdict={"passed":True,"metrics":metrics},source_revision="demo/investor-single-specimen-e2e")
    return model,cfg,qualified,metrics,meta


def fit_all(bundle_dir:str|Path,out_dir:str|Path,*,device:str="auto") -> dict:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);bundle=load_fit_bundle(bundle_dir);dev=_device(device)
    print(f"DEMO FIT DEVICE = {dev}",flush=True)
    _,_,surface,iv,im=fit_iris(bundle,out,dev)
    _,_,skeleton,gv,gm=fit_geppetto(bundle,surface,out,dev)
    _,_,skin,ametrics,am=fit_arachne(bundle,surface,skeleton,out,dev)
    report={
        "schema":"RealSaS.SingleSpecimenLearnedFitClosure.v1","status":"PASS",
        "generalization_claim":False,"single_specimen_fit":True,"manual_output_injection":False,
        "device":str(dev),"observation_sha256":bundle.observation_sha256,"truth_sha256":bundle.truth_sha256,
        "iris":iv.metrics,"geppetto":gv.metrics,"arachne":ametrics,
        "surface_node_count":len(surface.surface_nodes),"qualified_joint_count":len(skeleton.joints),"qualified_skin_row_count":len(skin.rows),
        "checkpoints":{"iris":im,"geppetto":gm,"arachne":am},
    }
    _json(out/"FIT_CLOSURE_V1.json",report);return report
