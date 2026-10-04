from __future__ import annotations

"""C4.0 direct carrier-weight mechanical optimization oracle.

This court is deliberately not a model fit.

It freezes:
- Stage19 carrier topology/XYZ,
- A2-verified ATLAS rig,
- frozen MIRA semantic backbone/readout,
- all identity carrier weight rows.

It optimizes only the logits of non-identity compiled-carrier vertices against a
differentiable carrier-native G3/G3B surrogate. No teacher weights and no
diagnostic transfer ceiling enter the objective.

The diagnostic transfer ceiling is loaded only after optimization for comparison.
Hard Compiler G3/G3B remains the verdict authority.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import (
    derive_joint_frames_from_skeleton,
)
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mechanical_carrier_skin_v1 import (
    qualify_mechanical_carrier_skin_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    _pose_skin_matrices,
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_EDGE_RATIO,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from models.shared.joint_mechanical_loss_v1 import (
    JointMechanicalLossConfigV1,
    joint_mechanical_loss_v1,
)
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_carrier,
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.audit_knight_carrier_normal_authority_c31_v1 import (
    _diagnostic_evidence,
    _transport_gsa_normals,
)
from tools.audit_knight_mira_support_mixture_c32_v1 import (
    _legacy_transfer_ceiling,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


def _non_identity_mask(candidate, ordered_vertex_ids):
    by_id={str(v.candidate_vertex_id):v for v in candidate.vertices}
    mask=[]
    for vid in ordered_vertex_ids:
        v=by_id[str(vid)]
        coeffs=tuple(v.support_binding.coefficients)
        identity=(
            str(v.support_binding.mode)=="IDENTITY_SURFACE_NODE"
            and len(coeffs)==1
            and abs(float(coeffs[0][1])-1.0)<=1e-12
        )
        mask.append(not identity)
    return np.asarray(mask,bool)


def _face_indices(candidate, ordered_vertex_ids):
    row={str(v):i for i,v in enumerate(ordered_vertex_ids)}
    out=[]
    for face in candidate.faces:
        ids=tuple(map(str,face))
        if len(ids)!=3 or len(set(ids))!=3 or any(v not in row for v in ids):
            raise RuntimeError("C40_FACE_AXIS_DRIFT")
        out.append(tuple(row[v] for v in ids))
    return np.asarray(out,np.int64)


def _local_problem(positions, faces, base_weights, non_identity):
    affected=np.any(non_identity[faces],axis=1)
    affected_faces=faces[affected]
    if not len(affected_faces):
        raise RuntimeError("C40_NO_NON_IDENTITY_AFFECTED_FACE")
    used=np.asarray(sorted(set(affected_faces.reshape(-1).tolist())),np.int64)
    remap=np.full(len(positions),-1,np.int64)
    remap[used]=np.arange(len(used),dtype=np.int64)
    local_faces=remap[affected_faces]
    local_non=non_identity[used]
    active=np.nonzero(local_non)[0].astype(np.int64)
    if not len(active):
        raise RuntimeError("C40_NO_ACTIVE_VERTEX")
    return {
        "global_vertex_indices":used,
        "local_faces":local_faces,
        "active_local_indices":active,
        "rest":np.asarray(positions,np.float64)[used],
        "base_weights":np.asarray(base_weights,np.float64)[used],
        "affected_face_count":int(len(affected_faces)),
    }


def _probe_matrices(skeleton,cameras):
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    joint_ids=tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    rows=[]
    ids=[]
    rest=_pose_skin_matrices(
        skeleton,frames,joint_id=None,local_axis_index=None,degrees=0.0
    )
    rows.append(np.stack([rest[j] for j in joint_ids],axis=0))
    ids.append("REST")
    for jid in sorted(joint_ids):
        for axis_i,axis_name in enumerate(("X","Y","Z")):
            for sign in (-1.0,1.0):
                deg=sign*10.0
                by=_pose_skin_matrices(
                    skeleton,frames,joint_id=jid,
                    local_axis_index=axis_i,degrees=deg,
                )
                rows.append(np.stack([by[j] for j in joint_ids],axis=0))
                ids.append(f"{jid}:LOCAL_{axis_name}:{deg:+g}")
    return tuple(ids),np.asarray(rows,np.float64)


def _compose_weights(base,active_idx,active_logits):
    active=torch.softmax(active_logits,dim=-1)
    inserted=torch.zeros_like(base).index_copy(0,active_idx,active)
    mask=torch.zeros((base.shape[0],1),device=base.device,dtype=base.dtype)
    mask=mask.index_fill(0,active_idx,1.0)
    return base*(1.0-mask)+inserted,active


def _surrogate_eval(rest,faces,weights,mats,cfg,probe_batch):
    totals=[]
    max_condition=0.0
    min_area=float("inf")
    max_area=0.0
    max_edge=0.0
    with torch.no_grad():
        for start in range(0,len(mats),probe_batch):
            out=joint_mechanical_loss_v1(
                rest,faces,weights,mats[start:start+probe_batch],config=cfg
            )
            totals.append(float(out["loss"].detach().cpu()))
            max_condition=max(max_condition,float(out["max_condition"].detach().cpu()))
            min_area=min(min_area,float(out["min_area_ratio"].detach().cpu()))
            max_area=max(max_area,float(out["max_area_ratio"].detach().cpu()))
            max_edge=max(max_edge,float(out["max_edge_ratio"].detach().cpu()))
    return {
        "mean_batch_loss":float(np.mean(totals)),
        "max_condition":max_condition,
        "min_area_ratio":min_area,
        "max_area_ratio":max_area,
        "max_edge_ratio":max_edge,
    }


def _hard_eval(
    *,
    weights,
    candidate,
    evidence,
    skeleton,
    query,
    surface,
    envelope,
    cameras,
    policy,
):
    skin=qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=evidence,
        skeleton=skeleton,
        vertex_ids=query.carrier_vertex_ids,
        joint_ids=query.joint_ids,
        weights=weights,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=0.1,
    )
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras.cameras,policy=policy,
    )
    g3b=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras.cameras,policy=policy,
        stress_all_faces=True,
    )
    return skin,g3,g3b


def main(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("C40_CUDA_NOT_AVAILABLE")
    if args.steps<1 or args.lr<=0 or args.probe_batch<1 or args.log_every<1:
        raise ValueError("C40_HYPERPARAM_INVALID")
    if args.trust_weight<0:
        raise ValueError("C40_TRUST_WEIGHT_NEGATIVE")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dev=torch.device(args.device)

    ctx=_ctx(args.authority_root,args.run_id)
    surface=rigging_surface_from_dict(stage_output_payload(
        ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    addressing=surface_addressing_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.SurfaceAddressingIR.v1"))
    static=static_mesh_qualification_from_dict(stage_output_payload(
        ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.StaticCanonicalMeshQualificationIR.v1"))
    base_carrier=build_mechanical_carrier_evidence_v1(
        candidate,static_qualification=static,surface_addressing=addressing)
    surface_tensor=tensorize_rigging_surface_v1(surface)
    signed_n,signed_v,_mass,_count=_transport_gsa_normals(
        candidate,surface_tensor,base_carrier.ordered_vertex_ids)
    evidence=_diagnostic_evidence(base_carrier,signed_n,signed_v)

    policy=mesh_policy_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    envelope=deformation_envelope_from_dict(stage_output_payload(
        ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    cameras=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))

    conditioning,ci,memory,tokens,decoder,mira_result,telemetry=_run_mira_backbone(
        surface=surface,skeleton=skeleton,arm_dir=args.arm_dir,
        device=args.device,query_chunk=args.backbone_query_chunk)
    query,base_weights=_decode_carrier(
        candidate=candidate,carrier=evidence,surface_tensor=surface_tensor,
        conditioning=conditioning,memory=memory,tokens=tokens,decoder=decoder,
        device=args.device,chunk=args.readout_chunk)
    base_weights=np.asarray(base_weights,np.float64)

    non_identity=_non_identity_mask(candidate,query.carrier_vertex_ids)
    faces=_face_indices(candidate,query.carrier_vertex_ids)
    local=_local_problem(
        np.asarray(evidence.positions,np.float64),faces,base_weights,non_identity)
    probe_ids,probe_np=_probe_matrices(skeleton,cameras.cameras)

    rest=torch.as_tensor(local["rest"],device=dev,dtype=torch.float32)
    local_faces=torch.as_tensor(local["local_faces"],device=dev,dtype=torch.long)
    base=torch.as_tensor(local["base_weights"],device=dev,dtype=torch.float32)
    active_idx=torch.as_tensor(
        local["active_local_indices"],device=dev,dtype=torch.long)
    base_active=base[active_idx].detach()
    initial_logits=torch.log(base_active.clamp_min(1e-12))
    logits=torch.nn.Parameter(initial_logits.clone())
    optimizer=torch.optim.Adam([logits],lr=float(args.lr))
    mats=torch.as_tensor(probe_np,device=dev,dtype=torch.float32)

    cfg=JointMechanicalLossConfigV1(
        max_condition=float(policy.g3_max_dynamic_condition_number),
        min_area_ratio=float(policy.g3_min_dynamic_area_ratio),
        max_area_ratio=float(policy.g3_max_dynamic_area_ratio),
        max_edge_ratio=float(DEFAULT_MAX_EDGE_RATIO),
    )

    generator=torch.Generator(device="cpu").manual_seed(args.seed)
    permutation=torch.randperm(len(mats),generator=generator).tolist()
    cursor=0
    best=None
    history=[]

    initial_weights,_=_compose_weights(base,active_idx,logits)
    initial_eval=_surrogate_eval(
        rest,local_faces,initial_weights,mats,cfg,args.probe_batch)
    print("C40_INITIAL="+json.dumps(initial_eval,sort_keys=True),flush=True)

    for step in range(1,args.steps+1):
        if cursor+args.probe_batch>len(permutation):
            permutation=torch.randperm(len(mats),generator=generator).tolist()
            cursor=0
        ids=permutation[cursor:cursor+args.probe_batch]
        cursor+=args.probe_batch
        batch=mats[torch.as_tensor(ids,device=dev,dtype=torch.long)]

        optimizer.zero_grad(set_to_none=True)
        weights,active=_compose_weights(base,active_idx,logits)
        mech=joint_mechanical_loss_v1(
            rest,local_faces,weights,batch,config=cfg)
        trust=(active-base_active).abs().sum(-1).mean()
        total=mech["loss"]+float(args.trust_weight)*trust
        if not bool(torch.isfinite(total)):
            raise RuntimeError("C40_NONFINITE_LOSS")
        total.backward()
        torch.nn.utils.clip_grad_norm_([logits],max_norm=10.0)
        optimizer.step()

        if step==1 or step%args.log_every==0 or step==args.steps:
            weights,_=_compose_weights(base,active_idx,logits)
            ev=_surrogate_eval(
                rest,local_faces,weights,mats,cfg,args.probe_batch)
            with torch.no_grad():
                trust_now=float(
                    (weights[active_idx]-base_active).abs().sum(-1).mean().cpu())
                max_l1=float(
                    (weights[active_idx]-base_active).abs().sum(-1).max().cpu())
            row={
                "step":step,
                **ev,
                "trust_mean_row_l1":trust_now,
                "trust_max_row_l1":max_l1,
            }
            history.append(row)
            key=(ev["mean_batch_loss"],trust_now,step)
            if best is None or key<best[0]:
                best=(key,logits.detach().cpu().clone(),row)
            print("C40_STEP="+json.dumps(row,sort_keys=True),flush=True)

    if best is None:
        raise RuntimeError("C40_NO_BEST_STATE")
    with torch.no_grad():
        logits.copy_(best[1].to(dev))
        local_final,_=_compose_weights(base,active_idx,logits)

    full=base_weights.copy()
    global_used=np.asarray(local["global_vertex_indices"],np.int64)
    active_local=np.asarray(local["active_local_indices"],np.int64)
    active_global=global_used[active_local]
    full[active_global]=local_final[active_idx].detach().cpu().numpy().astype(np.float64)
    full=full/full.sum(axis=1,keepdims=True)

    # Hard verdict on the exact full carrier.
    skin,g3,g3b=_hard_eval(
        weights=full,candidate=candidate,evidence=evidence,skeleton=skeleton,
        query=query,surface=surface,envelope=envelope,cameras=cameras,policy=policy)

    # Evaluation-only semantic ceiling. Never used by the optimizer.
    legacy=_decode_legacy(
        conditioning=conditioning,ci=ci,memory=memory,tokens=tokens,
        decoder=decoder,device=args.device,chunk=args.readout_chunk)
    ceiling=_legacy_transfer_ceiling(query=query,legacy_weights=legacy)

    delta=np.abs(full-base_weights).sum(axis=1)
    to_ceiling=np.abs(full-ceiling).sum(axis=1)
    identity=~non_identity
    hard=_summarize_g3(g3,g3b)
    report={
        "schema":"RealSaS.MIRACarrierDirectWeightMechanicalOracle.v1",
        "status":(
            "PASS_DIRECT_WEIGHT_ORACLE__HEAD_FIT_JUSTIFIED"
            if hard["g3_passed"] and hard["g3b_passed"]
            else "FAIL_DIRECT_WEIGHT_ORACLE__DO_NOT_FIT_HEAD_YET"
        ),
        "run_id":args.run_id,
        "carrier_topology_hash":evidence.topology_hash,
        "diagnostic_signed_normal_evidence_hash":evidence.carrier_evidence_hash,
        "mira_checkpoint_sha256":mira_result["model_sha256"],
        "query_hash":query.query_hash,
        "training_used":False,
        "direct_weight_optimization_used":True,
        "teacher_weight_objective_used":False,
        "diagnostic_ceiling_used_by_optimizer":False,
        "product_authority_minted":False,
        "identity_row_count":int(identity.sum()),
        "non_identity_row_count":int(non_identity.sum()),
        "affected_face_count":int(local["affected_face_count"]),
        "probe_count":len(probe_ids),
        "probe_ids":list(probe_ids),
        "steps":int(args.steps),
        "lr":float(args.lr),
        "probe_batch":int(args.probe_batch),
        "trust_weight":float(args.trust_weight),
        "initial_surrogate":initial_eval,
        "best_surrogate":best[2],
        "hard_mechanics":hard,
        "correction":{
            "identity_max_row_l1":float(delta[identity].max(initial=0.0)),
            "non_identity_mean_row_l1":float(delta[non_identity].mean()),
            "non_identity_p95_row_l1":float(np.quantile(delta[non_identity],0.95)),
            "non_identity_max_row_l1":float(delta[non_identity].max()),
            "optimized_to_ceiling_mean_row_l1":float(to_ceiling[non_identity].mean()),
            "optimized_to_ceiling_p95_row_l1":float(np.quantile(to_ceiling[non_identity],0.95)),
        },
        "history":history,
        "backbone_query_chunk_telemetry":telemetry,
        "decision":(
            "PROCEED_TO_C4_1_CARRIER_NATIVE_BOUNDED_CORRECTION_HEAD"
            if hard["g3_passed"] and hard["g3b_passed"]
            else "REVISIT_DIFFERENTIABLE_OBJECTIVE_OR_OWNER_ATTRIBUTION"
        ),
        "claim_boundary":[
            "Only non-identity carrier weight logits are optimized.",
            "Identity carrier rows are exact hard-frozen base MIRA outputs.",
            "No teacher skin weights or diagnostic transfer ceiling enter the objective.",
            "Hard Compiler G3/G3B is the verdict authority.",
            "This court does not mint product skin authority.",
        ],
    }

    write(args.out_dir/"OPTIMIZED_CARRIER_SKIN.json",skin.to_dict())
    write(args.out_dir/"G3.json",g3.to_dict())
    write(args.out_dir/"G3B.json",g3b)
    np.savez_compressed(
        args.out_dir/"OPTIMIZED_CARRIER_WEIGHTS.npz",
        weights=full,
        base_weights=base_weights,
        diagnostic_ceiling=ceiling,
        vertex_ids=np.asarray(query.carrier_vertex_ids),
        joint_ids=np.asarray(query.joint_ids),
        non_identity_mask=non_identity.astype(np.uint8),
    )
    write(args.out_dir/"REPORT.json",report)
    print("MIRA_C40_RESULT="+json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--arm-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--seed",type=int,default=20261004)
    ap.add_argument("--steps",type=int,default=300)
    ap.add_argument("--lr",type=float,default=0.03)
    ap.add_argument("--probe-batch",type=int,default=12)
    ap.add_argument("--log-every",type=int,default=25)
    ap.add_argument("--trust-weight",type=float,default=0.0005)
    ap.add_argument("--backbone-query-chunk",type=int,default=256)
    ap.add_argument("--readout-chunk",type=int,default=128)
    args=ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        write(args.out_dir/"ERROR.json",{
            "type":type(exc).__name__,
            "message":str(exc),
        })
        raise
