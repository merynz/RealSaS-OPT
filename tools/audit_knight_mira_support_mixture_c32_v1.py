from __future__ import annotations

"""C3.2 zero-train MIRA support-mixture causal court.

Purpose:
C3/C3N established that Stage19 unoriented face-cross normals corrupt the frozen
MIRA readout and that transported signed GSA normals restore exact identity-row
parity, but non-identity carrier vertices retain mechanical residual.

This court isolates the remaining representation choice on the exact same
carrier geometry and signed-normal counterfactual:

A) LATENT_PREBLEND
   support-weighted GSA memory -> nonlinear V6 readout (current bridge)

B) LOGIT_POSTBLEND
   V6 readout is evaluated once per admitted source support using exact carrier
   geometry/pair features; logits are then support-weighted and softmaxed

C) PROB_POSTBLEND
   same per-support carrier query, but normalized probability rows are blended

D) LEGACY_WEIGHT_TRANSFER_CEILING
   diagnostic only: blend legacy GSA-domain predicted weights through the same
   mechanical support. This is not a product proposal; it proves the available
   support field can close mechanics on this carrier.

No fitting. No teacher input. No topology mutation. No product promotion.
"""

import argparse
import json
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
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mechanical_carrier_skin_v1 import (
    qualify_mechanical_carrier_skin_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
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
from models.mira.carrier_query_v1 import (
    build_mira_mechanical_carrier_query_v1,
    transport_surface_memory_to_carrier_v1,
)
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.audit_knight_carrier_normal_authority_c31_v1 import (
    _diagnostic_evidence,
    _transport_gsa_normals,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


def _normalize_rows(weights: np.ndarray) -> np.ndarray:
    w=np.asarray(weights,np.float64)
    if w.ndim!=2 or not np.isfinite(w).all() or np.any(w<0.0):
        raise RuntimeError("C32_WEIGHT_MATRIX_INVALID")
    s=w.sum(axis=1,keepdims=True)
    if np.any(s<=1e-12):
        raise RuntimeError("C32_ZERO_WEIGHT_ROW")
    return w/s


def _decode_latent_preblend(*, query, memory, tokens, decoder, device, chunk):
    qmemory=transport_surface_memory_to_carrier_v1(memory,query)
    with torch.inference_mode(), torch.autocast(
        device,dtype=torch.bfloat16,enabled=(device=="cuda")
    ):
        _,pred=decoder.decode_all(
            qmemory,
            torch.as_tensor(query.geometry7,device=device),
            torch.as_tensor(query.pair_geometry,device=device),
            tokens,
            torch.as_tensor(query.legal_pair,device=device),
            chunk=chunk,
        )
    return _normalize_rows(pred.float().cpu().numpy())


def _decode_support_expanded(*, query, memory, tokens, decoder, device, chunk):
    row=np.asarray(query.support_row_index,np.int64)
    src=np.asarray(query.support_source_index,np.int64)
    coeff=np.asarray(query.support_coefficients,np.float64)
    if not (len(row)==len(src)==len(coeff)) or len(row)==0:
        raise RuntimeError("C32_SUPPORT_ARRAY_DRIFT")

    mem=memory[torch.as_tensor(src,dtype=torch.long,device=memory.device)]
    geom=torch.as_tensor(np.asarray(query.geometry7,np.float32)[row],device=device)
    pair=torch.as_tensor(np.asarray(query.pair_geometry,np.float32)[row],device=device)
    legal=torch.as_tensor(np.asarray(query.legal_pair,bool)[row],device=device)
    with torch.inference_mode(), torch.autocast(
        device,dtype=torch.bfloat16,enabled=(device=="cuda")
    ):
        logits,pred=decoder.decode_all(mem,geom,pair,tokens,legal,chunk=chunk)
    return (
        row,
        coeff,
        logits.float().cpu().numpy().astype(np.float64),
        pred.float().cpu().numpy().astype(np.float64),
    )


def _aggregate_postblend(*, query, row, coeff, logits, probs):
    V=query.vertex_count
    J=query.joint_count
    if logits.shape!=(len(row),J) or probs.shape!=(len(row),J):
        raise RuntimeError("C32_EXPANDED_OUTPUT_SHAPE")
    logit_mix=np.zeros((V,J),np.float64)
    prob_mix=np.zeros((V,J),np.float64)
    mass=np.zeros(V,np.float64)
    for k,qi in enumerate(row.tolist()):
        a=float(coeff[k])
        logit_mix[qi]+=a*logits[k]
        prob_mix[qi]+=a*probs[k]
        mass[qi]+=a
    if not np.allclose(mass,1.0,rtol=0.0,atol=1e-6):
        raise RuntimeError("C32_SUPPORT_MASS_DRIFT")
    shifted=logit_mix-logit_mix.max(axis=1,keepdims=True)
    exp=np.exp(shifted)
    logit_weights=_normalize_rows(exp)
    prob_weights=_normalize_rows(prob_mix)
    return logit_weights,prob_weights


def _legacy_transfer_ceiling(*, query, legacy_weights):
    row=np.asarray(query.support_row_index,np.int64)
    src=np.asarray(query.support_source_index,np.int64)
    coeff=np.asarray(query.support_coefficients,np.float64)
    V=query.vertex_count
    J=legacy_weights.shape[1]
    out=np.zeros((V,J),np.float64)
    mass=np.zeros(V,np.float64)
    for k,qi in enumerate(row.tolist()):
        a=float(coeff[k])
        out[qi]+=a*legacy_weights[int(src[k])]
        mass[qi]+=a
    if not np.allclose(mass,1.0,rtol=0.0,atol=1e-6):
        raise RuntimeError("C32_CEILING_SUPPORT_MASS_DRIFT")
    return _normalize_rows(out)


def _skin_and_mechanics(
    *,
    name,
    weights,
    candidate,
    evidence,
    skeleton,
    query,
    surface,
    envelope,
    cameras,
    policy,
    out_dir,
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
    write(out_dir/(name+"_skin.json"),skin.to_dict())
    write(out_dir/(name+"_g3.json"),g3.to_dict())
    write(out_dir/(name+"_g3b.json"),g3b)
    return skin,g3,g3b


def _vertex_mode(candidate):
    return {
        str(v.candidate_vertex_id):str(v.support_binding.mode)
        for v in candidate.vertices
    }


def _residual_localization(*, candidate, query, report, direct, ceiling):
    modes=_vertex_mode(candidate)
    cidx={str(v):i for i,v in enumerate(query.carrier_vertex_ids)}
    delta=np.abs(np.asarray(direct)-np.asarray(ceiling)).sum(axis=1)
    unsafe=set(map(int,report.get("unsafe_face_indices") or ()))
    categories={
        "all_identity":[],
        "touches_non_identity":[],
        "two_plus_non_identity":[],
    }
    for fi,face in enumerate(candidate.faces):
        non=sum(modes[str(v)]!="IDENTITY_SURFACE_NODE" for v in face)
        if non==0: key="all_identity"
        elif non>=2: key="two_plus_non_identity"
        else: key="touches_non_identity"
        categories[key].append(fi)
    out={}
    for key,faces in categories.items():
        if not faces:
            out[key]={"face_count":0,"unsafe_count":0,"unsafe_rate":None}
            continue
        arr=np.asarray(faces,np.int64)
        max_delta=np.asarray([
            max(delta[cidx[str(v)]] for v in candidate.faces[int(fi)])
            for fi in arr
        ],np.float64)
        mask=np.asarray([int(fi) in unsafe for fi in arr],bool)
        out[key]={
            "face_count":int(len(arr)),
            "unsafe_count":int(mask.sum()),
            "unsafe_rate":float(mask.mean()),
            "vertex_weight_delta_to_ceiling_mean":float(max_delta.mean()),
            "vertex_weight_delta_to_ceiling_p95":float(np.quantile(max_delta,0.95)),
            "unsafe_vertex_weight_delta_to_ceiling_mean":(
                float(max_delta[mask].mean()) if mask.any() else None
            ),
            "unsafe_vertex_weight_delta_to_ceiling_p95":(
                float(np.quantile(max_delta[mask],0.95)) if mask.any() else None
            ),
        }
    return out


def main(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("C32_CUDA_NOT_AVAILABLE")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

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
    base=build_mechanical_carrier_evidence_v1(
        candidate,static_qualification=static,surface_addressing=addressing)
    surface_tensor=tensorize_rigging_surface_v1(surface)
    signed_n,signed_v,_valid_mass,_support_count=_transport_gsa_normals(
        candidate,surface_tensor,base.ordered_vertex_ids)
    evidence=_diagnostic_evidence(base,signed_n,signed_v)

    policy=mesh_policy_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    envelope=deformation_envelope_from_dict(stage_output_payload(
        ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    cameras=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))

    conditioning,ci,memory,tokens,decoder,mira_result,telemetry=_run_mira_backbone(
        surface=surface,skeleton=skeleton,arm_dir=args.arm_dir,
        device=args.device,query_chunk=args.backbone_query_chunk,
    )
    legacy=_decode_legacy(
        conditioning=conditioning,ci=ci,memory=memory,tokens=tokens,
        decoder=decoder,device=args.device,chunk=args.readout_chunk,
    )
    query=build_mira_mechanical_carrier_query_v1(
        candidate=candidate,carrier_evidence=evidence,
        surface_tensor=surface_tensor,conditioning=conditioning,
    )

    pre=_decode_latent_preblend(
        query=query,memory=memory,tokens=tokens,decoder=decoder,
        device=args.device,chunk=args.readout_chunk)
    row,coeff,expanded_logits,expanded_probs=_decode_support_expanded(
        query=query,memory=memory,tokens=tokens,decoder=decoder,
        device=args.device,chunk=args.readout_chunk)
    logit_post,prob_post=_aggregate_postblend(
        query=query,row=row,coeff=coeff,
        logits=expanded_logits,probs=expanded_probs)
    ceiling=_legacy_transfer_ceiling(query=query,legacy_weights=legacy)

    arms={}
    raw_weights={
        "LATENT_PREBLEND":pre,
        "LOGIT_POSTBLEND":logit_post,
        "PROB_POSTBLEND":prob_post,
        "LEGACY_WEIGHT_TRANSFER_CEILING":ceiling,
    }
    reports={}
    for name,w in raw_weights.items():
        print("C32_ARM_BEGIN="+name,flush=True)
        skin,g3,g3b=_skin_and_mechanics(
            name=name,weights=w,candidate=candidate,evidence=evidence,
            skeleton=skeleton,query=query,surface=surface,
            envelope=envelope,cameras=cameras,policy=policy,
            out_dir=args.out_dir,
        )
        np.savez_compressed(
            args.out_dir/(name+"_weights.npz"),
            weights=w,
            vertex_ids=np.asarray(query.carrier_vertex_ids),
            joint_ids=np.asarray(query.joint_ids),
        )
        rep={
            **_summarize_g3(g3,g3b),
            "skin_lineage_hash":skin.skin_lineage_hash,
            "row_l1_to_ceiling_mean":float(np.abs(w-ceiling).sum(axis=1).mean()),
            "row_l1_to_ceiling_p95":float(np.quantile(np.abs(w-ceiling).sum(axis=1),0.95)),
            "row_l1_to_ceiling_max":float(np.abs(w-ceiling).sum(axis=1).max()),
        }
        reports[name]=rep
        print("C32_ARM_RESULT="+json.dumps({name:rep},sort_keys=True),flush=True)

    reports["LATENT_PREBLEND"]["residual_localization"]=_residual_localization(
        candidate=candidate,query=query,
        report=read_json(args.out_dir/"LATENT_PREBLEND_g3b.json"),
        direct=pre,ceiling=ceiling,
    )

    unsafe_pre=int(reports["LATENT_PREBLEND"]["g3b_unsafe_face_count"])
    unsafe_logit=int(reports["LOGIT_POSTBLEND"]["g3b_unsafe_face_count"])
    unsafe_prob=int(reports["PROB_POSTBLEND"]["g3b_unsafe_face_count"])
    unsafe_ceil=int(reports["LEGACY_WEIGHT_TRANSFER_CEILING"]["g3b_unsafe_face_count"])
    best_zero=min(unsafe_pre,unsafe_logit,unsafe_prob)

    result={
        "schema":"RealSaS.MIRASupportMixtureCausalCourt.v1",
        "status":"MEASURED__NO_TRAIN__NO_PROMOTION_CLAIM",
        "run_id":args.run_id,
        "carrier_topology_hash":evidence.topology_hash,
        "diagnostic_signed_normal_evidence_hash":evidence.carrier_evidence_hash,
        "mira_checkpoint_sha256":mira_result["model_sha256"],
        "query_hash":query.query_hash,
        "support_relation_count":int(len(query.support_row_index)),
        "carrier_vertex_count":int(query.vertex_count),
        "non_identity_vertex_count":int(sum(
            str(v.support_binding.mode)!="IDENTITY_SURFACE_NODE"
            for v in candidate.vertices
        )),
        "backbone_query_chunk_telemetry":telemetry,
        "arms":reports,
        "training_used":False,
        "teacher_predictor_input_used":False,
        "topology_changed":False,
        "product_authority_minted":False,
        "legacy_weight_transfer_is_diagnostic_ceiling_only":True,
        "decision":(
            "ZERO_TRAIN_POSTBLEND_CLOSES_MECHANICS"
            if best_zero==0 else
            "LATENT_PREBLEND_IS_MATERIAL_OWNER"
            if best_zero<unsafe_pre else
            "SUPPORT_MIXTURE_ORDER_NOT_SUFFICIENT"
        ),
        "fit_authorization_hint":(
            "KEEP_C4_BLOCKED"
            if best_zero==0 else
            "C4_DECODER_TAIL_FIT_NOW_JUSTIFIED_IF_NO_OTHER_QUERY_CONFOUND_REMAINS"
            if unsafe_ceil==0 else
            "CEILING_ITSELF_FAILS__DO_NOT_FIT"
        ),
    }
    write(args.out_dir/"REPORT.json",result)
    print("MIRA_C32_RESULT="+json.dumps(result,sort_keys=True),flush=True)


def read_json(path:Path):
    return json.loads(path.read_text())


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--arm-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--backbone-query-chunk",type=int,default=256)
    ap.add_argument("--readout-chunk",type=int,default=128)
    args=ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        write(args.out_dir/"ERROR.json",{
            "type":type(exc).__name__,"message":str(exc)
        })
        raise
