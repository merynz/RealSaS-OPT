from __future__ import annotations

"""Fail-closed Mage FIT1 runner for audited Arachne A1 v4.

Supersedes the unrun v3 treatment before optimizer construction. Product-time
predictor input is only rich RiggingSurfaceIR + Compiler-qualified skeleton +
exact camera binding. Teacher W is target/evaluation only.
"""
import argparse, inspect, json, math, random
from pathlib import Path
import numpy as np
import torch

from compiler.realsas_compiler_core.skin import qualify_skin
from experiments.arachne_a1_v3_fit1.run_arachne_a1_v3_fit1 import (
    EXPECTED_A0_MODEL_SHA, EXPECTED_BANK_SHA, EXPECTED_A0_CONFIG_HASH,
    EXPECTED_SURFACE_LINEAGE, EXPECTED_SKELETON_LINEAGE,
    sha_file, write_json, _atomic_torch_save, load_cameras, build_surface,
    load_skeleton, load_bank, load_frozen_codec, conditioning_to_torch,
    decode_logits, metrics, make_skin_proposal,
)
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4
from models.arachne.v4.loss_v4 import ArachneA1LossConfigV4, arachne_a1_behavior_loss_v4, normalize_joint_fields, articulated_deformation_ratio_loss
from models.arachne.v4.articulated_probe_v1 import build_articulated_probe_transforms

SCHEMA="RealSaS.ArachneA1V7NativeFIT1.v2"
SEED=20260911
EXPECTED_A1_CONFIG_HASH="738528eb7d94d8c8c69c9caf5838e4cf01341ddad287d5e29a78b735a53d3f96"
EXPECTED_A1_PARAMETER_COUNT=138_053_153
EXPECTED_LOSS_CONFIG_HASH="1e609f04bafcfaa592c732503e4d3054a614cd6e72dda5c7fad911f34e6efae6"
MAX_STEPS=12_288; MIN_CLOSURE_STEP=2048; CHECK_EVERY=256; REQUIRED_STABLE=3
WARMUP_STEPS=512; LR=5e-5; LR_FLOOR=.1; WEIGHT_DECAY=1e-4; GRAD_CLIP=1.0
TRAIN_GSA_ROWS=192; TRAIN_DENSE_ROWS=192; FIT1_P95_MAX=.05; FIT1_DEFORM_MAX=.05
COMPILER_TOTAL_CORR_MAX=1e-4


def lr_factor(step:int)->float:
    s=max(1,int(step))
    if s<=WARMUP_STEPS: return float(s)/WARMUP_STEPS
    q=min(max((s-WARMUP_STEPS)/float(MAX_STEPS-WARMUP_STEPS),0.),1.)
    return LR_FLOOR+(1-LR_FLOOR)*.5*(1+math.cos(math.pi*q))


def camera_yaw_fourier(cameras, device):
    rows=[]
    for i,c in enumerate(cameras):
        if int(c["view_index"])!=i: raise RuntimeError("CAMERA_ORDER_DRIFT")
        y=math.radians(float(c["yaw_deg"]))
        rows.append([math.sin(y),math.cos(y),math.sin(2*y),math.cos(2*y)])
    return torch.tensor(rows,device=device,dtype=torch.float32)[None]


def joint_world_and_parent(skeleton, conditioning, device):
    by_id={str(j.canonical_joint_id): j for j in skeleton.joints}
    ids=tuple(conditioning.joint_ids[0])
    if set(ids)!=set(by_id): raise RuntimeError("A1_V4_JOINT_WORLD_ID_DRIFT")
    jw=np.asarray([[by_id[jid].position for jid in ids]],np.float32)
    return torch.as_tensor(jw,device=device,dtype=torch.float32), torch.as_tensor(conditioning.parent_indices,device=device,dtype=torch.long)


def preflight_equivariance_v4(model, ci, *, tol=2e-5):
    model.eval()
    with torch.no_grad():
        base=model(**ci).field_tokens.float()
        N=ci["surface_mask"].shape[1]
        p=torch.arange(N-1,-1,-1,device=base.device); inv=torch.empty_like(p); inv[p]=torch.arange(N,device=base.device)
        cs=dict(ci)
        for n in ("surface_positions_normalized","surface_normals","surface_normal_valid","surface_support_views","surface_raster_xy","surface_raster_valid","surface_observed","surface_completed","surface_mask"):
            cs[n]=ci[n][:,p]
        cs["edge_index"]=inv[ci["edge_index"]]; cs["support_anchor_matrix"]=ci["support_anchor_matrix"][:,:,p]
        cs["pair_geometry"]=ci["pair_geometry"][:,p]; cs["pair_mask"]=ci["pair_mask"][:,p]
        sd=float((base-model(**cs).field_tokens.float()).abs().max().cpu())
        J=ci["joint_mask"].shape[1]; q=torch.arange(J-1,-1,-1,device=base.device); qi=torch.empty_like(q); qi[q]=torch.arange(J,device=base.device)
        cj=dict(ci)
        for n in ("joint_positions_normalized","joint_mask","root_mask","deform_root_mask","joint_depth_normalized"):
            cj[n]=ci[n][:,q]
        op=ci["parent_indices"][:,q]; cj["parent_indices"]=torch.where(op>=0,qi[op.clamp_min(0)],op)
        cj["support_anchor_matrix"]=ci["support_anchor_matrix"][:,q]; cj["pair_geometry"]=ci["pair_geometry"][:,:,q]; cj["pair_mask"]=ci["pair_mask"][:,:,q]
        jd=float((base[:,q]-model(**cj).field_tokens.float()).abs().max().cpu())
        vp=torch.tensor([3,0,7,1,6,2,5,4],device=base.device)
        cv=dict(ci)
        for n in ("surface_support_views","surface_raster_xy","surface_raster_valid"):
            cv[n]=ci[n][:,:,vp]
        cv["view_yaw_fourier"]=ci["view_yaw_fourier"][:,vp]
        vd=float((base-model(**cv).field_tokens.float()).abs().max().cpu())
    if max(sd,jd,vd)>tol: raise RuntimeError(f"A1_EQUIVARIANCE_FAIL::{sd}::{jd}::{vd}")
    return {"tolerance":tol,"surface_max_abs":sd,"joint_max_abs":jd,"view_binding_max_abs":vd,"support_anchor_remap_covered":True}


def run(args):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    device=torch.device("cuda"); gpu=torch.cuda.get_device_name(0)
    if "A100" not in gpu.upper() or not torch.cuda.is_bf16_supported(): raise RuntimeError(f"A100_BF16_REQUIRED::{gpu}")
    if torch.cuda.mem_get_info()[0] < 18*1024**3: raise RuntimeError("A1_V4_REQUIRES_AT_LEAST_18_GIB_FREE_VRAM")
    prereg_path=Path(args.prereg); prereg=json.loads(prereg_path.read_text())
    if prereg.get("schema")!=SCHEMA+".Preregistration.v1" or prereg.get("status")!="AUTHORIZED": raise RuntimeError("A1_V4_PREREG_NOT_AUTHORIZED")
    prereg_sha=sha_file(prereg_path)
    data_dir=Path(args.data_dir); cameras=load_cameras(data_dir); surface=build_surface(Path(args.zero_surface),cameras); skeleton=load_skeleton(Path(args.skeleton)); bank=load_bank(Path(args.bank)); codec,_=load_frozen_codec(Path(args.a0_model),device)
    conditioning=ArachneRichConditioningAdapterV3(require_scene_first=True)([surface],[skeleton])
    if conditioning.source_surface_hashes!=(EXPECTED_SURFACE_LINEAGE,) or conditioning.source_skeleton_hashes!=(EXPECTED_SKELETON_LINEAGE,): raise RuntimeError("A1_V4_LINEAGE_DRIFT")
    if int(conditioning.surface_mask[0].sum())!=950 or int(conditioning.edge_mask[0].sum())!=2813 or int(conditioning.joint_mask[0].sum())!=22: raise RuntimeError("A1_V4_CARDINALITY_DRIFT")
    if conditioning.relation_kind_vocab != ("SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",): raise RuntimeError("A1_V4_RELATION_KIND_CONTRACT_DRIFT")
    geom_delta=float(np.max(np.abs(conditioning.geometry7[0]-bank["gsa_geometry7"].astype(np.float32))))
    if geom_delta>5e-5: raise RuntimeError(f"A1_V4_CODEC_FRAME_PARITY_FAIL::{geom_delta}")
    if tuple(map(str,bank["surface_ids"]))!=tuple(conditioning.surface_ids[0]) or tuple(map(str,bank["joint_ids"]))!=tuple(conditioning.joint_ids[0]): raise RuntimeError("A1_V4_ID_ORDER_DRIFT")
    if not np.array_equal(bank["condition_query_indices"].astype(np.int64),conditioning.condition_query_indices[0]): raise RuntimeError("A1_V4_CONDITION_QUERY_DRIFT")
    cfg=ArachneA1ConfigV4(); lcfg=ArachneA1LossConfigV4()
    if cfg.config_hash!=EXPECTED_A1_CONFIG_HASH or lcfg.config_hash!=EXPECTED_LOSS_CONFIG_HASH: raise RuntimeError("A1_V4_CONFIG_HASH_DRIFT")
    model=ArachneA1V4(cfg).to(device=device,dtype=torch.float32)
    if model.parameter_count!=EXPECTED_A1_PARAMETER_COUNT: raise RuntimeError(f"A1_V4_PARAMETER_COUNT_DRIFT::{model.parameter_count}")
    sig=set(inspect.signature(model.forward).parameters)
    forbidden={x for x in sig if any(k in x.lower() for k in ("teacher","truth","weight","source_id","character","condition_token"))}
    if forbidden: raise RuntimeError(f"PREDICTOR_INPUT_FIREWALL_FAIL::{sorted(forbidden)}")
    ci=conditioning_to_torch(conditioning,device); ci["view_yaw_fourier"]=camera_yaw_fourier(cameras,device)
    eq=preflight_equivariance_v4(model,ci)
    jw,pi=joint_world_and_parent(skeleton,conditioning,device); artT=build_articulated_probe_transforms(jw,pi,ci["joint_mask"])
    q=torch.arange(jw.shape[1]-1,-1,-1,device=device); qi=torch.empty_like(q); qi[q]=torch.arange(len(q),device=device); op=pi[:,q]; pi2=torch.where(op>=0,qi[op.clamp_min(0)],op)
    art2=build_articulated_probe_transforms(jw[:,q],pi2,ci["joint_mask"][:,q]); probe_delta=float((artT[:,:,q]-art2).abs().max().cpu())
    if probe_delta>1e-6: raise RuntimeError(f"ARTICULATED_PROBE_JOINT_EQUIVARIANCE_FAIL::{probe_delta}")
    ggeom=torch.as_tensor(bank["gsa_geometry7"][None],device=device,dtype=torch.float32); qidx=torch.as_tensor(bank["condition_query_indices"][None],device=device,dtype=torch.long)
    with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
        ct=codec.encode_condition(ggeom,qidx).detach(); prepared=codec.decoder.prepare_condition(ct).detach()
    prepared=prepared.clone()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lr_factor)
    ggeom_np=np.asarray(bank["gsa_geometry7"],np.float32); gtruth=np.asarray(bank["gsa_teacher_weights"],np.float32); gworld=np.asarray(bank["gsa_world"],np.float32); gmask=np.asarray(bank["gsa_supervision_mask"],bool)
    dgeom=np.asarray(bank["dense8k_geometry7"],np.float32); dtruth=np.asarray(bank["dense8k_teacher_weights"],np.float32); dworld=np.asarray(bank["dense8k_world"],np.float32)
    hgeom=np.asarray(bank["holdout_geometry7"],np.float32); htruth=np.asarray(bank["holdout_teacher_weights"],np.float32); hworld=np.asarray(bank["holdout_world"],np.float32); gpool=np.where(gmask)[0]
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); trace_path=out/"ARACHNE_A1_V4_FIT1_TRACE.json"; model_path=out/"ARACHNE_A1_V4_FIT1_MODEL_ONLY_FP32.pt"; result_path=out/"ARACHNE_A1_V4_FIT1_RESULT.json"
    trace=[]; best=None; stable=0; closure=None
    def predict_all(geom):
        model.eval()
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
            tok=model(**ci).field_tokens; logits=decode_logits(codec,prepared,tok,geom); pred=normalize_joint_fields(torch.sigmoid(logits),ci["joint_mask"]).squeeze(0)
        return pred.float().cpu().numpy()
    def articulated_metric(pred,truth,world,mask):
        wp=torch.as_tensor(pred[None],device=device,dtype=torch.float32); wt=torch.as_tensor(truth[None],device=device,dtype=torch.float32); rw=torch.as_tensor(world[None],device=device,dtype=torch.float32); rm=torch.as_tensor(mask[None],device=device,dtype=torch.bool)
        with torch.no_grad(): r,m,e=articulated_deformation_ratio_loss(wp,wt,rw,artT,rm)
        return {"ratio":float(r.cpu()),"teacher_motion_rms":float(m.cpu()),"error_rms":float(e.cpu())}
    def evaluate(step):
        gp=predict_all(ggeom_np); hp=predict_all(hgeom); gm=metrics(gp,gtruth,gworld,device,gmask); hm=metrics(hp,htruth,hworld,device)
        ga=articulated_metric(gp,gtruth,gworld,gmask); ha=articulated_metric(hp,htruth,hworld,np.ones(len(hp),bool))
        prop=make_skin_proposal(gp,conditioning,cfg.config_hash); qskin=qualify_skin(surface,skeleton,prop,max_simplex_repair_l1=1e-5,max_total_correction_l1=COMPILER_TOTAL_CORR_MAX)
        qr=qskin.qualification_report; comp={"status":"PASS","row_count":len(qskin.rows),"skin_lineage_hash":qskin.skin_lineage_hash,"qualification_report":qr}
        passed=gm["row_l1_p95"]<=FIT1_P95_MAX and gm["deformation_error_ratio"]<=FIT1_DEFORM_MAX and len(qskin.rows)==950 and float(qr["total_correction_l1"])<=COMPILER_TOTAL_CORR_MAX
        return {"step":int(step),"gsa950":gm,"gsa950_articulated":ga,"disjoint_holdout":hm,"holdout_articulated":ha,"compiler_skin":comp,"fit1_gate":bool(passed)}
    row0=evaluate(0); row0["stable_streak"]=0; trace.append(row0); write_json(trace_path,{"schema":SCHEMA+".Trace.v1","prereg_sha256":prereg_sha,"rows":trace}); print("A1V4_CHECK="+json.dumps(row0,sort_keys=True),flush=True)
    for step in range(1,MAX_STEPS+1):
        model.train(); rng=np.random.default_rng(SEED+step*1009); gi=rng.choice(gpool,TRAIN_GSA_ROWS,replace=False); di=rng.choice(len(dgeom),TRAIN_DENSE_ROWS,replace=False)
        qgeom=np.concatenate([ggeom_np[gi],dgeom[di]]); truth_np=np.concatenate([gtruth[gi],dtruth[di]]); qworld=np.concatenate([gworld[gi],dworld[di]])
        pm=rng.permutation(len(qgeom)); qgeom,truth_np,qworld=qgeom[pm],truth_np[pm],qworld[pm]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
            tok=model(**ci).field_tokens; logits=decode_logits(codec,prepared,tok,qgeom,chunk=len(qgeom))
        truth=torch.as_tensor(truth_np[None],device=device,dtype=torch.float32); rest=torch.as_tensor(qworld[None],device=device,dtype=torch.float32); rm=torch.ones((1,len(qgeom)),device=device,dtype=torch.bool)
        losses=arachne_a1_behavior_loss_v4(logits,truth,rm,joint_mask=ci["joint_mask"],rest_world=rest,articulated_transforms=artT,config=lcfg)
        losses["total"].backward(); gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP))
        if not math.isfinite(gn): raise FloatingPointError("NONFINITE_A1_V4_GRADIENT")
        opt.step(); sched.step()
        if step%CHECK_EVERY: continue
        er=evaluate(step); er["losses"]={k:float(v.detach().cpu()) for k,v in losses.items()}; er["grad_norm"]=gn; er["lr"]=float(opt.param_groups[0]["lr"])
        stable=stable+1 if er["fit1_gate"] and step>=MIN_CLOSURE_STEP else 0; er["stable_streak"]=stable; trace.append(er); print("A1V4_CHECK="+json.dumps(er,sort_keys=True),flush=True)
        score=(max(float(er["gsa950"]["row_l1_p95"]),float(er["gsa950"]["deformation_error_ratio"])),float(er["gsa950_articulated"]["ratio"]),float(er["gsa950"]["row_l1_p95"]))
        if best is None or score<best:
            payload={"schema":SCHEMA+".ModelOnly.v1","prereg_sha256":prereg_sha,"a0_model_sha256":EXPECTED_A0_MODEL_SHA,"a1_supervision_bank_sha256":EXPECTED_BANK_SHA,"a1_config_hash":cfg.config_hash,"loss_config_hash":lcfg.config_hash,"parameter_count":model.parameter_count,"step":step,"score":list(score),"model":model.state_dict(),"product_evaluation_performed":False}
            _atomic_torch_save(payload,model_path); best=score
        write_json(trace_path,{"schema":SCHEMA+".Trace.v1","prereg_sha256":prereg_sha,"rows":trace})
        if stable>=REQUIRED_STABLE: closure=step; break
    if not model_path.exists(): raise RuntimeError("A1_V4_MODEL_CHECKPOINT_MISSING")
    payload=torch.load(model_path,map_location="cpu",weights_only=False); model.load_state_dict(payload["model"],strict=True); final=evaluate(int(payload["step"])); status="PASS" if closure is not None and final["fit1_gate"] else "FAIL"
    result={"schema":SCHEMA+".Result.v1","status":status,"prereg_sha256":prereg_sha,"seed":SEED,"a0":{"model_sha256":EXPECTED_A0_MODEL_SHA,"supervision_bank_sha256":EXPECTED_BANK_SHA,"field_tokens":4,"latent_channels":512,"ordered_latent_alignment_used":False},"a1":{"architecture_id":cfg.architecture_id,"config_hash":cfg.config_hash,"parameter_count":model.parameter_count,"loss_config_hash":lcfg.config_hash,"model_sha256":sha_file(model_path),"best_step":int(payload["step"]),"closure_step":closure,"equivariance_preflight":eq,"articulated_probe_joint_permutation_delta":probe_delta},"final_best_checkpoint_evaluation":final,"teacher_boundary":"OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT","holdout_role":"DIAGNOSTIC_ONLY__SAME_CHARACTER_DISJOINT_SURFACE__NOT_UNSEEN","product_pass_claimed":False,"unseen_generalization_claimed":False,"trace_path":str(trace_path),"model_path":str(model_path)}
    write_json(result_path,result); print("A1V4_RESULT="+json.dumps(result,sort_keys=True),flush=True)
    if status!="PASS": raise AssertionError("A1_V4_FIT1_DID_NOT_CLOSE")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",required=True); p.add_argument("--zero-surface",required=True); p.add_argument("--skeleton",required=True); p.add_argument("--a0-model",required=True); p.add_argument("--bank",required=True); p.add_argument("--prereg",required=True); p.add_argument("--output-dir",required=True); run(p.parse_args())
if __name__=="__main__": main()
