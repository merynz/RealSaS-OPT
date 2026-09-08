from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch

from models.skin_field_codec.v3 import SkinFieldCodecV3, SkinTokensStrengthConfigV3, scalar_field_loss_v3

SCHEMA = "RealSaS.ArachneMageA0SkinTokensStrengthV3Run.v1"
SEED = 20260908
MAX_STEPS = 4096
CHECK_EVERY = 64
CHECKPOINT_EVERY = 256
REQUIRED_STABLE = 3
LR = 2e-4
WEIGHT_DECAY = 1e-4
EXPECTED_CONFIG_HASH = "50a8e4d10f8ab2912c9f8cbb6922046f5b126571935dfc532072e4b5ee89c382"
EXPECTED_PARAM_COUNT = 273_296_903
EXPECTED_CACHE_SHA = "db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd"
EXPECTED_CACHE_BINDING_SHA = "c7e3bf10fc8edf16f862b4ce58b3aabfeb2aabf14cc8e744e53e3afc0764ac9f"
EXPECTED_TARGET_BINDING_SHA = "ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf"
EXPECTED_TEACHER_WEIGHTS_SHA = "7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d"
EXPECTED_SUPERVISED_ROWS = 934
EXPECTED_LOW_ROWS = 16
RESULT_NAME = "ARACHNE_MAGE_A0_SKINTOKENS_STRENGTH_V3_RESULT.json"
PROGRESS_NAME = "ARACHNE_MAGE_A0_SKINTOKENS_STRENGTH_V3_PROGRESS.pt"
CHECKPOINT_NAME = "ARACHNE_MAGE_A0_SKINTOKENS_STRENGTH_V3_CODEC_CHECKPOINT.pt"


def sha_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def raw_array_sha(a: np.ndarray) -> str: return hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()
def write_json(path: Path, obj: object) -> None: path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float)+"\n", encoding="utf-8")


def reset_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def load_cache(path: Path) -> dict[str,np.ndarray]:
    if sha_file(path) != EXPECTED_CACHE_SHA: raise RuntimeError("V3_CACHE_SHA_DRIFT")
    with np.load(path,allow_pickle=False) as z: d={k:np.asarray(z[k]) for k in z.files}
    req=("surface_ids","joint_ids","surface_features","joint_features","teacher_weights","teacher_supervision_mask","rest_points_world","parent_indices")
    if any(k not in d for k in req): raise RuntimeError("V3_CACHE_SCHEMA_MISSING")
    if d["surface_features"].shape!=(950,20) or d["teacher_weights"].shape!=(950,22): raise RuntimeError("V3_CACHE_SHAPE_DRIFT")
    if int(d["teacher_supervision_mask"].sum())!=EXPECTED_SUPERVISED_ROWS or int((~d["teacher_supervision_mask"].astype(bool)).sum())!=EXPECTED_LOW_ROWS: raise RuntimeError("V3_SUPERVISION_MASK_DRIFT")
    if raw_array_sha(np.asarray(d["teacher_weights"],np.float32))!=EXPECTED_TEACHER_WEIGHTS_SHA: raise RuntimeError("V3_TEACHER_WEIGHT_CONTENT_DRIFT")
    return d


def geometry_from_cache(cache) -> np.ndarray:
    sf=np.asarray(cache["surface_features"],np.float32)
    geom=np.concatenate([sf[:,0:3]*2.0,sf[:,12:15],sf[:,15:16]],axis=1).astype(np.float32)
    if geom.shape!=(950,7) or not np.isfinite(geom).all(): raise RuntimeError("V3_GEOMETRY_CONTRACT_DRIFT")
    return geom


def deterministic_fps(points: np.ndarray,count: int,first_index: int|None=None)->np.ndarray:
    p=np.asarray(points,np.float64); n=len(p)
    if count<=0 or n<=0: raise ValueError("FPS requires positive cardinality")
    if count>n: return np.resize(deterministic_fps(p,n,first_index),count).astype(np.int64)
    out=np.empty(count,np.int64); farthest=int(0 if first_index is None else first_index); dist=np.full(n,np.inf)
    if farthest<0 or farthest>=n: raise ValueError("FPS first index out of bounds")
    for i in range(count):
        out[i]=farthest; d=np.square(p-p[farthest]).sum(1); dist=np.minimum(dist,d); farthest=int(np.argmax(dist))
    return out


def condition_query_indices(geometry,count=384): return deterministic_fps(geometry[:,:3],count)


def field_context_indices(geometry,truth,supervised,minimum=128,maximum=384):
    active=np.where(supervised&(truth>1e-8))[0]
    if len(active)==0: raise RuntimeError("V3_NO_ACTIVE_FIELD_SUPPORT")
    if len(active)>maximum:
        local_first=int(np.argmax(truth[active])); return active[deterministic_fps(geometry[active,:3],maximum,local_first)].astype(np.int64)
    chosen=list(map(int,active))
    if len(chosen)<minimum:
        cand=np.where(supervised)[0]; a=geometry[active,:3].astype(np.float64); c=geometry[cand,:3].astype(np.float64)
        near=np.sqrt(np.square(c[:,None,:]-a[None,:,:]).sum(-1)).min(1); order=np.lexsort((cand,near)); seen=set(chosen)
        for oi in order:
            idx=int(cand[oi])
            if idx not in seen:
                chosen.append(idx); seen.add(idx)
                if len(chosen)>=minimum: break
    return np.asarray(chosen[:maximum],np.int64)


def tiny_config():
    return replace(SkinTokensStrengthConfigV3(),field_tokens=2,condition_tokens=8,latent_channels=32,encoder_width=32,decoder_width=64,attention_heads=4,encoder_layers=2,decoder_layers=3,ffn_ratio=2,fsq_levels=(4,4,4),strict_strength_contract=False,architecture_id="TEST_ONLY.RealSaS.SkinFieldCodec.v3")


def make_model(device,disposable_tiny=False,dtype=torch.float32):
    cfg=tiny_config() if disposable_tiny else SkinTokensStrengthConfigV3(); model=SkinFieldCodecV3(cfg).to(device=device,dtype=dtype)
    if not disposable_tiny:
        if cfg.config_hash!=EXPECTED_CONFIG_HASH: raise RuntimeError("V3_CONFIG_HASH_DRIFT")
        if sum(p.numel() for p in model.parameters())!=EXPECTED_PARAM_COUNT: raise RuntimeError("V3_PARAMETER_COUNT_DRIFT")
    return model


def encode_one_field_inputs(cache,geometry,j,device,dtype):
    truth=np.asarray(cache["teacher_weights"][:,j],np.float32); supervised=np.asarray(cache["teacher_supervision_mask"],bool); idx=field_context_indices(geometry,truth,supervised)
    obs=np.concatenate([geometry[idx],truth[idx,None]],1).astype(np.float32)
    return torch.as_tensor(obs[None],device=device,dtype=dtype),torch.as_tensor(truth[None],device=device,dtype=dtype),torch.as_tensor(supervised[None],device=device,dtype=torch.bool),idx


def train_step(model,opt,sched,cache,geometry,step,device,dtype):
    j=(step-1)%22; g=torch.as_tensor(geometry[None],device=device,dtype=dtype); qidx=torch.as_tensor(condition_query_indices(geometry,model.config.condition_tokens)[None],device=device,dtype=torch.long); obs,truth,supervised,ctx=encode_one_field_inputs(cache,geometry,j,device,dtype)
    model.train(); opt.zero_grad(set_to_none=True); amp=device.type=="cuda" and dtype in (torch.float16,torch.bfloat16)
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
        pred,continuous,quantized,token_ids,cond=model.forward_field(obs,g,qidx); losses=scalar_field_loss_v3(pred.float(),truth.float(),supervised); total=losses["total"]
    total.backward(); opt.step(); sched.step()
    return {"joint_index":j,"context_rows":int(len(ctx)),"loss":float(total.detach().cpu()),"bce":float(losses["bce"].detach().cpu()),"l1":float(losses["l1"].detach().cpu()),"token_id_min":int(token_ids.min().detach().cpu()),"token_id_max":int(token_ids.max().detach().cpu()),"lr":float(opt.param_groups[0]["lr"])}


def full_decode(model,cache,geometry,device,dtype):
    g=torch.as_tensor(geometry[None],device=device,dtype=dtype); qidx=torch.as_tensor(condition_query_indices(geometry,model.config.condition_tokens)[None],device=device,dtype=torch.long); cond=model.encode_condition(g,qidx); cols=[]
    for j in range(22):
        obs,_,_,_=encode_one_field_inputs(cache,geometry,j,device,dtype); _,zq,_=model.encode_field(obs); cols.append(model.decode_field(zq,cond,g)[0])
    return model.normalize_joint_fields(torch.stack(cols,-1)[None])[0]


def matrix_metrics(model,cache,geometry,surface,sk,sids,jids,transforms,device,dtype):
    model.eval()
    with torch.no_grad(): pred=full_decode(model,cache,geometry,device,dtype).float()
    truth=torch.as_tensor(cache["teacher_weights"],dtype=torch.float32,device=device); auth_np=np.asarray(cache["teacher_supervision_mask"],bool); auth=torch.as_tensor(auth_np,dtype=torch.bool,device=device); raw=pred.detach().cpu().numpy().astype(np.float64); tn=truth.detach().cpu().numpy().astype(np.float64); row=np.abs(raw-tn).sum(1)
    prop=hist.proposal_from_matrix(raw,sids,jids,surface,sk); q=hist.qualify_skin(surface,sk,prop,max_simplex_repair_l1=1e-6,max_total_correction_l1=1e-4,negative_tolerance=1e-8,max_influences=None); qw=hist.qualified_matrix(q,sids,jids); qrow=np.abs(qw-tn).sum(1)
    rest=torch.as_tensor(cache["rest_points_world"][None],dtype=torch.float32,device=device); rr,motion,err=hist.deformation_ratio(rest,truth[None],pred[None],transforms,auth[None]); qr,_,qerr=hist.deformation_ratio(rest,truth[None],torch.as_tensor(qw[None],dtype=torch.float32,device=device),transforms,auth[None]); report=q.qualification_report; corr=[float(r.correction_l1) for r in q.rows]
    return {"finite":bool(np.isfinite(raw).all()),"negative_weight_count":int((raw<-1e-8).sum()),"raw_row_l1_mean_auth":float(row[auth_np].mean()),"raw_row_l1_p95_auth":float(np.quantile(row[auth_np],.95)),"qualified_row_l1_mean_auth":float(qrow[auth_np].mean()),"qualified_row_l1_p95_auth":float(np.quantile(qrow[auth_np],.95)),"raw_deformation_error_ratio_auth":rr,"qualified_deformation_error_ratio_auth":qr,"teacher_motion_rms_auth":motion,"raw_deformation_error_rms_auth":err,"qualified_deformation_error_rms_auth":qerr,"raw_simplex_max_abs_residual":float(np.max(np.abs(raw.sum(1)-1.))),"qualified_simplex_max_abs_residual":float(np.max(np.abs(qw.sum(1)-1.))),"qualified_row_count":len(q.rows),"compiler_total_correction_l1":float(report["total_correction_l1"]),"compiler_mean_row_correction_l1":float(np.mean(corr)) if corr else 0.,"compiler_max_row_correction_l1":max(corr,default=0.),"compiler_corrected_row_count":int(report["corrected_row_count"]),"compiler_total_sparsification_discarded_mass":float(report["total_sparsification_discarded_mass"])}


def is_pass(m):
    return bool(m["finite"] and m["negative_weight_count"]==0 and m["qualified_row_count"]==950 and m["raw_row_l1_p95_auth"]<=.05 and m["qualified_row_l1_p95_auth"]<=.05 and m["raw_deformation_error_ratio_auth"]<=.05 and m["qualified_deformation_error_ratio_auth"]<=.05 and m["raw_simplex_max_abs_residual"]<=1e-6 and m["qualified_simplex_max_abs_residual"]<=1e-6 and m["compiler_total_correction_l1"]<=1e-4 and m["compiler_mean_row_correction_l1"]<=1e-7 and m["compiler_max_row_correction_l1"]<=1e-6 and m["compiler_total_sparsification_discarded_mass"]==0.)


def save_progress(path,model,opt,sched,step,streak,trace,dtype_name):
    torch.save({"schema":SCHEMA+".Progress","step":step,"streak":streak,"trace":trace,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"config_hash":EXPECTED_CONFIG_HASH,"parameter_count":EXPECTED_PARAM_COUNT,"cache_sha256":EXPECTED_CACHE_SHA,"binding_sha256":EXPECTED_TARGET_BINDING_SHA,"dtype":dtype_name},path)


def load_progress(path,model,opt,sched,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("schema")!=SCHEMA+".Progress" or ck.get("config_hash")!=EXPECTED_CONFIG_HASH or ck.get("parameter_count")!=EXPECTED_PARAM_COUNT: raise RuntimeError("V3_PROGRESS_FINGERPRINT_DRIFT")
    if ck.get("cache_sha256")!=EXPECTED_CACHE_SHA or ck.get("binding_sha256")!=EXPECTED_TARGET_BINDING_SHA: raise RuntimeError("V3_PROGRESS_DATA_BINDING_DRIFT")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
    for st in opt.state.values():
        for k,v in st.items():
            if torch.is_tensor(v): st[k]=v.to(device)
    random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"]); torch.set_rng_state(ck["torch_rng"])
    if torch.cuda.is_available() and ck.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return int(ck["step"]),int(ck["streak"]),list(ck["trace"])


def run_disposable_tiny(cache_path,output_dir):
    cache=load_cache(cache_path); geometry=geometry_from_cache(cache); reset_seed(); device=torch.device("cpu"); model=make_model(device,True,torch.float32); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8); losses=[]
    for step in range(1,4): losses.append(train_step(model,opt,sch,cache,geometry,step,device,torch.float32))
    report={"schema":SCHEMA+".DisposableTiny","status":"PASS_DISPOSABLE_TINY_FORWARD_FSQ_DECODE_BACKWARD","steps":3,"losses":losses,"main_v3_scientific_optimizer_steps":0}; write_json(output_dir/"ARACHNE_MAGE_A0_V3_DISPOSABLE_TINY_PREFLIGHT.json",report); return report


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,required=True); ap.add_argument("--zero-surface",type=Path); ap.add_argument("--camera-dir",type=Path); ap.add_argument("--qualified-skeleton",type=Path); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--disposable-tiny",action="store_true"); ap.add_argument("--full-preflight",action="store_true"); ap.add_argument("--require-cuda",action="store_true"); args=ap.parse_args(argv); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.disposable_tiny:
        print("V3_PREFLIGHT_TINY_START",flush=True); r=run_disposable_tiny(args.cache,args.output_dir); print("V3_PREFLIGHT_TINY="+json.dumps(r,sort_keys=True),flush=True); return 0
    global hist; import run_arachne_mage_a0_historical_1cc449 as hist
    if args.zero_surface is None or args.camera_dir is None or args.qualified_skeleton is None: raise RuntimeError("V3_FULL_INPUT_PATHS_REQUIRED")
    if args.require_cuda and not torch.cuda.is_available(): raise RuntimeError("V3_CUDA_REQUIRED")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and torch.cuda.get_device_properties(0).total_memory<18*1024**3: raise RuntimeError("V3_MAIN_REQUIRES_AT_LEAST_18_GIB_VRAM")
    dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device.type=="cuda" else torch.float32)
    cache=load_cache(args.cache); geometry=geometry_from_cache(cache); surface=hist.load_surface(args.zero_surface,args.camera_dir); sk=hist.load_skeleton(args.qualified_skeleton); sids,jids=hist.validate_binding(surface,sk,cache); transforms=hist.probe_transforms(len(jids),device); reset_seed(); model=make_model(device,False,dtype)
    if args.full_preflight:
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_STEPS); one=train_step(model,opt,sch,cache,geometry,1,device,dtype); report={"schema":SCHEMA+".FullPreflight","status":"PASS_FULL_V3_REAL_MAGE_ONE_FIELD_FORWARD_FSQ_DECODE_BACKWARD","device":str(device),"dtype":str(dtype),"parameter_count":EXPECTED_PARAM_COUNT,"config_hash":EXPECTED_CONFIG_HASH,"one_step":one,"main_v3_scientific_optimizer_steps":0}; write_json(args.output_dir/"ARACHNE_MAGE_A0_V3_FULL_PREFLIGHT.json",report); print("V3_FULL_PREFLIGHT="+json.dumps(report,sort_keys=True),flush=True); return 0
    result_path=args.output_dir/RESULT_NAME; progress_path=args.output_dir/PROGRESS_NAME; final_ck=args.output_dir/CHECKPOINT_NAME
    if result_path.exists():
        d=json.loads(result_path.read_text()); status=d.get("status")
        if status in ("A0_V3_TERMINAL_PASS","NO_A0_V3_TERMINAL_CLOSURE"): print("V3_EXISTING_TERMINAL_RESULT="+status,flush=True); return 0
        raise RuntimeError("V3_EXISTING_RESULT_STATUS_DRIFT")
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_STEPS); start=streak=0; trace=[]
    if progress_path.exists(): start,streak,trace=load_progress(progress_path,model,opt,sch,device); print(f"V3_RESUME step={start} streak={streak}",flush=True)
    else: print("V3_FRESH_START",flush=True)
    print(f"V3_MODEL params={EXPECTED_PARAM_COUNT} dtype={dtype} device={device} max_steps={MAX_STEPS}",flush=True); closure=None; last=None
    for step in range(start+1,MAX_STEPS+1):
        tr=train_step(model,opt,sch,cache,geometry,step,device,dtype)
        if step==1 or step%CHECK_EVERY==0:
            m=matrix_metrics(model,cache,geometry,surface,sk,sids,jids,transforms,device,dtype); passed=is_pass(m); streak=streak+1 if passed else 0; rec={"step":step,"pass":passed,"streak":streak,"train":tr,**m}; trace.append(rec); last=m; print("A0_V3_CHECK "+json.dumps({k:rec[k] for k in ("step","pass","streak","raw_row_l1_p95_auth","raw_deformation_error_ratio_auth","qualified_row_l1_p95_auth")},sort_keys=True),flush=True)
            if streak>=REQUIRED_STABLE: closure=step; break
        if step%CHECKPOINT_EVERY==0: save_progress(progress_path,model,opt,sch,step,streak,trace,str(dtype)); print(f"A0_V3_CHECKPOINT step={step}",flush=True)
    final_step=closure if closure is not None else MAX_STEPS
    if last is None or trace[-1]["step"]!=final_step: last=matrix_metrics(model,cache,geometry,surface,sk,sids,jids,transforms,device,dtype)
    status="A0_V3_TERMINAL_PASS" if closure is not None else "NO_A0_V3_TERMINAL_CLOSURE"; ck_sha=None
    if closure is not None:
        torch.save({"schema":SCHEMA+".QualifiedCodec","model":model.state_dict(),"config":asdict(model.config),"config_hash":EXPECTED_CONFIG_HASH,"parameter_count":EXPECTED_PARAM_COUNT},final_ck); ck_sha=sha_file(final_ck)
    result={"schema":SCHEMA,"status":status,"final_step":final_step,"closure_step":closure,"terminal_streak":streak,"required_terminal_streak":REQUIRED_STABLE,"config_hash":EXPECTED_CONFIG_HASH,"parameter_count":EXPECTED_PARAM_COUNT,"cache_sha256":EXPECTED_CACHE_SHA,"cache_binding_sha256":EXPECTED_CACHE_BINDING_SHA,"binding_sha256":EXPECTED_TARGET_BINDING_SHA,"teacher_weights_content_sha256":EXPECTED_TEACHER_WEIGHTS_SHA,"final_metrics":last,"qualified_codec_checkpoint_sha256":ck_sha,"a1_optimizer_authorized":status=="A0_V3_TERMINAL_PASS","trace":trace}; write_json(result_path,result)
    if progress_path.exists() and status=="A0_V3_TERMINAL_PASS": progress_path.unlink()
    print("A0_V3_FINAL="+json.dumps({"status":status,"final_step":final_step,"closure_step":closure,"streak":streak,"p95":last["raw_row_l1_p95_auth"],"deform":last["raw_deformation_error_ratio_auth"]},sort_keys=True),flush=True); return 0

if __name__=="__main__": raise SystemExit(main())
