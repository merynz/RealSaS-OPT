from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


EXPECTED_V6_SOURCE_SHA256 = "39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01"
EXPECTED_V6_PREREG_SHA256 = "187f5c1bac1e813ebdfd1bbb76fa6def3d6543b4c7abc5609d217b3ece5fd597"
EXPECTED_BANK_SHA256 = "26b6891ff81b9bfc3405461394159547fa76b417852a44518425b4e1de7e67d3"
EXPECTED_SOURCE_PROGRESS_SHA256 = "49e189b9ed2ab473a9e36e47fdd0b1e0d6342be5883734a1067a717fdd4c457f"
EXPECTED_SCHEDULE_SHA256 = "910e79a11dd70210108e164ed6e2b0c1a4b1fbb69632b80f546c68a9bb5331e9"


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):
            h.update(b)
    return h.hexdigest()


def hash_tensor(t: torch.Tensor) -> str:
    a=t.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def hash_state_dict(sd) -> str:
    h=hashlib.sha256()
    for k in sorted(sd):
        v=sd[k]
        h.update(k.encode()+b"\0")
        h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_module(path: Path):
    spec=importlib.util.spec_from_file_location("realsas_exact_knight_v6_diag",str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("V6_IMPORT_FAIL")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v6-source",type=Path,required=True)
    ap.add_argument("--v6-prereg",type=Path,required=True)
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--skeleton-json",type=Path,required=True)
    ap.add_argument("--teacher-bank",type=Path,required=True)
    ap.add_argument("--source-progress",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=256)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--deterministic",action="store_true")
    args=ap.parse_args()

    for p,expected,name in (
        (args.v6_source,EXPECTED_V6_SOURCE_SHA256,"v6_source"),
        (args.v6_prereg,EXPECTED_V6_PREREG_SHA256,"v6_prereg"),
        (args.teacher_bank,EXPECTED_BANK_SHA256,"teacher_bank"),
        (args.source_progress,EXPECTED_SOURCE_PROGRESS_SHA256,"source_progress"),
    ):
        got=sha_file(p)
        if got!=expected:
            raise RuntimeError(f"{name.upper()}_SHA_DRIFT::{got}")

    mod=load_module(args.v6_source)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    device=torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.cuda.empty_cache()
    free,total=torch.cuda.mem_get_info()
    if free < 24*1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A100_24GIB_BF16_REQUIRED")

    prereg=json.loads(args.v6_prereg.read_text())
    surface=mod.rigging_surface_from_dict(json.loads(args.surface_json.read_text()))
    skeleton=mod.qualified_skeleton_from_dict(json.loads(args.skeleton_json.read_text()))
    bank_path=args.teacher_bank

    torch.manual_seed(mod.SEED)
    np.random.seed(mod.SEED)
    random.seed(mod.SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

    conditioning=mod.ArachneRichConditioningAdapterV3(require_scene_first=True)([surface],[skeleton])
    ci=mod.base._conditioning_to_torch(conditioning,device)
    truth,valid,teacher_binding=mod.base._teacher_for_conditioning(bank_path,conditioning,skeleton)
    world=np.asarray(conditioning.surface_positions_world[0],np.float32)
    world_t=torch.as_tensor(world,device=device,dtype=torch.float32)
    truth_t=torch.as_tensor(truth,device=device,dtype=torch.float32)
    joint_world,parent,joint_mask=mod.base._joint_world(skeleton,conditioning,device)
    articulated=mod.build_articulated_probe_transforms(joint_world,parent,joint_mask)

    source_model,source_report=mod._load_backbone_from_v5_progress(args.source_progress,device)
    with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
        raw=source_model.backbone(**ci)
        geom=source_model.geometry7_from_surface(
            ci["surface_positions_normalized"],ci["surface_normals"],ci["surface_normal_valid"]
        )[0].float()
        pair_geometry=ci["pair_geometry"][0].float()
        legal=(ci["pair_mask"].bool() & ci["surface_mask"][:,:,None].bool() & ci["joint_mask"][:,None,:].bool())[0]
        field_tokens=raw.field_tokens[0].float()
        surface_memory=raw.surface_memory[0].float()
    source_model.cpu()
    torch.cuda.empty_cache()

    feature_hashes={
        "surface_memory":hash_tensor(surface_memory),
        "field_tokens":hash_tensor(field_tokens),
        "geometry7":hash_tensor(geom),
        "pair_geometry":hash_tensor(pair_geometry),
        "legal":hashlib.sha256(legal.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }
    feature_hashes["bundle"]=hashlib.sha256("".join(feature_hashes[k] for k in sorted(feature_hashes)).encode()).hexdigest()

    torch.manual_seed(mod.DECODER_SEED)
    np.random.seed(mod.DECODER_SEED)
    decoder=mod.ArachneV6RawReadout(
        surface_dim=int(surface_memory.shape[-1]),
        token_dim=int(field_tokens.shape[-1]),
        pair_dim=int(pair_geometry.shape[-1]),
        geom_dim=int(geom.shape[-1]),
        relation_dim=mod.RELATION_DIM,
    ).to(device=device,dtype=torch.float32)
    init_hash=hash_state_dict(decoder.state_dict())

    clean=np.flatnonzero(np.asarray(valid,bool))
    schedule=mod._sample_schedule(clean)
    schedule_hash=mod._schedule_digest(schedule)
    if schedule_hash!=EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(f"SCHEDULE_SHA_DRIFT::{schedule_hash}")

    opt=torch.optim.AdamW(decoder.parameters(),lr=mod.DECODER_LR,weight_decay=mod.WEIGHT_DECAY)
    checkpoints={}
    for step in range(1,int(args.steps)+1):
        decoder.train()
        idx_np=schedule[step-1].astype(np.int64)
        idx=torch.as_tensor(idx_np,device=device,dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
            logits,pred=decoder.forward_rows(idx,surface_memory,geom,pair_geometry,field_tokens,legal)
            losses=mod._objective(logits,pred,truth_t[idx],world_t[idx],articulated)
        losses["total"].backward()
        grad=float(torch.nn.utils.clip_grad_norm_(decoder.parameters(),mod.GRAD_CLIP))
        opt.step()
        lr=mod._cosine_lr(step)
        for group in opt.param_groups:
            group["lr"]=lr
        if step in (1,64,256,int(args.steps)):
            checkpoints[str(step)]={
                "decoder_state_sha256":hash_state_dict(decoder.state_dict()),
                "loss":float(losses["total"].detach().cpu()),
                "grad_norm":float(grad),
            }

    eval_row,_,_=mod._evaluate(
        decoder,surface_memory,geom,pair_geometry,field_tokens,legal,
        truth,valid,surface,skeleton,conditioning,world,
        joint_world,parent,joint_mask,device,int(args.steps)
    )

    props=torch.cuda.get_device_properties(0)
    receipt={
        "schema":"RealSaS.KnightArachneV6ReproducibilityReplica.v1",
        "status":"PASS_REPLICA_CAPTURE",
        "steps":int(args.steps),
        "runtime":{
            "gpu":props.name,
            "torch":torch.__version__,
            "cuda_runtime":torch.version.cuda,
            "numpy":np.__version__,
            "pythonhashseed":os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_requested":bool(args.deterministic),
            "deterministic_algorithms":bool(torch.are_deterministic_algorithms_enabled()),
            "tf32_matmul":bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn":bool(torch.backends.cudnn.allow_tf32),
        },
        "source_report":source_report,
        "teacher_binding":teacher_binding,
        "schedule_sha256":schedule_hash,
        "feature_hashes":feature_hashes,
        "decoder_init_sha256":init_hash,
        "checkpoints":checkpoints,
        "eval":{
            "row_l1_p95":float(eval_row["metrics"]["row_l1_p95"]),
            "legacy_deformation_ratio":float(eval_row["metrics"]["legacy_deformation_ratio"]),
            "articulated_deformation_ratio":float(eval_row["metrics"]["articulated_deformation_ratio"]),
            "dominant_accuracy":float(eval_row["metrics"]["dominant_accuracy"]),
        },
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("ARACHNE_V6_REPRO_REPLICA="+json.dumps(receipt,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
