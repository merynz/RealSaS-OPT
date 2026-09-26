from __future__ import annotations

"""Knight Arachne V5 demo-witness transfer fit.

This runner is deliberately scoped to the single Knight demo witness. It consumes
only the qualified RiggingSurfaceIR + Compiler-qualified Geppetto skeleton as
predictor inputs. Exact source skin is teacher-only objective/evaluation truth.

Execution policy:
- sealed Mage FIT2 Arachne V5 checkpoint is transfer initialization only;
- perform a zero-shot Knight evaluation before constructing the optimizer;
- if zero-shot closes every frozen science/Compiler gate, stop without updates;
- otherwise fine-tune end-to-end V5 with the frozen preregistered apparatus;
- require three consecutive full PASS checks;
- Compiler.qualify_skin remains the only skin authority.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
    write_ir_json,
)
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.articulated_probe_v1 import build_articulated_probe_transforms
from models.arachne.v4.loss_v4 import articulated_deformation_ratio_loss
from models.arachne.v5.arachne_candidate_v5 import ArachneA1V5


SCHEMA = "RealSaS.KnightArachneV5DemoTransferFit.v1"
ARCHITECTURE_ID = "RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5"
EXPECTED_INIT_SHA256 = "8749843ee673f5c4b84b5d58a6285bd108e9867af5d87bd4afcf2a5e64d98353"
EXPECTED_SURFACE_LINEAGE = "452cf6564a2cde35a4d6d420021491d231f90cec267df9f41d4424dc8d74895a"
EXPECTED_SKELETON_LINEAGE = "c2727bcf4f4b7f4bfbc8e4af24b8d731dd0892beee10608862f0943e4daebcce"
EXPECTED_SURFACE_N = 12090
EXPECTED_SURFACE_E = 36469
EXPECTED_JOINTS = 28
EXPECTED_PARAMETER_COUNT = 138_378_466

SEED = 20260925
MAX_STEPS = 16384
CHECK_EVERY = 256
REQUIRED_STABLE = 3
MIN_FINETUNE_GATE_STEP = 2048
PROGRESS_EVERY = 1024
ROWS_PER_STEP = 384
BACKBONE_LR = 5.0e-5
DECODER_LR = 3.0e-4
BACKBONE_WARMUP_STEPS = 512
BACKBONE_LR_FLOOR = 0.10
DECODER_LR_FLOOR = 0.03
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0

TRAINING_APPARATUS_ID = "KNIGHT_ARACHNE_V5_TRANSFER_SEALED_FIT2_OBJECTIVE_V2"

ROW_L1_P95_MAX = 0.05
LEGACY_DEFORM_MAX = 0.05
ARTICULATED_DEFORM_MAX = 0.05
COMPILER_TOTAL_CORR_MAX = 1.0e-4
COMPILER_ROW_CORR_MAX = 1.0e-5
SIMPLEX_MAX = 1.0e-6


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def _runtime() -> dict:
    out = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "numpy_version": str(np.__version__),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        out["gpu0"] = {
            "name": str(p.name),
            "compute_capability": [int(p.major), int(p.minor)],
            "total_memory_bytes": int(total),
            "free_memory_bytes": int(free),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    return out


def _conditioning_to_torch(c, device: torch.device) -> dict:
    names = (
        "surface_positions_normalized","surface_normals","surface_normal_valid",
        "surface_support_views","surface_raster_xy","surface_raster_valid",
        "surface_observed","surface_completed","surface_mask",
        "edge_index","edge_features","edge_mask",
        "joint_positions_normalized","joint_mask","parent_indices","root_mask",
        "deform_root_mask","joint_depth_normalized","support_anchor_matrix",
        "pair_geometry","pair_mask",
    )
    bools = {
        "surface_normal_valid","surface_support_views","surface_raster_valid",
        "surface_observed","surface_completed","surface_mask","edge_mask",
        "joint_mask","root_mask","deform_root_mask","support_anchor_matrix","pair_mask",
    }
    ints = {"edge_index","parent_indices"}
    out = {}
    for name in names:
        dtype = torch.bool if name in bools else torch.long if name in ints else torch.float32
        out[name] = torch.as_tensor(getattr(c, name), device=device, dtype=dtype)
    out["view_yaw_fourier"] = torch.as_tensor(c.view_yaw_code, device=device, dtype=torch.float32)
    return out


def _extract_state(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise RuntimeError("ARACHNE_INIT_CHECKPOINT_STATE_NOT_FOUND")


def _load_init(model: ArachneA1V5, path: Path) -> dict:
    if _sha(path) != EXPECTED_INIT_SHA256:
        raise RuntimeError("ARACHNE_INIT_CHECKPOINT_SHA_DRIFT")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)

    # The sealed Mage FIT2 V5 artifact is a ModelOnly payload with separate
    # backbone/decoder state dicts (not a monolithic ArachneA1V5 state dict).
    # Preserve that historical serialization exactly; bind it into the current
    # composite model without rewriting or weakening strict key checks.
    split_loaded = False
    if isinstance(payload, dict):
        b = payload.get("backbone")
        d = payload.get("decoder")
        if (
            isinstance(b, dict) and b and all(torch.is_tensor(v) for v in b.values())
            and isinstance(d, dict) and d and all(torch.is_tensor(v) for v in d.values())
        ):
            model.backbone.load_state_dict(b, strict=True)
            model.decoder.load_state_dict(d, strict=True)
            split_loaded = True

    if not split_loaded:
        state = _extract_state(payload)
        if state and all(str(k).startswith("module.") for k in state):
            state = {str(k)[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

    return {
        "checkpoint_sha256": EXPECTED_INIT_SHA256,
        "payload_schema": str(payload.get("schema") or "") if isinstance(payload, dict) else "",
        "payload_step": None if not isinstance(payload, dict) else payload.get("step"),
        "serialization_mode": "SPLIT_BACKBONE_DECODER" if split_loaded else "MONOLITHIC_STATE_DICT",
    }


def _qualified_source_index_map(skeleton, conditioning) -> np.ndarray:
    by_id = {str(j.canonical_joint_id): j for j in skeleton.joints}
    out = []
    for jid in conditioning.joint_ids[0]:
        joint = by_id[str(jid)]
        raw = str(joint.source_proposal_id)
        if not raw.startswith("P:GRS:"):
            raise RuntimeError(f"ARACHNE_KNIGHT_SOURCE_PROPOSAL_ID_DRIFT:{raw}")
        out.append(int(raw.rsplit(":", 1)[1]))
    arr = np.asarray(out, dtype=np.int64)
    if sorted(arr.tolist()) != list(range(EXPECTED_JOINTS)):
        raise RuntimeError(f"ARACHNE_KNIGHT_PROPOSAL_INDEX_PERMUTATION_DRIFT:{arr.tolist()}")
    return arr


def _teacher_for_conditioning(bank_path: Path, conditioning, skeleton) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(bank_path, allow_pickle=False) as z:
        required = {"weights","surface_ids","teacher_valid_mask"}
        if not required.issubset(z.files):
            raise RuntimeError(f"ARACHNE_TEACHER_BANK_SCHEMA_MISSING:{sorted(required-set(z.files))}")
        w0 = np.asarray(z["weights"], dtype=np.float64)
        valid0 = np.asarray(z["teacher_valid_mask"], dtype=np.uint8).astype(bool)
        s0 = tuple(map(str, z["surface_ids"].tolist()))
    if w0.shape != (len(s0), EXPECTED_JOINTS) or valid0.shape != (len(s0),):
        raise RuntimeError(f"ARACHNE_TEACHER_BANK_SHAPE_DRIFT:{w0.shape}:{valid0.shape}")
    row_of = {sid:i for i,sid in enumerate(s0)}
    wanted = tuple(map(str, conditioning.surface_ids[0]))
    if len(row_of) != len(s0) or any(sid not in row_of for sid in wanted):
        raise RuntimeError("ARACHNE_TEACHER_SURFACE_ID_BINDING_DRIFT")
    rows = np.asarray([row_of[sid] for sid in wanted], dtype=np.int64)
    cols = _qualified_source_index_map(skeleton, conditioning)
    w = w0[rows][:, cols]
    valid = valid0[rows]
    if int(np.count_nonzero(valid)) < 192:
        raise RuntimeError("ARACHNE_TEACHER_CLEAN_ROWS_INSUFFICIENT")
    w = np.maximum(w, 0.0)
    sums = w.sum(1, keepdims=True)
    if np.any(sums <= 1e-12):
        raise RuntimeError("ARACHNE_TEACHER_ZERO_ROW")
    w /= sums
    residual = np.abs(w.sum(1)-1.0)
    if not np.isfinite(w).all() or float(residual.max(initial=0.0)) > 1e-10:
        raise RuntimeError("ARACHNE_TEACHER_NUMERIC_DRIFT")
    return w.astype(np.float32), valid.astype(bool), {
        "bank_sha256": _sha(bank_path),
        "row_reindexed": bool(rows.tolist() != list(range(len(rows)))),
        "canonical_joint_to_target_index": cols.tolist(),
        "max_simplex_residual": float(residual.max(initial=0.0)),
        "clean_row_count": int(np.count_nonzero(valid)),
        "coverage": float(np.mean(valid)),
        "teacher_valid_mask_applied": True,
    }


def _joint_world(skeleton, conditioning, device: torch.device):
    by_id = {str(j.canonical_joint_id): j for j in skeleton.joints}
    ids = tuple(map(str, conditioning.joint_ids[0]))
    world = np.asarray([[by_id[jid].position for jid in ids]], np.float32)
    return (
        torch.as_tensor(world, device=device, dtype=torch.float32),
        torch.as_tensor(conditioning.parent_indices, device=device, dtype=torch.long),
        torch.as_tensor(conditioning.joint_mask, device=device, dtype=torch.bool),
    )


def _legacy_probe_transforms(j: int, device: torch.device) -> torch.Tensor:
    poses = 4
    t = torch.eye(4, dtype=torch.float32, device=device)[None,None].repeat(1,poses,j,1,1)
    for ji in range(j):
        u = float(ji + 1) / float(j)
        t[0,1,ji,0,3] = 0.10 * u
        t[0,1,ji,1,3] = 0.035 * (-1 if ji % 2 else 1)
        t[0,2,ji,1,3] = 0.085 * u
        t[0,2,ji,2,3] = 0.030 * (ji - (j - 1) / 2)
        t[0,3,ji,0,3] = -0.055 * (ji - (j - 1) / 2)
        t[0,3,ji,2,3] = 0.070 * u
    return t


def _lbs(rest, weights, transforms):
    ones = torch.ones((*rest.shape[:2],1), dtype=rest.dtype, device=rest.device)
    hom = torch.cat([rest, ones], -1)
    moved = torch.einsum("bpjac,bnc->bpjna", transforms, hom)[...,:3]
    return torch.einsum("bnj,bpjna->bpna", weights, moved)


def _deform_ratio(rest, truth, pred, transforms, row_mask):
    td = _lbs(rest, truth, transforms)
    pd = _lbs(rest, pred, transforms)
    rp = rest[:,None].expand_as(td)
    m = row_mask[:,None,:,None].to(td.dtype)
    den = (m.sum()*td.shape[1]*td.shape[-1]).clamp_min(1.0)
    motion = torch.sqrt((((td-rp)*m).square().sum()/den).clamp_min(1e-12))
    err = torch.sqrt((((pd-td)*m).square().sum()/den).clamp_min(1e-12))
    return float((err/motion.clamp_min(1e-6)).detach().cpu()), float(motion.detach().cpu()), float(err.detach().cpu())


def _decode_all(model: ArachneA1V5, ci: dict, *, chunk: int = 1024) -> torch.Tensor:
    raw = model.backbone(**ci)
    geom = model.geometry7_from_surface(
        ci["surface_positions_normalized"], ci["surface_normals"], ci["surface_normal_valid"]
    )
    legal = ci["pair_mask"].bool() & ci["surface_mask"][:,:,None].bool() & ci["joint_mask"][:,None,:].bool()
    pieces = []
    for start in range(0, geom.shape[1], int(chunk)):
        stop = min(start + int(chunk), geom.shape[1])
        _, w = model.decoder(
            geom[:,start:stop],
            ci["pair_geometry"][:,start:stop],
            legal[:,start:stop],
            raw.field_tokens,
        )
        pieces.append(w.float())
    return torch.cat(pieces, 1)


def _canonicalize_rows(weights: np.ndarray) -> tuple[np.ndarray, dict]:
    raw = np.asarray(weights, dtype=np.float64)
    clipped = np.maximum(raw, 0.0)
    sums = clipped.sum(1, keepdims=True)
    if np.any(sums <= 1e-15):
        raise RuntimeError("ARACHNE_PRED_ZERO_ROW")
    final = clipped / sums
    delta = np.abs(final - raw).sum(1)
    return final, {
        "method":"FLOAT64_ROW_SUM_NORMALIZATION_V1",
        "max_row_l1":float(delta.max(initial=0.0)),
        "total_l1":float(delta.sum()),
        "simplex_max_abs_residual":float(np.max(np.abs(final.sum(1)-1.0))),
    }


def _proposal(surface, skeleton, conditioning, weights64: np.ndarray, provenance: str) -> SkinProposalIR:
    sids = tuple(map(str, conditioning.surface_ids[0]))
    jids = tuple(map(str, conditioning.joint_ids[0]))
    if weights64.shape != (len(sids), len(jids)):
        raise RuntimeError("ARACHNE_PROPOSAL_SHAPE_DRIFT")
    influences = tuple(
        SkinInfluenceProposal(sid, jid, float(weights64[si,ji]))
        for si,sid in enumerate(sids)
        for ji,jid in enumerate(jids)
    )
    return SkinProposalIR(
        influences,
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        model_provenance=provenance,
        metadata={
            "candidate_architecture":ARCHITECTURE_ID,
            "field_tokens":4,
            "dense_proposal":True,
            "compiler_owns_qualification":True,
            "teacher_input_used":False,
            "demo_witness":True,
        },
    )


def _metrics(pred: np.ndarray, truth: np.ndarray, valid_mask: np.ndarray, world: np.ndarray, joint_world, parent, joint_mask, device) -> dict:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (len(pred),) or not bool(valid.any()):
        raise RuntimeError("ARACHNE_TEACHER_VALID_MASK_DRIFT")
    row_all = np.abs(np.asarray(pred,np.float64)-np.asarray(truth,np.float64)).sum(1)
    row = row_all[valid]
    pt = torch.as_tensor(np.asarray(pred,np.float32)[None], device=device)
    tt = torch.as_tensor(np.asarray(truth,np.float32)[None], device=device)
    rest = torch.as_tensor(np.asarray(world,np.float32)[None], device=device)
    mask = torch.as_tensor(valid[None], device=device, dtype=torch.bool)
    legacy = _legacy_probe_transforms(pred.shape[1], device)
    legacy_ratio, legacy_motion, legacy_err = _deform_ratio(rest,tt,pt,legacy,mask)
    articulated = build_articulated_probe_transforms(joint_world,parent,joint_mask)
    with torch.no_grad():
        art_ratio, art_motion, art_err = articulated_deformation_ratio_loss(pt,tt,rest,articulated,mask)
    return {
        "rows_total":int(len(pred)),
        "rows_evaluated":int(np.count_nonzero(valid)),
        "teacher_coverage":float(np.mean(valid)),
        "row_l1_mean":float(row.mean()),
        "row_l1_p50":float(np.quantile(row,.50)),
        "row_l1_p90":float(np.quantile(row,.90)),
        "row_l1_p95":float(np.quantile(row,.95)),
        "row_l1_p99":float(np.quantile(row,.99)),
        "cvar10":float(row[row>=np.quantile(row,.90)].mean()),
        "dominant_accuracy":float((pred[valid].argmax(1)==truth[valid].argmax(1)).mean()),
        "simplex_max_abs_residual":float(np.max(np.abs(np.asarray(pred,np.float64).sum(1)-1.0))),
        "legacy_deformation_ratio":legacy_ratio,
        "legacy_teacher_motion_rms":legacy_motion,
        "legacy_error_rms":legacy_err,
        "articulated_deformation_ratio":float(art_ratio.cpu()),
        "articulated_teacher_motion_rms":float(art_motion.cpu()),
        "articulated_error_rms":float(art_err.cpu()),
    }


def _science_pass(m: dict) -> bool:
    return (
        bool(np.isfinite(list(v for v in m.values() if isinstance(v,(int,float)))).all())
        and m["row_l1_p95"] <= ROW_L1_P95_MAX
        and m["legacy_deformation_ratio"] <= LEGACY_DEFORM_MAX
        and m["articulated_deformation_ratio"] <= ARTICULATED_DEFORM_MAX
        and m["simplex_max_abs_residual"] <= SIMPLEX_MAX
    )


def _evaluate(model, ci, truth, valid_mask, surface, skeleton, conditioning, world, joint_world, parent, joint_mask, device, step, provenance):
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        pred_t = _decode_all(model,ci,chunk=1024)
    pred = pred_t[0].float().cpu().numpy()
    m = _metrics(pred,truth,valid_mask,world,joint_world,parent,joint_mask,device)
    science = _science_pass(m)
    compiler = {"pass":False,"attempted":False}
    canonicalization = None
    qualified = None
    if science:
        w64, canonicalization = _canonicalize_rows(pred)
        proposal = _proposal(surface,skeleton,conditioning,w64,provenance)
        compiler["attempted"] = True
        try:
            qualified = qualify_skin(
                surface,skeleton,proposal,
                max_simplex_repair_l1=COMPILER_ROW_CORR_MAX,
                max_total_correction_l1=COMPILER_TOTAL_CORR_MAX,
                negative_tolerance=1e-8,
            )
            report = dict(qualified.qualification_report)
            compiler = {
                "pass":len(qualified.rows)==EXPECTED_SURFACE_N and float(report["total_correction_l1"])<=COMPILER_TOTAL_CORR_MAX,
                "attempted":True,
                "row_count":len(qualified.rows),
                "skin_lineage_hash":qualified.skin_lineage_hash,
                "qualification_report":report,
            }
        except Exception as exc:
            compiler = {"pass":False,"attempted":True,"error":repr(exc)}
    return {
        "step":int(step),
        "metrics":m,
        "science_gate":bool(science),
        "compiler":compiler,
        "canonicalization":canonicalization,
        "full_gate":bool(science and compiler.get("pass") is True),
    }, pred, qualified


def _backbone_lr_factor(step: int) -> float:
    s=max(1,int(step))
    if s<=BACKBONE_WARMUP_STEPS:
        return float(s)/float(BACKBONE_WARMUP_STEPS)
    q=min(max((s-BACKBONE_WARMUP_STEPS)/float(MAX_STEPS-BACKBONE_WARMUP_STEPS),0.0),1.0)
    return BACKBONE_LR_FLOOR + (1.0-BACKBONE_LR_FLOOR)*0.5*(1.0+math.cos(math.pi*q))


def _decoder_lr_factor(step: int) -> float:
    s=max(1,int(step))
    q=min(max(s/float(MAX_STEPS),0.0),1.0)
    return DECODER_LR_FLOOR + (1.0-DECODER_LR_FLOOR)*0.5*(1.0+math.cos(math.pi*q))


def _sample_loss(model, ci, truth_np, valid_mask_np, surface_world_np, articulated_transforms, rng, device):
    clean=np.flatnonzero(np.asarray(valid_mask_np,dtype=bool))
    if len(clean)<1:
        raise RuntimeError("ARACHNE_TEACHER_VALID_MASK_EMPTY")
    idx_np=rng.choice(clean,size=min(ROWS_PER_STEP,len(clean)),replace=False)
    idx=torch.as_tensor(idx_np,device=device,dtype=torch.long)
    raw=model.backbone(**ci)
    geom_all=model.geometry7_from_surface(
        ci["surface_positions_normalized"],ci["surface_normals"],ci["surface_normal_valid"]
    )
    legal_all=ci["pair_mask"].bool() & ci["surface_mask"][:,:,None].bool() & ci["joint_mask"][:,None,:].bool()
    logits,pred=model.decoder(
        geom_all[:,idx],
        ci["pair_geometry"][:,idx],
        legal_all[:,idx],
        raw.field_tokens,
    )

    # Exact direct-simplex loss semantics from the sealed Mage FIT2 V5 closure.
    # The previous Knight runner accidentally removed dice + blend-boundary terms
    # and downweighted categorical CE 10x; this is a causal apparatus correction,
    # not a gate relaxation.
    t=torch.as_tensor(truth_np[idx_np],device=device,dtype=torch.float32)
    p=pred[0].float()
    z=logits[0].float()
    rm=torch.ones((len(idx_np),),device=device,dtype=torch.bool)
    logp=torch.log_softmax(z,-1)
    ce=(-(t*logp).sum(-1)[rm]).mean()
    mse=F.mse_loss(p[rm],t[rm])
    numer=2*(p*t).sum(0)+1e-4
    denom=p.square().sum(0)+t.square().sum(0)+1e-4
    dice=(1-numer/denom).mean()
    row=(p-t).abs().sum(-1)
    rv=row[rm]
    coupled=rv.mean()
    k=max(1,int((rv.numel()+9)//10))
    hard=torch.topk(rv,k=k,largest=True).values.mean()
    blend=(1-t.max(-1).values).clamp_min(0)
    bw=blend[rm]
    boundary=(rv*bw).sum()/bw.sum().clamp_min(1e-8) if bool((bw>0).any()) else rv.new_zeros(())
    rest=torch.as_tensor(surface_world_np[idx_np][None],device=device,dtype=torch.float32)
    art,motion,err=articulated_deformation_ratio_loss(
        p[None],t[None],rest,articulated_transforms,rm[None]
    )
    total=ce + 0.1*mse + dice + coupled + 0.5*hard + 0.5*boundary + 0.25*art
    return {
        "total":total,
        "categorical_ce":ce,
        "mse":mse,
        "dice":dice,
        "normalized_row_l1":coupled,
        "hard_tail_cvar10":hard,
        "blend_boundary_l1":boundary,
        "articulated_deformation_ratio":art,
        "articulated_teacher_motion_rms":motion,
        "articulated_deformation_error_rms":err,
    }

def run(args) -> dict:
    prereg_path=Path(args.prereg).resolve()
    surface_path=Path(args.surface_json).resolve()
    skeleton_path=Path(args.skeleton_json).resolve()
    bank_path=Path(args.teacher_bank).resolve()
    init_path=Path(args.init_checkpoint).resolve()
    outdir=Path(args.output_dir).resolve()
    outdir.mkdir(parents=True,exist_ok=True)

    print("ARACHNE_KNIGHT_STAGE=PREREG_BEGIN", flush=True)
    prereg=_read_json(prereg_path)
    if prereg.get("schema")!="RealSaS.KnightArachneV5DemoTransferPreregistration.v2" or prereg.get("status")!="FROZEN_BEFORE_KNIGHT_ARACHNE_OPTIMIZER_STEP_1":
        raise RuntimeError("ARACHNE_KNIGHT_PREREG_DRIFT")
    if prereg.get("training",{}).get("apparatus_id")!=TRAINING_APPARATUS_ID:
        raise RuntimeError("ARACHNE_KNIGHT_TRAINING_APPARATUS_DRIFT")

    print("ARACHNE_KNIGHT_STAGE=PREREG_PASS", flush=True)
    surface=rigging_surface_from_dict(_read_json(surface_path))
    skeleton=qualified_skeleton_from_dict(_read_json(skeleton_path))
    if surface.geometry_lineage_hash!=EXPECTED_SURFACE_LINEAGE or len(surface.surface_nodes)!=EXPECTED_SURFACE_N or len(surface.local_relations)!=EXPECTED_SURFACE_E:
        raise RuntimeError("ARACHNE_KNIGHT_SURFACE_AUTHORITY_DRIFT")
    if skeleton.skeleton_lineage_hash!=EXPECTED_SKELETON_LINEAGE or len(skeleton.joints)!=EXPECTED_JOINTS:
        raise RuntimeError("ARACHNE_KNIGHT_SKELETON_AUTHORITY_DRIFT")

    if not torch.cuda.is_available():
        raise RuntimeError("ARACHNE_KNIGHT_CUDA_REQUIRED")
    device=torch.device("cuda")
    free,total=torch.cuda.mem_get_info()
    if not torch.cuda.is_bf16_supported() or free < 28*1024**3:
        raise RuntimeError(f"ARACHNE_KNIGHT_REQUIRES_BF16_AND_28GIB_FREE::{_runtime()}")

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    print("ARACHNE_KNIGHT_STAGE=AUTHORITY_PASS", flush=True)
    conditioning=ArachneRichConditioningAdapterV3(require_scene_first=True)([surface],[skeleton])
    if int(conditioning.surface_mask[0].sum())!=EXPECTED_SURFACE_N or int(conditioning.edge_mask[0].sum())!=EXPECTED_SURFACE_E or int(conditioning.joint_mask[0].sum())!=EXPECTED_JOINTS:
        raise RuntimeError("ARACHNE_KNIGHT_CONDITIONING_CARDINALITY_DRIFT")
    ci=_conditioning_to_torch(conditioning,device)
    truth,teacher_valid,teacher_binding=_teacher_for_conditioning(bank_path,conditioning,skeleton)
    world=np.asarray(conditioning.surface_positions_world[0],np.float32)
    joint_world,parent,joint_mask=_joint_world(skeleton,conditioning,device)
    articulated=build_articulated_probe_transforms(joint_world,parent,joint_mask)

    print("ARACHNE_KNIGHT_STAGE=CONDITIONING_PASS", flush=True)
    model=ArachneA1V5().to(device=device,dtype=torch.float32)
    if model.parameter_count!=EXPECTED_PARAMETER_COUNT or model.config.architecture_id!=ARCHITECTURE_ID:
        raise RuntimeError(f"ARACHNE_KNIGHT_MODEL_CONTRACT_DRIFT::{model.parameter_count}")
    print("ARACHNE_KNIGHT_STAGE=MODEL_CONTRACT_PASS", flush=True)
    init_report=_load_init(model,init_path)
    print("ARACHNE_KNIGHT_STAGE=INIT_CHECKPOINT_PASS "+json.dumps(init_report,sort_keys=True), flush=True)
    model=model.to(device=device,dtype=torch.float32)

    preflight={
        "schema":SCHEMA+".Preflight.v1",
        "status":"PASS_PREFLIGHT",
        "prereg_sha256":_sha(prereg_path),
        "surface_lineage_hash":surface.geometry_lineage_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "surface_nodes":EXPECTED_SURFACE_N,
        "surface_edges":EXPECTED_SURFACE_E,
        "joint_count":EXPECTED_JOINTS,
        "architecture_id":ARCHITECTURE_ID,
        "parameter_count":model.parameter_count,
        "training_apparatus_id":TRAINING_APPARATUS_ID,
        "training_apparatus_id":TRAINING_APPARATUS_ID,
        "initialization":init_report,
        "teacher_binding":teacher_binding,
        "teacher_boundary":"OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT",
        "runtime":_runtime(),
        "generalization_claimed":False,
        "product_authority_claimed":False,
    }
    _write_json(outdir/"ARACHNE_KNIGHT_PREFLIGHT.json",preflight)

    started=time.monotonic()
    trace=[]
    zero,pred,qualified=_evaluate(
        model,ci,truth,teacher_valid,surface,skeleton,conditioning,world,joint_world,parent,joint_mask,
        device,0,"KNIGHT_ARACHNE_V5_ZERO_SHOT_FROM_MAGE_FIT2"
    )
    zero["stable_streak"]=0
    trace.append(zero)
    print("ARACHNE_KNIGHT_ZERO_SHOT="+json.dumps(zero,sort_keys=True),flush=True)

    closure_mode=None
    closure_step=None
    stable=0
    final_pred=pred
    final_qualified=qualified
    if zero["full_gate"]:
        closure_mode="ZERO_SHOT_PASS"
        closure_step=0
    else:
        optimizer=torch.optim.AdamW([
            {"params":model.backbone.parameters(),"lr":BACKBONE_LR},
            {"params":model.decoder.parameters(),"lr":DECODER_LR},
        ],weight_decay=WEIGHT_DECAY)
        progress_path=outdir/"ARACHNE_KNIGHT_PROGRESS.pt"
        fingerprint=sha256(json.dumps({
            "apparatus":TRAINING_APPARATUS_ID,
            "prereg_sha256":_sha(prereg_path),
            "teacher_bank_sha256":_sha(bank_path),
            "init_checkpoint_sha256":_sha(init_path),
            "surface_lineage":surface.geometry_lineage_hash,
            "skeleton_lineage":skeleton.skeleton_lineage_hash,
        },sort_keys=True,separators=(",",":")).encode()).hexdigest()
        start_step=0
        if progress_path.is_file():
            prog=torch.load(progress_path,map_location="cpu",weights_only=False)
            if prog.get("fingerprint")!=fingerprint:
                raise RuntimeError("ARACHNE_KNIGHT_PROGRESS_FINGERPRINT_DRIFT")
            model.load_state_dict(prog["model"],strict=True)
            optimizer.load_state_dict(prog["optimizer"])
            start_step=int(prog["step"])
            stable=int(prog.get("stable",0))
            trace=list(prog.get("trace",trace))
            print(f"ARACHNE_KNIGHT_RESUME_FROM_STEP={start_step}",flush=True)

        rng=np.random.default_rng(SEED + start_step*1009)
        for step in range(start_step+1,MAX_STEPS+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                losses=_sample_loss(model,ci,truth,teacher_valid,world,articulated,rng,device)
            losses["total"].backward()
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP))
            if not math.isfinite(grad):
                raise FloatingPointError("ARACHNE_KNIGHT_NONFINITE_GRADIENT")
            optimizer.step()
            optimizer.param_groups[0]["lr"]=BACKBONE_LR*_backbone_lr_factor(step)
            optimizer.param_groups[1]["lr"]=DECODER_LR*_decoder_lr_factor(step)

            if step % 64 == 0:
                print(
                    "ARACHNE_KNIGHT_TRAIN="
                    +json.dumps({
                        "step":step,
                        "loss":float(losses["total"].detach().cpu()),
                        "grad_norm":grad,
                        "lr_backbone":float(optimizer.param_groups[0]["lr"]),
                        "lr_decoder":float(optimizer.param_groups[1]["lr"]),
                    },sort_keys=True),
                    flush=True,
                )

            if step % CHECK_EVERY == 0:
                row,pred,qualified=_evaluate(
                    model,ci,truth,teacher_valid,surface,skeleton,conditioning,world,joint_world,parent,joint_mask,
                    device,step,"KNIGHT_ARACHNE_V5_TRANSFER_FINETUNE"
                )
                stable=stable+1 if row["full_gate"] and step>=MIN_FINETUNE_GATE_STEP else 0
                row["stable_streak"]=stable
                row["grad_norm"]=grad
                row["lr_backbone"]=float(optimizer.param_groups[0]["lr"])
                row["lr_decoder"]=float(optimizer.param_groups[1]["lr"])
                row["losses"]={k:float(v.detach().cpu()) for k,v in losses.items()}
                trace.append(row)
                final_pred=pred
                final_qualified=qualified
                print("ARACHNE_KNIGHT_CHECK="+json.dumps(row,sort_keys=True),flush=True)
                _write_json(outdir/"ARACHNE_KNIGHT_TRACE.json",{"schema":SCHEMA+".Trace.v2","rows":trace})
                if stable>=REQUIRED_STABLE:
                    closure_mode="TRANSFER_FINETUNE_PASS"
                    closure_step=step
                    break

            if step % PROGRESS_EVERY == 0 and step < MAX_STEPS:
                tmp=Path(str(progress_path)+".tmp")
                torch.save({
                    "schema":SCHEMA+".Progress.v2",
                    "fingerprint":fingerprint,
                    "step":step,
                    "stable":stable,
                    "trace":trace,
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                },tmp)
                tmp.replace(progress_path)
                print(f"ARACHNE_KNIGHT_PROGRESS_SAVED={step}",flush=True)

    if closure_mode is None:
        result={
            "schema":SCHEMA+".Result.v1","status":"NO_TERMINAL_CLOSURE",
            "closure_step":None,"required_consecutive_full_passes":REQUIRED_STABLE,
            "trace":trace,"initialization":init_report,"teacher_boundary":"OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT",
            "product_authority_claimed":False,"generalization_claimed":False,
        }
        _write_json(outdir/"ARACHNE_KNIGHT_RESULT.json",result)
        raise AssertionError("ARACHNE_KNIGHT_NO_TERMINAL_CLOSURE")

    # Re-evaluate closure model and mint final Compiler authority.
    final_eval,final_pred,final_qualified=_evaluate(
        model,ci,truth,teacher_valid,surface,skeleton,conditioning,world,joint_world,parent,joint_mask,
        device,int(closure_step),f"KNIGHT_ARACHNE_V5_{closure_mode}"
    )
    if not final_eval["full_gate"] or final_qualified is None:
        raise RuntimeError("ARACHNE_KNIGHT_FINAL_REQUALIFICATION_FAIL")

    w64,canon=_canonicalize_rows(final_pred)
    weights_path=outdir/"ARACHNE_KNIGHT_CANONICAL_SKIN_WEIGHTS_F64.npz"
    np.savez_compressed(
        weights_path,
        weights=w64,
        surface_ids=np.asarray(conditioning.surface_ids[0]),
        canonical_joint_ids=np.asarray(conditioning.joint_ids[0]),
    )
    skin_path=outdir/"ARACHNE_KNIGHT_QUALIFIED_SKIN_IR.json"
    write_ir_json(skin_path,final_qualified)

    checkpoint_path=outdir/"ARACHNE_KNIGHT_V5_MODEL_FINAL_FP32.pt"
    torch.save({
        "schema":SCHEMA+".Model.v1",
        "architecture_id":ARCHITECTURE_ID,
        "model":model.state_dict(),
        "closure_mode":closure_mode,
        "closure_step":int(closure_step),
        "surface_lineage_hash":surface.geometry_lineage_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "teacher_bank_sha256":_sha(bank_path),
        "init_checkpoint_sha256":EXPECTED_INIT_SHA256,
        "historical_checkpoint_loaded":True,
        "teacher_predictor_input_used":False,
        "product_authority_claimed":False,
        "generalization_claimed":False,
    },checkpoint_path)

    result={
        "schema":SCHEMA+".Result.v1",
        "status":"PASS",
        "closure_mode":closure_mode,
        "closure_step":int(closure_step),
        "required_consecutive_full_passes":REQUIRED_STABLE,
        "terminal_check_steps":[int(x["step"]) for x in trace[-REQUIRED_STABLE:]] if closure_step else [0],
        "architecture_id":ARCHITECTURE_ID,
        "parameter_count":model.parameter_count,
        "surface_lineage_hash":surface.geometry_lineage_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "skin_lineage_hash":final_qualified.skin_lineage_hash,
        "qualified_skin_sha256":_sha(skin_path),
        "canonical_weights_sha256":_sha(weights_path),
        "model_sha256":_sha(checkpoint_path),
        "final_evaluation":final_eval,
        "canonicalization":canon,
        "teacher_binding":teacher_binding,
        "teacher_boundary":"OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT",
        "historical_checkpoint_loaded":True,
        "initialization_checkpoint_sha256":EXPECTED_INIT_SHA256,
        "wall_seconds":float(time.monotonic()-started),
        "runtime":_runtime(),
        "product_authority_claimed":False,
        "generalization_claimed":False,
        "mesh_pass_claimed":False,
        "motion_pass_claimed":False,
    }
    _write_json(outdir/"ARACHNE_KNIGHT_TRACE.json",{"schema":SCHEMA+".Trace.v1","rows":trace})
    _write_json(outdir/"ARACHNE_KNIGHT_RESULT.json",result)
    print("ARACHNE_KNIGHT_V5_DEMO_PASS="+json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prereg",required=True)
    ap.add_argument("--surface-json",required=True)
    ap.add_argument("--skeleton-json",required=True)
    ap.add_argument("--teacher-bank",required=True)
    ap.add_argument("--init-checkpoint",required=True)
    ap.add_argument("--output-dir",required=True)
    run(ap.parse_args())


if __name__=="__main__":
    main()
