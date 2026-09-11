from __future__ import annotations

"""Fail-closed Mage FIT1 runner for V7-native Arachne A1.

This runner trains only the product-time predictor. The selected K4 A0 codec is
loaded from the frozen closure checkpoint and every codec parameter remains
frozen. Teacher skin appears only in the target/evaluation bank.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    SkinInfluenceProposal,
    SkinProposalIR,
)
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v3.arachne_candidate_v3 import ArachneA1ConfigV3, ArachneA1V3
from models.arachne.v3.loss_v3 import (
    ArachneA1LossConfigV3,
    arachne_a1_behavior_loss_v3,
    normalize_joint_fields,
)
from models.skin_field_codec.v7.skin_field_codec_v7 import (
    ArachneSkinFieldConfigV7,
    SkinFieldCodecV7,
)


SCHEMA = "RealSaS.ArachneA1V7NativeFIT1.v1"
SEED = 20260911

EXPECTED_A0_MODEL_SHA = "8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7"
EXPECTED_BANK_SHA = "b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a"
EXPECTED_A0_CONFIG_HASH = "e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1"
EXPECTED_A1_CONFIG_HASH = "2ad623648cbe94d5684416f2ddac779dc343661ca9102d526998b37c62865621"
EXPECTED_A1_PARAMETER_COUNT = 54031920
EXPECTED_LOSS_CONFIG_HASH = "c7657fe161eaacfea28e38a1645c744f7ab7bd8c49aceabe1eacfd5f74a78458"

EXPECTED_ZERO_SHA = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
EXPECTED_SKELETON_FILE_SHA = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"
EXPECTED_SURFACE_LINEAGE = "67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"
IRIS_CHECKPOINT_SHA = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
SOURCE_RUN_ID = "20260904T220929Z"
CAMERA_SHA = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)

MAX_STEPS = 8192
MIN_CLOSURE_STEP = 2048
CHECK_EVERY = 256
REQUIRED_STABLE = 3
WARMUP_STEPS = 256
LR = 1.0e-4
LR_FLOOR = 0.1
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
TRAIN_GSA_ROWS = 192
TRAIN_DENSE_ROWS = 192
FIT1_P95_MAX = 0.05
FIT1_DEFORM_MAX = 0.05
EVAL_CHUNK = 1024


def sha_file(path: Path, chunk: int = 8 << 20) -> str:
    h = sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_torch_save(payload, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sha_file(path)


def load_cameras(data_dir: Path) -> tuple[dict, ...]:
    cams = []
    for i, expected in enumerate(CAMERA_SHA):
        p = data_dir / f"V{i}.camera.json"
        if not p.exists() or sha_file(p) != expected:
            raise RuntimeError(f"CAMERA_SHA_DRIFT_V{i}")
        cams.append(json.loads(p.read_text(encoding="utf-8")))
    cams.sort(key=lambda x: int(x["view_index"]))
    if [int(c["view_index"]) for c in cams] != list(range(8)):
        raise RuntimeError("CAMERA_VIEW_INDEX_DRIFT")
    if [int(c["yaw_deg"]) for c in cams] != [0,45,90,135,180,225,270,315]:
        raise RuntimeError("CAMERA_YAW_CONTRACT_DRIFT")
    return tuple(cams)


def build_surface(zero_path: Path, cameras: tuple[dict, ...]):
    if sha_file(zero_path) != EXPECTED_ZERO_SHA:
        raise RuntimeError("ZERO_SURFACE_SHA_DRIFT")
    with np.load(zero_path, allow_pickle=False) as z:
        world = np.asarray(z["vertices"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
        hints = np.asarray(z["normals"], np.float64)
    center = np.asarray(cameras[0]["center"], np.float64)
    half = float(cameras[0]["half_extent"])
    surface = rigging_surface_from_scene_first_zero_mesh_v1(
        (world - center[None, :]) / half,
        faces,
        hints,
        cameras,
        normalization_center=center,
        normalization_half_extent=half,
        authority_label="IRIS_SCENE_FIRST_SIGNED_V3_PROMOTED_MAGE_FIT",
        source_run_id=SOURCE_RUN_ID,
        source_checkpoint_sha256=IRIS_CHECKPOINT_SHA,
        source_zero_surface_sha256=EXPECTED_ZERO_SHA,
        target_nodes=1024,
        normal_k=64,
        visibility_depth_tolerance_norm=0.02,
        metadata={
            "camera_contract": "CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG",
            "teacher_truth_used": False,
            "geppetto_reference_strength_fit1": True,
        },
    )
    if surface.geometry_lineage_hash != EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError(f"SURFACE_LINEAGE_DRIFT::{surface.geometry_lineage_hash}")
    if len(surface.surface_nodes) != 950 or len(surface.local_relations) != 2813:
        raise RuntimeError("SURFACE_WITNESS_CARDINALITY_DRIFT")
    return surface


def load_skeleton(path: Path) -> QualifiedSkeletonIRV2:
    if sha_file(path) != EXPECTED_SKELETON_FILE_SHA:
        raise RuntimeError("QUALIFIED_SKELETON_FILE_SHA_DRIFT")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("skeleton_lineage_hash") != EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("SKELETON_LINEAGE_DRIFT")
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(x["canonical_joint_id"]),
            position=tuple(map(float, x["position"])),
            parent_canonical_id=None if x.get("parent_canonical_id") is None else str(x["parent_canonical_id"]),
            support_surface_ids=tuple(map(str, x.get("support_surface_ids", ()))),
            source_proposal_id=str(x.get("source_proposal_id", "")),
        )
        for x in d["joints"]
    )
    if len(joints) != 22:
        raise RuntimeError("QUALIFIED_JOINT_COUNT_DRIFT")
    deform = tuple(map(str, d.get("deform_root_ids", ())))
    if not deform:
        deform = tuple(j.canonical_joint_id for j in joints if j.parent_canonical_id is None)
    return QualifiedSkeletonIRV2(
        joints=joints,
        deform_root_ids=deform,
        assembly_root_binding=dict(d.get("assembly_root_binding", {})),
        qualification_report=dict(d.get("qualification_report", {})),
        skeleton_lineage_hash=str(d["skeleton_lineage_hash"]),
    )


def load_bank(path: Path) -> dict[str, np.ndarray]:
    if sha_file(path) != EXPECTED_BANK_SHA:
        raise RuntimeError("A1_SUPERVISION_BANK_SHA_DRIFT")
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    required = {
        "surface_ids", "joint_ids",
        "gsa_geometry7", "gsa_world", "gsa_teacher_weights", "gsa_supervision_mask",
        "dense8k_geometry7", "dense8k_world", "dense8k_teacher_weights",
        "holdout_geometry7", "holdout_world", "holdout_teacher_weights",
        "condition_query_indices",
    }
    if not required.issubset(d):
        raise RuntimeError(f"A1_BANK_SCHEMA_MISSING::{sorted(required-set(d))}")
    if d["gsa_geometry7"].shape != (950,7) or d["gsa_teacher_weights"].shape != (950,22):
        raise RuntimeError("A1_BANK_GSA_SHAPE_DRIFT")
    if d["dense8k_geometry7"].shape[1:] != (7,) or d["dense8k_teacher_weights"].shape[1:] != (22,):
        raise RuntimeError("A1_BANK_DENSE_SHAPE_DRIFT")
    if d["holdout_geometry7"].shape[1:] != (7,) or d["holdout_teacher_weights"].shape[1:] != (22,):
        raise RuntimeError("A1_BANK_HOLDOUT_SHAPE_DRIFT")
    if d["condition_query_indices"].shape != (384,):
        raise RuntimeError("A1_BANK_CONDITION_QUERY_DRIFT")
    return d


def load_frozen_codec(path: Path, device: torch.device) -> tuple[SkinFieldCodecV7, dict]:
    if sha_file(path) != EXPECTED_A0_MODEL_SHA:
        raise RuntimeError("A0_K4_MODEL_SHA_DRIFT")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("token_count", -1)) != 4:
        raise RuntimeError("A0_K4_TOKEN_COUNT_DRIFT")
    if payload.get("config_hash") != EXPECTED_A0_CONFIG_HASH:
        raise RuntimeError("A0_K4_CONFIG_HASH_DRIFT")
    cfg = ArachneSkinFieldConfigV7()
    if cfg.config_hash != EXPECTED_A0_CONFIG_HASH:
        raise RuntimeError("LOCAL_V7_CONFIG_HASH_DRIFT")
    codec = SkinFieldCodecV7(cfg)
    codec.load_state_dict(payload["model"], strict=True)
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    codec = codec.to(device=device, dtype=torch.float32)
    if any(p.requires_grad for p in codec.parameters()):
        raise RuntimeError("FROZEN_CODEC_HAS_TRAINABLE_PARAMETER")
    return codec, payload


def conditioning_to_torch(c, device):
    names = (
        "surface_positions_normalized", "surface_normals", "surface_normal_valid",
        "surface_support_views", "surface_raster_xy", "surface_raster_valid",
        "surface_observed", "surface_completed", "surface_mask",
        "edge_index", "edge_features", "edge_mask",
        "joint_positions_normalized", "joint_mask", "parent_indices",
        "root_mask", "deform_root_mask", "joint_depth_normalized",
        "support_anchor_matrix", "pair_geometry", "pair_mask",
    )
    bool_names = {
        "surface_normal_valid", "surface_support_views", "surface_raster_valid",
        "surface_observed", "surface_completed", "surface_mask", "edge_mask",
        "joint_mask", "root_mask", "deform_root_mask", "support_anchor_matrix", "pair_mask",
    }
    int_names = {"edge_index", "parent_indices"}
    out = {}
    for name in names:
        a = getattr(c, name)
        dtype = torch.bool if name in bool_names else torch.long if name in int_names else torch.float32
        out[name] = torch.as_tensor(a, device=device, dtype=dtype)
    return out


def decode_logits(codec, prepared_condition, field_tokens, query_geometry, *, chunk=EVAL_CHUNK):
    if field_tokens.ndim != 4 or field_tokens.shape[0] != 1:
        raise ValueError("field token shape drift")
    q = torch.as_tensor(query_geometry, device=field_tokens.device, dtype=torch.float32)
    pieces = []
    for start in range(0, len(q), int(chunk)):
        qg = q[start:start+chunk][None]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            qemb = codec.geometry_embedding(qg).detach()
        per_joint = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            for j in range(field_tokens.shape[1]):
                per_joint.append(codec.decoder.query_logits(prepared_condition, field_tokens[:, j], qemb))
        pieces.append(torch.stack(per_joint, dim=-1).float())
    return torch.cat(pieces, dim=1)


def lr_factor(step: int) -> float:
    s = max(1, int(step))
    if s <= WARMUP_STEPS:
        return float(s) / float(WARMUP_STEPS)
    q = (s - WARMUP_STEPS) / float(MAX_STEPS - WARMUP_STEPS)
    q = min(max(q, 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * q))


def probe_transforms(j: int, device):
    poses = 4
    t = torch.eye(4, dtype=torch.float32, device=device)[None, None].repeat(1, poses, j, 1, 1)
    for ji in range(j):
        u = float(ji + 1) / float(j)
        t[0, 1, ji, 0, 3] = 0.10 * u
        t[0, 1, ji, 1, 3] = 0.035 * (-1.0 if ji % 2 else 1.0)
        t[0, 2, ji, 1, 3] = 0.085 * u
        t[0, 2, ji, 2, 3] = 0.030 * (ji - (j - 1) / 2.0)
        t[0, 3, ji, 0, 3] = -0.055 * (ji - (j - 1) / 2.0)
        t[0, 3, ji, 2, 3] = 0.070 * u
    return t


def verified_lbs(rest_points, weights, transforms):
    ones = torch.ones((*rest_points.shape[:2], 1), dtype=rest_points.dtype, device=rest_points.device)
    hom = torch.cat([rest_points, ones], -1)
    transformed = torch.einsum("bpjac,bnc->bpjna", transforms, hom)[..., :3]
    return torch.einsum("bnj,bpjna->bpna", weights, transformed)


def deformation_ratio(rest, wtruth, wpred, transforms, mask):
    td = verified_lbs(rest, wtruth, transforms)
    pd = verified_lbs(rest, wpred, transforms)
    rp = rest[:, None].expand_as(td)
    m = mask[:, None, :, None].to(td.dtype)
    denom = (m.sum() * td.shape[1] * td.shape[-1]).clamp_min(1.0)
    motion = torch.sqrt((((td-rp)*m).square().sum()/denom).clamp_min(1e-12))
    err = torch.sqrt((((pd-td)*m).square().sum()/denom).clamp_min(1e-12))
    return float((err/motion.clamp_min(1e-6)).cpu()), float(motion.cpu()), float(err.cpu())


def metrics(pred, truth, world, device, mask=None):
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    mask = np.ones(len(pred), bool) if mask is None else np.asarray(mask, bool)
    row = np.abs(pred-truth).sum(1)
    dom = float((pred[mask].argmax(1) == truth[mask].argmax(1)).mean())
    top3 = np.argpartition(-pred[mask], kth=2, axis=1)[:, :3]
    td = truth[mask].argmax(1)
    top3inc = float(np.mean([td[i] in top3[i] for i in range(len(td))]))
    rest = torch.as_tensor(np.asarray(world,np.float32)[None], device=device)
    wt = torch.as_tensor(truth.astype(np.float32)[None], device=device)
    wp = torch.as_tensor(pred.astype(np.float32)[None], device=device)
    mt = torch.as_tensor(mask[None], device=device, dtype=torch.bool)
    deform, motion, err = deformation_ratio(rest, wt, wp, probe_transforms(pred.shape[1], device), mt)
    return {
        "rows_total": int(len(pred)),
        "rows_evaluated": int(mask.sum()),
        "row_l1_mean": float(row[mask].mean()),
        "row_l1_p50": float(np.quantile(row[mask], .50)),
        "row_l1_p90": float(np.quantile(row[mask], .90)),
        "row_l1_p95": float(np.quantile(row[mask], .95)),
        "row_l1_p99": float(np.quantile(row[mask], .99)),
        "cvar10": float(row[mask][row[mask] >= np.quantile(row[mask], .90)].mean()),
        "deformation_error_ratio": deform,
        "deformation_error_rms": err,
        "teacher_motion_rms": motion,
        "dominant_accuracy": dom,
        "teacher_dominant_top3_inclusion": top3inc,
        "simplex_max_abs_residual": float(np.max(np.abs(pred.sum(1)-1.0))),
    }


def make_skin_proposal(pred, conditioning, model_provenance: str):
    p = np.asarray(pred, np.float64)
    sids = conditioning.surface_ids[0]
    jids = conditioning.joint_ids[0]
    if p.shape != (len(sids), len(jids)):
        raise ValueError("skin proposal shape drift")
    inf = []
    for si, sid in enumerate(sids):
        for ji, jid in enumerate(jids):
            inf.append(SkinInfluenceProposal(str(sid), str(jid), float(p[si, ji])))
    return SkinProposalIR(
        tuple(inf),
        conditioning.source_surface_hashes[0],
        conditioning.source_skeleton_hashes[0],
        model_provenance=model_provenance,
        metadata={
            "candidate_architecture": "RealSaS.Arachne.A1.RichQualifiedSurfaceSkeleton.v3",
            "field_tokens": 4,
            "dense_proposal": True,
            "compiler_owns_qualification": True,
            "teacher_input_used": False,
        },
    )


def preflight_equivariance(model, ci, *, tol=2e-4):
    model.eval()
    with torch.no_grad():
        base = model(**ci).field_tokens.float()
        N = ci["surface_mask"].shape[1]
        p = torch.arange(N-1, -1, -1, device=base.device)
        inv = torch.empty_like(p); inv[p] = torch.arange(N, device=base.device)
        cs = dict(ci)
        for name in (
            "surface_positions_normalized","surface_normals","surface_normal_valid",
            "surface_support_views","surface_raster_xy","surface_raster_valid",
            "surface_observed","surface_completed","surface_mask",
        ):
            cs[name] = ci[name][:, p]
        cs["edge_index"] = inv[ci["edge_index"]]
        cs["support_anchor_matrix"] = ci["support_anchor_matrix"][:, :, p]
        cs["pair_geometry"] = ci["pair_geometry"][:, p]
        cs["pair_mask"] = ci["pair_mask"][:, p]
        s_alt = model(**cs).field_tokens.float()
        s_delta = float((base-s_alt).abs().max().cpu())

        J = ci["joint_mask"].shape[1]
        q = torch.arange(J-1, -1, -1, device=base.device)
        qinv = torch.empty_like(q); qinv[q] = torch.arange(J, device=base.device)
        cj = dict(ci)
        for name in (
            "joint_positions_normalized","joint_mask","root_mask",
            "deform_root_mask","joint_depth_normalized",
        ):
            cj[name] = ci[name][:, q]
        old_parent = ci["parent_indices"][:, q]
        new_parent = torch.where(old_parent >= 0, qinv[old_parent.clamp_min(0)], old_parent)
        cj["parent_indices"] = new_parent
        cj["support_anchor_matrix"] = ci["support_anchor_matrix"][:, q]
        cj["pair_geometry"] = ci["pair_geometry"][:, :, q]
        cj["pair_mask"] = ci["pair_mask"][:, :, q]
        j_alt = model(**cj).field_tokens.float()
        j_delta = float((base[:, q] - j_alt).abs().max().cpu())

    if s_delta > tol:
        raise RuntimeError(f"SURFACE_PERMUTATION_EQUIVARIANCE_FAIL::{s_delta}")
    if j_delta > tol:
        raise RuntimeError(f"JOINT_PERMUTATION_EQUIVARIANCE_FAIL::{j_delta}")
    return {"tolerance": float(tol), "surface_max_abs": s_delta, "joint_max_abs": j_delta}


def run(args) -> dict:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name.upper():
        raise RuntimeError(f"A100_REQUIRED__FOUND={gpu_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16_REQUIRED")

    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("schema") != SCHEMA + ".Preregistration.v1" or prereg.get("status") != "AUTHORIZED":
        raise RuntimeError("A1_PREREG_NOT_AUTHORIZED")
    prereg_sha = sha_file(prereg_path)

    data_dir = Path(args.data_dir)
    cameras = load_cameras(data_dir)
    surface = build_surface(Path(args.zero_surface), cameras)
    skeleton = load_skeleton(Path(args.skeleton))
    bank = load_bank(Path(args.bank))
    codec, _a0_payload = load_frozen_codec(Path(args.a0_model), device)

    adapter = ArachneRichConditioningAdapterV3(require_scene_first=True)
    conditioning = adapter([surface], [skeleton])
    if conditioning.source_surface_hashes != (EXPECTED_SURFACE_LINEAGE,):
        raise RuntimeError("A1_CONDITIONING_SURFACE_LINEAGE_DRIFT")
    if conditioning.source_skeleton_hashes != (EXPECTED_SKELETON_LINEAGE,):
        raise RuntimeError("A1_CONDITIONING_SKELETON_LINEAGE_DRIFT")
    if conditioning.geometry7.shape != (1,950,7):
        raise RuntimeError("A1_RICH_GEOMETRY_SHAPE_DRIFT")
    geom_delta = float(np.max(np.abs(conditioning.geometry7[0] - bank["gsa_geometry7"].astype(np.float32))))
    if geom_delta > 5e-5:
        raise RuntimeError(f"A1_CODEC_FRAME_PARITY_FAIL::{geom_delta}")
    if tuple(map(str, bank["surface_ids"])) != tuple(conditioning.surface_ids[0]):
        raise RuntimeError("A1_SURFACE_ID_ORDER_DRIFT")
    if tuple(map(str, bank["joint_ids"])) != tuple(conditioning.joint_ids[0]):
        raise RuntimeError("A1_JOINT_ID_ORDER_DRIFT")
    if not np.array_equal(bank["condition_query_indices"].astype(np.int64), conditioning.condition_query_indices[0]):
        raise RuntimeError("A1_CONDITION_QUERY_INDEX_DRIFT")

    a1_cfg = ArachneA1ConfigV3()
    loss_cfg = ArachneA1LossConfigV3()
    if a1_cfg.config_hash != EXPECTED_A1_CONFIG_HASH:
        raise RuntimeError("A1_CONFIG_HASH_DRIFT")
    if loss_cfg.config_hash != EXPECTED_LOSS_CONFIG_HASH:
        raise RuntimeError("A1_LOSS_CONFIG_HASH_DRIFT")
    model = ArachneA1V3(a1_cfg).to(device=device, dtype=torch.float32)
    if model.parameter_count != EXPECTED_A1_PARAMETER_COUNT:
        raise RuntimeError(f"A1_PARAMETER_COUNT_DRIFT::{model.parameter_count}")

    ci = conditioning_to_torch(conditioning, device)
    eq = preflight_equivariance(model, ci)

    ggeom = torch.as_tensor(bank["gsa_geometry7"][None], device=device, dtype=torch.float32)
    qidx = torch.as_tensor(bank["condition_query_indices"][None], device=device, dtype=torch.long)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        condition_tokens = codec.encode_condition(ggeom, qidx).detach()
        prepared_condition = codec.decoder.prepare_condition(condition_tokens).detach()
    condition_tokens = condition_tokens.clone()
    prepared_condition = prepared_condition.clone()

    # Optimizer creation is deliberately after all authority/data/equivariance gates.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    gsa_geom = np.asarray(bank["gsa_geometry7"], np.float32)
    gsa_truth = np.asarray(bank["gsa_teacher_weights"], np.float32)
    gsa_world = np.asarray(bank["gsa_world"], np.float32)
    gsa_mask = np.asarray(bank["gsa_supervision_mask"], bool)
    dense_geom = np.asarray(bank["dense8k_geometry7"], np.float32)
    dense_truth = np.asarray(bank["dense8k_teacher_weights"], np.float32)
    hold_geom = np.asarray(bank["holdout_geometry7"], np.float32)
    hold_truth = np.asarray(bank["holdout_teacher_weights"], np.float32)
    hold_world = np.asarray(bank["holdout_world"], np.float32)
    gsa_pool = np.where(gsa_mask)[0].astype(np.int64)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "ARACHNE_A1_V7_NATIVE_FIT1_TRACE.json"
    model_path = out_dir / "ARACHNE_A1_V7_NATIVE_FIT1_MODEL_ONLY_FP32.pt"
    result_path = out_dir / "ARACHNE_A1_V7_NATIVE_FIT1_RESULT.json"
    trace = []
    best_score = None
    best_sha = None
    stable = 0
    closure_step = None

    def predict_all(geometry):
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            tok = model(**ci).field_tokens
            logits = decode_logits(codec, prepared_condition, tok, geometry)
            pred = normalize_joint_fields(torch.sigmoid(logits), ci["joint_mask"]).squeeze(0)
        return pred.float().cpu().numpy()

    def evaluate(step):
        gp = predict_all(gsa_geom)
        hp = predict_all(hold_geom)
        gm = metrics(gp, gsa_truth, gsa_world, device, gsa_mask)
        hm = metrics(hp, hold_truth, hold_world, device)
        proposal = make_skin_proposal(gp, conditioning, a1_cfg.config_hash)
        qskin = qualify_skin(surface, skeleton, proposal)
        compiler = {
            "status": "PASS",
            "row_count": len(qskin.rows),
            "skin_lineage_hash": qskin.skin_lineage_hash,
            "qualification_report": qskin.qualification_report,
        }
        passed = (
            gm["row_l1_p95"] <= FIT1_P95_MAX
            and gm["deformation_error_ratio"] <= FIT1_DEFORM_MAX
            and int(compiler["row_count"]) == 950
        )
        return gp, {
            "step": int(step),
            "gsa950": gm,
            "disjoint_holdout": hm,
            "compiler_skin": compiler,
            "fit1_gate": bool(passed),
        }

    _, row0 = evaluate(0)
    row0["stable_streak"] = 0
    trace.append(row0)
    write_json(trace_path, {"schema":SCHEMA+".Trace.v1","prereg_sha256":prereg_sha,"rows":trace})
    print("A1_CHECK=" + json.dumps(row0, sort_keys=True), flush=True)

    for step in range(1, MAX_STEPS + 1):
        model.train()
        rng = np.random.default_rng(SEED + step * 1009)
        gi = np.asarray(rng.choice(gsa_pool, size=TRAIN_GSA_ROWS, replace=False), np.int64)
        di = np.asarray(rng.choice(len(dense_geom), size=TRAIN_DENSE_ROWS, replace=False), np.int64)
        qgeom_np = np.concatenate([gsa_geom[gi], dense_geom[di]], axis=0)
        truth_np = np.concatenate([gsa_truth[gi], dense_truth[di]], axis=0)
        perm = rng.permutation(len(qgeom_np))
        qgeom_np = qgeom_np[perm]
        truth_np = truth_np[perm]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            raw = model(**ci)
            logits = decode_logits(codec, prepared_condition, raw.field_tokens, qgeom_np, chunk=len(qgeom_np))
        truth = torch.as_tensor(truth_np[None], device=device, dtype=torch.float32)
        row_mask = torch.ones((1,len(qgeom_np)), device=device, dtype=torch.bool)
        losses = arachne_a1_behavior_loss_v3(logits, truth, row_mask, joint_mask=ci["joint_mask"], config=loss_cfg)
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("NONFINITE_A1_GRADIENT")
        optimizer.step()
        scheduler.step()

        if step % CHECK_EVERY:
            continue

        _, erow = evaluate(step)
        erow["losses"] = {k: float(v.detach().cpu()) for k,v in losses.items()}
        erow["grad_norm"] = grad_norm
        erow["lr"] = float(optimizer.param_groups[0]["lr"])
        stable = stable + 1 if erow["fit1_gate"] and step >= MIN_CLOSURE_STEP else 0
        erow["stable_streak"] = stable
        trace.append(erow)
        print("A1_CHECK=" + json.dumps(erow, sort_keys=True), flush=True)

        _p95 = float(erow["gsa950"]["row_l1_p95"])
        _def = float(erow["gsa950"]["deformation_error_ratio"])
        score = (max(_p95, _def), _p95, _def)
        if best_score is None or score < best_score:
            payload = {
                "schema": SCHEMA + ".ModelOnly.v1",
                "prereg_sha256": prereg_sha,
                "a0_model_sha256": EXPECTED_A0_MODEL_SHA,
                "a1_supervision_bank_sha256": EXPECTED_BANK_SHA,
                "a1_config_hash": a1_cfg.config_hash,
                "loss_config_hash": loss_cfg.config_hash,
                "parameter_count": model.parameter_count,
                "step": int(step),
                "score": list(score),
                "model": model.state_dict(),
                "product_evaluation_performed": False,
            }
            best_sha = _atomic_torch_save(payload, model_path)
            best_score = score

        write_json(trace_path, {"schema":SCHEMA+".Trace.v1","prereg_sha256":prereg_sha,"rows":trace})
        if stable >= REQUIRED_STABLE:
            closure_step = step
            break

    if not model_path.exists() or best_sha is None:
        raise RuntimeError("A1_MODEL_CHECKPOINT_MISSING")

    try:
        best_payload = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        best_payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    _, final = evaluate(int(best_payload["step"]))

    status = "PASS" if closure_step is not None else "FAIL"
    if status == "PASS" and not bool(final["fit1_gate"]):
        raise RuntimeError("SAVED_BEST_CHECKPOINT_NOT_FIT1_PASS")
    result = {
        "schema": SCHEMA + ".Result.v1",
        "status": status,
        "prereg_sha256": prereg_sha,
        "seed": SEED,
        "a0": {
            "model_sha256": EXPECTED_A0_MODEL_SHA,
            "config_hash": EXPECTED_A0_CONFIG_HASH,
            "supervision_bank_sha256": EXPECTED_BANK_SHA,
            "field_tokens": 4,
            "latent_channels": 512,
            "ordered_latent_alignment_used": False,
        },
        "rich_conditioning": {
            "surface_lineage_hash": conditioning.source_surface_hashes[0],
            "skeleton_lineage_hash": conditioning.source_skeleton_hashes[0],
            "conditioning_hash": conditioning.conditioning_hashes[0],
            "surface_nodes": int(conditioning.surface_mask[0].sum()),
            "gsa_edges": int(conditioning.edge_mask[0].sum()),
            "joints": int(conditioning.joint_mask[0].sum()),
            "surface_codec_frame_max_abs_delta": geom_delta,
        },
        "a1": {
            "architecture_id": a1_cfg.architecture_id,
            "config_hash": a1_cfg.config_hash,
            "parameter_count": model.parameter_count,
            "loss_config_hash": loss_cfg.config_hash,
            "model_sha256": sha_file(model_path),
            "best_step": int(best_payload["step"]),
            "closure_step": closure_step,
            "equivariance_preflight": eq,
        },
        "final_best_checkpoint_evaluation": final,
        "fit1_gate": {
            "row_l1_p95_max": FIT1_P95_MAX,
            "deformation_error_ratio_max": FIT1_DEFORM_MAX,
            "stable_observations": REQUIRED_STABLE,
            "minimum_closure_step": MIN_CLOSURE_STEP,
        },
        "holdout_role": "DIAGNOSTIC_ONLY__SAME_CHARACTER_DISJOINT_SURFACE__NOT_UNSEEN",
        "teacher_boundary": "OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT",
        "deformation_probe_role": "A0_CONTINUITY_DIAGNOSTIC_ONLY__NOT_TRAINING_LOSS",
        "compiler_skin_qualification_required": True,
        "product_evaluation_performed": False,
        "product_pass_claimed": False,
        "unseen_generalization_claimed": False,
        "trace_path": str(trace_path),
        "model_path": str(model_path),
    }
    write_json(result_path, result)
    print("A1_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if status != "PASS":
        raise AssertionError("A1_FIT1_DID_NOT_CLOSE")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--a0-model", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
