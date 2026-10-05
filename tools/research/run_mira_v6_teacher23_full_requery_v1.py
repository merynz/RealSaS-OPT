from __future__ import annotations

"""Frozen MIRA v6 full re-query on the fixed raw-teacher23 deform rig.

No training occurs. Predictor inputs are only shipping-legal GSA + the fixed
Compiler-qualified teacher23 test rig. Teacher skin is used to validate the test
rig membership before inference and is opened as a continuous surface field only
after predictions have been persisted, for evaluation metrics.

This is a research substitution court, not product/generalization authority.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.rig import qualify_skeleton
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4
from tools.research.materialize_teacher23_mira_query_preflight_v1 import (
    EXPECTED_BACKBONE_ARCH,
    EXPECTED_BACKBONE_PARAMS,
    EXPECTED_DECODER_PARAMS,
    EXPECTED_MIRA_ARCH,
    EXPECTED_MIRA_MODEL_SHA256,
    EXPECTED_MIRA_SCHEMA,
    EXPECTED_RUNNER_SHA256,
    EXPECTED_SURFACE_LINEAGE,
    build_teacher23_proposal,
)
from tools.research.materialize_teacher23_mira_query_preflight_v2 import (
    extract_exact_v6_symbol_closure,
)
from tools.research.run_tessa_knight_continuous_skin_motion_witness_v1 import (
    continuous_skin_field,
)


SCHEMA = "RealSaS.MIRAV6Teacher23FullRequery.v1"
EXPECTED_JOINTS = 23
EXPECTED_SURFACE_N = 12090
EXPECTED_SURFACE_E = 36469


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def phase(name: str, **payload) -> None:
    print("MIRA23_REQUERY_PHASE=" + json.dumps({"phase": name, **payload}, sort_keys=True), flush=True)


def conditioning_to_torch(conditioning, device: torch.device) -> dict[str, torch.Tensor]:
    float_names = (
        "surface_positions_normalized", "surface_normals", "surface_raster_xy",
        "edge_features", "joint_positions_normalized", "joint_depth_normalized",
        "pair_geometry", "view_yaw_code",
    )
    bool_names = (
        "surface_normal_valid", "surface_support_views", "surface_raster_valid",
        "surface_observed", "surface_completed", "surface_mask", "edge_mask",
        "joint_mask", "root_mask", "deform_root_mask", "support_anchor_matrix",
        "pair_mask",
    )
    long_names = ("edge_index", "parent_indices")
    out: dict[str, torch.Tensor] = {}
    for name in float_names:
        out[name if name != "view_yaw_code" else "view_yaw_fourier"] = torch.as_tensor(
            getattr(conditioning, name), device=device, dtype=torch.float32
        )
    for name in bool_names:
        out[name] = torch.as_tensor(getattr(conditioning, name), device=device, dtype=torch.bool)
    for name in long_names:
        out[name] = torch.as_tensor(getattr(conditioning, name), device=device, dtype=torch.long)
    return out


def runtime() -> dict:
    info = {
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info["gpu0"] = {
            "name": str(p.name),
            "compute_capability": [int(p.major), int(p.minor)],
            "total_bytes": int(total),
            "free_bytes": int(free),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--teacher-npz", type=Path, required=True)
    ap.add_argument("--mira-checkpoint", type=Path, required=True)
    ap.add_argument("--exact-v6-runner", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--decode-chunk", type=int, default=512)
    args = ap.parse_args(argv)

    surface_path = args.surface_json.resolve()
    teacher_path = args.teacher_npz.resolve()
    checkpoint_path = args.mira_checkpoint.resolve()
    runner_path = args.exact_v6_runner.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    phase("VERIFY_AUTHORITIES")
    model_sha = sha256(checkpoint_path)
    runner_sha = sha256(runner_path)
    if model_sha != EXPECTED_MIRA_MODEL_SHA256:
        raise RuntimeError(f"MIRA_MODEL_SHA_DRIFT::{model_sha}")
    if runner_sha != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(f"MIRA_RUNNER_SHA_DRIFT::{runner_sha}")
    surface = rigging_surface_from_dict(load_json(surface_path))
    if str(surface.geometry_lineage_hash) != EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("MIRA23_REQUERY_SURFACE_LINEAGE_DRIFT")
    if len(surface.surface_nodes) != EXPECTED_SURFACE_N or len(surface.local_relations) != EXPECTED_SURFACE_E:
        raise RuntimeError("MIRA23_REQUERY_SURFACE_CARDINALITY_DRIFT")

    phase("QUALIFY_FIXED_TEACHER23_TEST_RIG")
    proposal, binding = build_teacher23_proposal(surface, teacher_path)
    skeleton = qualify_skeleton(
        surface,
        proposal,
        run_ilp_shadow=False,
        authority_bindings={
            "court_scope": "MIRA_ONLY_FIXED_TEACHER23_RIG",
            "support_policy": "GSA_NEAREST8_TO_RAW_TEACHER_BONE_HEAD_V1",
        },
    )
    if len(skeleton.joints) != EXPECTED_JOINTS:
        raise RuntimeError("MIRA23_REQUERY_QUALIFIED_JOINT_COUNT_DRIFT")

    phase("BUILD_SHIPPING_LEGAL_CONDITIONING")
    conditioning = ArachneRichConditioningAdapterV3(require_scene_first=True)([surface], [skeleton])
    n = int(conditioning.surface_mask[0].sum())
    e = int(conditioning.edge_mask[0].sum())
    j = int(conditioning.joint_mask[0].sum())
    if (n, e, j) != (EXPECTED_SURFACE_N, EXPECTED_SURFACE_E, EXPECTED_JOINTS):
        raise RuntimeError(f"MIRA23_REQUERY_CONDITIONING_CARDINALITY_DRIFT::{n}::{e}::{j}")

    if not torch.cuda.is_available():
        raise RuntimeError("MIRA23_REQUERY_CUDA_REQUIRED")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("MIRA23_REQUERY_BF16_REQUIRED")
    free, _ = torch.cuda.mem_get_info()
    if free < 24 * 1024**3:
        raise RuntimeError(f"MIRA23_REQUERY_REQUIRES_24GIB_FREE::{runtime()}")
    device = torch.device("cuda")

    phase("STRICT_LOAD_FROZEN_V6")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != EXPECTED_MIRA_SCHEMA or ckpt.get("architecture_id") != EXPECTED_MIRA_ARCH:
        raise RuntimeError("MIRA23_REQUERY_CHECKPOINT_AUTHORITY_DRIFT")
    if ckpt.get("backbone_architecture_id") != EXPECTED_BACKBONE_ARCH:
        raise RuntimeError("MIRA23_REQUERY_BACKBONE_ARCH_DRIFT")
    if ckpt.get("teacher_predictor_input_used") is not False:
        raise RuntimeError("MIRA23_REQUERY_TEACHER_INPUT_BOUNDARY_DRIFT")

    backbone = ArachneA1V4(ArachneA1ConfigV4()).cpu().float()
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    if int(sum(p.numel() for p in backbone.parameters())) != EXPECTED_BACKBONE_PARAMS:
        raise RuntimeError("MIRA23_REQUERY_BACKBONE_PARAM_DRIFT")

    exact = extract_exact_v6_symbol_closure(runner_path)
    decoder = exact.ArachneV6RawReadout(
        surface_dim=int(backbone.config.model_dim),
        token_dim=int(backbone.config.latent_channels),
        pair_dim=int(backbone.config.pair_geometry_dim),
        geom_dim=7,
        relation_dim=int(exact.RELATION_DIM),
    ).cpu().float()
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    if int(sum(p.numel() for p in decoder.parameters())) != EXPECTED_DECODER_PARAMS:
        raise RuntimeError("MIRA23_REQUERY_DECODER_PARAM_DRIFT")

    phase("MOVE_FROZEN_MODELS_TO_A100", runtime=runtime())
    backbone = backbone.to(device=device, dtype=torch.float32).eval()
    decoder = decoder.to(device=device, dtype=torch.float32).eval()
    ci = conditioning_to_torch(conditioning, device)

    torch.manual_seed(20260926)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    phase("BACKBONE_FORWARD_BEGIN", surface_nodes=n, joints=j)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        raw = backbone(**ci)
    surface_memory = raw.surface_memory[0].float()
    field_tokens = raw.field_tokens[0].float()
    phase(
        "BACKBONE_FORWARD_END",
        seconds=time.time() - t0,
        surface_memory_shape=list(surface_memory.shape),
        field_tokens_shape=list(field_tokens.shape),
        cuda_peak_gib=torch.cuda.max_memory_allocated() / (1024**3),
    )

    geom = torch.as_tensor(conditioning.geometry7[0], device=device, dtype=torch.float32)
    pair_geometry = ci["pair_geometry"][0].float()
    legal = (
        ci["pair_mask"].bool()
        & ci["surface_mask"][:, :, None].bool()
        & ci["joint_mask"][:, None, :].bool()
    )[0]

    phase("V6_READOUT_BEGIN", decode_chunk=int(args.decode_chunk))
    with torch.no_grad():
        logits, pred = decoder.decode_all(
            surface_memory,
            geom,
            pair_geometry,
            field_tokens,
            legal,
            chunk=int(args.decode_chunk),
        )
    pred = pred.float()
    if tuple(pred.shape) != (EXPECTED_SURFACE_N, EXPECTED_JOINTS):
        raise RuntimeError(f"MIRA23_REQUERY_OUTPUT_SHAPE_DRIFT::{pred.shape}")
    if not torch.isfinite(pred).all() or bool((pred < 0).any()):
        raise RuntimeError("MIRA23_REQUERY_INVALID_WEIGHT_OUTPUT")
    simplex = (pred.sum(-1) - 1.0).abs()
    simplex_max = float(simplex.max().item())
    if simplex_max > 1e-6:
        raise RuntimeError(f"MIRA23_REQUERY_SIMPLEX_FAIL::{simplex_max}")

    weights = pred.detach().cpu().numpy().astype(np.float64)
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    prediction_path = out / "MIRA_V6_TEACHER23_GSA_WEIGHTS_F64.npz"
    np.savez_compressed(
        prediction_path,
        weights=weights,
        logits=logits_np,
        surface_ids=np.asarray(conditioning.surface_ids[0]),
        canonical_joint_ids=np.asarray(conditioning.joint_ids[0]),
    )
    prediction_sha = sha256(prediction_path)
    phase("PREDICTION_SEALED_BEFORE_TEACHER_FIELD_EVAL", sha256=prediction_sha, simplex_max=simplex_max)

    # Teacher skin is opened only after frozen predictions are persisted. This is
    # evaluation-only and cannot influence predictor features or weights.
    phase("CONTINUOUS_TEACHER_FIELD_EVAL_BEGIN")
    with np.load(teacher_path, allow_pickle=False) as z:
        source_vertices = np.asarray(z["vertices_source"], dtype=np.float64)
        source_faces = np.asarray(z["faces"], dtype=np.int64)
        source_skin = np.asarray(z["skin"], dtype=np.float64)[:, :EXPECTED_JOINTS]
    gsa_world = np.asarray(conditioning.surface_positions_world[0, :n], dtype=np.float64)
    teacher_field, field_distance, field_face, field_bary, _ = continuous_skin_field(
        gsa_world,
        source_vertices,
        source_faces,
        source_skin,
    )
    teacher_simplex = np.abs(teacher_field.sum(-1) - 1.0)
    row_l1 = np.abs(weights - teacher_field).sum(axis=1)
    dominant_pred = np.argmax(weights, axis=1)
    dominant_teacher = np.argmax(teacher_field, axis=1)
    metrics = {
        "row_l1_mean": float(row_l1.mean()),
        "row_l1_p50": float(np.percentile(row_l1, 50)),
        "row_l1_p90": float(np.percentile(row_l1, 90)),
        "row_l1_p95": float(np.percentile(row_l1, 95)),
        "row_l1_p99": float(np.percentile(row_l1, 99)),
        "dominant_accuracy": float(np.mean(dominant_pred == dominant_teacher)),
        "prediction_simplex_max_abs_residual": simplex_max,
        "teacher_field_simplex_max_abs_residual": float(teacher_simplex.max()),
        "teacher_surface_projection_distance_mean": float(field_distance.mean()),
        "teacher_surface_projection_distance_p95": float(np.percentile(field_distance, 95)),
        "teacher_surface_projection_distance_max": float(field_distance.max()),
        "rows_evaluated": int(n),
    }
    np.savez_compressed(
        out / "MIRA_V6_TEACHER23_CONTINUOUS_FIELD_EVAL.npz",
        teacher_weights=teacher_field,
        row_l1=row_l1,
        source_face_index=field_face,
        barycentric=field_bary,
        projection_distance=field_distance,
    )
    phase("CONTINUOUS_TEACHER_FIELD_EVAL_END", **metrics)

    result = {
        "schema": SCHEMA,
        "status": "PASS_REQUERY_EXECUTION",
        "scope": "MIRA_ONLY_FIXED_TEACHER23_RESEARCH_COURT__NOT_PRODUCT_AUTHORITY",
        "inputs": {
            "surface_sha256": sha256(surface_path),
            "surface_lineage_hash": str(surface.geometry_lineage_hash),
            "teacher_npz_sha256": sha256(teacher_path),
            "mira_checkpoint_sha256": model_sha,
            "exact_v6_runner_sha256": runner_sha,
        },
        "fixed_test_rig": {
            "raw_teacher_bone_count": 41,
            "selected_raw_bone_indices": binding["selected_raw_bone_indices"],
            "joint_count": len(skeleton.joints),
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "teacher_skin_role_before_prediction": "RIG_MEMBERSHIP_VALIDATION_ONLY__NOT_PREDICTOR_FEATURE",
        },
        "prediction": {
            "weights_sha256": prediction_sha,
            "shape": list(weights.shape),
            "simplex_max_abs_residual": simplex_max,
            "backbone_parameter_count": EXPECTED_BACKBONE_PARAMS,
            "decoder_parameter_count": EXPECTED_DECODER_PARAMS,
            "historical_28_joint_guard_is_decoder_authority": False,
            "runtime_joint_count": EXPECTED_JOINTS,
        },
        "continuous_teacher_field_evaluation": metrics,
        "causal_boundary": {
            "teacher_skin_predictor_input_used": False,
            "teacher_field_opened_after_prediction_seal": True,
            "fbx_object_packaging_authority": False,
            "carrier_identity": "GEOMETRY_COMPONENT_SUPPORT_SEMANTICS",
        },
        "runtime": runtime(),
        "claims": {
            "training_performed": False,
            "product_authority": False,
            "generalization": False,
            "tessa_carrier_skin_query_completed": False,
        },
        "next": "TRANSPORT_FROZEN_MIRA_SEMANTIC_MEMORY_TO_TESSA_CARRIER_AND_RUN_DYNAMIC_COURT",
    }
    write_json(out / "MIRA_V6_TEACHER23_FULL_REQUERY_RESULT.json", result)
    phase("COMPLETE", next=result["next"], result=str(out / "MIRA_V6_TEACHER23_FULL_REQUERY_RESULT.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
