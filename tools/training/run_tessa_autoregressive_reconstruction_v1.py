from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from compiler.realsas_compiler_core.geometry_artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict
from models.tessa.v1 import (
    TESSAConfigV1,
    TESSAV1,
    build_teacher_asset_sequence_v1,
    build_tessa_conditioning_v1,
    decode_tessa_asset_sequence_v1,
    quantize_tessa_xyz_v1,
)


RESULT_SCHEMA = "RealSaS.TESSAAutoregressiveReconstructionResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.TESSASupervisedCheckpoint.v2"
G3_NUMERICAL_MIN_ANGLE_DEG = 7.5
G3_NUMERICAL_MAX_ASPECT = 16.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_teacher(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        vertex_key = "vertices_source" if "vertices_source" in z.files else "vertices"
        face_key = "faces" if "faces" in z.files else "faces_source"
        if vertex_key not in z.files or face_key not in z.files:
            raise ValueError(f"TESSA_TEACHER_NPZ_KEYS_MISSING:{z.files}")
        return np.asarray(z[vertex_key], dtype=np.float64), np.asarray(z[face_key], dtype=np.int64)


def normalize_teacher_vertices(
    vertices: np.ndarray,
    *,
    center: tuple[float, float, float],
    scale: float,
) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    out = (v - c[None, :]) / float(scale)
    if np.any(out < -0.5000001) or np.any(out > 0.5000001):
        raise ValueError("TESSA_TEACHER_OUTSIDE_CANONICAL_NORMALIZATION")
    return out


def _reshape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, n, d = x.shape
    hd = d // heads
    return x.view(b, n, heads, hd).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, n, hd = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * hd)


def _precompute_cross_kv(model: TESSAV1, memory: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in model.decoder:
        mha = block.cross
        d = int(mha.embed_dim)
        w = mha.in_proj_weight
        b = mha.in_proj_bias
        kb = None if b is None else b[d : 2 * d]
        vb = None if b is None else b[2 * d :]
        k = F.linear(memory, w[d : 2 * d], kb)
        v = F.linear(memory, w[2 * d :], vb)
        cached.append((_reshape_heads(k, mha.num_heads), _reshape_heads(v, mha.num_heads)))
    return cached


def _cross_attention_one(
    block,
    x: torch.Tensor,
    cached_kv: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    mha = block.cross
    d = int(mha.embed_dim)
    w = mha.in_proj_weight
    b = mha.in_proj_bias
    qb = None if b is None else b[:d]
    q = F.linear(block.norm2(x), w[:d], qb)
    q = _reshape_heads(q, mha.num_heads)
    k, v = cached_kv
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    y = _merge_heads(y)
    return mha.out_proj(y)


def _decoder_step_cached(
    *,
    model: TESSAV1,
    token_id: int,
    absolute_position: int,
    self_kv: list[tuple[torch.Tensor, torch.Tensor] | None],
    cross_kv: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    cfg = model.cfg
    device = model.token_embedding.weight.device
    ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    pos = torch.tensor(
        [[int(absolute_position) % int(cfg.local_attention_window)]],
        dtype=torch.long,
        device=device,
    )
    x = model.token_embedding(ids) + model.position_embedding(pos)
    new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    for li, block in enumerate(model.decoder):
        n1 = block.norm1(x)
        qkv = block.self_attn.qkv(n1).view(
            1, 1, 3, cfg.n_heads, cfg.d_model // cfg.n_heads
        )
        q, k_new, v_new = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k_new = k_new.transpose(1, 2).contiguous()
        v_new = v_new.transpose(1, 2).contiguous()
        prior = self_kv[li]
        if prior is None:
            k = k_new
            v = v_new
        else:
            k = torch.cat((prior[0], k_new), dim=2)
            v = torch.cat((prior[1], v_new), dim=2)
        if k.shape[2] > cfg.local_attention_window:
            k = k[:, :, -cfg.local_attention_window :]
            v = v[:, :, -cfg.local_attention_window :]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + block.self_attn.out(_merge_heads(y))
        x = x + _cross_attention_one(block, x, cross_kv[li])
        x = x + block.mlp(block.norm3(x))
        new_cache.append((k, v))

    logits = model.lm_head(model.final_norm(x))[:, -1, :]
    return logits, new_cache


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in np.asarray(faces, dtype=np.int64).tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            e = tuple(sorted((int(u), int(v))))
            counts[e] = counts.get(e, 0) + 1
    return counts


def _triangle_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    degenerate = 0
    min_angle = float("inf")
    max_aspect = 0.0
    for ia, ib, ic in f.tolist():
        p0, p1, p2 = v[int(ia)], v[int(ib)], v[int(ic)]
        a = float(np.linalg.norm(p1 - p2))
        b = float(np.linalg.norm(p0 - p2))
        c = float(np.linalg.norm(p0 - p1))
        area2 = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        if min(a, b, c) <= 1e-15 or area2 <= 1e-15:
            degenerate += 1
            min_angle = 0.0
            max_aspect = float("inf")
            continue
        angles = []
        for opposite, s1, s2 in ((a, b, c), (b, a, c), (c, a, b)):
            cosine = (s1 * s1 + s2 * s2 - opposite * opposite) / (2.0 * s1 * s2)
            cosine = min(1.0, max(-1.0, cosine))
            angles.append(math.degrees(math.acos(cosine)))
        min_angle = min(min_angle, min(angles))
        longest = max(a, b, c)
        aspect = (longest * longest) / area2
        max_aspect = max(max_aspect, aspect)
    return {
        "degenerate_face_count": int(degenerate),
        "min_triangle_angle_deg": float(min_angle if math.isfinite(min_angle) else 0.0),
        "max_aspect_longest_over_min_altitude": float(max_aspect),
    }


def _mesh_identity_sets(decoded, cfg: TESSAConfigV1) -> tuple[set[tuple], set[tuple]]:
    q = quantize_tessa_xyz_v1(decoded.vertices_normalized, cfg.coordinate_bins)
    vertex_keys: set[tuple] = set()
    for vi, row in enumerate(q.tolist()):
        vertex_keys.add((int(decoded.vertex_component_indices[vi]), *map(int, row)))
    face_keys: set[tuple] = set()
    for fi, face in enumerate(decoded.faces.tolist()):
        comp = int(decoded.face_component_indices[fi])
        verts = [tuple(map(int, q[int(vi)].tolist())) for vi in face]
        face_keys.add((comp, *tuple(sorted(verts))))
    return vertex_keys, face_keys


def _set_pr(a: set, b: set) -> tuple[float, float, float]:
    # a=generated, b=teacher
    inter = len(a & b)
    precision = inter / max(len(a), 1)
    recall = inter / max(len(b), 1)
    union = len(a | b)
    iou = inter / max(union, 1)
    return float(precision), float(recall), float(iou)


def _token_metrics(generated: list[int], teacher: tuple[int, ...]) -> dict[str, float | int | bool]:
    n = min(len(generated), len(teacher))
    matches = sum(int(generated[i] == teacher[i]) for i in range(n))
    prefix = 0
    for i in range(n):
        if generated[i] != teacher[i]:
            break
        prefix += 1
    return {
        "generated_token_count": int(len(generated)),
        "teacher_token_count": int(len(teacher)),
        "length_exact": bool(len(generated) == len(teacher)),
        "position_exact_fraction_on_overlap": float(matches / max(n, 1)),
        "common_prefix_tokens": int(prefix),
        "common_prefix_fraction_of_teacher": float(prefix / max(len(teacher), 1)),
        "exact_sequence_match": bool(tuple(generated) == tuple(teacher)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-json", required=True)
    ap.add_argument("--normalization-json", required=True)
    ap.add_argument("--teacher-npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-generation-tokens", type=int, default=70000)
    ap.add_argument("--log-every", type=int, default=256)
    args = ap.parse_args()
    if args.max_generation_tokens < 2 or args.log_every < 1:
        raise ValueError("TESSA_T1B_ARGUMENT_INVALID")

    surface_path = Path(args.surface_json)
    normalization_path = Path(args.normalization_json)
    teacher_path = Path(args.teacher_npz)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("TESSA_T1B_PHASE", json.dumps({"phase": "LOAD_AND_VERIFY_INPUTS"}), flush=True)
    surface = rigging_surface_from_dict(json.loads(surface_path.read_text(encoding="utf-8")))
    normalization = normalization_domain_from_dict(json.loads(normalization_path.read_text(encoding="utf-8")))
    conditioning = build_tessa_conditioning_v1(surface, normalization=normalization)
    teacher_vertices, teacher_faces = load_teacher(teacher_path)
    normalized_teacher = normalize_teacher_vertices(
        teacher_vertices,
        center=conditioning.center,
        scale=conditioning.scale,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("TESSA_T1B_CHECKPOINT_SCHEMA_MISMATCH")
    cfg = TESSAConfigV1(**dict(state["config"]))
    meta = dict(state.get("metadata") or {})
    expected = {
        "surface_sha256": sha256_file(surface_path),
        "normalization_sha256": sha256_file(normalization_path),
        "normalization_hash": conditioning.normalization_hash,
        "teacher_sha256": sha256_file(teacher_path),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"TESSA_T1B_INPUT_CONTRACT_MISMATCH:{key}")
    if int(state.get("step", -1)) != 1024:
        raise ValueError("TESSA_T1B_REQUIRES_SEALED_1024_STEP_CHECKPOINT")

    teacher_sequence = build_teacher_asset_sequence_v1(normalized_teacher, teacher_faces, cfg=cfg)
    teacher_decoded = decode_tessa_asset_sequence_v1(teacher_sequence.token_ids, cfg=cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("TESSA_T1B_CUDA_REQUIRED")
    device = torch.device("cuda")
    model = TESSAV1(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    surface_features = conditioning.features.unsqueeze(0).to(device)
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(
        "TESSA_T1B_PREFLIGHT",
        json.dumps(
            {
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_step": int(state["step"]),
                "teacher_tokens": len(teacher_sequence.token_ids),
                "teacher_faces": int(teacher_sequence.face_count),
                "teacher_components": int(teacher_sequence.component_count),
                "max_generation_tokens": int(args.max_generation_tokens),
                "gpu": torch.cuda.get_device_name(0),
                "bf16": use_bf16,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    print("TESSA_T1B_PHASE", json.dumps({"phase": "AUTOREGRESSIVE_GENERATION"}), flush=True)
    generated = [int(cfg.BOS)]
    self_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(model.decoder)
    confidence_sum = 0.0
    confidence_min = 1.0
    first_mismatch: int | None = None
    invalid_early_token: int | None = None
    started = time.time()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        memory = model.encode_surface(surface_features)
        cross_kv = _precompute_cross_kv(model, memory)
        while len(generated) < int(args.max_generation_tokens):
            pos = len(generated) - 1
            logits, self_kv = _decoder_step_cached(
                model=model,
                token_id=generated[-1],
                absolute_position=pos,
                self_kv=self_kv,
                cross_kv=cross_kv,
            )
            probs = torch.softmax(logits.float(), dim=-1)
            conf, nxt = probs.max(dim=-1)
            next_id = int(nxt.item())
            confidence = float(conf.item())
            confidence_sum += confidence
            confidence_min = min(confidence_min, confidence)
            generated.append(next_id)

            new_index = len(generated) - 1
            if first_mismatch is None and new_index < len(teacher_sequence.token_ids):
                if next_id != int(teacher_sequence.token_ids[new_index]):
                    first_mismatch = int(new_index)
                    print(
                        "TESSA_T1B_FIRST_MISMATCH",
                        json.dumps(
                            {
                                "token_index": new_index,
                                "generated": next_id,
                                "teacher": int(teacher_sequence.token_ids[new_index]),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            if next_id == cfg.EOS:
                break
            if next_id in (cfg.PAD, cfg.BOS):
                invalid_early_token = next_id
                break

            if len(generated) % int(args.log_every) == 0:
                elapsed = max(time.time() - started, 1e-9)
                expected_total = len(teacher_sequence.token_ids)
                rate = (len(generated) - 1) / elapsed
                remaining = max(expected_total - len(generated), 0)
                eta = remaining / max(rate, 1e-9)
                print(
                    "TESSA_T1B_PROGRESS",
                    json.dumps(
                        {
                            "tokens": len(generated),
                            "teacher_tokens": expected_total,
                            "progress_vs_teacher": len(generated) / expected_total,
                            "tokens_per_second": rate,
                            "elapsed_seconds": elapsed,
                            "eta_seconds_vs_teacher_length": eta,
                            "first_mismatch": first_mismatch,
                            "mean_greedy_confidence": confidence_sum / max(len(generated) - 1, 1),
                            "min_greedy_confidence": confidence_min,
                            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                            "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    elapsed = max(time.time() - started, 1e-9)
    eos_reached = bool(generated and generated[-1] == cfg.EOS)
    print(
        "TESSA_T1B_GENERATION_DONE",
        json.dumps(
            {
                "tokens": len(generated),
                "eos_reached": eos_reached,
                "invalid_early_token": invalid_early_token,
                "first_mismatch": first_mismatch,
                "seconds": elapsed,
                "tokens_per_second": (len(generated) - 1) / elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    token_metrics = _token_metrics(generated, teacher_sequence.token_ids)
    strict_decode_pass = False
    decode_error: str | None = None
    decoded = None
    if eos_reached and invalid_early_token is None:
        print("TESSA_T1B_PHASE", json.dumps({"phase": "STRICT_GRAMMAR_DECODE"}), flush=True)
        try:
            decoded = decode_tessa_asset_sequence_v1(generated, cfg=cfg)
            strict_decode_pass = True
        except Exception as exc:  # preserve exact scientific failure signature
            decode_error = f"{type(exc).__name__}:{exc}"
    else:
        decode_error = "GENERATION_DID_NOT_REACH_VALID_EOS"

    static_report: dict = {
        "strict_decode_pass": strict_decode_pass,
        "decode_error": decode_error,
        "orientation_qualified": False,
        "support_binding_qualified": False,
        "full_compiler_six_gate_qualification_run": False,
        "full_compiler_six_gate_qualification_reason": (
            "TESSA_PROPOSAL_NOT_YET_SUPPORT_BOUND; G1/G3/G3B/G4/G5 REQUIRE COMPILER AUTHORITY"
        ),
    }

    reconstruction_gate = False
    if decoded is not None:
        print("TESSA_T1B_PHASE", json.dumps({"phase": "DECODED_MESH_STATIC_PREFLIGHT"}), flush=True)
        edge_counts = _edge_incidence(decoded.faces)
        nonmanifold = sum(int(n > 2) for n in edge_counts.values())
        boundary = sum(int(n == 1) for n in edge_counts.values())
        face_keys = [tuple(sorted(map(int, row))) for row in decoded.faces.tolist()]
        duplicate_faces = len(face_keys) - len(set(face_keys))
        tri = _triangle_metrics(decoded.vertices_normalized, decoded.faces)
        gen_vertices, gen_faces = _mesh_identity_sets(decoded, cfg)
        teacher_vertices_set, teacher_faces_set = _mesh_identity_sets(teacher_decoded, cfg)
        vp, vr, viou = _set_pr(gen_vertices, teacher_vertices_set)
        fp, fr, fiou = _set_pr(gen_faces, teacher_faces_set)
        intrinsic_pass = bool(
            tri["degenerate_face_count"] == 0
            and duplicate_faces == 0
            and nonmanifold == 0
            and float(tri["min_triangle_angle_deg"]) + 1e-12 >= G3_NUMERICAL_MIN_ANGLE_DEG
            and float(tri["max_aspect_longest_over_min_altitude"]) - 1e-12 <= G3_NUMERICAL_MAX_ASPECT
        )
        static_report.update(
            {
                "decoded_vertex_count": int(len(decoded.vertices_normalized)),
                "decoded_face_count": int(len(decoded.faces)),
                "decoded_component_count": int(decoded.component_count),
                "decoded_restart_count": int(decoded.restart_count),
                "duplicate_face_count": int(duplicate_faces),
                "nonmanifold_edge_count": int(nonmanifold),
                "boundary_edge_count": int(boundary),
                **tri,
                "teacher_vertex_set_precision": vp,
                "teacher_vertex_set_recall": vr,
                "teacher_vertex_set_iou": viou,
                "teacher_face_set_precision": fp,
                "teacher_face_set_recall": fr,
                "teacher_face_set_iou": fiou,
                "compiler_intrinsic_g3_min_angle_threshold_deg": G3_NUMERICAL_MIN_ANGLE_DEG,
                "compiler_intrinsic_g3_max_aspect_threshold": G3_NUMERICAL_MAX_ASPECT,
                "compiler_intrinsic_static_preflight_pass": intrinsic_pass,
            }
        )
        reconstruction_gate = bool(
            token_metrics["exact_sequence_match"]
            and fp == 1.0
            and fr == 1.0
            and vp == 1.0
            and vr == 1.0
            and int(decoded.component_count) == int(teacher_decoded.component_count)
            and int(len(decoded.faces)) == int(len(teacher_decoded.faces))
        )

    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS" if reconstruction_gate else "FAIL",
        "court": "T1B_AUTOREGRESSIVE_RECONSTRUCTION",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_optimizer_step": int(state["step"]),
        "surface_sha256": expected["surface_sha256"],
        "normalization_sha256": expected["normalization_sha256"],
        "normalization_hash": expected["normalization_hash"],
        "teacher_sha256": expected["teacher_sha256"],
        "generation": {
            "eos_reached": eos_reached,
            "invalid_early_token": invalid_early_token,
            "first_mismatch_token_index": first_mismatch,
            "seconds": elapsed,
            "tokens_per_second": (len(generated) - 1) / elapsed,
            "mean_greedy_confidence": confidence_sum / max(len(generated) - 1, 1),
            "min_greedy_confidence": confidence_min,
            "max_generation_tokens": int(args.max_generation_tokens),
            "greedy_decode": True,
            "teacher_tokens_fed_to_model": False,
        },
        "token_reconstruction": token_metrics,
        "static_mesh_report": static_report,
        "t1b_reconstruction_pass": reconstruction_gate,
        "product_authority_claimed": False,
        "generalization_claimed": False,
        "mechanical_consequence_training_included": False,
        "next_required_rung": (
            "T1C_COMPILER_SUPPORT_BINDING_AND_STATIC_QUALIFICATION"
            if reconstruction_gate
            else "T1B_FAILURE_ANALYSIS"
        ),
    }
    result_path = out_dir / "TESSA_T1B_AUTOREGRESSIVE_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.save(out_dir / "TESSA_T1B_GENERATED_TOKENS.npy", np.asarray(generated, dtype=np.int64))
    if decoded is not None:
        np.savez_compressed(
            out_dir / "TESSA_T1B_DECODED_MESH.npz",
            vertices_normalized=np.asarray(decoded.vertices_normalized, dtype=np.float32),
            faces=np.asarray(decoded.faces, dtype=np.int64),
            vertex_component_indices=np.asarray(decoded.vertex_component_indices, dtype=np.int64),
            face_component_indices=np.asarray(decoded.face_component_indices, dtype=np.int64),
        )
    print("TESSA_T1B_FINAL", json.dumps(result, sort_keys=True), flush=True)
    return 0 if reconstruction_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
