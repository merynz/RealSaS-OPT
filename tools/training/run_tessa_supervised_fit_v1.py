from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from compiler.realsas_compiler_core.geometry_artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict
from models.tessa.v1 import (
    TESSAConfigV1,
    TESSAV1,
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
    build_tessa_conditioning_v1,
)


SCHEMA = "RealSaS.TESSASupervisedFitResult.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_teacher_vertices(vertices: np.ndarray, *, center: tuple[float, float, float], scale: float) -> np.ndarray:
    """Map world-space teacher vertices into the Stage08-bound TESSA token frame.

    `scale` is 2*NormalizationDomainIR.half_extent, so Stage08 normalized
    coordinates [-1,+1] map exactly to TESSA coordinates [-0.5,+0.5]. No
    teacher-derived refit, padding or clipping is allowed.
    """
    v = np.asarray(vertices, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("TESSA_TEACHER_VERTICES_INVALID")
    out = (v - c[None, :]) / float(scale)
    if np.any(out < -0.5000001) or np.any(out > 0.5000001):
        lo = out.min(axis=0).tolist()
        hi = out.max(axis=0).tolist()
        raise ValueError(f"TESSA_TEACHER_OUTSIDE_CANONICAL_NORMALIZATION:{lo}:{hi}")
    return out


def load_teacher(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        vertex_key = "vertices_source" if "vertices_source" in z.files else "vertices"
        face_key = "faces" if "faces" in z.files else "faces_source"
        if vertex_key not in z.files or face_key not in z.files:
            raise ValueError(f"TESSA_TEACHER_NPZ_KEYS_MISSING:{z.files}")
        vertices = np.asarray(z[vertex_key], dtype=np.float64)
        faces = np.asarray(z[face_key], dtype=np.int64)
    return vertices, faces


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: Path, *, model, optimizer, epoch: int, step: int, cfg: TESSAConfigV1, metadata: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "RealSaS.TESSASupervisedCheckpoint.v1",
            "epoch": int(epoch),
            "step": int(step),
            "config": asdict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        tmp,
    )
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-json", required=True)
    ap.add_argument("--normalization-json", required=True)
    ap.add_argument("--teacher-npz", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--small-smoke", action="store_true")
    args = ap.parse_args()

    if args.epochs < 1 or args.grad_accum < 1 or args.target_tokens < 32:
        raise ValueError("TESSA_TRAIN_ARGUMENT_INVALID")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    surface_path = Path(args.surface_json)
    normalization_path = Path(args.normalization_json)
    teacher_path = Path(args.teacher_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_payload = json.loads(surface_path.read_text(encoding="utf-8"))
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    surface = rigging_surface_from_dict(surface_payload)
    normalization = normalization_domain_from_dict(normalization_payload)
    conditioning = build_tessa_conditioning_v1(surface, normalization=normalization)
    teacher_vertices, teacher_faces = load_teacher(teacher_path)
    normalized_teacher = normalize_teacher_vertices(
        teacher_vertices,
        center=conditioning.center,
        scale=conditioning.scale,
    )

    cfg = TESSAConfigV1()
    if args.small_smoke:
        cfg = TESSAConfigV1(
            d_model=128,
            n_heads=8,
            surface_layers=2,
            decoder_layers=2,
            mlp_ratio=2,
            surface_latent_count=96,
            local_attention_window=256,
            query_chunk_size=64,
            max_faces=32768,
            max_vertices=32768,
        )
    teacher_sequence = build_teacher_asset_sequence_v1(normalized_teacher, teacher_faces, cfg=cfg)
    windows = build_truncated_teacher_windows_v1(
        teacher_sequence,
        cfg=cfg,
        target_tokens=int(args.target_tokens),
    )
    if not windows:
        raise RuntimeError("TESSA_NO_TRAINING_WINDOWS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TESSAV1(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    latest = output_dir / "TESSA_V1_LATEST.pt"
    start_epoch = 0
    global_step = 0

    manifest = {
        "surface_sha256": sha256_file(surface_path),
        "normalization_sha256": sha256_file(normalization_path),
        "normalization_hash": conditioning.normalization_hash,
        "coordinate_frame": conditioning.coordinate_frame,
        "tessa_world_scale": float(conditioning.scale),
        "teacher_sha256": sha256_file(teacher_path),
        "surface_geometry_lineage_hash": conditioning.source_geometry_lineage_hash,
        "teacher_vertex_count": int(len(teacher_vertices)),
        "teacher_face_count": int(len(teacher_faces)),
        "teacher_component_count": int(teacher_sequence.component_count),
        "teacher_restart_count": int(teacher_sequence.restart_count),
        "global_sequence_tokens": int(len(teacher_sequence.token_ids)),
        "training_window_count": int(len(windows)),
        "target_tokens_per_window": int(args.target_tokens),
        "causal_context_tokens": int(cfg.local_attention_window),
        "artificial_topology_chart_split": False,
        "teacher_refit_or_clip_performed": False,
        "model_parameter_count": int(parameter_count(model)),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "bf16": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        "seed": int(args.seed),
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }

    if args.resume and latest.is_file():
        state = torch.load(latest, map_location="cpu")
        if state.get("schema") != "RealSaS.TESSASupervisedCheckpoint.v1":
            raise ValueError("TESSA_RESUME_SCHEMA_MISMATCH")
        if dict(state.get("config") or {}) != asdict(cfg):
            raise ValueError("TESSA_RESUME_CONFIG_MISMATCH")
        old_meta = dict(state.get("metadata") or {})
        for key in (
            "surface_sha256",
            "normalization_sha256",
            "normalization_hash",
            "coordinate_frame",
            "teacher_sha256",
            "global_sequence_tokens",
            "target_tokens_per_window",
            "causal_context_tokens",
        ):
            if old_meta.get(key) != manifest.get(key):
                raise ValueError(f"TESSA_RESUME_INPUT_CONTRACT_MISMATCH:{key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["step"])
        print("TESSA_RESUME", start_epoch, global_step, flush=True)

    surface_features = conditioning.features.unsqueeze(0).to(device)
    use_bf16 = bool(manifest["bf16"])
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    (output_dir / "TESSA_V1_FIT_INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TESSA_PREFLIGHT", json.dumps(manifest, sort_keys=True), flush=True)

    epoch_records = []
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(windows)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        scored_token_count = 0
        t0 = time.time()
        for local_index, window_index in enumerate(order):
            window = windows[window_index]
            ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.tensor(window.labels, dtype=torch.long, device=device).unsqueeze(0)
            scored = int((labels != cfg.PAD).sum().item())
            if scored < 1:
                raise RuntimeError("TESSA_WINDOW_HAS_NO_SCORED_TOKENS")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                out = model(
                    surface_features=surface_features,
                    input_ids=ids,
                    labels=labels,
                    sequence_position_offset=window.sequence_position_offset,
                )
                if out.loss is None:
                    raise RuntimeError("TESSA_LOSS_MISSING")
                loss = out.loss / float(args.grad_accum)
            loss.backward()
            raw_loss = float(out.loss.detach().cpu())
            epoch_loss_sum += raw_loss * scored
            scored_token_count += scored
            do_step = ((local_index + 1) % args.grad_accum == 0) or (local_index + 1 == len(order))
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            print(
                "TESSA_WINDOW",
                json.dumps(
                    {
                        "epoch": epoch,
                        "window": int(window_index),
                        "sequence_position_offset": window.sequence_position_offset,
                        "input_tokens": int(ids.numel()),
                        "scored_tokens": scored,
                        "loss": raw_loss,
                        "global_step": global_step,
                        "cuda_peak_bytes": peak,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elapsed = max(time.time() - t0, 1e-9)
        mean_loss = epoch_loss_sum / max(scored_token_count, 1)
        record = {
            "epoch": epoch,
            "mean_scored_token_loss": mean_loss,
            "scored_tokens": scored_token_count,
            "seconds": elapsed,
            "scored_tokens_per_second": scored_token_count / elapsed,
            "global_step": global_step,
        }
        epoch_records.append(record)
        print("TESSA_EPOCH", json.dumps(record, sort_keys=True), flush=True)
        save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            cfg=cfg,
            metadata=manifest,
        )
        numbered = output_dir / f"TESSA_V1_EPOCH_{epoch:04d}.pt"
        if epoch == args.epochs - 1 or epoch % 5 == 0:
            save_checkpoint(
                numbered,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                cfg=cfg,
                metadata=manifest,
            )

    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "training_class": "T1_SUPERVISED_TOPOLOGY_FIT",
        "input_manifest": manifest,
        "epochs": epoch_records,
        "latest_checkpoint": latest.name,
        "latest_checkpoint_sha256": sha256_file(latest),
        "product_authority_claimed": False,
        "generalization_claimed": False,
        "mechanical_consequence_training_included": False,
        "next_required_rung": "T2_MECHANICAL_CONSEQUENCE_FIT",
    }
    (output_dir / "TESSA_V1_FIT_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TESSA_SUPERVISED_FIT_PASS", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
