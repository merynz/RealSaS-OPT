#!/usr/bin/env python3
from __future__ import annotations

"""Train the shared SURF-A1 extraction-only decoder over exact sealed V8 H."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from models.iris.vnext.surface_rescue_raster_v1 import SurfA1RasterConfigV1
from models.iris.vnext.surface_rescue_training_v1 import SurfA1ScheduleV1, extract_rescued_surface_v1, module_state_sha256_v1, surf_a1_objective_v1
from models.iris.vnext.surface_rescue_v1 import FrozenHSurfaceRescueV1
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head, load_exact_v8_model, load_json, load_subject_source, sha256_file
from tools.iris_vnext.surf_a1_io_v1 import camera_tensors_v1, load_camera_bundles_v1, load_raster_truth_v1, load_surf_a1_binding_v1

SCHEMA = "RealSaS.SurfA1FrozenHTrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.SurfA1FrozenHCheckpoint.v1"


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _checkpoint_payload(*, decoder, optimizer, completed_steps: int, scene_hash: str, source_sha: str, prereg_sha: str, binding_sha: str) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "completed_steps": int(completed_steps),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "v8_scene_core_sha256": scene_hash,
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "input_binding_sha256": binding_sha,
    }


def _restore_if_present(path: Path, *, decoder, optimizer, scene_hash: str, source_sha: str, prereg_sha: str, binding_sha: str) -> int:
    if not path.is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("SURF_A1_RESUME_SCHEMA_DRIFT")
    expected = {
        "v8_scene_core_sha256": scene_hash,
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "input_binding_sha256": binding_sha,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("SURF_A1_RESUME_AUTHORITY_DRIFT:" + key)
    decoder.load_state_dict(payload["decoder"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if payload.get("cuda_rng"):
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    completed = int(payload.get("completed_steps", -1))
    if completed < 0 or completed > SurfA1ScheduleV1().total_steps:
        raise RuntimeError("SURF_A1_RESUME_STEP_INVALID")
    print(f"SURF_A1_RESUME completed_steps={completed}", flush=True)
    return completed


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    source_sha = str(args.expected_source_sha).lower()
    if git_head(repo) != source_sha:
        raise RuntimeError("SURF_A1_TRAIN_SOURCE_SHA_DRIFT")
    prereg_path = Path(args.preregistration).resolve()
    binding_path = Path(args.binding).resolve()
    prereg = load_json(prereg_path)
    if prereg.get("schema") != "RealSaS.SurfA1FrozenHPreregistration.v1":
        raise ValueError("SURF_A1_TRAIN_PREREG_SCHEMA_INVALID")
    binding = load_surf_a1_binding_v1(binding_path)
    prereg_sha = sha256_file(prereg_path)
    binding_sha = sha256_file(binding_path)
    preflight = load_json(Path(args.preflight).resolve())
    if preflight.get("status") != "PASS_SURF_A1_A100_PREFLIGHT__TRAINING_AUTHORIZED" or preflight.get("training_authorized") is not True:
        raise RuntimeError("SURF_A1_TRAIN_PREFLIGHT_NOT_AUTHORIZED")
    if preflight.get("execution_source_sha") != source_sha:
        raise RuntimeError("SURF_A1_TRAIN_PREFLIGHT_SOURCE_DRIFT")

    manifest_path = Path(args.prepared_manifest).resolve()
    manifest = load_json(manifest_path)
    rows = list(manifest.get("subjects") or ())
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4" or tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("SURF_A1_TRAIN_MANIFEST_INVALID")
    if sha256_file(manifest_path) != binding["prepared"]["manifest"]["sha256"]:
        raise ValueError("SURF_A1_TRAIN_MANIFEST_SHA_DRIFT")

    device = torch.device("cuda")
    torch.manual_seed(26101924)
    torch.cuda.manual_seed_all(26101924)
    v8 = load_exact_v8_model(Path(args.v8_checkpoint), device)
    scene_hash = module_state_sha256_v1(v8.scene_core)
    if preflight.get("v8_scene_core_sha256") != scene_hash:
        raise RuntimeError("SURF_A1_TRAIN_SCENE_HASH_PREFLIGHT_DRIFT")
    extractor = PinnedFlexiCubesV1(Path(args.flexicubes_root), device=device, weight_scale=v8.config.flexicubes_weight_scale)
    decoder = FrozenHSurfaceRescueV1().to(device=device)
    decoder.train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=0.01)
    schedule = SurfA1ScheduleV1()
    raster_config = SurfA1RasterConfigV1(image_size=256, silhouette_sigma=1.0e-4, faces_per_pixel=24)
    cameras = load_camera_bundles_v1(Path(args.stage01_bundle), binding)

    print("[SURF-A1] caching exact sealed V8 scene planes + training truth for 8 subjects", flush=True)
    cache: dict[str, dict[str, Any]] = {}
    prepared_root = Path(args.prepared_root).resolve()
    for index, row in enumerate(rows):
        subject_id = str(row["subject_id"])
        shard_path = prepared_root / str(row["shard_file"])
        source = load_subject_source(shard_path, str(row["shard_sha256"]), device)
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        teacher_surface = shard.get("teacher_surface_points_iris")
        if not isinstance(teacher_surface, torch.Tensor) or teacher_surface.shape != (4096, 3):
            raise ValueError("SURF_A1_TRAIN_TEACHER_SURFACE_SHAPE_DRIFT:" + subject_id)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            planes = v8.encode_scene(source["image_tokens"], source["token_xy"], source["camera_features"]).float().detach()
        cache[subject_id] = {
            "scene_planes": planes,
            "teacher_surface": teacher_surface.to(device=device, dtype=torch.float32),
            "raster": load_raster_truth_v1(Path(args.raster_truth_root), subject_id, device),
        }
        print(f"[SURF-A1] cache {index+1}/8 {subject_id} planes={tuple(planes.shape)}", flush=True)
    if module_state_sha256_v1(v8.scene_core) != scene_hash:
        raise RuntimeError("SURF_A1_TRAIN_SCENE_MUTATED_DURING_CACHE")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mirror_root = Path(args.checkpoint_mirror_dir).resolve()
    mirror_root.mkdir(parents=True, exist_ok=True)
    local_latest = output_root / "checkpoint_latest.pt"
    mirror_latest = mirror_root / "checkpoint_latest.pt"
    if not local_latest.is_file() and mirror_latest.is_file():
        shutil.copy2(mirror_latest, local_latest)
    completed = _restore_if_present(
        local_latest, decoder=decoder, optimizer=optimizer, scene_hash=scene_hash,
        source_sha=source_sha, prereg_sha=prereg_sha, binding_sha=binding_sha,
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(completed, schedule.total_steps):
        step_started = time.perf_counter()
        resolution = schedule.resolution(step)
        lr = schedule.learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        subject_index = step % len(SUBJECT_IDS)
        subject_id = SUBJECT_IDS[subject_index]
        subject_cycle = step // len(SUBJECT_IDS)
        view0 = (2 * subject_cycle) % 8
        view1 = (view0 + 3) % 8
        item = cache[subject_id]
        raster = item["raster"]
        view_rows = []
        for view_index in (view0, view1):
            view_rows.append({
                "camera": camera_tensors_v1(cameras[subject_id], view_index, device),
                "mask": raster["mask"][view_index],
                "depth_normalized": raster["depth_normalized"][view_index],
                "normal_world": raster["normal_world"][view_index],
                "depth_normal_valid": raster["depth_normal_valid"][view_index],
            })
        optimizer.zero_grad(set_to_none=True)
        surface = extract_rescued_surface_v1(
            v8, decoder, item["scene_planes"], extractor,
            resolution=resolution, cube_chunk=int(args.cube_chunk), training=True,
        )
        losses = surf_a1_objective_v1(
            surface,
            teacher_surface_points_iris=item["teacher_surface"],
            raster_views=view_rows,
            raster_config=raster_config,
        )
        if not bool(torch.isfinite(losses["total"]).item()):
            raise RuntimeError("SURF_A1_TRAIN_NONFINITE_LOSS")
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
            raise RuntimeError("SURF_A1_TRAIN_NONFINITE_GRADIENT")
        optimizer.step()
        done = step + 1
        step_seconds = time.perf_counter() - step_started
        if done % 10 == 0 or done in (1, 400, 1000, 2400):
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            scalars = {key: float(value.detach().float().item()) for key, value in losses.items()}
            print(
                "SURF_A1_STEP",
                f"step={done}/{schedule.total_steps}", f"res={resolution}", f"subject={subject_id}",
                f"views={view0},{view1}", f"lr={lr:.8g}", f"total={scalars['total']:.6f}",
                f"chamfer={scalars['surface_chamfer']:.6f}", f"mask={scalars['silhouette']:.6f}",
                f"depth={scalars['depth']:.6f}", f"normal={scalars['normal']:.6f}",
                f"overlap={scalars['depth_normal_overlap_fraction']:.4f}",
                f"grad={float(grad_norm):.5f}", f"step_s={step_seconds:.3f}",
                f"gpu_alloc_gib={mem_alloc:.2f}", f"gpu_reserved_gib={mem_reserved:.2f}", f"gpu_peak_gib={mem_peak:.2f}",
                flush=True,
            )
        if done % 200 == 0 or done == schedule.total_steps:
            if module_state_sha256_v1(v8.scene_core) != scene_hash:
                raise RuntimeError("SURF_A1_TRAIN_FROZEN_SCENE_CORE_HASH_DRIFT")
            payload = _checkpoint_payload(
                decoder=decoder, optimizer=optimizer, completed_steps=done,
                scene_hash=scene_hash, source_sha=source_sha, prereg_sha=prereg_sha, binding_sha=binding_sha,
            )
            _atomic_torch_save(payload, local_latest)
            mirror_tmp = mirror_latest.with_suffix(".pt.tmp")
            shutil.copy2(local_latest, mirror_tmp)
            os.replace(mirror_tmp, mirror_latest)
            latest = {
                "schema": "RealSaS.SurfA1CheckpointPointer.v1",
                "completed_steps": done,
                "checkpoint_file": mirror_latest.name,
                "checkpoint_sha256": sha256_file(mirror_latest),
                "execution_source_sha": source_sha,
                "v8_scene_core_sha256": scene_hash,
            }
            _atomic_json(latest, mirror_root / "LATEST.json")
            print(f"SURF_A1_CHECKPOINT step={done} sha={latest['checkpoint_sha256'][:16]} mirror={mirror_latest}", flush=True)
        del surface, losses

    if module_state_sha256_v1(v8.scene_core) != scene_hash:
        raise RuntimeError("SURF_A1_TRAIN_FINAL_FROZEN_SCENE_CORE_HASH_DRIFT")
    final_checkpoint_sha = sha256_file(local_latest)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED_SURF_A1_2400_STEPS__AWAITING_INDEPENDENT_COURT",
        "execution_source_sha": source_sha,
        "preregistration_sha256": prereg_sha,
        "input_binding_sha256": binding_sha,
        "v8_scene_core_sha256": scene_hash,
        "optimizer_steps_executed": schedule.total_steps,
        "final_resolution": 48,
        "final_checkpoint_file": local_latest.name,
        "final_checkpoint_sha256": final_checkpoint_sha,
        "elapsed_seconds": float(time.perf_counter() - started),
        "teacher_truth_in_inference_forward": False,
        "tessa_invoked": False,
        "compiler_qualified": False,
        "scientific_pass_claimed": False,
        "next_action": "RUN_INDEPENDENT_SURF_A1_REPRESENTATION_RESCUE_COURT",
    }
    _atomic_json(result, output_root / "SURF_A1_TRAINING_RESULT.json")
    _atomic_json(result, mirror_root / "SURF_A1_TRAINING_RESULT.json")
    print("SURF_A1_TRAINING_COMPLETE", json.dumps({"checkpoint": final_checkpoint_sha[:16], "scene_core": scene_hash[:12], "elapsed_s": result["elapsed_seconds"]}, sort_keys=True), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--binding", default="canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json")
    p.add_argument("--preregistration", default="canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json")
    p.add_argument("--preflight", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--stage01-bundle", required=True)
    p.add_argument("--raster-truth-root", required=True)
    p.add_argument("--flexicubes-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint-mirror-dir", required=True)
    p.add_argument("--cube-chunk", type=int, default=2048)
    train(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
