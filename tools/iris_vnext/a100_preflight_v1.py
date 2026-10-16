from __future__ import annotations

"""A100 engineering preflight for IRIS vNext Fit8.

This is deliberately not a scientific Fit8 run. It uses the exact prepared B=8
cohort but a short preflight-only policy/checkpoint lineage, measures real GPU
memory and step wall time, and separately exercises pinned FlexiCubes with an
analytic zero-surface so random-model topology does not masquerade as a science
failure. No preflight checkpoint is reusable by the scientific run.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time

import torch

from models.iris.vnext.cohort_v1 import VNextFit8PolicyV1, VNextStageV1
from models.iris.vnext.cohort_v2 import IRISVNextFit8TrainerV2
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from tools.iris_vnext.input_binding_v1 import sha256_file


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", type=Path, required=True)
    p.add_argument("--flexicubes-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry-steps", type=int, default=3)
    return p.parse_args()


def _require_prepared_cohort(root: Path):
    manifest_path = root / "COHORT_MANIFEST.json"
    receipt_path = root / "PREPARATION_RECEIPT.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("VNEXT_A100_PREPARED_COHORT_RECEIPT_REQUIRED")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "RealSaS.IRISVNextFit8PreparationReceipt.v2":
        raise ValueError("VNEXT_A100_PREPARATION_RECEIPT_SCHEMA_INVALID")
    if receipt.get("cohort_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("VNEXT_A100_COHORT_MANIFEST_HASH_DRIFT")
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("VNEXT_A100_COHORT_MANIFEST_SCHEMA_INVALID")
    if manifest.get("builder") != "tools/iris_vnext/build_fit8_vnext_shards_v2.py":
        raise ValueError("VNEXT_A100_COHORT_BUILDER_AUTHORITY_DRIFT")
    if manifest.get("legacy_optimizer_state_reused") is not False:
        raise ValueError("VNEXT_A100_LEGACY_OPTIMIZER_REUSE_FORBIDDEN")
    if receipt.get("scientific_fit_started") is not False:
        raise ValueError("VNEXT_A100_PREPARATION_ALREADY_CLAIMS_FIT")
    return manifest, receipt, manifest_path


def _cuda_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _analytic_flexicubes_probe(root: Path, cfg: IRISVNextConfigV1):
    extractor = PinnedFlexiCubesV1(
        root, device="cuda", weight_scale=cfg.flexicubes_weight_scale
    )
    vertices, cubes = extractor.construct_iris_grid(cfg.extraction_training_resolution)
    vertices = vertices.cuda().float().requires_grad_(True)
    cubes = cubes.cuda()
    field = vertices.square().sum(-1).sqrt() - 0.55
    raw = torch.zeros((cubes.shape[0], 21), device="cuda", requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    surface, faces, l_dev = extractor.extract(
        vertices_iris=vertices,
        field=field,
        cubes=cubes,
        raw_parameters=raw,
        resolution=cfg.extraction_training_resolution,
        training=True,
    )
    objective = surface.square().mean() + (l_dev.mean() if l_dev.numel() else surface.sum() * 0.0)
    objective.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if surface.numel() == 0 or faces.numel() == 0:
        raise RuntimeError("VNEXT_A100_FLEXICUBES_EMPTY_ANALYTIC_SURFACE")
    if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
        raise RuntimeError("VNEXT_A100_FLEXICUBES_PARAMETER_GRADIENT_INVALID")
    if vertices.grad is None or not bool(torch.isfinite(vertices.grad).all()):
        raise RuntimeError("VNEXT_A100_FLEXICUBES_VERTEX_GRADIENT_INVALID")
    return {
        "resolution": int(cfg.extraction_training_resolution),
        "grid_vertices": int(vertices.shape[0]),
        "cubes": int(cubes.shape[0]),
        "surface_vertices": int(surface.shape[0]),
        "surface_faces": int(faces.shape[0]),
        "raw_parameter_gradient_l1": float(raw.grad.abs().mean().detach().cpu()),
        "vertex_gradient_l1": float(vertices.grad.abs().mean().detach().cpu()),
        "elapsed_seconds": elapsed,
        "memory": _cuda_snapshot(),
        "scientific_result": "NONE__ANALYTIC_ENGINEERING_COURT_ONLY",
    }


def main():
    args = _args()
    if args.geometry_steps <= 0:
        raise ValueError("VNEXT_A100_GEOMETRY_STEPS_INVALID")
    if not torch.cuda.is_available():
        raise RuntimeError("VNEXT_A100_CUDA_REQUIRED")
    device_name = torch.cuda.get_device_name(0)
    if "A100" not in device_name.upper():
        raise RuntimeError(f"VNEXT_A100_DEVICE_REQUIRED:{device_name}")
    cohort_root = args.cohort_dir.resolve()
    manifest, preparation, manifest_path = _require_prepared_cohort(cohort_root)
    cfg = IRISVNextConfigV1()
    if int(manifest["topology_coordinate_bins"]) != int(cfg.tessa_decoder_config().coordinate_bins):
        raise ValueError("VNEXT_A100_TOPOLOGY_GRAMMAR_BIN_DRIFT")
    if preparation.get("topology_teacher_collision_policy") != manifest.get("topology_teacher_collision_policy"):
        raise ValueError("VNEXT_A100_TOPOLOGY_COLLISION_POLICY_DRIFT")
    policy = replace(
        VNextFit8PolicyV1(),
        checkpoint_interval=max(1000, args.geometry_steps + 1),
        eval_interval=max(1000, args.geometry_steps + 1),
        stages=(VNextStageV1("PREFLIGHT_G1_GEOMETRY", args.geometry_steps, field_weight=1.0),),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix="realsas_vnext_a100_preflight_") as tmp:
        trainer = IRISVNextFit8TrainerV2(
            data_root=cohort_root,
            manifest=manifest,
            checkpoint_root=Path(tmp) / "discarded_checkpoints",
            device="cuda",
            model_config=cfg,
            policy=policy,
            flexicubes_root=args.flexicubes_root.resolve(),
        )
        parameter_count = sum(p.numel() for p in trainer.model.parameters())
        trainable_parameter_count = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        step_rows = []
        for _ in range(args.geometry_steps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            row = trainer.one_step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            if row["batch_subjects"] != 8 or row["shared_scene_encodes"] != 1:
                raise RuntimeError("VNEXT_A100_BATCH8_EXECUTION_DRIFT")
            if row["subject_serial_model_forward"] is not False or row["optimizer_steps"] != 1:
                raise RuntimeError("VNEXT_A100_OPTIMIZER_SEMANTICS_DRIFT")
            if not torch.isfinite(torch.tensor(row["loss"])):
                raise RuntimeError("VNEXT_A100_NONFINITE_GEOMETRY_STEP")
            step_rows.append(
                {
                    "global_step": int(row["global_step"]),
                    "loss": float(row["loss"]),
                    "wall_seconds": wall,
                    "memory": _cuda_snapshot(),
                }
            )
        flexi = _analytic_flexicubes_probe(args.flexicubes_root.resolve(), cfg)

    report = {
        "schema": "RealSaS.IRISVNextA100EngineeringPreflight.v1",
        "device": {
            "name": device_name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "model": {
            "parameter_count": int(parameter_count),
            "trainable_parameter_count": int(trainable_parameter_count),
            "config": cfg.__dict__,
        },
        "cohort": {
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_value_sha256": manifest["manifest_value_sha256"],
            "preparation_receipt_sha256": sha256_file(cohort_root / "PREPARATION_RECEIPT.json"),
            "subject_count": 8,
            "topology_coordinate_bins": int(manifest["topology_coordinate_bins"]),
        },
        "geometry_optimizer_probe": {
            "steps": step_rows,
            "batch_subjects": 8,
            "shared_scene_encodes_per_step": 1,
            "subject_serial_model_forward": False,
            "optimizer_steps_per_step": 1,
        },
        "flexicubes_analytic_probe": flexi,
        "preflight_checkpoint_reusable": False,
        "legacy_5400_checkpoint_reused": False,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
        "status": "PASS",
    }
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(
        "VNEXT_A100_ENGINEERING_PREFLIGHT_PASS",
        f"device={device_name}",
        f"parameters={parameter_count}",
        f"report_sha256={sha256_file(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
