from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from models.iris.v2.checkpoint_v2 import save_checkpoint_v2
from models.iris.v2.eval_v2 import iris_v2_scientific_metrics
from models.iris.v2.iris_apparatus_v2 import IrisDINOv2SApparatusV2
from models.iris.v2.observation_contract_v2 import ObservationContractV2, camera_from_renderer_json_v2
from models.iris.v2.observation_evidence_emitter_v2 import emit_observation_evidence_v2, EmissionPolicyV2
from models.iris.v2.q_domain_v2 import build_production_observation_ray_lattice_v2
from models.iris.v2.resource_contract_v2 import estimate_apparatus_materialization_v2
from models.iris.v2.train_v2 import iris_v2_loss, train_step_production_v2
from compiler.realsas_compiler_core.substrate.iris_v2 import compile_surface_v2, attach_dtb_nd1_from_evidence
from experiments.first_family_fit_v1.run_first_family_observation_v1 import load_spec, stage_authority


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def domain_to(domain, device):
    kw = {}
    for f in fields(domain):
        value = getattr(domain, f.name)
        kw[f.name] = value.to(device) if torch.is_tensor(value) else value
    return type(domain)(**kw)


def load_observation(observation_root: Path, spec: dict) -> tuple[torch.Tensor, ObservationContractV2, dict]:
    manifest_path = observation_root / "PREFIT_OBSERVATION_MANIFEST_V1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("asset_id") != spec["asset_id"] or manifest.get("candidate_id") != spec["candidate_id"]:
        raise RuntimeError("first-family observation identity drift")
    if manifest.get("raster_authority") != spec["observation"]["raster_authority"]:
        raise RuntimeError("first-family raster authority drift")
    if not manifest.get("exact_eight_views") or len(manifest.get("views", [])) != 8:
        raise RuntimeError("first-family observation must contain exact eight views")

    images = []
    cameras = []
    rgba_hashes = []
    for v in range(8):
        rgba_path = observation_root / f"V{v}" / "RGBA.png"
        camera_path = observation_root / f"V{v}" / "camera.json"
        expected = manifest["views"][v]
        if sha256_file(rgba_path) != expected["rgba_sha256"]:
            raise RuntimeError(f"RGBA SHA drift V{v}")
        if sha256_file(camera_path) != expected["camera_sha256"]:
            raise RuntimeError(f"camera SHA drift V{v}")
        with Image.open(rgba_path) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise RuntimeError(f"RGBA contract drift V{v}")
            images.append(np.asarray(im, dtype=np.uint8).copy())
        cameras.append(camera_from_renderer_json_v2(camera_path, v))
        rgba_hashes.append(expected["rgba_sha256"])

    contract = ObservationContractV2(
        spec["asset_id"],
        tuple(cameras),
        tuple(rgba_hashes),
        raster_authority=spec["observation"]["raster_authority"],
    )
    rgba = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    return rgba, contract, manifest


def build_teacher_from_master_raster(*, authority_root: Path, authority_manifest: dict, contract: ObservationContractV2, anchor_stride: int, depth_bins: int) -> tuple[object, torch.Tensor, torch.Tensor, dict]:
    with np.load(authority_root / "primary_geometry.npz", allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.shape != tuple(authority_manifest["primary_geometry"]["vertices"]):
        raise RuntimeError("canonical vertex shape drift")
    if faces.shape != tuple(authority_manifest["primary_geometry"]["faces"]):
        raise RuntimeError("canonical face shape drift")

    cam = contract.cameras[0]
    depths = cam.depth_for_point(vertices)
    dmin = float(np.min(depths)); dmax = float(np.max(depths))
    margin = max((dmax - dmin) * 0.08, 1e-3)
    depth_values = torch.linspace(dmin - margin, dmax + margin, int(depth_bins), dtype=torch.float32)
    domain = build_production_observation_ray_lattice_v2((contract,), anchor_view_index=0, anchor_stride_px=int(anchor_stride), depth_values=depth_values)

    raster_path = authority_root / "V0" / "raster_authority.npz"
    with np.load(raster_path, allow_pickle=False) as z:
        pix = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
        tri = np.asarray(z["triangle_id"], dtype=np.int64).reshape(-1)
        buv = np.asarray(z["barycentric_uv"], dtype=np.float64)
    order = np.argsort(pix)
    pix_sorted = pix[order]

    q_grid = domain.anchor_grid[0].detach().cpu().numpy()
    xy = cam.grid_to_pixel_center(q_grid)
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    if np.max(np.abs(xy[:, 0] - xi)) > 1e-5 or np.max(np.abs(xy[:, 1] - yi)) > 1e-5:
        raise RuntimeError("production lattice did not map to integer native pixels")
    qpix = yi * 1024 + xi
    pos = np.searchsorted(pix_sorted, qpix)
    in_range = pos < len(pix_sorted)
    hit = np.zeros(len(qpix), dtype=bool)
    hit[in_range] = pix_sorted[pos[in_range]] == qpix[in_range]

    teacher = np.zeros(len(qpix), dtype=np.float32)
    hit_q = np.flatnonzero(hit)
    if len(hit_q):
        rows = order[pos[hit_q]]
        tids = tri[rows]
        uv = buv[rows]
        w = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], axis=1)
        points = (vertices[faces[tids]] * w[:, :, None]).sum(axis=1)
        teacher[hit_q] = np.asarray(cam.depth_for_point(points), dtype=np.float32)

    teacher_depth = torch.from_numpy(teacher).unsqueeze(0)
    teacher_support = torch.from_numpy(hit).unsqueeze(0)
    supported = int(hit.sum())
    if supported < 64:
        raise RuntimeError(f"too few teacher-supported production rays: {supported}")
    spacing = float((depth_values[1] - depth_values[0]).abs()) if len(depth_values) > 1 else float("nan")
    telemetry = {
        "construction_authority": domain.construction_authority,
        "anchor_stride_px": int(anchor_stride),
        "q_count": int(domain.anchor_view.shape[1]),
        "depth_bins": int(depth_bins),
        "depth_min": float(depth_values.min()),
        "depth_max": float(depth_values.max()),
        "depth_spacing": spacing,
        "teacher_supported_q": supported,
        "teacher_supported_fraction_full_frame": float(hit.mean()),
        "teacher_source": "MASTER_V0_VISIBLE_RASTER_AUTHORITY_LABELS_FIXED_CAMERA_ONLY_Q_LATTICE",
        "learner_q_selection_uses_teacher_or_alpha": False,
        "raster_authority_sha256": sha256_file(raster_path),
    }
    return domain, teacher_depth, teacher_support, telemetry


def normalized_metrics(metrics: dict, contract: ObservationContractV2) -> dict:
    scale = float(2.0 * contract.cameras[0].half_extent)
    out = dict(metrics)
    for src, dst in (("coverage_mae", "coverage_mae_norm"), ("coverage_p95_abs", "coverage_p95_norm"), ("coverage_tail_mean", "coverage_tail_mean_norm")):
        if src in metrics:
            out[dst] = float(metrics[src]) / scale
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--observation-root", required=True)
    authority = ap.add_mutually_exclusive_group(required=True)
    authority.add_argument("--master-root")
    authority.add_argument("--authority-zip")
    ap.add_argument("--authority-manifest", required=True)
    ap.add_argument("--dino-source", required=True)
    ap.add_argument("--dino-weight", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--anchor-stride", type=int)
    ap.add_argument("--depth-bins", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--check-every", type=int)
    ap.add_argument("--resource-probe-only", action="store_true")
    ap.add_argument("--historical-p95-norm-goal", type=float)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    spec = load_spec(Path(args.spec).resolve())
    iris_cfg = spec["iris"]
    anchor_stride = int(args.anchor_stride if args.anchor_stride is not None else iris_cfg["anchor_stride_px"])
    depth_bins = int(args.depth_bins if args.depth_bins is not None else iris_cfg["depth_bins"])
    max_steps = int(args.max_steps if args.max_steps is not None else iris_cfg["max_steps"])
    check_every = int(args.check_every if args.check_every is not None else iris_cfg["check_every"])
    historical_p95_norm_goal = float(args.historical_p95_norm_goal if args.historical_p95_norm_goal is not None else iris_cfg["historical_p95_norm_goal"])
    seed = int(args.seed if args.seed is not None else iris_cfg["seed"])

    out_root = Path(args.out_root).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root).resolve(); shutil.rmtree(work_root, ignore_errors=True); work_root.mkdir(parents=True)
    authority_manifest = json.loads(Path(args.authority_manifest).read_text(encoding="utf-8"))
    authority_root, authority_source = stage_authority(zip_path=Path(args.authority_zip).resolve() if args.authority_zip else None, master_root=Path(args.master_root).resolve() if args.master_root else None, work=work_root / "authority_stage", manifest=authority_manifest, spec=spec)

    torch.manual_seed(seed); np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")

    rgba_cpu, contract, observation_manifest = load_observation(Path(args.observation_root).resolve(), spec)
    domain_cpu, teacher_depth_cpu, teacher_support_cpu, teacher_telemetry = build_teacher_from_master_raster(authority_root=authority_root, authority_manifest=authority_manifest, contract=contract, anchor_stride=anchor_stride, depth_bins=depth_bins)
    prereg = {
        "schema": "RealSaS.FirstFamily.IRISFit.v1",
        "family_label": spec["family_label"],
        "asset_id": spec["asset_id"],
        "candidate_id": spec["candidate_id"],
        "observation_contract_hash": contract.contract_hash,
        "observation_manifest_sha256": sha256_file(Path(args.observation_root) / "PREFIT_OBSERVATION_MANIFEST_V1.json"),
        "authority_source": authority_source,
        "teacher": teacher_telemetry,
        "seed": seed,
        "hidden_dim": int(iris_cfg["hidden_dim"]),
        "max_modes": int(iris_cfg["max_modes"]),
        "anchor_stride_px": anchor_stride,
        "depth_bins": depth_bins,
        "max_steps": max_steps,
        "check_every": check_every,
        "fit_gate_note": "single-family fit witness only; not a generalization claim",
        "historical_p95_norm_goal": historical_p95_norm_goal,
    }
    write_json(out_root / "IRIS_FIT_PREREG_V1.json", prereg)

    apparatus = IrisDINOv2SApparatusV2.from_authority_files(Path(args.dino_source), Path(args.dino_weight), device=device, hidden_dim=int(iris_cfg["hidden_dim"]), max_modes=int(iris_cfg["max_modes"]), foundation_view_chunk=1)
    optimizer = torch.optim.AdamW(apparatus.trainable_parameters(), lr=2e-4, weight_decay=1e-4)
    init_meta = save_checkpoint_v2(out_root / "IRIS_STEP000000_INIT.pt", model=apparatus, optimizer=optimizer, step=0, config=prereg, source_contract_hash=apparatus.source_contract_hash)

    estimate = estimate_apparatus_materialization_v2(apparatus, domain_cpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    resource = {"device": torch.cuda.get_device_name(0), "free_bytes_after_dino_load": int(free_bytes), "total_bytes": int(total_bytes), "forward_live_lower_bound_bytes": int(estimate.forward_live_lower_bound_bytes), "forward_live_lower_bound_gib": estimate.forward_live_lower_bound_gib, "fits_forward_lower_bound": bool(estimate.forward_live_lower_bound_bytes <= free_bytes), "note": "lower bound excludes backward, native/foundation feature maps, params, allocator overhead"}
    write_json(out_root / "IRIS_RESOURCE_PREFLIGHT_V1.json", resource)
    if args.resource_probe_only:
        write_json(out_root / "IRIS_FIT_RESULT_V1.json", {"status": "PASS_RESOURCE_PROBE_ONLY" if resource["fits_forward_lower_bound"] else "RESOURCE_BLOCKED_ON_THIS_GPU_EXPECTED", "optimizer_steps": 0, "resource": resource, "initial_checkpoint": init_meta})
        return
    if not resource["fits_forward_lower_bound"]:
        write_json(out_root / "IRIS_FIT_RESULT_V1.json", {"status": "RESOURCE_BLOCKED_BEFORE_ZERO_STEP", "optimizer_steps": 0, "resource": resource, "initial_checkpoint": init_meta})
        raise RuntimeError("IRIS current materialization is provably too large for this GPU; move identical run to A100-class GPU")

    domain = domain_to(domain_cpu, device)
    rgba = rgba_cpu.to(device, non_blocking=True)
    teacher_depth = teacher_depth_cpu.to(device)
    teacher_support = teacher_support_cpu.to(device)

    optimizer.zero_grad(set_to_none=True)
    apparatus.train()
    output = apparatus(rgba, domain)
    losses = iris_v2_loss(output, domain, teacher_depth, teacher_support)
    losses["total"].backward()
    bad_grad = [n for n, p in apparatus.learner.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    if bad_grad:
        raise RuntimeError(f"IRIS_ZERO_STEP_GRAD_FAIL:{bad_grad[:10]}")
    zero_metrics = normalized_metrics(iris_v2_scientific_metrics(output, domain, teacher_depth, teacher_support), contract)
    write_json(out_root / "IRIS_ZERO_STEP_METRICS_V1.json", {"losses": {k: float(v.detach().cpu()) for k, v in losses.items()}, "metrics": zero_metrics})
    save_checkpoint_v2(out_root / "IRIS_STEP000000_ZERO_STEP.pt", model=apparatus, optimizer=optimizer, step=0, config=prereg, source_contract_hash=apparatus.source_contract_hash)
    optimizer.zero_grad(set_to_none=True)
    del output, losses
    torch.cuda.empty_cache()

    history = []
    streak = 0
    final_output = None
    for step in range(1, max_steps + 1):
        train_metrics = train_step_production_v2(apparatus, optimizer, {"images": rgba, "domain": domain, "teacher_depth": teacher_depth, "teacher_support": teacher_support})
        if step == 1 or step % check_every == 0 or step == max_steps:
            apparatus.eval()
            with torch.no_grad():
                final_output = apparatus(rgba, domain)
                scientific = normalized_metrics(iris_v2_scientific_metrics(final_output, domain, teacher_depth, teacher_support), contract)
            record = {"step": step, "train": train_metrics, "eval": scientific}
            history.append(record)
            write_json(out_root / "IRIS_HISTORY_V1.json", history)
            save_checkpoint_v2(out_root / "IRIS_CHECKPOINT_LATEST.pt", model=apparatus, optimizer=optimizer, step=step, config=prereg, source_contract_hash=apparatus.source_contract_hash)
            streak = streak + 1 if scientific["coverage_p95_norm"] <= historical_p95_norm_goal else 0
            print(json.dumps({"IRIS": record, "fit_streak": streak}, sort_keys=True), flush=True)
            if streak >= 3:
                break

    if final_output is None:
        raise RuntimeError("IRIS fit produced no evaluation checkpoint")
    if streak < 3:
        write_json(out_root / "IRIS_FIT_RESULT_V1.json", {"status": "IRIS_FIT_FAIL", "optimizer_steps": history[-1]["step"], "last": history[-1], "history": history})
        raise RuntimeError(f"IRIS_FIT_FAIL:{history[-1]}")

    evidence = emit_observation_evidence_v2((contract,), domain, final_output.modes, final_output.refined_depth, final_output.depth_output, policy=EmissionPolicyV2(0.05, True))[0]
    surface = compile_surface_v2(evidence)
    surface = attach_dtb_nd1_from_evidence(evidence, surface)
    torch.save({"schema": "RealSaS.FirstFamily.IRISSurfacePickle.v1", "family_label": spec["family_label"], "asset_id": spec["asset_id"], "surface": surface, "evidence": evidence, "source_contract_hash": apparatus.source_contract_hash}, out_root / "IRIS_SURFACE_AND_EVIDENCE.pt")
    result = {"status": "PASS_IRIS_FIT_AND_S", "family_label": spec["family_label"], "asset_id": spec["asset_id"], "optimizer_steps": history[-1]["step"], "last": history[-1], "surface_nodes": len(surface.surface_nodes), "surface_relations": len(surface.local_relations), "source_contract_hash": apparatus.source_contract_hash, "final_checkpoint_sha256": sha256_file(out_root / "IRIS_CHECKPOINT_LATEST.pt")}
    write_json(out_root / "IRIS_FIT_RESULT_V1.json", result)
    if result["surface_nodes"] < 64 or result["surface_relations"] < 1:
        raise RuntimeError(f"IRIS_SURFACE_FAIL:{result}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
