from __future__ import annotations

"""Fail-closed Mage FIT1 runner for the reference-strength Geppetto arm.

All inputs are local paths. Download/mount orchestration belongs to the notebook,
not to the scientific runner.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_loss_v1 import (
    GeppettoReferenceStrengthLossConfigV1,
    prepare_reference_strength_target_v1,
    reference_strength_loss_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    GeppettoReferenceStrengthNoLearnedSlotV1,
    assert_no_learned_view_slot_identity_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    build_mechanical_core_target_v1,
    target_content_sha256_v1,
    world_heads_from_rest_world_source_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)


SCHEMA = "RealSaS.GeppettoReferenceStrengthFIT1.v1"
SEED = 20260907
MAX_STEPS = 16384
CHECK_EVERY = 64
TERMINAL_CHECKS = 48
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RESOURCE_STEP_LIMIT = 128  # experiment execution budget only; not product K
DIFFUSION_SEEDS = (11, 23, 47, 89)
TARGET_COUNT_EXPECTED_FOR_PINNED_FIT = 22
TARGET_HASH = "0b5a25c877116de60b710b7bb2a7848f30988e1622e2eda8084cad21c8ca23c9"
ZERO_SURFACE_SHA = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
NORMALIZED_CORPUS_SHA = "528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f"
IRIS_CHECKPOINT_SHA = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
SOURCE_RUN_ID = "20260904T220929Z"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def _load_cameras(paths: list[Path]) -> tuple[dict, ...]:
    cameras = []
    for path in paths:
        cameras.append(json.loads(path.read_text()))
    cameras.sort(key=lambda x: int(x["view_index"]))
    if [int(c["view_index"]) for c in cameras] != list(range(8)):
        raise RuntimeError("camera set must contain exact view indices 0..7")
    if [int(c.get("yaw_deg", -999)) for c in cameras] != [0, 45, 90, 135, 180, 225, 270, 315]:
        raise RuntimeError("canonical 8-view yaw contract drift")
    resolution = {int(c["resolution"]) for c in cameras}
    if resolution != {1024}:
        raise RuntimeError(f"camera resolution drift:{resolution}")
    centers = np.asarray([c["center"] for c in cameras], dtype=np.float64)
    halves = np.asarray([c["half_extent"] for c in cameras], dtype=np.float64)
    if not np.allclose(centers, centers[0], atol=1e-10, rtol=0.0):
        raise RuntimeError("camera center drift")
    if not np.allclose(halves, halves[0], atol=1e-10, rtol=0.0):
        raise RuntimeError("camera half-extent drift")
    return tuple(cameras)


def _build_surface(zero_path: Path, cameras: tuple[dict, ...]):
    if _sha(zero_path) != ZERO_SURFACE_SHA:
        raise RuntimeError("pinned zero-surface SHA mismatch")
    with np.load(zero_path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(f"zero-surface payload drift:{z.files}")
        world = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        hints = np.asarray(z["normals"], dtype=np.float64)
    center = np.asarray(cameras[0]["center"], dtype=np.float64)
    half = float(cameras[0]["half_extent"])
    vertices_normalized = (world - center[None, :]) / half
    surface = rigging_surface_from_scene_first_zero_mesh_v1(
        vertices_normalized,
        faces,
        hints,
        cameras,
        normalization_center=center,
        normalization_half_extent=half,
        authority_label="IRIS_SCENE_FIRST_SIGNED_V3_PROMOTED_MAGE_FIT",
        source_run_id=SOURCE_RUN_ID,
        source_checkpoint_sha256=IRIS_CHECKPOINT_SHA,
        source_zero_surface_sha256=ZERO_SURFACE_SHA,
        target_nodes=1024,
        normal_k=64,
        visibility_depth_tolerance_norm=0.02,
        metadata={
            "camera_contract": "CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG",
            "teacher_truth_used": False,
            "geppetto_reference_strength_fit1": True,
        },
    )
    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if tensor.node_count != 950:
        raise RuntimeError(f"known Mage GSA node witness drift:{tensor.node_count}")
    if tensor.edge_count != 2813:
        raise RuntimeError(f"known Mage GSA topology witness drift:{tensor.edge_count}")
    if not tensor.raster_valid.any() or not tensor.support.any():
        raise RuntimeError("production raster/support evidence unexpectedly empty")
    return surface, tensor


def _build_target(corpus_path: Path, tensor):
    if _sha(corpus_path) != NORMALIZED_CORPUS_SHA:
        raise RuntimeError("pinned normalized corpus SHA mismatch")
    with np.load(corpus_path, allow_pickle=False) as z:
        required = {"parents", "deform_mask", "skin", "rest_world_source"}
        if not required.issubset(z.files):
            raise RuntimeError(f"target corpus fields missing:{sorted(required-set(z.files))}")
        heads_world = world_heads_from_rest_world_source_v1(z["rest_world_source"])
        target = build_mechanical_core_target_v1(
            parents=z["parents"],
            deform_mask=z["deform_mask"],
            skin=z["skin"],
            bone_heads_world=heads_world,
        )
    if target.count != TARGET_COUNT_EXPECTED_FOR_PINNED_FIT:
        raise RuntimeError(f"pinned Mage target count drift:{target.count}")
    got = target_content_sha256_v1(target)
    if got != TARGET_HASH:
        raise RuntimeError(f"pinned Mage target hash drift:{got}")
    prepared = prepare_reference_strength_target_v1(
        tensor, target, support_target_k=8
    )
    return target, prepared


def _matched_metrics(pred_world: np.ndarray, target_world: np.ndarray, scale: float):
    pred = np.asarray(pred_world, dtype=np.float64)
    truth = np.asarray(target_world, dtype=np.float64)
    if pred.shape != truth.shape:
        return None
    d = np.linalg.norm(pred[:, None, :] - truth[None, :, :], axis=-1) / float(scale)
    r, c = linear_sum_assignment(d)
    dist = d[r, c]
    return {
        "pred_rows": r.astype(int),
        "target_rows": c.astype(int),
        "mae_norm": float(dist.mean()),
        "p95_norm": float(np.quantile(dist, 0.95)),
    }


def _qualified_topology_metrics(qualified, target, matching) -> dict:
    qj = list(qualified.joints)
    target_parent = np.asarray(target.parent_indices, dtype=np.int64)
    target_root = np.asarray(target.root_mask, dtype=bool)
    pred_to_target = {
        int(pr): int(tr)
        for pr, tr in zip(matching["pred_rows"].tolist(), matching["target_rows"].tolist())
    }
    canonical_to_pred = {str(j.canonical_joint_id): i for i, j in enumerate(qj)}
    root_ok = 0
    parent_ok = 0
    parent_total = 0
    illegal = 0
    unsupported = 0
    for pred_i, joint in enumerate(qj):
        tchild = pred_to_target[pred_i]
        proot = joint.parent_canonical_id is None
        root_ok += int(proot == bool(target_root[tchild]))
        unsupported += int(len(joint.support_surface_ids) == 0)
        if joint.parent_canonical_id is not None:
            ppred = canonical_to_pred.get(str(joint.parent_canonical_id))
            if ppred is None:
                illegal += 1
            else:
                tparent_pred = pred_to_target[ppred]
                parent_total += 1
                parent_ok += int(tparent_pred == int(target_parent[tchild]))
        elif int(target_parent[tchild]) >= 0:
            parent_total += 1
    return {
        "root_accuracy": float(root_ok / max(len(qj), 1)),
        "parent_accuracy": float(parent_ok / max(parent_total, 1)),
        "illegal_parent_count": int(illegal),
        "unsupported_joint_count": int(unsupported),
        "deform_root_count": int(len(qualified.deform_root_ids)),
    }


def _evaluate_seed(model, surface, tensor, target, seed: int) -> dict:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    proposal = model.propose(
        tensor,
        resource_step_limit=min(RESOURCE_STEP_LIMIT, tensor.node_count),
        generator=gen,
    )
    count = len(proposal.joints)
    report = {
        "seed": int(seed),
        "generated_count": int(count),
        "target_count": int(target.count),
        "teacher_feedback_used": False,
    }
    if count != target.count:
        report.update({"status": "COUNT_FAIL", "pass": False})
        return report
    qualified = qualify_skeleton_v2(surface, proposal)
    pred_world = np.asarray([j.position for j in qualified.joints], dtype=np.float64)
    matching = _matched_metrics(pred_world, target.positions_world, tensor.normalization_scale)
    assert matching is not None
    topo = _qualified_topology_metrics(qualified, target, matching)
    finite = bool(np.isfinite(pred_world).all())
    passed = bool(
        matching["p95_norm"] <= 0.05
        and matching["mae_norm"] <= 0.03
        and len(qualified.joints) == target.count
        and topo["deform_root_count"] == 1
        and topo["root_accuracy"] == 1.0
        and topo["parent_accuracy"] == 1.0
        and topo["illegal_parent_count"] == 0
        and topo["unsupported_joint_count"] == 0
        and finite
    )
    report.update({
        "status": "PASS" if passed else "STRUCTURAL_FAIL",
        "pass": passed,
        "matched_mae_norm": matching["mae_norm"],
        "matched_p95_norm": matching["p95_norm"],
        **topo,
        "finite": finite,
        "proposal": proposal.to_dict(),
        "qualified": qualified.to_dict(),
    })
    return report


def _check(model, surface, tensor, target, step: int) -> dict:
    model.eval()
    assert_no_learned_view_slot_identity_v1(model)
    rows = []
    with torch.no_grad():
        for seed in DIFFUSION_SEEDS:
            rows.append(_evaluate_seed(model, surface, tensor, target, seed))
    passed = all(bool(x.get("pass", False)) for x in rows)
    return {
        "step": int(step),
        "pass": passed,
        "diffusion_seed_reports": rows,
        "learned_absolute_view_slot_parameter_count": 0,
    }


def _save_visual(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    best = report["diffusion_seed_reports"][0]
    if "qualified" not in best:
        return
    joints = best["qualified"]["joints"]
    p = {j["canonical_joint_id"]: np.asarray(j["position"], dtype=float) for j in joints}
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for j in joints:
        q = p[j["canonical_joint_id"]]
        ax.scatter([q[0]], [q[1]], [q[2]], s=18)
        parent = j.get("parent_canonical_id")
        if parent is not None and parent in p:
            r = p[parent]
            ax.plot([q[0], r[0]], [q[1], r[1]], [q[2], r[2]])
    ax.set_title("Qualified Geppetto FIT1 skeleton")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    camera_paths = [Path(x) for x in args.cameras]
    cameras = _load_cameras(camera_paths)
    surface, tensor = _build_surface(Path(args.zero_surface), cameras)
    target, prepared = _build_target(Path(args.normalized_corpus), tensor)

    model = GeppettoReferenceStrengthNoLearnedSlotV1().to(device)
    assert_no_learned_view_slot_identity_v1(model)
    loss_cfg = GeppettoReferenceStrengthLossConfigV1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(SEED)
    trace = []
    terminal_streak = 0
    closure_step = None

    teacher_pos = torch.as_tensor(
        prepared.positions_normalized, device=device, dtype=torch.float32
    )
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_surface(
            tensor,
            decode_steps=prepared.count,
            teacher_positions_normalized=teacher_pos,
            generator=train_gen,
        )
        losses = reference_strength_loss_v1(out, prepared, config=loss_cfg)
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()

        if step != 1 and step % CHECK_EVERY:
            continue
        check = _check(model, surface, tensor, target, step)
        check["losses"] = {k: float(v.detach().cpu()) for k, v in losses.items()}
        check["grad_norm"] = grad_norm
        terminal_streak = terminal_streak + 1 if check["pass"] else 0
        check["terminal_streak"] = terminal_streak
        trace.append(check)
        print("GEPPETTO_RS_CHECK=" + json.dumps(check, sort_keys=True), flush=True)
        if terminal_streak >= TERMINAL_CHECKS:
            closure_step = step
            break

    final_check = _check(model, surface, tensor, target, closure_step or MAX_STEPS)
    status = "FIT1_TERMINAL_PASS" if closure_step is not None else "NO_TERMINAL_CLOSURE"
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "GEPPETTO_REFERENCE_STRENGTH_FIT1_CHECKPOINT.pt"
    torch.save({
        "schema": SCHEMA,
        "model": model.state_dict(),
        "config": asdict(model.config),
        "config_hash": model.config.config_hash,
        "repo_head": _repo_head(),
        "target_hash": TARGET_HASH,
        "tensorization_hash": tensor.tensorization_hash,
        "step": closure_step or MAX_STEPS,
    }, checkpoint_path)
    checkpoint_sha = _sha(checkpoint_path)
    result = {
        "schema": SCHEMA,
        "status": status,
        "repo_head": _repo_head(),
        "seed": SEED,
        "architecture_id": model.config.architecture_id,
        "config_hash": model.config.config_hash,
        "loss_config": asdict(loss_cfg),
        "source_run_id": SOURCE_RUN_ID,
        "zero_surface_sha256": ZERO_SURFACE_SHA,
        "normalized_corpus_sha256": NORMALIZED_CORPUS_SHA,
        "target_content_sha256": TARGET_HASH,
        "surface_node_count": tensor.node_count,
        "surface_edge_count": tensor.edge_count,
        "surface_source_hash": tensor.source_surface_hash,
        "tensorization_hash": tensor.tensorization_hash,
        "certificate_hash": tensor.certificate_hash,
        "resource_step_limit_execution_only": RESOURCE_STEP_LIMIT,
        "fixed_product_control_cap": False,
        "learned_absolute_view_slot_parameter_count": 0,
        "teacher_feedback_used": False,
        "closure_step": closure_step,
        "terminal_streak": terminal_streak,
        "required_terminal_checks": TERMINAL_CHECKS,
        "final_check": final_check,
        "trace": trace,
        "checkpoint_sha256": checkpoint_sha,
        "generalization_claim": False,
        "promotion_authorized": False,
    }
    (outdir / "GEPPETTO_REFERENCE_STRENGTH_FIT1_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _save_visual(final_check, outdir / "GEPPETTO_REFERENCE_STRENGTH_FIT1_SKELETON.png")
    print("GEPPETTO_RS_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if status != "FIT1_TERMINAL_PASS":
        raise AssertionError(status)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--normalized-corpus", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
