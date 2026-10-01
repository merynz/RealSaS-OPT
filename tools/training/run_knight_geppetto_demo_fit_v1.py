from __future__ import annotations

"""Knight DEMO_WITNESS Geppetto fresh-fit runner.

Consumes only the qualified Stage15 RiggingSurfaceIR as learner input.
Teacher target is training/evaluation-only. Free-running evaluation receives no
teacher target. The runner emits the exact Stage27 external-fit contract:
checkpoint, result, SkeletonProposalIR and ModelFitExecutionReceipt.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    model_fit_preregistration_from_dict,
    rigging_surface_qualification_from_dict,
)
from compiler.realsas_compiler_core.rig import qualify_skeleton
from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_loss_v1 import (
    GeppettoReferenceStrengthLossConfigV1,
    prepare_reference_strength_target_v1,
    reference_strength_loss_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    MechanicalCoreTargetV1,
    target_content_sha256_v1,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    ARCHITECTURE_ID,
    GeppettoReferenceStrengthNoLearnedSlotV1,
    assert_no_learned_view_slot_identity_v1,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)


SCHEMA = "RealSaS.KnightGeppettoDemoFit.v1"
SEED = 20260907
MAX_STEPS = 16384
CHECK_EVERY = 64
TERMINAL_CHECKS = 48
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RESOURCE_STEP_LIMIT = 128
DIFFUSION_SEEDS = (11, 23, 47, 89)
SELECTED_PROPOSAL_SEED = DIFFUSION_SEEDS[0]
MAX_MATCHED_MAE_NORM = 0.03
MAX_MATCHED_P95_NORM = 0.05


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_environment() -> dict:
    cuda_available = bool(torch.cuda.is_available())
    gpu = None
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": str(props.name),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [
                int(props.major),
                int(props.minor),
            ],
        }
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "") or ""),
        "torch_cuda_runtime": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": (
            None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "cuda_available": cuda_available,
        "gpu0": gpu,
        "sys_executable": str(sys.executable),
    }


def _load_target(npz_path: Path, report_path: Path) -> tuple[MechanicalCoreTargetV1, dict]:
    report = _load_json(report_path)
    if report.get("schema") != "RealSaS.KnightGeppettoMechanicalCoreTarget.v1":
        raise RuntimeError("GEPPETTO_KNIGHT_TARGET_REPORT_SCHEMA_DRIFT")
    if report.get("status") != "PASS":
        raise RuntimeError("GEPPETTO_KNIGHT_TARGET_REPORT_NOT_PASS")
    if report.get("product_authority_claimed") is not False:
        raise RuntimeError("GEPPETTO_KNIGHT_TARGET_PRODUCT_AUTHORITY_FORBIDDEN")
    if str(report["output_npz"]["sha256"]) != _sha(npz_path):
        raise RuntimeError("GEPPETTO_KNIGHT_TARGET_NPZ_SHA_DRIFT")
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "positions_world",
            "parent_indices",
            "root_mask",
            "source_indices_provenance_only",
            "skin_mass",
        }
        if not required.issubset(set(data.files)):
            raise RuntimeError("GEPPETTO_KNIGHT_TARGET_ARRAYS_MISSING")
        target = MechanicalCoreTargetV1(
            positions_world=np.asarray(data["positions_world"], dtype=np.float32),
            parent_indices=np.asarray(data["parent_indices"], dtype=np.int64),
            root_mask=np.asarray(data["root_mask"], dtype=np.uint8).astype(bool),
            source_indices_provenance_only=np.asarray(
                data["source_indices_provenance_only"], dtype=np.int64
            ),
            skin_mass=np.asarray(data["skin_mass"], dtype=np.float64),
        )
    got = target_content_sha256_v1(target)
    if got != str(report["target_content_sha256"]):
        raise RuntimeError("GEPPETTO_KNIGHT_TARGET_CONTENT_HASH_DRIFT")
    if int(report["target_count"]) != target.count:
        raise RuntimeError("GEPPETTO_KNIGHT_TARGET_COUNT_DRIFT")
    if int(np.count_nonzero(target.root_mask)) != 1:
        raise RuntimeError("GEPPETTO_KNIGHT_FROZEN_SINGLE_ROOT_PREFLIGHT_FAILED")
    return target, report


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
        for pr, tr in zip(
            matching["pred_rows"].tolist(),
            matching["target_rows"].tolist(),
        )
    }
    canonical_to_pred = {
        str(j.canonical_joint_id): i for i, j in enumerate(qj)
    }
    root_ok = 0
    parent_ok = 0
    parent_total = 0
    illegal = 0
    unsupported = 0
    for pred_i, joint in enumerate(qj):
        tchild = pred_to_target[pred_i]
        predicted_root = joint.parent_canonical_id is None
        root_ok += int(predicted_root == bool(target_root[tchild]))
        unsupported += int(len(joint.support_surface_ids) == 0)
        if joint.parent_canonical_id is not None:
            parent_pred = canonical_to_pred.get(str(joint.parent_canonical_id))
            if parent_pred is None:
                illegal += 1
            else:
                target_parent_pred = pred_to_target[parent_pred]
                parent_total += 1
                parent_ok += int(
                    target_parent_pred == int(target_parent[tchild])
                )
        elif int(target_parent[tchild]) >= 0:
            parent_total += 1
    return {
        "root_accuracy": float(root_ok / max(len(qj), 1)),
        "parent_accuracy": float(parent_ok / max(parent_total, 1)),
        "illegal_parent_count": int(illegal),
        "unsupported_joint_count": int(unsupported),
        "canonical_root_count": int(
            sum(j.parent_canonical_id is None for j in qj)
        ),
    }


def _evaluate_seed(
    model,
    surface,
    tensor,
    target,
    seed: int,
    *,
    include_payload: bool = False,
) -> dict:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    proposal = model.propose(
        tensor,
        resource_step_limit=min(RESOURCE_STEP_LIMIT, tensor.node_count),
        generator=generator,
    )
    count = len(proposal.joints)
    report = {
        "seed": int(seed),
        "generated_count": int(count),
        "target_count": int(target.count),
        "teacher_feedback_used": False,
        "teacher_inference_inputs_used": False,
    }
    if count != target.count:
        report.update({"status": "COUNT_FAIL", "pass": False})
        return report

    try:
        qualified = qualify_skeleton(surface, proposal, run_ilp_shadow=False)
    except Exception as exc:
        report.update(
            {
                "status": "COMPILER_QUALIFICATION_FAIL",
                "pass": False,
                "qualification_error": f"{type(exc).__name__}:{exc}",
            }
        )
        if include_payload:
            report["proposal"] = proposal.to_dict()
        return report

    pred_world = np.asarray(
        [joint.position for joint in qualified.joints], dtype=np.float64
    )
    matching = _matched_metrics(
        pred_world,
        target.positions_world,
        tensor.normalization_scale,
    )
    if matching is None:
        report.update({"status": "MATCH_SHAPE_FAIL", "pass": False})
        return report
    topology = _qualified_topology_metrics(qualified, target, matching)
    finite = bool(np.isfinite(pred_world).all())
    passed = bool(
        matching["p95_norm"] <= MAX_MATCHED_P95_NORM
        and matching["mae_norm"] <= MAX_MATCHED_MAE_NORM
        and len(qualified.joints) == target.count
        and topology["canonical_root_count"] == 1
        and topology["root_accuracy"] == 1.0
        and topology["parent_accuracy"] == 1.0
        and topology["illegal_parent_count"] == 0
        and topology["unsupported_joint_count"] == 0
        and finite
    )
    report.update(
        {
            "status": "PASS" if passed else "STRUCTURAL_FAIL",
            "pass": passed,
            "matched_mae_norm": matching["mae_norm"],
            "matched_p95_norm": matching["p95_norm"],
            **topology,
            "finite": finite,
        }
    )
    if include_payload:
        report["proposal"] = proposal.to_dict()
        report["qualified"] = qualified.to_dict()
    return report


def _check(
    model,
    surface,
    tensor,
    target,
    step: int,
    *,
    include_payload: bool = False,
) -> dict:
    model.eval()
    assert_no_learned_view_slot_identity_v1(model)
    rows = []
    with torch.no_grad():
        for seed in DIFFUSION_SEEDS:
            rows.append(
                _evaluate_seed(
                    model,
                    surface,
                    tensor,
                    target,
                    seed,
                    include_payload=include_payload,
                )
            )
    return {
        "step": int(step),
        "pass": all(bool(row.get("pass", False)) for row in rows),
        "diffusion_seed_reports": rows,
        "learned_absolute_view_slot_parameter_count": 0,
        "teacher_feedback_used": False,
        "teacher_inference_inputs_used": False,
    }


def _render_final_skeleton_evidence(
    final_check: dict,
    *,
    camera_set_path: Path,
    observation_dir: Path,
    outdir: Path,
) -> dict:
    rows = {
        int(row["seed"]): row
        for row in final_check.get("diffusion_seed_reports") or ()
    }
    selected = rows.get(SELECTED_PROPOSAL_SEED)
    if selected is None or selected.get("pass") is not True:
        raise RuntimeError("GEPPETTO_VISUAL_EVIDENCE_SELECTED_SEED_NOT_PASS")
    qualified = dict(selected.get("qualified") or {})
    joints = list(qualified.get("joints") or ())
    if not joints:
        raise RuntimeError("GEPPETTO_VISUAL_EVIDENCE_SKELETON_EMPTY")

    camera_set = qualified_camera_set_from_dict(_load_json(camera_set_path))
    joint_ids = [str(row["canonical_joint_id"]) for row in joints]
    index_by_id = {jid: i for i, jid in enumerate(joint_ids)}
    if len(index_by_id) != len(joints):
        raise RuntimeError("GEPPETTO_VISUAL_EVIDENCE_DUPLICATE_JOINT_ID")
    world = np.asarray([row["position"] for row in joints], dtype=np.float64)
    if world.shape != (len(joints), 3) or not np.isfinite(world).all():
        raise RuntimeError("GEPPETTO_VISUAL_EVIDENCE_JOINT_POSITION_INVALID")

    visual_dir = outdir / "visual_evidence"
    visual_dir.mkdir(parents=True, exist_ok=True)
    views = []
    rendered = []
    for camera in sorted(camera_set.cameras, key=lambda row: int(row.view_index)):
        vi = int(camera.view_index)
        source = observation_dir / f"V{vi}.png"
        if not source.is_file():
            raise RuntimeError(f"GEPPETTO_VISUAL_EVIDENCE_SOURCE_MISSING:V{vi}")
        image = Image.open(source).convert("RGB")
        projected = np.asarray(project_points_xyz_v3(world, camera), dtype=np.float64)
        if projected.shape != (len(joints), 3) or not np.isfinite(projected).all():
            raise RuntimeError(f"GEPPETTO_VISUAL_EVIDENCE_PROJECTION_INVALID:V{vi}")
        if np.any(projected[:, 2] <= 0.0):
            raise RuntimeError(f"GEPPETTO_VISUAL_EVIDENCE_BEHIND_CAMERA:V{vi}")
        draw = ImageDraw.Draw(image)
        xy = projected[:, :2]
        for child, row in enumerate(joints):
            parent_id = row.get("parent_canonical_id")
            if parent_id is None:
                continue
            parent = index_by_id.get(str(parent_id))
            if parent is None:
                raise RuntimeError("GEPPETTO_VISUAL_EVIDENCE_PARENT_ID_DRIFT")
            draw.line(
                [
                    (float(xy[parent, 0]), float(xy[parent, 1])),
                    (float(xy[child, 0]), float(xy[child, 1])),
                ],
                fill=(255, 80, 80),
                width=4,
            )
        for i, row in enumerate(joints):
            x, y = map(float, xy[i])
            radius = 6
            fill = (80, 220, 120) if row.get("parent_canonical_id") is None else (255, 230, 80)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
                outline=(20, 20, 20),
                width=2,
            )
        target = visual_dir / f"V{vi}_QUALIFIED_SKELETON.png"
        image.save(target)
        rendered.append(image)
        views.append(
            {
                "view_index": vi,
                "path": str(target),
                "sha256": _sha(target),
                "source_raster_sha256": _sha(source),
            }
        )

    width = max(image.width for image in rendered)
    height = max(image.height for image in rendered)
    sheet = Image.new("RGB", (width * 2, height * 4), (0, 0, 0))
    for index, image in enumerate(rendered):
        sheet.paste(image, ((index % 2) * width, (index // 2) * height))
    sheet_path = visual_dir / "KNIGHT_GEPPETTO_8VIEW_CONTACT_SHEET.png"
    sheet.save(sheet_path)
    return {
        "schema": "RealSaS.KnightGeppettoVisualEvidence.v1",
        "selected_proposal_seed": SELECTED_PROPOSAL_SEED,
        "gate_role": "HUMAN_REVIEW_EVIDENCE_ONLY__NOT_CHECKPOINT_SELECTION",
        "views": views,
        "contact_sheet": {
            "path": str(sheet_path),
            "sha256": _sha(sheet_path),
        },
    }


def _selected_proposal(final_check: dict) -> dict:
    rows = {
        int(row["seed"]): row
        for row in final_check.get("diffusion_seed_reports") or ()
    }
    row = rows.get(SELECTED_PROPOSAL_SEED)
    if row is None or row.get("pass") is not True or "proposal" not in row:
        raise RuntimeError("GEPPETTO_SELECTED_PREDECLARED_SEED_NOT_PASS")
    return dict(row["proposal"])


def run(args) -> dict:
    surface_path = Path(args.surface_json).expanduser().resolve()
    qualification_path = Path(args.surface_qualification_json).expanduser().resolve()
    target_path = Path(args.teacher_target_npz).expanduser().resolve()
    target_report_path = Path(args.teacher_target_report).expanduser().resolve()
    prereg_path = Path(args.preregistration_ir).expanduser().resolve()
    experiment_prereg_path = Path(args.experiment_prereg).expanduser().resolve()
    model_source_path = Path(args.model_source).expanduser().resolve()
    camera_set_path = Path(args.camera_set_json).expanduser().resolve()
    observation_dir = Path(args.observation_dir).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    prereg = model_fit_preregistration_from_dict(_load_json(prereg_path))
    if prereg.lane != "GEPPETTO":
        raise RuntimeError("GEPPETTO_PREREG_LANE_DRIFT")
    if prereg.architecture_id != ARCHITECTURE_ID:
        raise RuntimeError("GEPPETTO_PREREG_ARCHITECTURE_DRIFT")
    if int(prereg.random_seed) != SEED:
        raise RuntimeError("GEPPETTO_PREREG_SEED_DRIFT")
    if _sha(model_source_path) != prereg.model_source_sha256:
        raise RuntimeError("GEPPETTO_MODEL_SOURCE_SHA_DRIFT")
    if set(prereg.expected_output_contract) != {
        "CHECKPOINT",
        "RESULT",
        "SKELETON_PROPOSAL",
    }:
        raise RuntimeError("GEPPETTO_PREREG_OUTPUT_CONTRACT_DRIFT")

    surface = rigging_surface_from_dict(_load_json(surface_path))
    qualification = rigging_surface_qualification_from_dict(
        _load_json(qualification_path)
    )
    if qualification.surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("GEPPETTO_SURFACE_QUALIFICATION_BINDING_DRIFT")
    upstream = dict(prereg.upstream_bindings)
    expected_upstream = {
        "rigging_surface": surface.geometry_lineage_hash,
        "rigging_surface_qualification": qualification.qualification_hash,
    }
    if upstream != expected_upstream:
        raise RuntimeError("GEPPETTO_PREREG_UPSTREAM_BINDING_DRIFT")

    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if tensor.tensorization_hash != qualification.tensorization_hash:
        raise RuntimeError("GEPPETTO_TENSORIZATION_HASH_DRIFT")
    if tensor.certificate_hash != qualification.certificate_hash:
        raise RuntimeError("GEPPETTO_TENSORIZATION_CERTIFICATE_DRIFT")
    target, target_report = _load_target(target_path, target_report_path)
    experiment_prereg = _load_json(experiment_prereg_path)
    if experiment_prereg.get("schema") != "RealSaS.KnightGeppettoDemoFitPreregistration.v1":
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_SCHEMA_DRIFT")
    if experiment_prereg.get("status") != "FROZEN_BEFORE_OPTIMIZER_STEP_1":
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_NOT_FROZEN")
    if experiment_prereg.get("product_authority_claimed") is not False:
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_PRODUCT_AUTHORITY_FORBIDDEN")
    bindings = dict(experiment_prereg.get("bindings") or {})
    expected_bindings = {
        "rigging_surface_lineage_hash": surface.geometry_lineage_hash,
        "rigging_surface_qualification_hash": qualification.qualification_hash,
        "tensorization_hash": tensor.tensorization_hash,
        "certificate_hash": tensor.certificate_hash,
        "teacher_target_content_sha256": target_content_sha256_v1(target),
    }
    for key, expected in expected_bindings.items():
        if str(bindings.get(key) or "") != str(expected):
            raise RuntimeError(f"GEPPETTO_EXPERIMENT_PREREG_BINDING_DRIFT:{key}")
    apparatus = dict(experiment_prereg.get("apparatus") or {})
    expected_apparatus = {
        "architecture_id": ARCHITECTURE_ID,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "check_every": CHECK_EVERY,
        "terminal_checks": TERMINAL_CHECKS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "resource_step_limit_execution_only": RESOURCE_STEP_LIMIT,
        "selected_proposal_seed_predeclared": SELECTED_PROPOSAL_SEED,
        "matched_mae_norm_max": MAX_MATCHED_MAE_NORM,
        "matched_p95_norm_max": MAX_MATCHED_P95_NORM,
    }
    for key, expected in expected_apparatus.items():
        actual = apparatus.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-15:
                raise RuntimeError(f"GEPPETTO_EXPERIMENT_PREREG_APPARATUS_DRIFT:{key}")
        elif actual != expected:
            raise RuntimeError(f"GEPPETTO_EXPERIMENT_PREREG_APPARATUS_DRIFT:{key}")
    if tuple(map(int, apparatus.get("diffusion_eval_seeds") or ())) != DIFFUSION_SEEDS:
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_DIFFUSION_SEEDS_DRIFT")
    if apparatus.get("fresh_from_scratch") is not True:
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_FRESH_INIT_REQUIRED")
    if apparatus.get("historical_checkpoint_loaded") is not False:
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_HISTORICAL_CHECKPOINT_FORBIDDEN")
    if apparatus.get("teacher_feedback_during_free_running_inference") is not False:
        raise RuntimeError("GEPPETTO_EXPERIMENT_PREREG_TEACHER_FEEDBACK_FORBIDDEN")
    experiment_prereg_sha = _sha(experiment_prereg_path)

    prepared = prepare_reference_strength_target_v1(
        tensor,
        target,
        support_target_k=8,
    )

    preflight = {
        "schema": SCHEMA,
        "status": "PASS_PREFLIGHT",
        "architecture_id": ARCHITECTURE_ID,
        "preregistration_binding_hash": prereg.preregistration_hash,
        "model_source_sha256": prereg.model_source_sha256,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "surface_qualification_hash": qualification.qualification_hash,
        "surface_node_count": tensor.node_count,
        "surface_edge_count": tensor.edge_count,
        "tensorization_hash": tensor.tensorization_hash,
        "certificate_hash": tensor.certificate_hash,
        "target_content_sha256": target_content_sha256_v1(target),
        "target_count": target.count,
        "experiment_preregistration_sha256": experiment_prereg_sha,
        "target_root_count": int(np.count_nonzero(target.root_mask)),
        "teacher_target_report_sha256": _sha(target_report_path),
        "teacher_supervision_during_training": True,
        "teacher_feedback_during_free_running_inference": False,
        "historical_checkpoint_loaded": False,
        "fresh_from_scratch": True,
        "max_steps": MAX_STEPS,
        "check_every": CHECK_EVERY,
        "terminal_checks": TERMINAL_CHECKS,
        "diffusion_eval_seeds": list(DIFFUSION_SEEDS),
        "selected_proposal_seed_predeclared": SELECTED_PROPOSAL_SEED,
        "thresholds": {
            "matched_mae_norm_max": MAX_MATCHED_MAE_NORM,
            "matched_p95_norm_max": MAX_MATCHED_P95_NORM,
            "root_accuracy_required": 1.0,
            "parent_accuracy_required": 1.0,
            "all_diffusion_seeds_must_pass": True,
        },
        "runtime_environment": _runtime_environment(),
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    (outdir / "GEPPETTO_KNIGHT_PREFLIGHT.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        print("GEPPETTO_KNIGHT_PREFLIGHT_PASS")
        print(json.dumps(preflight, sort_keys=True))
        return preflight

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_GEPPETTO_KNIGHT_FIT")
    started = time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")

    model = GeppettoReferenceStrengthNoLearnedSlotV1().to(device)
    assert_no_learned_view_slot_identity_v1(model)
    loss_cfg = GeppettoReferenceStrengthLossConfigV1()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(SEED)
    teacher_pos = torch.as_tensor(
        prepared.positions_normalized,
        device=device,
        dtype=torch.float32,
    )

    trace = []
    terminal_streak = 0
    closure_step = None
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_surface(
            tensor,
            decode_steps=prepared.count,
            teacher_positions_normalized=teacher_pos,
            generator=train_generator,
        )
        losses = reference_strength_loss_v1(out, prepared, config=loss_cfg)
        losses["total"].backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        )
        if not math.isfinite(grad_norm):
            raise FloatingPointError("GEPPETTO_KNIGHT_NONFINITE_GRADIENT")
        optimizer.step()

        if step != 1 and step % CHECK_EVERY:
            continue
        check = _check(model, surface, tensor, target, step)
        check["losses"] = {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        }
        check["grad_norm"] = grad_norm
        terminal_streak = terminal_streak + 1 if check["pass"] else 0
        check["terminal_streak"] = terminal_streak
        trace.append(check)
        print(
            "GEPPETTO_KNIGHT_CHECK=" + json.dumps(check, sort_keys=True),
            flush=True,
        )
        if terminal_streak >= TERMINAL_CHECKS:
            closure_step = step
            break

    final_step = closure_step or MAX_STEPS
    final_check = _check(
        model,
        surface,
        tensor,
        target,
        final_step,
        include_payload=True,
    )
    if closure_step is None or final_check.get("pass") is not True:
        result = {
            "schema": SCHEMA,
            "status": "NO_TERMINAL_CLOSURE",
            "preregistration_binding_hash": prereg.preregistration_hash,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "target_content_sha256": target_content_sha256_v1(target),
            "final_step": final_step,
            "terminal_streak": terminal_streak,
            "required_terminal_checks": TERMINAL_CHECKS,
            "final_check": final_check,
            "trace": trace,
            "teacher_inference_inputs_used": False,
            "product_authority_claimed": False,
            "generalization_claimed": False,
        }
        result_path = outdir / "GEPPETTO_KNIGHT_RESULT.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise AssertionError("GEPPETTO_KNIGHT_NO_TERMINAL_CLOSURE")

    visual_evidence = _render_final_skeleton_evidence(
        final_check,
        camera_set_path=camera_set_path,
        observation_dir=observation_dir,
        outdir=outdir,
    )

    checkpoint_path = outdir / "GEPPETTO_KNIGHT_CHECKPOINT.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "model": model.state_dict(),
            "config": asdict(model.config),
            "config_hash": model.config.config_hash,
            "architecture_id": model.config.architecture_id,
            "seed": SEED,
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "surface_qualification_hash": qualification.qualification_hash,
            "tensorization_hash": tensor.tensorization_hash,
            "certificate_hash": tensor.certificate_hash,
            "target_content_sha256": target_content_sha256_v1(target),
            "step": final_step,
            "historical_checkpoint_loaded": False,
            "teacher_feedback_used": False,
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha(checkpoint_path)

    proposal_payload = _selected_proposal(final_check)
    proposal_path = outdir / "GEPPETTO_KNIGHT_SKELETON_PROPOSAL.json"
    proposal_path.write_text(
        json.dumps(proposal_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    proposal_sha = _sha(proposal_path)

    wall_seconds = float(time.monotonic() - started)
    result = {
        "schema": SCHEMA,
        "status": "TERMINAL_PASS",
        "architecture_id": model.config.architecture_id,
        "config_hash": model.config.config_hash,
        "seed": SEED,
        "preregistration_binding_hash": prereg.preregistration_hash,
        "model_source_sha256": prereg.model_source_sha256,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "surface_qualification_hash": qualification.qualification_hash,
        "surface_node_count": tensor.node_count,
        "surface_edge_count": tensor.edge_count,
        "tensorization_hash": tensor.tensorization_hash,
        "certificate_hash": tensor.certificate_hash,
        "target_content_sha256": target_content_sha256_v1(target),
        "target_count": target.count,
        "teacher_target_report_sha256": _sha(target_report_path),
        "experiment_preregistration_sha256": experiment_prereg_sha,
        "loss_config": asdict(loss_cfg),
        "closure_step": closure_step,
        "terminal_streak": terminal_streak,
        "required_terminal_checks": TERMINAL_CHECKS,
        "diffusion_eval_seeds": list(DIFFUSION_SEEDS),
        "selected_proposal_seed": SELECTED_PROPOSAL_SEED,
        "all_diffusion_seeds_passed": True,
        "final_check": final_check,
        "checkpoint_sha256": checkpoint_sha,
        "proposal_sha256": proposal_sha,
        "visual_evidence": visual_evidence,
        "teacher_supervision_during_training": True,
        "teacher_feedback_during_free_running_inference": False,
        "teacher_inference_inputs_used": False,
        "historical_checkpoint_loaded": False,
        "fresh_from_scratch": True,
        "wall_seconds": wall_seconds,
        "runtime_environment": _runtime_environment(),
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    result_path = outdir / "GEPPETTO_KNIGHT_RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_sha = _sha(result_path)

    receipt = {
        "schema": "RealSaS.ModelFitExecutionReceipt.v1",
        "lane": "GEPPETTO",
        "status": "PASS",
        "architecture_id": prereg.architecture_id,
        "preregistration_binding_hash": prereg.preregistration_hash,
        "model_source_sha256": prereg.model_source_sha256,
        "upstream_bindings": dict(prereg.upstream_bindings),
        "checkpoint_sha256": checkpoint_sha,
        "result_sha256": result_sha,
        "proposal_sha256": proposal_sha,
        "teacher_inference_inputs_used": False,
        "wall_seconds": wall_seconds,
        "timing": {
            "training_wall_seconds": wall_seconds,
            "closure_step": closure_step,
        },
        "metadata": {
            "run_mode": "KNIGHT_DEMO_WITNESS_FRESH_FIT",
            "teacher_supervision_during_training": True,
            "teacher_feedback_during_free_running_inference": False,
            "historical_checkpoint_loaded": False,
            "all_frozen_diffusion_seeds_passed": True,
            "selected_proposal_seed_predeclared": SELECTED_PROPOSAL_SEED,
            "runtime_environment": _runtime_environment(),
            "product_authority_minted": False,
            "generalization_claimed": False,
        },
    }
    receipt_path = outdir / "GEPPETTO_KNIGHT_EXECUTION_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "RealSaS.KnightGeppettoFitOutputManifest.v1",
        "status": "PASS",
        "files": {
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha,
            },
            "result": {
                "path": result_path.name,
                "sha256": result_sha,
            },
            "proposal": {
                "path": proposal_path.name,
                "sha256": proposal_sha,
            },
            "execution_receipt": {
                "path": receipt_path.name,
                "sha256": _sha(receipt_path),
            },
        },
        "product_authority_claimed": False,
    }
    (outdir / "GEPPETTO_KNIGHT_OUTPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("GEPPETTO_KNIGHT_TERMINAL_PASS")
    print(json.dumps(manifest, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-json", required=True)
    parser.add_argument("--surface-qualification-json", required=True)
    parser.add_argument("--teacher-target-npz", required=True)
    parser.add_argument("--teacher-target-report", required=True)
    parser.add_argument("--preregistration-ir", required=True)
    parser.add_argument("--experiment-prereg", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--camera-set-json", required=True)
    parser.add_argument("--observation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
