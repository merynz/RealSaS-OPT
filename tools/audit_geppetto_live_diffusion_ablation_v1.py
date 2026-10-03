from __future__ import annotations

"""Measure the causal contribution of live Geppetto residual diffusion.

The current shipping-faithful Knight Geppetto already uses conditional residual
diffusion. This court compares its normal inference against a coarse-only
counterfactual on the exact same checkpoint, surface, latent control states,
STOP/root/support evidence and decoded cardinality.

The coarse counterfactual changes only:
- emitted XYZ := coarse XYZ
- final all-pair parent evidence is recomputed from the same states + coarse XYZ

Recurrence is unchanged because the live model already feeds coarse XYZ, not the
diffusion-refined sample, back into later generation.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.types import (
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
    sha256_file,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    GeppettoReferenceStrengthNoLearnedSlotV1,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import (
    GeppettoReferenceStrengthConfigV1,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import (
    read,
    verified_checkpoint,
    write,
)


SEEDS = (11, 23, 47, 89)


def _load_teacher_target(fit_run: Path):
    candidates = (
        fit_run / "artifacts/26_GEPPETTO_FIT_PREREGISTERED/GEPPETTO_KNIGHT_MECHANICAL_CORE_TARGET.npz",
        fit_run / "artifacts/26_GEPPETTO_FIT_PREREGISTERED/KNIGHT_GEPPETTO_MECHANICAL_CORE_TARGET.npz",
        fit_run / "artifacts/27_GEPPETTO_FIT/GEPPETTO_KNIGHT_MECHANICAL_CORE_TARGET.npz",
    )
    for p in candidates:
        if p.is_file():
            with np.load(p, allow_pickle=False) as z:
                if "positions_world" in z.files:
                    return np.asarray(z["positions_world"], dtype=np.float64), str(p)
    return None, None


def _proposal_from_raw(model, surface, out, count: int, *, coarse_only: bool):
    if coarse_only:
        pos = out.coarse_positions_normalized[:, :count]
        parent = model._all_pair_parent_logits(
            out.control_states[:, :count], pos
        )
    else:
        pos = out.positions_normalized[:, :count]
        parent = out.all_pair_parent_logits[:, :count, :count]

    ids = tuple(f"P:GDIFF:{i:04d}" for i in range(count))
    sp = torch.sigmoid(out.support_presence_logits[0, :count])
    ex = torch.sigmoid(out.existence_logits[0, :count])
    roots = torch.sigmoid(out.root_logits[0, :count])
    salience = torch.sigmoid(out.salience_logits[0, :count])

    joints = []
    for i in range(count):
        pn = pos[0, i].detach().cpu().numpy()
        world = (
            pn.astype(np.float64) * float(surface.normalization_scale)
            + np.asarray(surface.normalization_center, dtype=np.float64)
        )
        sigma = torch.exp(out.position_log_sigma[0, i]).mean()
        support_ids = ()
        if float(sp[i]) >= model.config.support_presence_probability:
            top = min(model.config.support_topk, surface.node_count)
            idx = torch.argsort(
                out.support_logits[0, i],
                descending=True,
                stable=True,
            )[:top].tolist()
            support_ids = tuple(surface.surface_ids[j] for j in idx)
        confidence = float((ex[i] * torch.exp(-sigma)).clamp(0.0, 1.0))
        joints.append(
            SkeletonProposalJoint(
                proposal_id=ids[i],
                position=tuple(map(float, world)),
                root_score=float(roots[i]),
                confidence=confidence,
                support_surface_ids=support_ids,
                metadata={
                    "generation_index_internal_only": i,
                    "mechanical_salience_probability": float(salience[i]),
                    "position_variant":
                        "COARSE_ONLY_COUNTERFACTUAL"
                        if coarse_only
                        else "LIVE_CONDITIONAL_RESIDUAL_DIFFUSION",
                    "canonical_authority": False,
                },
            )
        )

    edges = []
    for child in range(count):
        for par in range(count):
            if child == par:
                continue
            score = float(torch.sigmoid(parent[0, child, par]))
            edges.append(
                SkeletonProposalEdge(
                    edge_id=f"E:{ids[par]}->{ids[child]}",
                    parent_proposal_id=ids[par],
                    child_proposal_id=ids[child],
                    score=score,
                    confidence=max(score, 1.0-score),
                    hard_required=False,
                    hard_forbidden=False,
                    reason=(
                        "GEPPETTO_LIVE_DIFFUSION_ABLATION_ALL_PAIR_PARENT"
                    ),
                    metadata={
                        "compiler_owns_tree_selection": True,
                        "coarse_only_counterfactual": bool(coarse_only),
                    },
                )
            )

    return SkeletonProposalIR(
        joints=tuple(joints),
        edges=tuple(edges),
        surface_binding_hash=surface.source_surface_hash,
        model_provenance=model.config.config_hash,
        metadata={
            "candidate_architecture": model.config.architecture_id,
            "conditional_residual_diffusion":
                not coarse_only,
            "coarse_only_counterfactual": bool(coarse_only),
            "compiler_owns_root_tree_ids": True,
            "product_authority_claimed": False,
        },
    )


def _match(pred, truth, scale):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if truth is None or pred.shape != truth.shape:
        return None
    d = np.linalg.norm(
        pred[:, None, :] - truth[None, :, :], axis=-1
    ) / float(scale)
    r, c = linear_sum_assignment(d)
    dist = d[r, c]
    return {
        "mae_norm": float(dist.mean()),
        "p95_norm": float(np.quantile(dist, 0.95)),
        "max_norm": float(dist.max()),
    }


def _parent_signature(qualified):
    by = {
        str(j.canonical_joint_id): str(j.source_proposal_id)
        for j in qualified.joints
    }
    return {
        str(j.source_proposal_id):
            None
            if j.parent_canonical_id is None
            else by.get(str(j.parent_canonical_id))
        for j in qualified.joints
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GEPPETTO_DIFFUSION_ABLATION_CUDA_REQUIRED")

    ctx = _ctx(args.authority_root, args.run_id)
    surface_ir = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    surface = tensorize_rigging_surface_v1(surface_ir)

    execution = read(
        args.fit_run / "artifacts/27_GEPPETTO_FIT/model_fit_execution.json"
    )
    if sha256_file(Path(inspect.getfile(
        GeppettoReferenceStrengthNoLearnedSlotV1
    ))) != execution["model_source_sha256"]:
        raise RuntimeError("GEPPETTO_DIFFUSION_ABLATION_MODEL_SOURCE_DRIFT")

    ckpt = verified_checkpoint(
        Path(execution["checkpoint_path"]),
        execution["checkpoint_sha256"],
    )
    config = GeppettoReferenceStrengthConfigV1(**ckpt["config"])
    if config.config_hash != ckpt["config_hash"]:
        raise RuntimeError("GEPPETTO_DIFFUSION_ABLATION_CONFIG_DRIFT")
    if "CausalDiffusion" not in config.architecture_id:
        raise RuntimeError("GEPPETTO_LIVE_ARCHITECTURE_NOT_DIFFUSION")

    teacher_world, teacher_path = _load_teacher_target(args.fit_run)

    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = GeppettoReferenceStrengthNoLearnedSlotV1(config).cuda().eval()
    model.load_state_dict(ckpt["model"], strict=True)

    rows = []
    for seed in SEEDS:
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        with torch.inference_mode():
            out, count = model.generate_surface(
                surface,
                resource_step_limit=min(128, surface.node_count),
                generator=gen,
            )
            live_prop = _proposal_from_raw(
                model, surface, out, count, coarse_only=False
            )
            coarse_prop = _proposal_from_raw(
                model, surface, out, count, coarse_only=True
            )

        live = qualify_skeleton(surface_ir, live_prop, run_ilp_shadow=False)
        coarse = qualify_skeleton(surface_ir, coarse_prop, run_ilp_shadow=False)

        live_world = np.asarray([j.position for j in live.joints], np.float64)
        coarse_world = np.asarray([j.position for j in coarse.joints], np.float64)
        paired = (
            np.linalg.norm(live_world - coarse_world, axis=1)
            / float(surface.normalization_scale)
            if live_world.shape == coarse_world.shape
            else np.asarray([], dtype=np.float64)
        )
        live_parent = _parent_signature(live)
        coarse_parent = _parent_signature(coarse)
        shared = set(live_parent) & set(coarse_parent)
        row = {
            "seed": int(seed),
            "joint_count_live": len(live.joints),
            "joint_count_coarse": len(coarse.joints),
            "live_teacher": (
                None if teacher_world is None
                else _match(live_world, teacher_world, surface.normalization_scale)
            ),
            "coarse_teacher": (
                None if teacher_world is None
                else _match(coarse_world, teacher_world, surface.normalization_scale)
            ),
            "live_vs_coarse_same_shape":
                bool(live_world.shape == coarse_world.shape),
            "live_vs_coarse_position_rmse_norm":
                (
                    None if paired.size == 0
                    else float(np.sqrt(np.mean(paired**2)))
                ),
            "live_vs_coarse_position_max_norm":
                None if paired.size == 0 else float(paired.max()),
            "changed_compiler_parent_count":
                int(sum(live_parent[k] != coarse_parent[k] for k in shared)),
            "shared_generation_id_count": int(len(shared)),
        }
        rows.append(row)
        print(
            "GEPPETTO_LIVE_DIFFUSION_ABLATION_SEED="
            + json.dumps(row, sort_keys=True),
            flush=True,
        )

        write(args.out_dir / f"seed_{seed}_live_proposal.json", live_prop.to_dict())
        write(args.out_dir / f"seed_{seed}_coarse_proposal.json", coarse_prop.to_dict())
        write(args.out_dir / f"seed_{seed}_live_skeleton.json", live.to_dict())
        write(args.out_dir / f"seed_{seed}_coarse_skeleton.json", coarse.to_dict())

    comparable = [
        r for r in rows
        if r["live_teacher"] is not None and r["coarse_teacher"] is not None
    ]
    live_better = None
    if comparable:
        live_better = {
            "mae_all_seeds": bool(all(
                r["live_teacher"]["mae_norm"]
                <= r["coarse_teacher"]["mae_norm"]
                for r in comparable
            )),
            "p95_all_seeds": bool(all(
                r["live_teacher"]["p95_norm"]
                <= r["coarse_teacher"]["p95_norm"]
                for r in comparable
            )),
            "mean_mae_delta_coarse_minus_live": float(np.mean([
                r["coarse_teacher"]["mae_norm"]
                - r["live_teacher"]["mae_norm"]
                for r in comparable
            ])),
            "mean_p95_delta_coarse_minus_live": float(np.mean([
                r["coarse_teacher"]["p95_norm"]
                - r["live_teacher"]["p95_norm"]
                for r in comparable
            ])),
        }

    report = {
        "schema":"RealSaS.GeppettoLiveDiffusionAblationCourt.v1",
        "status":"MEASURED__NO_PROMOTION_CLAIM",
        "checkpoint_sha256":execution["checkpoint_sha256"],
        "architecture_id":config.architecture_id,
        "diffusion_train_steps":config.diffusion_train_steps,
        "diffusion_sample_steps":config.diffusion_sample_steps,
        "diffusion_residual_clip":config.diffusion_residual_clip,
        "teacher_target_path":teacher_path,
        "seeds":list(SEEDS),
        "rows":rows,
        "teacher_comparison_summary":live_better,
        "training_used":False,
        "product_authority_minted":False,
        "claim_boundary":[
            "The live model already contains conditional residual diffusion.",
            "Both arms share exact encoder, recurrence, control states, STOP/root/support evidence and cardinality.",
            "Coarse-only recomputes final all-pair parent evidence from coarse XYZ.",
            "This court measures diffusion contribution; it does not evaluate richer-rig target policy.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print(
        "GEPPETTO_LIVE_DIFFUSION_ABLATION_RESULT="
        + json.dumps(report, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--fit-run",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(
            args.out_dir/"ERROR.json",
            {"type":type(exc).__name__,"message":str(exc)},
        )
        raise
