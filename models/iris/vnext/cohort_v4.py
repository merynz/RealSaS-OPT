from __future__ import annotations

"""IRIS vNext Fit8 trainer v4.

V4 keeps the proven surgical migration from the failed v2 step-1601 source,
ends the dedicated G2 surface stage at the empirically stable 400-update point,
reallocates the removed 800 updates to joint finetuning, and executes all
probability BCE terms outside autocast in FP32.  The latter is required by
PyTorch because BCELoss/BCE are explicitly unsafe under CUDA autocast.
"""

from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .cohort_v1 import (
    VNextFit8PolicyV1,
    _REQUIRED_APPEARANCE,
    _REQUIRED_MECH,
    _REQUIRED_RELATION,
    _sha,
    _vh,
)
from .cohort_v3 import IRISVNextFit8TrainerV3, vnext_fit8_policy_v2
from .contracts_v1 import IRISVNextConfigV1
from .surface_extraction_v1 import extract_surface_batch_v1, extracted_surface_loss_v1


def vnext_fit8_policy_v3() -> VNextFit8PolicyV1:
    """Court/authority-log selected curriculum; total remains exactly 12k."""
    base = vnext_fit8_policy_v2()
    stages = list(base.stages)
    g2 = stages[1]
    j1 = stages[-1]
    if g2.stage_id != "G2_FLEXICUBES_SURFACE" or g2.steps != 1200:
        raise RuntimeError("VNEXT_V4_G2_BASE_POLICY_DRIFT")
    if j1.stage_id != "J1_JOINT_FINETUNE" or j1.steps != 2400:
        raise RuntimeError("VNEXT_V4_J1_BASE_POLICY_DRIFT")
    # Authority run was healthy through dedicated-G2 stage_step 400, then the
    # surface field over-complexified sharply after ~600.  Stop the dedicated
    # objective at 400 and spend those 800 updates in the multi-objective J1.
    stages[1] = replace(g2, steps=400)
    stages[-1] = replace(j1, steps=3200)
    policy = replace(base, stages=tuple(stages))
    policy.validate()
    if policy.total_steps != 12000:
        raise RuntimeError("VNEXT_V4_TOTAL_STEP_DRIFT")
    return policy


def _source_identity_v4(root: Path):
    from .cohort_v3 import _source_identity_v3
    value = dict(_source_identity_v3(root))
    value["models/iris/vnext/cohort_v4.py"] = _sha(root / "models/iris/vnext/cohort_v4.py")
    return value


def _probability_bce_fp32(predicted: torch.Tensor, truth: torch.Tensor, *, reduction="none"):
    """Probability BCE with autocast explicitly disabled.

    Existing heads intentionally expose calibrated probabilities rather than
    logits.  Preserve that public/model contract and only move this loss op to
    FP32 instead of changing the heads to logits.
    """
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        return F.binary_cross_entropy(
            predicted.float().clamp(1e-6, 1.0 - 1e-6),
            truth.float(),
            reduction=reduction,
        )


def mechanical_demand_loss_v2(predicted, truth, *, valid_mask=None):
    required = (
        "stretch_major",
        "stretch_minor",
        "tangent_direction_2theta",
        "bend_demand",
        "rigidity_probability",
    )
    if any(key not in predicted or key not in truth for key in required):
        raise ValueError("IRIS_VNEXT_MECHANICAL_TRUTH_INCOMPLETE")

    def reduce(value):
        if valid_mask is None:
            return value.mean()
        mask = valid_mask.to(dtype=value.dtype)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * (
            value.shape[-1] if value.ndim > valid_mask.ndim else 1
        )
        return (value * mask).sum() / denom

    stretch = F.smooth_l1_loss(
        predicted["stretch_major"], truth["stretch_major"], reduction="none"
    ) + F.smooth_l1_loss(
        predicted["stretch_minor"], truth["stretch_minor"], reduction="none"
    )
    tangent_pred = F.normalize(
        predicted["tangent_direction_2theta"], dim=-1, eps=1e-6
    )
    tangent_truth = F.normalize(
        truth["tangent_direction_2theta"], dim=-1, eps=1e-6
    )
    tangent = 1.0 - (tangent_pred * tangent_truth).sum(-1)
    bend = F.smooth_l1_loss(
        predicted["bend_demand"], truth["bend_demand"], reduction="none"
    )
    rigid = _probability_bce_fp32(
        predicted["rigidity_probability"], truth["rigidity_probability"], reduction="none"
    )
    return reduce(stretch) + reduce(tangent) + reduce(bend) + reduce(rigid)


def graphics_relation_loss_v2(predicted, truth):
    keys = (
        "same_drawable_affinity",
        "cut_probability",
        "attachment_affinity",
        "alpha_boundary_probability",
        "appearance_boundary_probability",
        "front_probability",
        "occlusion_probability",
        "visibility_probability_a",
        "visibility_probability_b",
    )
    if any(key not in truth for key in keys):
        missing = [key for key in keys if key not in truth]
        raise ValueError("IRIS_VNEXT_RELATION_TRUTH_INCOMPLETE:" + ",".join(missing))

    fallback = torch.isclose(
        truth["front_probability"],
        torch.full_like(truth["front_probability"], 0.5),
        atol=1e-6,
        rtol=0.0,
    )
    observed = ~fallback
    always = torch.ones_like(fallback, dtype=torch.bool)
    validity = {
        "same_drawable_affinity": always,
        "cut_probability": always,
        "attachment_affinity": (
            (truth["attachment_affinity"] > 0.5)
            | (truth["same_drawable_affinity"] > 0.5)
        ),
        "alpha_boundary_probability": observed,
        "appearance_boundary_probability": always,
        "front_probability": observed,
        "occlusion_probability": observed,
        "visibility_probability_a": always,
        "visibility_probability_b": observed | (truth["visibility_probability_b"] > 0.5),
    }
    terms = []
    for key in keys:
        if key not in predicted:
            raise ValueError("IRIS_VNEXT_RELATION_PREDICTION_MISSING:" + key)
        mask = validity[key]
        if not bool(mask.any()):
            continue
        raw = _probability_bce_fp32(predicted[key], truth[key], reduction="none")
        m = mask.to(dtype=raw.dtype)
        terms.append((raw * m).sum() / m.sum().clamp_min(1.0))
    if not terms:
        raise ValueError("IRIS_VNEXT_RELATION_TRUTH_EMPTY")
    return sum(terms) / len(terms)


class IRISVNextFit8TrainerV4(IRISVNextFit8TrainerV3):
    architecture_id = "RealSaS.IRISVNextFit8Trainer.v4"

    def __init__(
        self,
        *,
        data_root,
        manifest,
        checkpoint_root,
        device,
        model_config=IRISVNextConfigV1(),
        policy=None,
        repo_root=None,
        flexicubes_root=None,
    ):
        policy = vnext_fit8_policy_v3() if policy is None else policy
        policy.validate()
        super().__init__(
            data_root=data_root,
            manifest=manifest,
            checkpoint_root=checkpoint_root,
            device=device,
            model_config=model_config,
            policy=policy,
            repo_root=repo_root,
            flexicubes_root=flexicubes_root,
        )
        self.binding = {
            **self.binding,
            "schema": "RealSaS.IRISVNextFit8Binding.v6",
            "source": _source_identity_v4(self.repo_root),
            "policy": asdict(policy),
            "curriculum_semantics": (
                "G1__G2_FIELD_ANCHORED_FLEXICUBES_EARLY_STOP_400__"
                "M1__A1__R1__T1__J1_EXTENDED_3200"
            ),
            "probability_loss_semantics": "PROBABILITY_BCE_FP32_AUTOCAST_DISABLED",
            "g2_authority_log_repair": {
                "previous_v3_collapse_observed_after_stage_step": 600,
                "dedicated_g2_steps": 400,
                "reallocated_joint_steps": 800,
                "total_steps_unchanged": 12000,
            },
        }
        self.binding_hash = _vh(self.binding)
        # V3 created the checkpoint store before this final binding was minted.
        # Rebind it now and restore only checkpoints that match V6.
        from .cohort_v1 import VNextCheckpointStoreV1
        self.store = VNextCheckpointStoreV1(Path(checkpoint_root), self.binding_hash)
        previous = self.store.load()
        if previous:
            self._restore(previous)
        else:
            self.step = 0
            self.stage_index = 0
            self.stage_step = 0
            self._apply_stage_lrs()

    def one_step(self):
        if self._advance_stage_if_needed():
            raise StopIteration("VNEXT_FIT8_COMPLETE")
        stage = self.stage
        next_step = self.step + 1
        view_index = next_step % 8
        source = self.cohort.tensors
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            planes = self.model.encode_scene(
                source["image_tokens"], source["token_xy"], source["camera_features"]
            )
            zero = planes.sum() * 0.0
            losses = {
                "field": zero,
                "extraction": zero,
                "extraction_chamfer": zero,
                "extraction_l_dev": zero,
                "mechanical": zero,
                "appearance": zero,
                "relations": zero,
                "topology": zero,
            }

            if stage.field_weight > 0:
                points, target = self.cohort.geometry_batch(self.policy, next_step)
                predicted = self.model.query_evidence(planes, points)
                losses["field"] = F.smooth_l1_loss(
                    predicted["field"].float(), target.float()
                )

            extracted = None
            if stage.extraction_weight > 0 or stage.topology_weight > 0:
                if self.extractor is None:
                    raise RuntimeError("VNEXT_FIT8_EXTRACTOR_MISSING")
                extracted = extract_surface_batch_v1(
                    self.model,
                    planes,
                    self.extractor,
                    resolution=self.model.config.extraction_training_resolution,
                    witness_count=self.model.config.surface_witness_count,
                    training=True,
                )
                if stage.extraction_weight > 0:
                    extraction = extracted_surface_loss_v1(
                        extracted,
                        source["teacher_surface_points_iris"],
                        l_dev_weight=self.policy.extraction_l_dev_weight,
                        flexible_weight_l1=self.policy.extraction_flexible_weight_l1,
                    )
                    losses["extraction"] = extraction["total"]
                    losses["extraction_chamfer"] = extraction["surface_chamfer"]
                    losses["extraction_l_dev"] = extraction["l_dev"]

            if stage.mechanical_weight > 0:
                batch = self.cohort.mechanical_batch(self.policy, next_step)
                predicted = self.model.query_evidence(
                    planes, batch["mechanical_points"]
                )["mechanical_demand"]
                losses["mechanical"] = mechanical_demand_loss_v2(
                    predicted, self._mechanical_truth(batch)
                )

            if stage.appearance_weight > 0:
                batch = self.cohort.appearance_batch(self.policy, next_step, view_index)
                local = self.model.query_evidence(planes, batch["appearance_points"])
                predicted = self.model.query_appearance(
                    local["H"], source["camera_features"][:, view_index]
                )
                rgba = F.smooth_l1_loss(
                    predicted["rgba_candidate"].float(), batch["appearance_rgba"].float()
                )
                alpha_boundary = _probability_bce_fp32(
                    predicted["alpha_boundary_probability"], batch["appearance_alpha_boundary"],
                    reduction="mean",
                )
                appearance_boundary = _probability_bce_fp32(
                    predicted["appearance_boundary_probability"], batch["appearance_boundary"],
                    reduction="mean",
                )
                losses["appearance"] = rgba + 0.25 * (
                    alpha_boundary + appearance_boundary
                )

            if stage.relation_weight > 0:
                batch = self.cohort.relation_batch(self.policy, next_step, view_index)
                feature_a = self.model.query_evidence(planes, batch["relation_point_a"])["H"]
                feature_b = self.model.query_evidence(planes, batch["relation_point_b"])["H"]
                static = self.model.relations.static_relations(feature_a, feature_b)
                cameras = source["camera_features"][:, view_index, None, :].expand(
                    -1, feature_a.shape[1], -1
                )
                viewed = self.model.relations.view_relations(feature_a, feature_b, cameras)
                predicted = {**static, **viewed}
                truth = {
                    key.removeprefix("relation_"): batch[key]
                    for key in _REQUIRED_RELATION[2:]
                }
                losses["relations"] = graphics_relation_loss_v2(predicted, truth)

            topology_offsets = None
            if stage.topology_weight > 0:
                if extracted is None:
                    raise RuntimeError("VNEXT_FIT8_MODEL_SURFACE_WITNESS_MISSING")
                input_ids, labels, topology_offsets = self.cohort.topology_batch(next_step)
                witness = self.model.query_evidence(planes, extracted.witness_xyz_iris)
                output = self.model.decode_topology(
                    scene_planes=planes,
                    witness_xyz=extracted.witness_xyz_tessa,
                    witness_normal=extracted.witness_normal,
                    witness_h=witness["H"],
                    witness_log_uncertainty=witness["log_uncertainty"],
                    witness_mechanical=witness["mechanical_demand"],
                    input_ids=input_ids.long(),
                    labels=labels.long(),
                    sequence_position_offset=topology_offsets.long(),
                )
                if output.loss is None:
                    raise RuntimeError("VNEXT_FIT8_TOPOLOGY_LOSS_MISSING")
                losses["topology"] = output.loss

            loss = (
                stage.field_weight * losses["field"]
                + stage.extraction_weight * losses["extraction"]
                + stage.mechanical_weight * losses["mechanical"]
                + stage.appearance_weight * losses["appearance"]
                + stage.relation_weight * losses["relations"]
                + stage.topology_weight * losses["topology"]
            )

        if not bool(torch.isfinite(loss)):
            raise RuntimeError("VNEXT_FIT8_NONFINITE_LOSS")
        loss.backward()
        self.optimizer.step()

        post_surface = None
        if stage.extraction_weight > 0 or stage.topology_weight > 0:
            try:
                post_surface = self._surface_gate()
            except Exception as exc:
                raise RuntimeError(
                    f"VNEXT_POST_UPDATE_SURFACE_GATE_FAILURE_AT_NEXT_STEP_{next_step}:{exc}"
                ) from exc

        self.step += 1
        self.stage_step += 1
        if self.step % self.policy.checkpoint_interval == 0:
            self.save()

        result = {
            "global_step": self.step,
            "stage": stage.stage_id,
            "stage_step": self.stage_step,
            "loss": float(loss.detach()),
            **{f"{key}_loss": float(value.detach()) for key, value in losses.items()},
            "view_index": view_index,
            "batch_subjects": 8,
            "shared_scene_encodes": 1,
            "subject_serial_model_forward": False,
            "optimizer_steps": 1,
            "elapsed_seconds": 0.0,
        }
        if topology_offsets is not None:
            result["topology_position_offsets"] = tuple(
                map(int, topology_offsets.detach().cpu().tolist())
            )
        if post_surface is not None:
            result["post_update_surface_court"] = post_surface
            result["surface_telemetry_phase"] = "POST_OPTIMIZER"
            result["extracted_vertex_counts"] = post_surface["vertices"]
            result["extracted_face_counts"] = post_surface["faces"]
        return result
