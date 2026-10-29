"""Sealed Truth139 unseen evaluator.

This evaluator has no optimizer and no checkpoint-selection path.  It consumes only:
* the exact final TRAIN checkpoint and its TRAINING_RESULT receipt;
* the physically separate UNSEEN manifest;
* the split-bound training recipe (policy only, no train rows);
* a tiny TRAIN-only mechanics statistics artifact containing two medians and two fixed scales.

Mechanics is evaluated three ways on the same unseen truth rows: real carrier, deterministic cyclic
shuffled carrier from another unseen subject, and a constant train-median prior.  This prevents a
good marginal mechanics distribution from masquerading as scene-conditioned prediction.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.iris.training.cohort_v1 import file_hash, value_hash
from models.iris.v5.direct_fstar_fit_v5 import direct_fstar_metrics_v5
from models.iris.v31.evidence_adapter_model_v2 import IRISEvidenceAdapterConfigV2
from models.iris.v31.fixed_snake_capacity_pilot_v1 import mechanics_target_log1p_v1
from models.iris.v31.mechanics_fixed_scale_student_t_v4 import (
    conditioning_gain_report_v4,
    fixed_scale_student_t_mechanical_loss_v4,
    mechanics_regression_metrics_v4,
)
from models.iris.v31.presentation_objective_v1 import appearance_rgba_loss_v1
from models.iris.v31.truth139_model_v1 import IRISV31Truth139V1
from models.iris.v31.truth139_trainer_v1 import Truth139SubjectV1


EVAL_MANIFEST_SCHEMA = "RealSaS.IRISV31Truth139UnseenEvalManifest.v1"
TRAIN_RESULT_SCHEMA = "RealSaS.IRISV31Truth139TrainingResult.v1"
CHECKPOINT_SCHEMA = "RealSaS.IRISV31Truth139Checkpoint.v1"
MECHANICS_STATS_SCHEMA = "RealSaS.IRISV31Truth139MechanicsTrainStatistics.v1"
RESULT_SCHEMA = "RealSaS.IRISV31Truth139UnseenEvaluation.v1"


def _require(value, code: str) -> None:
    if not value:
        raise ValueError(code)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def _normal_metrics(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    if predicted.shape != target.shape or predicted.shape[-1] != 3 or valid.shape != predicted.shape[:-1]:
        raise ValueError("TRUTH139_UNSEEN_NORMAL_METRIC_SHAPE")
    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("TRUTH139_UNSEEN_NORMAL_METRIC_EMPTY")
    p = F.normalize(predicted.float(), dim=-1, eps=1e-8)
    t = F.normalize(target.float(), dim=-1, eps=1e-8)
    cosine = torch.sum(p * t, dim=-1).clamp(-1.0, 1.0)[mask]
    angle = torch.rad2deg(torch.acos(cosine))
    return {
        "count": int(angle.numel()),
        "mean_angular_degrees": float(angle.mean().cpu()),
        "p95_angular_degrees": float(torch.quantile(angle, 0.95).cpu()),
        "mean_cosine": float(cosine.mean().cpu()),
    }


def _macro_worst(rows: list[dict], keys: tuple[str, ...], *, higher_is_worse: set[str] | None = None) -> dict:
    higher_is_worse = higher_is_worse or set(keys)
    if not rows:
        raise ValueError("TRUTH139_UNSEEN_AGGREGATE_EMPTY")
    out = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("TRUTH139_UNSEEN_AGGREGATE_NONFINITE:" + key)
        out["macro_" + key] = float(values.mean())
        out["worst_" + key] = float(values.max() if key in higher_is_worse else values.min())
    return out


def _constant_prior_raw(shape, median: tuple[float, float], *, device, dtype) -> torch.Tensor:
    out = torch.zeros(tuple(shape), device=device, dtype=dtype)
    if out.shape[-1] != 4:
        raise ValueError("TRUTH139_UNSEEN_PRIOR_SHAPE")
    out[..., 0] = float(median[0])
    out[..., 2] = float(median[1])
    return out


class Truth139UnseenEvaluatorV1:
    def __init__(
        self,
        *,
        unseen_manifest: dict,
        training_result: dict,
        recipe: dict,
        mechanics_statistics: dict,
        checkpoint_path,
        output_root,
        device: str = "cuda",
    ) -> None:
        _require(unseen_manifest.get("schema") == EVAL_MANIFEST_SCHEMA, "TRUTH139_UNSEEN_MANIFEST_SCHEMA")
        _require(training_result.get("schema") == TRAIN_RESULT_SCHEMA, "TRUTH139_UNSEEN_TRAIN_RESULT_SCHEMA")
        _require(mechanics_statistics.get("schema") == MECHANICS_STATS_SCHEMA, "TRUTH139_UNSEEN_MECHANICS_STATS_SCHEMA")
        _require(unseen_manifest.get("optimizer_updates_allowed") is False, "TRUTH139_UNSEEN_OPTIMIZER_FORBIDDEN")
        _require(unseen_manifest.get("checkpoint_selection_allowed") is False, "TRUTH139_UNSEEN_SELECTION_FORBIDDEN")
        split = str(unseen_manifest.get("split_identity_sha256") or "")
        _require(
            split
            and split == str(training_result.get("split_identity_sha256") or "")
            and split == str(mechanics_statistics.get("split_identity_sha256") or ""),
            "TRUTH139_UNSEEN_SPLIT_DRIFT",
        )
        commitment = str(unseen_manifest.get("unseen_subject_set_commitment_sha256") or "")
        _require(
            commitment == str(training_result.get("unseen_subject_set_commitment_sha256") or ""),
            "TRUTH139_UNSEEN_SUBJECT_COMMITMENT_DRIFT",
        )
        _require(training_result.get("unseen_opened") is False, "TRUTH139_TRAIN_RESULT_ALREADY_OPENED_UNSEEN")
        _require(training_result.get("heldout_or_unseen_used_for_gradient") is False, "TRUTH139_TRAIN_RESULT_UNSEEN_GRADIENT")
        _require(training_result.get("heldout_or_unseen_used_for_checkpoint_selection") is False, "TRUTH139_TRAIN_RESULT_UNSEEN_SELECTION")
        _require(recipe.get("selection_rule") == "FIXED_FINAL_STAGE_ENDPOINT__NO_CHECKPOINT_SHOPPING", "TRUTH139_UNSEEN_RECIPE_SELECTION")
        _require(recipe.get("split_identity_sha256") == split, "TRUTH139_UNSEEN_RECIPE_SPLIT")

        checkpoint = Path(checkpoint_path).resolve()
        _require(
            checkpoint.is_file()
            and file_hash(checkpoint) == str(training_result.get("selected_checkpoint_sha256") or ""),
            "TRUTH139_UNSEEN_CHECKPOINT_BYTES_DRIFT",
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _require(payload.get("schema") == CHECKPOINT_SCHEMA, "TRUTH139_UNSEEN_CHECKPOINT_SCHEMA")
        _require(payload.get("architecture_id") == IRISV31Truth139V1.architecture_id, "TRUTH139_UNSEEN_ARCHITECTURE_DRIFT")
        _require(payload.get("foundation_identity") == training_result.get("foundation_identity"), "TRUTH139_UNSEEN_FOUNDATION_IDENTITY_DRIFT")
        fixed_scale = tuple(map(float, payload.get("fixed_mechanics_scale") or ()))
        stats_scale = tuple(map(float, mechanics_statistics.get("fixed_scale") or ()))
        _require(len(fixed_scale) == 2 and fixed_scale == stats_scale, "TRUTH139_UNSEEN_MECHANICS_SCALE_DRIFT")
        median = tuple(map(float, mechanics_statistics.get("transformed_median") or ()))
        _require(len(median) == 2 and all(math.isfinite(x) for x in median), "TRUTH139_UNSEEN_MECHANICS_PRIOR_INVALID")

        self.device = torch.device(device)
        _require(self.device.type == "cuda" and torch.cuda.is_available(), "TRUTH139_UNSEEN_CUDA_REQUIRED")
        runtime = recipe["runtime"]
        torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(runtime["tf32"])
        torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = False

        config_row = dict(payload["config"])
        config_row["mechanical_channels"] = tuple(config_row["mechanical_channels"])
        config_row["pair_channels"] = tuple(config_row["pair_channels"])
        self.config = IRISEvidenceAdapterConfigV2(**config_row)
        self.model = IRISV31Truth139V1(self.config).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.eval().requires_grad_(False)
        self.fixed_scale = fixed_scale
        self.train_prior_median = median
        self.recipe = recipe
        self.manifest = unseen_manifest
        self.training_result = training_result
        self.mechanics_statistics = mechanics_statistics
        self.output = Path(output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)

        rows = list(unseen_manifest.get("subjects") or ())
        ids = [str(row.get("subject_id") or "") for row in rows]
        _require(ids and len(ids) == len(set(ids)) == int(unseen_manifest["subject_count"]), "TRUTH139_UNSEEN_SUBJECT_CARDINALITY")
        _require(value_hash(sorted(ids)) == commitment, "TRUTH139_UNSEEN_SUBJECT_COMMITMENT_CONTENT_DRIFT")
        self.ids = sorted(ids)
        by_id = {str(row["subject_id"]): row for row in rows}
        self.subjects = {
            subject_id: Truth139SubjectV1(by_id[subject_id], device=self.device, recipe=recipe)
            for subject_id in self.ids
        }
        _require(len(self.ids) >= 2, "TRUTH139_UNSEEN_SHUFFLE_REQUIRES_TWO_SUBJECTS")
        self.shuffle_map = {
            subject_id: self.ids[(index + 1) % len(self.ids)]
            for index, subject_id in enumerate(self.ids)
        }
        _require(all(self.shuffle_map[x] != x for x in self.ids), "TRUTH139_UNSEEN_SHUFFLE_SELF_MAP")

    @torch.no_grad()
    def _carrier(self, subject: Truth139SubjectV1):
        final, layer2, alpha, cameras = subject.observation()
        return self.model.encode(final=final, layer2=layer2, alpha=alpha, cameras=cameras), cameras

    @torch.no_grad()
    def _evaluate_real_subject(self, subject: Truth139SubjectV1, carrier, cameras) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        points = subject.geometry["heldout_points"].to(self.device).float()[None]
        target_fstar = subject.geometry["heldout_targets"].to(self.device).float()[None]
        gg, _ = self.model.query_states(carrier, points.to(carrier.dtype))
        pred_fstar = self.model.geometry(gg)[..., 0].float()
        geometry = direct_fstar_metrics_v5(pred_fstar, target_fstar, near_zero_band=0.03)

        normal_ids = np.flatnonzero(np.asarray(subject.normal["valid"], dtype=np.bool_))
        _require(len(normal_ids), "TRUTH139_UNSEEN_NORMAL_EMPTY:" + subject.subject_id)
        normal_xyz = torch.from_numpy(subject.normal["xyz"][normal_ids]).to(self.device)[None].float()
        normal_target = torch.from_numpy(subject.normal["normal_world"][normal_ids]).to(self.device)[None].float()
        normal_valid = torch.ones((1, len(normal_ids)), device=self.device, dtype=torch.bool)
        normal_pred = self.model.query_surface_normal(carrier, normal_xyz.to(carrier.dtype)).float()
        normals = _normal_metrics(normal_pred, normal_target, normal_valid)

        appearance_ids = np.flatnonzero(np.asarray(subject.presentation["appearance_valid"], dtype=np.bool_))
        _require(len(appearance_ids), "TRUTH139_UNSEEN_APPEARANCE_EMPTY:" + subject.subject_id)
        appearance_xyz = torch.from_numpy(subject.presentation["appearance_xyz"][appearance_ids]).to(self.device)[None].float()
        appearance_view = torch.from_numpy(subject.presentation["appearance_camera_index"][appearance_ids]).to(self.device).long()
        appearance_target = torch.from_numpy(subject.presentation["appearance_rgba"][appearance_ids]).to(self.device)[None].float()
        _, gp = self.model.query_states(carrier, appearance_xyz.to(carrier.dtype))
        cam = cameras[:, appearance_view, :].to(gp.dtype)
        rgba = self.model.appearance(torch.cat((gp, cam), dim=-1))[..., :4].sigmoid().float()
        appearance_valid = torch.ones((1, len(appearance_ids)), device=self.device, dtype=torch.bool)
        appearance_loss = appearance_rgba_loss_v1(rgba, appearance_target, appearance_valid)

        mechanics_ids = np.flatnonzero(np.asarray(subject.presentation["mechanical_valid"], dtype=np.bool_))
        _require(len(mechanics_ids), "TRUTH139_UNSEEN_MECHANICS_EMPTY:" + subject.subject_id)
        mechanics_xyz = torch.from_numpy(subject.presentation["mechanical_xyz"][mechanics_ids]).to(self.device)[None].float()
        mechanics_target_raw = torch.from_numpy(subject.presentation["mechanical_target"][mechanics_ids]).to(self.device)[None].float()
        mechanics_target = mechanics_target_log1p_v1(mechanics_target_raw)
        mechanics_state = self.model.query_mechanics_state(carrier, mechanics_xyz.to(carrier.dtype))
        mechanics_raw = self.model.mechanics(mechanics_state).float()
        mechanics_valid = torch.ones((1, len(mechanics_ids)), device=self.device, dtype=torch.bool)
        mechanics_metrics = mechanics_regression_metrics_v4(mechanics_raw, mechanics_target, mechanics_valid)
        mechanics_nll = fixed_scale_student_t_mechanical_loss_v4(
            mechanics_raw,
            mechanics_target,
            mechanics_valid,
            fixed_scale=self.fixed_scale,
            degrees_of_freedom=float(self.recipe["mechanics"]["degrees_of_freedom"]),
        )
        prior_raw = _constant_prior_raw(
            mechanics_raw.shape,
            self.train_prior_median,
            device=self.device,
            dtype=mechanics_raw.dtype,
        )
        prior_metrics = mechanics_regression_metrics_v4(prior_raw, mechanics_target, mechanics_valid)
        prior_nll = fixed_scale_student_t_mechanical_loss_v4(
            prior_raw,
            mechanics_target,
            mechanics_valid,
            fixed_scale=self.fixed_scale,
            degrees_of_freedom=float(self.recipe["mechanics"]["degrees_of_freedom"]),
        )

        real = {
            "subject_id": subject.subject_id,
            "geometry": geometry,
            "surface_normal": normals,
            "appearance": {"valid_count": int(len(appearance_ids)), "rgba_loss": float(appearance_loss.cpu())},
            "mechanics_model": {**mechanics_metrics, "fixed_scale_student_t_nll": float(mechanics_nll.cpu())},
            "mechanics_train_prior": {**prior_metrics, "fixed_scale_student_t_nll": float(prior_nll.cpu())},
        }
        return real, {"valid": mechanics_valid, "target": mechanics_target, "real_raw": mechanics_raw, "prior_raw": prior_raw}, mechanics_xyz, mechanics_target

    @torch.no_grad()
    def _shuffled_mechanics(self, *, target_subject: Truth139SubjectV1, donor_subject: Truth139SubjectV1, xyz: torch.Tensor, target: torch.Tensor) -> dict:
        donor_carrier, _ = self._carrier(donor_subject)
        state = self.model.query_mechanics_state(donor_carrier, xyz.to(donor_carrier.dtype))
        raw = self.model.mechanics(state).float()
        valid = torch.ones(target.shape[:-1], device=self.device, dtype=torch.bool)
        metrics = mechanics_regression_metrics_v4(raw, target, valid)
        nll = fixed_scale_student_t_mechanical_loss_v4(
            raw,
            target,
            valid,
            fixed_scale=self.fixed_scale,
            degrees_of_freedom=float(self.recipe["mechanics"]["degrees_of_freedom"]),
        )
        return {**metrics, "fixed_scale_student_t_nll": float(nll.cpu()), "donor_subject_id": donor_subject.subject_id}

    def evaluate(self) -> dict:
        per_subject = []
        mechanics_real_raw, mechanics_prior_raw, mechanics_shuffled_raw, mechanics_targets = [], [], [], []
        for subject_id in self.ids:
            subject = self.subjects[subject_id]
            carrier, cameras = self._carrier(subject)
            real, mechanics_packet, xyz, target = self._evaluate_real_subject(subject, carrier, cameras)
            donor = self.subjects[self.shuffle_map[subject_id]]
            shuffled = self._shuffled_mechanics(
                target_subject=subject,
                donor_subject=donor,
                xyz=xyz,
                target=target,
            )
            real["mechanics_shuffled_scene"] = shuffled
            per_subject.append(real)
            mechanics_real_raw.append(mechanics_packet["real_raw"].detach().cpu())
            mechanics_prior_raw.append(mechanics_packet["prior_raw"].detach().cpu())
            # Recompute only the small raw prediction for aggregation; donor carrier is not retained.
            donor_carrier, _ = self._carrier(donor)
            with torch.no_grad():
                state = self.model.query_mechanics_state(donor_carrier, xyz.to(donor_carrier.dtype))
                shuffled_raw = self.model.mechanics(state).float().detach().cpu()
            mechanics_shuffled_raw.append(shuffled_raw)
            mechanics_targets.append(target.detach().cpu())
            del carrier, donor_carrier
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        target_all = torch.cat(mechanics_targets, dim=1)
        valid_all = torch.ones(target_all.shape[:-1], dtype=torch.bool)
        model_metrics = mechanics_regression_metrics_v4(torch.cat(mechanics_real_raw, dim=1), target_all, valid_all)
        prior_metrics = mechanics_regression_metrics_v4(torch.cat(mechanics_prior_raw, dim=1), target_all, valid_all)
        shuffled_metrics = mechanics_regression_metrics_v4(torch.cat(mechanics_shuffled_raw, dim=1), target_all, valid_all)
        conditioning = conditioning_gain_report_v4(model_metrics, prior_metrics, shuffled_metrics)

        geometry_rows = [row["geometry"] for row in per_subject]
        normal_rows = [row["surface_normal"] for row in per_subject]
        appearance_rows = [row["appearance"] for row in per_subject]
        aggregate = {
            "geometry": _macro_worst(
                geometry_rows,
                ("mae", "rmse", "p95_abs", "sign_disagreement_fraction", "near_zero_mae"),
            ),
            "surface_normal": _macro_worst(
                normal_rows,
                ("mean_angular_degrees", "p95_angular_degrees"),
            ),
            "appearance": _macro_worst(appearance_rows, ("rgba_loss",)),
            "mechanics_conditioning": conditioning,
        }
        result = {
            "schema": RESULT_SCHEMA,
            "status": "UNSEEN_EVALUATED__NO_GENERALIZATION_PASS_THRESHOLD_APPLIED",
            "split_identity_sha256": self.manifest["split_identity_sha256"],
            "unseen_manifest_identity_sha256": self.manifest["manifest_identity_sha256"],
            "training_binding_sha256": self.training_result["binding_sha256"],
            "selected_checkpoint_sha256": self.training_result["selected_checkpoint_sha256"],
            "unseen_subject_count": len(self.ids),
            "shuffle_policy": "SORTED_UNSEEN_IDS__CYCLIC_NEXT_SUBJECT__NO_SELF_DONOR",
            "per_subject": per_subject,
            "aggregate": aggregate,
            "optimizer_created": False,
            "checkpoint_selection_performed": False,
            "unseen_gradient_updates": False,
            "qualification_threshold_applied": False,
            "generalization_pass_claimed": False,
            "product_pass_claimed": False,
        }
        _atomic_json(self.output / "UNSEEN_EVALUATION.json", result)
        return result
