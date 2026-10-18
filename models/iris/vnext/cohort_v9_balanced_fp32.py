from __future__ import annotations

"""Official support-aware FP32 execution authority for focused IRIS V9.

This layer keeps the deterministic FP32 authority from ``cohort_v9_fp32`` and
fixes two scientific issues that must not reach the A100 notebook:

1. projected alpha/appearance boundaries are rare, so ordinary within-family BCE
   would let negatives dominate the gradient;
2. current occlusion truth is positive-only, so it is evidence/metric support but
   not an identifiable binary training objective.

The official objective is grouped by independent semantic content:

``ordering`` = mean(front, visibility_a, visibility_b)
``alpha_boundary`` = class-balanced projected-alpha BCE
``appearance_boundary`` = class-balanced projected-appearance BCE

The three group losses are averaged equally.  Class weights are computed only
from the immutable TRAIN split for the current view.  Occlusion is metric-only
until negative authority exists.  Static same/cut/attachment remain abstentions.
"""

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from .cohort_v9 import (
    STATIC_RELATION_KEYS_V2,
    VIEW_RELATION_KEYS_V2,
    _family_metrics,
    _value_hash,
)
from .cohort_v9_fp32 import (
    IRISVNextFocusedRelationTrainerV9FP32,
    V9_NUMERIC_POLICY,
)
from .cohort_v1 import VNextCheckpointStoreV1


V9_SUPERVISION_POLICY = (
    "TRAIN_SPLIT_PER_VIEW_CLASS_BALANCED_GROUPED_BCE__"
    "OCCLUSION_METRIC_ONLY__STATIC_ABSTAIN_V1"
)
V9_ORDERING_FAMILIES = (
    "front_probability",
    "visibility_probability_a",
    "visibility_probability_b",
)
V9_BOUNDARY_FAMILIES = (
    "projected_alpha_boundary_probability",
    "projected_appearance_boundary_probability",
)
V9_OPTIMIZED_FAMILIES = V9_ORDERING_FAMILIES + V9_BOUNDARY_FAMILIES
V9_METRIC_ONLY_FAMILIES = ("occlusion_probability",)


def _binary_counts(target: torch.Tensor, valid: torch.Tensor) -> dict[str, int]:
    mask = valid.to(dtype=torch.bool)
    values = target.detach().float()[mask]
    if len(values) == 0:
        return {"valid": 0, "positive": 0, "negative": 0}
    binary = (values == 0.0) | (values == 1.0)
    if not bool(binary.all()):
        raise ValueError("VNEXT_V9_NONBINARY_PROBABILITY_TARGET")
    positive = int((values == 1.0).sum().item())
    return {
        "valid": int(len(values)),
        "positive": positive,
        "negative": int(len(values) - positive),
    }


def _collect_train_support(corpus) -> dict[str, Any]:
    families = V9_OPTIMIZED_FAMILIES + V9_METRIC_ONLY_FAMILIES
    result: dict[str, Any] = {}
    for family in families:
        total = {"valid": 0, "positive": 0, "negative": 0}
        per_view: dict[str, dict[str, int]] = {}
        for view_index in range(8):
            view_counts = {"valid": 0, "positive": 0, "negative": 0}
            for subject_index in range(8):
                pair_indices = corpus.train_pairs[subject_index][view_index]
                rows = corpus._rows_from_pairs(pair_indices)
                view = corpus.truth[subject_index]["views"][view_index]
                counts = _binary_counts(
                    torch.as_tensor(view["view_truth"][family])[rows],
                    torch.as_tensor(view["view_validity"][family], dtype=torch.bool)[rows],
                )
                for key in view_counts:
                    view_counts[key] += counts[key]
            per_view[str(view_index)] = view_counts
            for key in total:
                total[key] += view_counts[key]
        result[family] = {
            **total,
            "per_view": per_view,
            "optimized": family in V9_OPTIMIZED_FAMILIES,
            "metric_only": family in V9_METRIC_ONLY_FAMILIES,
        }

    for family in V9_OPTIMIZED_FAMILIES:
        row = result[family]
        if row["positive"] <= 0 or row["negative"] <= 0:
            raise ValueError("VNEXT_V9_OPTIMIZED_FAMILY_BINARY_SUPPORT_MISSING:" + family)
        for view_index in range(8):
            support = row["per_view"][str(view_index)]
            if support["positive"] <= 0 or support["negative"] <= 0:
                raise ValueError(
                    f"VNEXT_V9_OPTIMIZED_FAMILY_VIEW_BINARY_SUPPORT_MISSING:{family}:V{view_index}"
                )

    for family in V9_METRIC_ONLY_FAMILIES:
        row = result[family]
        if row["positive"] <= 0:
            raise ValueError("VNEXT_V9_METRIC_ONLY_FAMILY_HAS_NO_POSITIVE_SUPPORT:" + family)
        if row["negative"] != 0:
            raise ValueError("VNEXT_V9_OCCLUSION_POLICY_MUST_BE_REVIEWED_AFTER_NEGATIVE_AUTHORITY")
    return result


def _balanced_family_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    base = _family_metrics(probability, target, valid)
    p = probability.detach().float()[valid].clamp(1.0e-6, 1.0 - 1.0e-6)
    t = target.detach().float()[valid]
    if len(p) == 0:
        return {
            **base,
            "court_eligible": False,
            "positive_bce_mean": None,
            "negative_bce_mean": None,
            "balanced_bce_mean": None,
        }
    truth = t >= 0.5
    positive = int(truth.sum().item())
    negative = int(len(truth) - positive)
    raw = F.binary_cross_entropy(p, t, reduction="none")
    if not positive or not negative:
        return {
            **base,
            "court_eligible": False,
            "positive_bce_mean": float(raw[truth].mean().item()) if positive else None,
            "negative_bce_mean": float(raw[~truth].mean().item()) if negative else None,
            "balanced_bce_mean": None,
        }
    pos_bce = float(raw[truth].mean().item())
    neg_bce = float(raw[~truth].mean().item())
    return {
        **base,
        "court_eligible": True,
        "positive_bce_mean": pos_bce,
        "negative_bce_mean": neg_bce,
        "balanced_bce_mean": 0.5 * (pos_bce + neg_bce),
    }


class IRISVNextFocusedRelationTrainerV9BalancedFP32(IRISVNextFocusedRelationTrainerV9FP32):
    """Only optimizer authority intended for the V9 A100 notebook."""

    supervision_policy = V9_SUPERVISION_POLICY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.step != 0:
            # Existing checkpoints minted under an older supervision policy are
            # not compatible with this authority.
            raise RuntimeError("VNEXT_V9_BALANCED_POLICY_REQUIRES_FRESH_STEP_ZERO")

        self.training_support = _collect_train_support(self.corpus)
        self.binding = dict(self.binding)
        self.binding["supervision_execution"] = {
            "policy": V9_SUPERVISION_POLICY,
            "loss_groups": {
                "ordering": list(V9_ORDERING_FAMILIES),
                "alpha_boundary": [V9_BOUNDARY_FAMILIES[0]],
                "appearance_boundary": [V9_BOUNDARY_FAMILIES[1]],
            },
            "group_weighting": "EQUAL_THREE_GROUP_MEAN",
            "class_weighting": "TRAIN_SPLIT_PER_VIEW_INVERSE_CLASS_FREQUENCY",
            "optimized_families": list(V9_OPTIMIZED_FAMILIES),
            "metric_only_families": list(V9_METRIC_ONLY_FAMILIES),
            "static_families": "ABSTAIN__NO_GRADIENT",
            "train_support": self.training_support,
        }
        self.binding_hash = _value_hash(self.binding)
        final_root = self.store.root
        self.store = VNextCheckpointStoreV1(final_root, self.binding_hash)
        previous = self.store.load()
        if previous is not None:
            raise RuntimeError("VNEXT_V9_BALANCED_POLICY_CHECKPOINT_ROOT_MUST_BE_FRESH")
        self.step = 0
        self.save()

    def _official_loss(
        self,
        predicted: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
        validity: dict[str, torch.Tensor],
        view_index: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        family_losses: dict[str, torch.Tensor] = {}
        for family in V9_OPTIMIZED_FAMILIES:
            p = predicted[family]
            t = truth[family].to(device=p.device, dtype=p.dtype)
            mask = validity[family].to(device=p.device, dtype=torch.bool)
            if p.shape != t.shape or p.shape != mask.shape or not bool(mask.any()):
                raise ValueError("VNEXT_V9_BALANCED_LOSS_SHAPE_OR_VALIDITY_DRIFT:" + family)
            selected = t[mask]
            if not bool(((selected == 0.0) | (selected == 1.0)).all()):
                raise ValueError("VNEXT_V9_BALANCED_LOSS_NONBINARY_TARGET:" + family)

            support = self.training_support[family]["per_view"][str(int(view_index))]
            positive = int(support["positive"])
            negative = int(support["negative"])
            total = positive + negative
            if positive <= 0 or negative <= 0 or total <= 0:
                raise RuntimeError("VNEXT_V9_BALANCED_LOSS_SUPPORT_INVALID:" + family)
            positive_weight = float(total) / float(2 * positive)
            negative_weight = float(total) / float(2 * negative)

            raw = F.binary_cross_entropy(
                p.clamp(1.0e-6, 1.0 - 1.0e-6), t, reduction="none"
            )
            weights = torch.where(
                t >= 0.5,
                torch.full_like(t, positive_weight),
                torch.full_like(t, negative_weight),
            )
            family_losses[family] = (raw[mask] * weights[mask]).mean()

        ordering = torch.stack([family_losses[name] for name in V9_ORDERING_FAMILIES]).mean()
        alpha = family_losses[V9_BOUNDARY_FAMILIES[0]]
        appearance = family_losses[V9_BOUNDARY_FAMILIES[1]]
        final = torch.stack((ordering, alpha, appearance)).mean()
        breakdown = {
            "ordering": float(ordering.detach().item()),
            "alpha_boundary": float(alpha.detach().item()),
            "appearance_boundary": float(appearance.detach().item()),
        }
        return final, breakdown

    def one_step(self) -> dict[str, Any]:
        self._assert_numeric_policy()
        if self.step >= self.policy.maximum_optimizer_steps:
            raise StopIteration("VNEXT_V9_FOCUSED_RELATION_SCHEDULE_COMPLETE")
        next_step = self.step + 1
        view_index = next_step % 8
        batch = self.corpus.training_batch(
            step=next_step,
            view_index=view_index,
            pair_groups_per_subject=self.policy.batch_pair_groups_per_subject,
        )
        self.model.relations.view.train()
        self.model.relations.view_pair.train()
        self.optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in self.optimizer.param_groups[0]["params"]]
        predicted = self._predict(batch["point_a"], batch["point_b"], view_index)
        loss, loss_groups = self._official_loss(
            predicted, batch["truth"], batch["validity"], view_index
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("VNEXT_V9_NONFINITE_LOSS")
        loss.backward()

        gradient_sq = 0.0
        parameter_sq = 0.0
        for parameter in self.optimizer.param_groups[0]["params"]:
            p = parameter.detach().float()
            parameter_sq += float(torch.sum(p * p).item())
            if parameter.grad is not None:
                g = parameter.grad.detach().float()
                gradient_sq += float(torch.sum(g * g).item())
        self.optimizer.step()
        update_sq = 0.0
        for parameter, old in zip(self.optimizer.param_groups[0]["params"], before):
            delta = parameter.detach().float() - old.to(self.device, dtype=torch.float32)
            update_sq += float(torch.sum(delta * delta).item())
        self.step = next_step
        frozen = self._assert_frozen_non_regression()
        self._assert_numeric_policy()
        if self.step % self.policy.checkpoint_interval == 0:
            self.save()

        families = {
            name: _balanced_family_metrics(
                predicted[name], batch["truth"][name], batch["validity"][name]
            )
            for name in STATIC_RELATION_KEYS_V2 + VIEW_RELATION_KEYS_V2
        }
        parameter_l2 = math.sqrt(parameter_sq)
        update_l2 = math.sqrt(update_sq)
        return {
            "global_step": self.step,
            "view_index": view_index,
            "loss": float(loss.detach().item()),
            "loss_groups": loss_groups,
            "supervision_policy": V9_SUPERVISION_POLICY,
            "numeric_policy": V9_NUMERIC_POLICY,
            "family_metrics": families,
            "telemetry": {
                "group": "relation_view_branch",
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "parameter_l2": parameter_l2,
                "gradient_l2": math.sqrt(gradient_sq),
                "update_l2": update_l2,
                "update_parameter_ratio": update_l2 / max(parameter_l2, 1.0e-30),
            },
            "frozen_non_regression": frozen,
            "batch_subjects": 8,
            "pair_groups_per_subject": self.policy.batch_pair_groups_per_subject,
            "optimizer_steps": 1,
            "elapsed_seconds": time.monotonic() - self.started,
        }

    @torch.no_grad()
    def evaluate_heldout(self) -> dict[str, Any]:
        self._assert_numeric_policy()
        self.model.eval()
        families = STATIC_RELATION_KEYS_V2 + VIEW_RELATION_KEYS_V2
        all_probability: dict[str, list[torch.Tensor]] = {name: [] for name in families}
        all_truth: dict[str, list[torch.Tensor]] = {name: [] for name in families}
        all_validity: dict[str, list[torch.Tensor]] = {name: [] for name in families}
        per_view: dict[str, Any] = {}

        for view_index in range(8):
            batch = self.corpus.heldout_view_batch(view_index)
            predicted = self._predict(batch["point_a"], batch["point_b"], view_index)
            view_metrics: dict[str, Any] = {}
            for name in families:
                valid = batch["validity"][name] & batch["real"]
                p = predicted[name].detach().float().cpu()
                t = batch["truth"][name].detach().float().cpu()
                v = valid.detach().cpu()
                all_probability[name].append(p)
                all_truth[name].append(t)
                all_validity[name].append(v)
                view_metrics[name] = _balanced_family_metrics(p, t, v)
            per_view[str(view_index)] = view_metrics

        metrics: dict[str, Any] = {}
        for name in families:
            p = torch.cat([x.reshape(-1) for x in all_probability[name]])
            t = torch.cat([x.reshape(-1) for x in all_truth[name]])
            v = torch.cat([x.reshape(-1) for x in all_validity[name]])
            metrics[name] = _balanced_family_metrics(p, t, v)
            metrics[name]["optimized"] = name in V9_OPTIMIZED_FAMILIES
            metrics[name]["metric_only"] = name in V9_METRIC_ONLY_FAMILIES

        for name in V9_OPTIMIZED_FAMILIES:
            if not metrics[name]["court_eligible"]:
                raise RuntimeError("VNEXT_V9_OPTIMIZED_FAMILY_HELDOUT_COURT_INELIGIBLE:" + name)
        for name in V9_METRIC_ONLY_FAMILIES:
            if metrics[name]["court_eligible"]:
                raise RuntimeError("VNEXT_V9_OCCLUSION_BINARY_AUTHORITY_UNEXPECTED__REVIEW_POLICY")

        frozen = self._assert_frozen_non_regression()
        self._assert_numeric_policy()
        return {
            "schema": "RealSaS.IRISVNextFocusedRelationHeldOutMetrics.v2",
            "global_step": self.step,
            "split_policy": self.binding["split"]["policy"],
            "split_seed": self.binding["split"]["seed"],
            "holdout_fraction": self.binding["split"]["holdout_fraction"],
            "numeric_policy": V9_NUMERIC_POLICY,
            "supervision_policy": V9_SUPERVISION_POLICY,
            "families": metrics,
            "per_view_aggregate": per_view,
            "training_support": self.training_support,
            "court_policy": {
                "optimized_binary_families": list(V9_OPTIMIZED_FAMILIES),
                "metric_only_positive_support": list(V9_METRIC_ONLY_FAMILIES),
                "static_abstention_families": list(STATIC_RELATION_KEYS_V2),
                "subject_view_cells": "DIAGNOSTIC_ONLY__NO_PER_CELL_BINARY_PASS_REQUIREMENT",
            },
            "frozen_non_regression": frozen,
            "training_steps_executed": self.step,
            "scientific_fit_pass_claimed": False,
        }
