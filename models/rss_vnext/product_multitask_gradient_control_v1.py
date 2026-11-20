from __future__ import annotations

"""Opt-in, source-qualified multi-task gradient routing for ONE RSS.vNext M*.

This is a *bounded research candidate*, not a claim that PCGrad/Recon/GradNorm
was reproduced or won on a licensed corpus. Its default is the actual product
weighted sum. Only the explicitly selected SHARED owners can be rerouted.
No expert-private gradient, teacher validity mask, loss coefficient, parameter
freeze, optimizer moments, or discrete topology/FSQ token is modified.
"""
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

SCHEMA = "RealSaS.RSSvNext.ProductMultitaskGradientRouting.v1"
MODES = frozenset({"WEIGHTED_SUM", "BOUNDED_SHARED_PCGRAD"})
FAMILIES = ("GEOMETRY", "SEMANTIC", "GRAPHICS", "JOINT_RENDER", "MECHANICS", "POSED_MSTAR_RENDER", "POSED_LAYER_CLEARANCE", "CONTROL_INTERVENTION")
# Ownership is explicit. Introducing additional owners requires code review.
SHARED_OWNERS = (
    "iris.scene.",
    "tessa_seeded_head.",
    "surface_mechanics_context.",
)


@dataclass(frozen=True)
class ProductMultitaskGradientPolicyV1:
    mode: str = "WEIGHTED_SUM"
    shared_owner_prefixes: tuple[str, ...] = ("iris.scene.",)
    every_n_steps: int = 1
    projection_strength: float = 0.5
    max_candidate_norm_ratio: float = 1.25
    max_shared_elements: int = 24_000_000
    max_dense_probe_bytes: int = 768 * 1024 * 1024

    def validate(self) -> None:
        if (
            self.mode not in MODES
            or not isinstance(self.shared_owner_prefixes, tuple)
            or not self.shared_owner_prefixes
            or len(set(self.shared_owner_prefixes)) != len(self.shared_owner_prefixes)
            or not set(self.shared_owner_prefixes).issubset(SHARED_OWNERS)
            or type(self.every_n_steps) is not int or self.every_n_steps < 1
            or type(self.max_shared_elements) is not int
            or self.max_shared_elements < 1
            or type(self.max_dense_probe_bytes) is not int
            or self.max_dense_probe_bytes < 1024
            or any(
                type(v) not in (float, int) or not math.isfinite(v)
                for v in (self.projection_strength, self.max_candidate_norm_ratio)
            )
            or not 0.0 <= self.projection_strength <= 1.0
            or not 1.0 <= self.max_candidate_norm_ratio <= 2.0
        ):
            raise ValueError("RSS_VNEXT_MULTITASK_GRADIENT_POLICY_INVALID")

    @property
    def enabled(self) -> bool:
        return self.mode != "WEIGHTED_SUM"


def admitted_product_task_losses_v1(
    result, *,
    component_udf_weight: float,
    graphics_rgb_weight: float,
    graphics_joint_ray_weight: float,
    anchor_face_weight: float,
    mstar_pose_raster_weight: float = 0.,
    native_layer_clearance_weight: float = 0.,
    native_control_intervention_weight: float = 0.,
) -> dict[str, Tensor]:
    """Partition EXACTLY the qualified product total; never duplicate a subloss.

    All source registries and masks have already been validated by the owning
    product objective. We use its aggregate losses, not teacher-shaped inputs.
    An absent sparse family is UNKNOWN, not a zero-valued negative example.
    """
    total = result.total_loss
    if not isinstance(total, Tensor) or total.numel() != 1:
        raise RuntimeError("RSS_VNEXT_MULTITASK_TOTAL_LOSS_INVALID")
    if result.seeded_tessa_loss is None:
        raise RuntimeError("RSS_VNEXT_MULTITASK_PRODUCT_SEEDED_SOURCE_REQUIRED")
    if any(
        type(weight) not in (int, float) or not math.isfinite(weight)
        or weight < 0
        for weight in (
            component_udf_weight, graphics_rgb_weight,
            graphics_joint_ray_weight, anchor_face_weight,
            mstar_pose_raster_weight, native_layer_clearance_weight,
            native_control_intervention_weight,
        )
    ):
        raise ValueError("RSS_VNEXT_MULTITASK_LOSS_WEIGHTS_INVALID")

    def checked(value: Tensor | None, name: str) -> Tensor | None:
        if value is None:
            return None
        if (
            not isinstance(value, Tensor)
            or value.shape != total.shape
            or value.device != total.device
            or not bool(torch.isfinite(value.detach()).all())
        ):
            raise RuntimeError("RSS_VNEXT_MULTITASK_FAMILY_LOSS_INVALID:" + name)
        return value

    groups: dict[str, Tensor] = {}
    geo = checked(result.seeded_tessa_loss, "seeded_tessa")
    if component_udf_weight and result.component_udf_loss is not None:
        geo = geo + component_udf_weight * checked(
            result.component_udf_loss, "component_udf"
        )
    groups["GEOMETRY"] = geo

    sem = checked(result.product_semantic_loss, "semantic")
    if sem is not None:
        groups["SEMANTIC"] = sem

    # A first-hit visible RGB objective is NOT the same gradient task as
    # 3D ray-rendered RGB/alpha/semantic transmittance. The second objective
    # reaches multiple IRIS heads; folding it into Graphics would falsify
    # measured Geometry-vs-Graphics gradient conflict.
    if result.graphics_source_rgb_loss is not None:
        if graphics_rgb_weight <= 0:
            raise RuntimeError("RSS_VNEXT_MULTITASK_UNWEIGHTED_ACTIVE_RGB")
        groups["GRAPHICS"] = graphics_rgb_weight * checked(
            result.graphics_source_rgb_loss, "visible_rgb"
        )
    if result.realsas_joint_slrm_loss is not None:
        if graphics_joint_ray_weight <= 0:
            raise RuntimeError("RSS_VNEXT_MULTITASK_UNWEIGHTED_JOINT_RENDER")
        groups["JOINT_RENDER"] = graphics_joint_ray_weight * checked(
            result.realsas_joint_slrm_loss, "source_rgb_alpha_semantic_rays"
        )

    # Screen-space M* render backpropagates to Geometry, SLR, Graphics, AXIS
    # and MIRA. It is NOT only Graphics or the earlier shared-H ray objective.
    # Keeping it separate preserves honest inter-task gradient attribution.
    posed_raster_loss = getattr(result, "product_mstar_pose_raster_loss", None)
    if posed_raster_loss is not None:
        if mstar_pose_raster_weight <= 0:
            raise RuntimeError("RSS_VNEXT_MULTITASK_UNWEIGHTED_POSED_MSTAR")
        groups["POSED_MSTAR_RENDER"] = mstar_pose_raster_weight * checked(
            posed_raster_loss, "posed_mstar_render",
        )

    clearance_loss = getattr(result,"product_native_layer_clearance_loss",None)
    if clearance_loss is not None:
        if native_layer_clearance_weight <= 0:
            raise RuntimeError("RSS_VNEXT_MULTITASK_UNWEIGHTED_LAYER_CLEARANCE")
        groups["POSED_LAYER_CLEARANCE"] = native_layer_clearance_weight * checked(
            clearance_loss,"source_native_layer_clearance",
        )

    control_loss = getattr(result,"product_native_control_intervention_loss",None)
    if control_loss is not None:
        if native_control_intervention_weight <= 0:
            raise RuntimeError("RSS_VNEXT_MULTITASK_UNWEIGHTED_CONTROL_INTERVENTION")
        groups["CONTROL_INTERVENTION"] = native_control_intervention_weight * checked(
            control_loss,"source_native_control_intervention",
        )

    mechanical = []
    if result.product_axis_mira_loss is not None:
        mechanical.append(checked(result.product_axis_mira_loss, "axis_mira"))
    if result.anchor_face_correspondence_loss is not None:
        if anchor_face_weight <= 0:
            raise RuntimeError("RSS_VNEXT_MULTITASK_UNWEIGHTED_ANCHOR")
        mechanical.append(
            anchor_face_weight * checked(
                result.anchor_face_correspondence_loss, "anchor"
            )
        )
    if mechanical:
        groups["MECHANICS"] = sum(mechanical[1:], mechanical[0])

    reconstructed = sum(
        (groups[k] for k in FAMILIES if k in groups),
        torch.zeros_like(total),
    )
    if not bool(torch.allclose(
        reconstructed.detach(), total.detach(), rtol=1e-5, atol=1e-6
    )):
        raise RuntimeError("RSS_VNEXT_MULTITASK_LOSS_PARTITION_DRIFT")
    return groups


@dataclass
class PreparedProductGradientRouteV1:
    """Short-lived, graph-bound probe, consumed before optimizer.step()."""

    parameters: tuple[nn.Parameter, ...]
    baseline: Tensor
    candidate: Tensor
    names: tuple[str, ...]
    report: dict

    def apply_after_backward(self) -> dict:
        actual = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p))
            .detach().reshape(-1).to(dtype=torch.float32)
            for p in self.parameters
        ])
        if actual.shape != self.baseline.shape or not bool(
            torch.isfinite(actual).all()
        ):
            raise RuntimeError("RSS_VNEXT_MULTITASK_BACKWARD_NONFINITE_OR_DRIFT")
        # A partial/duplicate task decomposition may agree in scalar value
        # but not in gradient. Never overwrite such a production gradient.
        deviation = torch.linalg.vector_norm(actual - self.baseline)
        tolerance = 1e-5 + 5e-4 * torch.linalg.vector_norm(actual)
        if bool(deviation > tolerance):
            raise RuntimeError("RSS_VNEXT_MULTITASK_BACKWARD_GRAPH_DRIFT")
        if self.report["projection_applied"]:
            start = 0
            for param in self.parameters:
                size = param.numel()
                part = self.candidate[start:start + size].reshape(param.shape)
                start += size
                if param.grad is None:
                    if bool(torch.any(part != 0)):
                        raise RuntimeError(
                            "RSS_VNEXT_MULTITASK_CREATED_UNOWNED_GRADIENT"
                        )
                else:
                    param.grad.copy_(part.to(
                        device=param.grad.device, dtype=param.grad.dtype
                    ))
        return dict(self.report)


def _flatten_probe(
    group_loss: Tensor, parameters: tuple[nn.Parameter, ...]
) -> Tensor:
    grads = torch.autograd.grad(
        group_loss, parameters, retain_graph=True, allow_unused=True,
        create_graph=False,
    )
    return torch.cat([
        (g.detach() if g is not None else torch.zeros_like(p))
        .reshape(-1).to(dtype=torch.float32)
        for g, p in zip(grads, parameters)
    ])


def prepare_product_gradient_route_v1(
    *, named_parameters: Sequence[tuple[str, nn.Parameter]],
    task_losses: Mapping[str, Tensor],
    policy: ProductMultitaskGradientPolicyV1,
    global_step: int,
) -> PreparedProductGradientRouteV1 | None:
    """Probe ACTUAL qualified objective graph; default never alters update.

    The deterministic pair projection is an explicitly named PCGrad-inspired
    candidate. It is NOT the original randomized PCGrad implementation and
    is NOT a general CAGrad/MGDA solver. The step is bounded, and it falls
    back to the exact weighted sum if first-order *total* descent is lost.
    """
    policy.validate()
    if type(global_step) is not int or global_step < 0:
        raise ValueError("RSS_VNEXT_MULTITASK_GLOBAL_STEP_INVALID")
    if not policy.enabled or global_step % policy.every_n_steps:
        return None
    if not set(task_losses).issubset(FAMILIES):
        raise ValueError("RSS_VNEXT_MULTITASK_UNKNOWN_LOSS_FAMILY")
    active = [
        (name, task_losses[name])
        for name in FAMILIES
        if name in task_losses
        and isinstance(task_losses[name], Tensor)
        and task_losses[name].requires_grad
    ]
    if len(active) < 2:
        return None

    chosen = tuple((name, p) for name, p in named_parameters
                   if p.requires_grad and any(
                       name.startswith(prefix)
                       for prefix in policy.shared_owner_prefixes
                   ))
    if not chosen:
        raise RuntimeError("RSS_VNEXT_MULTITASK_SHARED_OWNER_EMPTY")
    parameters = tuple(p for _, p in chosen)
    element_count = sum(p.numel() for p in parameters)
    # K probes + baseline + candidate + backward comparison, all float32.
    required_bytes = 4 * element_count * (len(active) + 3)
    if (
        element_count > policy.max_shared_elements
        or required_bytes > policy.max_dense_probe_bytes
    ):
        raise RuntimeError("RSS_VNEXT_MULTITASK_GRADIENT_MEMORY_BUDGET")
    probes = {
        name: _flatten_probe(loss, parameters)
        for name, loss in active
    }
    live = {
        name: gradient for name, gradient in probes.items()
        if bool(torch.any(gradient != 0))
    }
    if len(live) < 2:
        return None
    base = torch.stack(list(live.values())).sum(dim=0)
    if not bool(torch.isfinite(base).all()):
        raise RuntimeError("RSS_VNEXT_MULTITASK_BASELINE_NONFINITE")
    zero = base.new_zeros(())
    conflicting = []
    names = tuple(live)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dot = torch.dot(live[a], live[b])
            if bool(dot < zero):
                conflicting.append((a, b))
    candidate = base.clone()
    applied = False
    if conflicting:
        # Task-private gradients remain untouched. Route the entire named
        # shared-owner vector (not biased sampled parameter fragments).
        projected = []
        for a in names:
            direction = live[a].clone()
            for b in names:
                if a == b:
                    continue
                reference = live[b]
                dot = torch.dot(direction, reference)
                if bool(dot < zero):
                    denom = torch.dot(reference, reference)
                    if bool(denom > zero):
                        direction = direction - (dot / denom) * reference
            projected.append(direction)
        proposed = torch.stack(projected).sum(dim=0)
        candidate = (
            (1 - policy.projection_strength) * base
            + policy.projection_strength * proposed
        )
        base_norm = torch.linalg.vector_norm(base)
        candidate_norm = torch.linalg.vector_norm(candidate)
        if bool(candidate_norm > policy.max_candidate_norm_ratio * base_norm):
            candidate = candidate * (
                policy.max_candidate_norm_ratio * base_norm
                / candidate_norm.clamp_min(1e-30)
            )
        if not bool(torch.isfinite(candidate).all()) or (
            bool(base_norm > zero)
            and bool(torch.dot(candidate, base) <= zero)
        ):
            candidate = base.clone()
        applied = not bool(torch.equal(candidate, base))
    baseline_norm = float(torch.linalg.vector_norm(base).item())
    report = {
        "schema": SCHEMA,
        "mode": policy.mode,
        "shared_owner_prefixes": list(policy.shared_owner_prefixes),
        "active_shared_task_families": list(names),
        "conflicting_task_pairs": [list(x) for x in conflicting],
        "projection_applied": applied,
        "baseline_shared_gradient_norm": baseline_norm,
        "candidate_shared_gradient_norm":
            float(torch.linalg.vector_norm(candidate).item()),
        "first_order_total_descent_preserved": bool(
            torch.dot(candidate, base) >= zero
        ),
        "shared_element_count": element_count,
        "probe_buffer_estimated_bytes": required_bytes,
        "sampled_gradient_court": False,
        "task_private_gradients_preserved": True,
        "discrete_topology_and_fsq_operators_unchanged": True,
        "independent_source_truth_inferred": False,
        "product_release_qualified": False,
        "global_step_before_update": global_step,
    }
    return PreparedProductGradientRouteV1(
        parameters=parameters, baseline=base, candidate=candidate,
        names=names, report=report,
    )


__all__ = [
    "SCHEMA", "FAMILIES", "SHARED_OWNERS",
    "ProductMultitaskGradientPolicyV1",
    "PreparedProductGradientRouteV1",
    "admitted_product_task_losses_v1",
    "prepare_product_gradient_route_v1",
]
