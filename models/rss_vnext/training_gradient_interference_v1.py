from __future__ import annotations

"""Read-only, source-neutral gradient interference court for RSS.vNext.

This module reports *observed gradients*, not optimization advice or quality.
It never changes .grad, model parameters, optimizer state, source evidence,
loss weights, or the actual backward pass. Defaults are intentionally off:
separate reverse passes are expensive on large H(x) graphs.

The policy deliberately samples named parameters within each module family.
It reports denominators/coverage so sampled cosine is NEVER represented as
the exact full-model cosine. Unknown/zero gradients have no defined cosine,
rather than an invented zero-conflict verdict.
"""

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


SCHEMA = "RealSaS.RSSvNext.TrainingGradientInterferenceCourt.v1"
_OWNER_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SOURCE_VISION", ("vision.",)),
    ("SOURCE_SEMANTIC_PRIOR", (
        "early_semantics.", "normal_condition_proj.",
        "semantic_evidence_to_memory.",
    )),
    ("SHARED_IRIS_SCENE", ("iris.scene.",)),
    ("SHARED_IRIS_FIELD", ("iris.",)),
    ("CROSS_EXPERT_BRIDGES", (
        "presentation_roles_to_memory.", "semantic_to_memory.",
        "region_graph_to_memory.", "articulation_prior.",
        "affordance_to_memory.", "graphics_to_memory.",
        "control_cross.", "control_norm.", "fc_surface_position.",
    )),
    ("SEMANTIC_HEAD", (
        "presentation_role_head.", "semantic.", "semantic_structure.",
        "relation.", "affordance.",
    )),
    ("GRAPHICS_HEAD", (
        "graphics.", "presentation.", "surface_refine.",
        "constructed_view_renderer.",
    )),
    ("TESSA_HEAD", ("tessa.", "mesh_bridge.")),
    ("AXIS_HEAD", ("axis.",)),
    ("MIRA_HEAD", ("mira.",)),
    ("SOCKET_HEAD", ("anchor_face_correspondence.",)),
    ("DYNAMIC_HEAD", ("dynamic_refine.", "deformation_quality.")),
)


@dataclass(frozen=True)
class RSSvNextGradientCourtPolicyV1:
    """Sampling and cadence are checkpoint-pinned trainer behavior.

    max_parameters_per_owner limits the number of *named tensors*, not layer
    count. max_elements_per_owner bounds selected scalar coordinates. A
    skipped larger tensor is never treated as a measured zero gradient.
    """

    every_n_steps: int = 0
    max_parameters_per_owner: int = 5
    max_elements_per_owner: int = 65536
    report_individual_parameters: bool = True

    def validate(self) -> None:
        if (
            type(self.every_n_steps) is not int or self.every_n_steps < 0
            or type(self.max_parameters_per_owner) is not int
            or not 1 <= self.max_parameters_per_owner <= 128
            or type(self.max_elements_per_owner) is not int
            or not 128 <= self.max_elements_per_owner <= 8_388_608
            or type(self.report_individual_parameters) is not bool
        ):
            raise ValueError("RSS_VNEXT_GRADIENT_COURT_POLICY_INVALID")


def _owner(name: str) -> str | None:
    for owner, prefixes in _OWNER_PREFIXES:
        if name.startswith(prefixes):
            return owner
    return None


def _selected_owners(
    parameters: Sequence[tuple[str, nn.Parameter]],
    policy: RSSvNextGradientCourtPolicyV1,
) -> dict[str, tuple[list[tuple[str, nn.Parameter]], int, int]]:
    domains: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, param in parameters:
        owner = _owner(name)
        if owner is not None and param.requires_grad:
            domains.setdefault(owner, []).append((name, param))
    selected: dict[str, tuple[list[tuple[str, nn.Parameter]], int, int]] = {}
    for owner, candidates in domains.items():
        kept: list[tuple[str, nn.Parameter]] = []
        n_elements = 0
        for name, param in candidates:
            count = param.numel()
            if (
                len(kept) >= policy.max_parameters_per_owner
                or count > policy.max_elements_per_owner - n_elements
            ):
                continue
            kept.append((name, param))
            n_elements += count
        if kept:
            selected[owner] = (kept, len(candidates), sum(p.numel() for _, p in candidates))
    return selected


def _cosine(a: Tensor, b: Tensor, na: float, nb: float) -> float | None:
    if na <= 0. or nb <= 0.:
        return None
    value = float(torch.dot(a, b).item()) / (na * nb)
    # Rounding close to +/-1 is harmless; far-out values are not.
    if value < -1.000001 or value > 1.000001 or not isfinite(value):
        raise RuntimeError("RSS_VNEXT_GRADIENT_COURT_COSINE_NUMERICS")
    return max(-1., min(1., value))


def audit_rss_vnext_gradient_interference_v1(
    *,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    loss_terms: Mapping[str, tuple[Tensor, float]],
    policy: RSSvNextGradientCourtPolicyV1,
) -> dict:
    """Take isolated task gradients on sampled owners without perturbing step.

    loss_terms holds ALREADY admitted and nonempty objective terms. All
    weights must be finite and strictly positive. Zero/absent terms are not
    made up by this court. Retain the graph so the caller's regular
    weighted-sum backward still runs exactly as before.
    """
    policy.validate()
    if policy.every_n_steps == 0:
        raise ValueError("RSS_VNEXT_GRADIENT_COURT_DISABLED")
    if not isinstance(loss_terms, Mapping) or not loss_terms:
        raise ValueError("RSS_VNEXT_GRADIENT_COURT_NO_ADMITTED_LOSSES")
    terms = sorted(loss_terms.items())
    for name, pair in terms:
        if (
            type(name) is not str or not name
            or type(pair) is not tuple or len(pair) != 2
            or not isinstance(pair[0], Tensor)
            or pair[0].numel() != 1 or not pair[0].requires_grad
            or type(pair[1]) not in (int, float)
            or not isfinite(pair[1]) or pair[1] <= 0.
        ):
            raise ValueError("RSS_VNEXT_GRADIENT_COURT_LOSS_DOMAIN")
    selected = _selected_owners(tuple(named_parameters), policy)
    if not selected:
        raise ValueError("RSS_VNEXT_GRADIENT_COURT_NO_AUDITABLE_PARAMETERS")

    result: dict = {
        "schema": SCHEMA,
        "authority": "OBSERVED_SAMPLED_GRADIENTS_NOT_FULL_MODEL_OR_QUALITY",
        "modifies_gradient": False,
        "uses_task_projection": False,
        "optimizer_step_performed": False,
        "loss_families": [name for name, _ in terms],
        "policy": asdict(policy),
        "owners": {},
        "global_pairwise_conflict_estimate": None,
    }
    for owner, (params, eligible_n, eligible_elements) in selected.items():
        tensor_params = tuple(p for _, p in params)
        by_term: dict[str, Tensor] = {}
        norms: dict[str, float] = {}
        weighted_norms: dict[str, float] = {}
        parameter_grads: dict[str, list[Tensor]] = {}
        for name, (loss, weight) in terms:
            raw = torch.autograd.grad(
                loss, tensor_params, retain_graph=True,
                create_graph=False, allow_unused=True,
                materialize_grads=False,
            )
            per_param = [
                (
                    g.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
                    if g is not None
                    else torch.zeros(p.numel(), dtype=torch.float64)
                )
                for (_, p), g in zip(params, raw)
            ]
            if any(not bool(torch.isfinite(g).all()) for g in per_param):
                raise RuntimeError("RSS_VNEXT_GRADIENT_COURT_NONFINITE_GRADIENT")
            vector = torch.cat(per_param)
            by_term[name] = vector
            parameter_grads[name] = per_param
            norm = float(torch.linalg.vector_norm(vector).item())
            norms[name] = norm
            weighted_norms[name] = norm * float(weight)

        pairs = []
        worst_parameters = []
        for i, (left, _) in enumerate(terms):
            for right, _ in terms[i+1:]:
                cos = _cosine(by_term[left], by_term[right], norms[left], norms[right])
                pairs.append({
                    "left": left, "right": right, "cosine": cos,
                    "negative_interference": cos is not None and cos < -1.e-6,
                })
                if policy.report_individual_parameters:
                    for index, (param_name, _) in enumerate(params):
                        a = parameter_grads[left][index]
                        b = parameter_grads[right][index]
                        param_cos = _cosine(
                            a, b,
                            float(torch.linalg.vector_norm(a).item()),
                            float(torch.linalg.vector_norm(b).item()),
                        )
                        if param_cos is not None and param_cos < -1.e-6:
                            worst_parameters.append({
                                "name": param_name, "left": left,
                                "right": right, "cosine": param_cos,
                            })
        usable = [p for p in pairs if p["cosine"] is not None]
        weighted = sum(
            (by_term[name] * float(weight)
             for name, (_, weight) in terms),
            torch.zeros_like(next(iter(by_term.values()))),
        )
        weighted_total = sum(weighted_norms.values())
        weighted_result = float(torch.linalg.vector_norm(weighted).item())
        cancellation_ratio = (
            weighted_result / weighted_total if weighted_total > 0. else None
        )
        if (
            cancellation_ratio is not None
            and (not isfinite(cancellation_ratio)
                 or cancellation_ratio > 1.000001)
        ):
            raise RuntimeError("RSS_VNEXT_GRADIENT_COURT_CANCELLATION_NUMERICS")
        nonzero = [v for v in weighted_norms.values() if v > 0.]
        result["owners"][owner] = {
            "sampled_parameter_names": [name for name, _ in params],
            "sampled_parameter_count": len(params),
            "eligible_parameter_count": eligible_n,
            "sampled_scalar_elements": sum(p.numel() for _, p in params),
            "eligible_scalar_elements": eligible_elements,
            "coverage_is_complete": len(params) == eligible_n,
            "per_family_gradient_norms": norms,
            "weighted_gradient_norms": weighted_norms,
            "pairwise": pairs,
            "pairwise_defined_count": len(usable),
            "pairwise_negative_count": sum(
                1 for p in usable if p["negative_interference"]
            ),
            "worst_pair_cosine": (
                min(p["cosine"] for p in usable) if usable else None
            ),
            "cancellation_ratio": (
                min(1., max(0., cancellation_ratio))
                if cancellation_ratio is not None else None
            ),
            "weighted_dominance_ratio": (
                max(nonzero) / min(nonzero) if len(nonzero) >= 2 else None
            ),
            "worst_sampled_parameters": sorted(
                worst_parameters,
                key=lambda entry: (entry["cosine"], entry["name"]),
            )[:20],
        }
    return result


__all__ = [
    "SCHEMA", "RSSvNextGradientCourtPolicyV1",
    "audit_rss_vnext_gradient_interference_v1",
]
