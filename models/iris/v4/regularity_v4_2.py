from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch


@dataclass(frozen=True)
class GlobalEikonalPolicyV42:
    maximum_points_per_step: int = 1024
    finite_difference_epsilon_normalized: float = 2.0 / 511.0
    top_level_weight: float = 0.02
    interior_guard_normalized: float = 2.0 / 511.0

    def validate(self) -> None:
        if int(self.maximum_points_per_step) <= 0:
            raise ValueError("maximum_points_per_step must be positive")
        for name in (
            "finite_difference_epsilon_normalized",
            "top_level_weight",
            "interior_guard_normalized",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.interior_guard_normalized < self.finite_difference_epsilon_normalized:
            raise ValueError("interior guard must be >= finite-difference epsilon")


@dataclass(frozen=True)
class RayLipschitzPolicyV42:
    maximum_directional_lipschitz: float = 1.0
    top_level_weight: float = 0.02
    squared_excess: bool = True

    def validate(self) -> None:
        if (
            not math.isfinite(float(self.maximum_directional_lipschitz))
            or float(self.maximum_directional_lipschitz) <= 0.0
        ):
            raise ValueError("maximum_directional_lipschitz must be finite and positive")
        if not math.isfinite(float(self.top_level_weight)) or float(self.top_level_weight) <= 0.0:
            raise ValueError("top_level_weight must be finite and positive")


def _deterministic_seed(fit_seed: int, training_step: int, lane: str) -> int:
    payload = f"{int(fit_seed)}:{int(training_step)}:{lane}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def deterministic_global_points_v42(
    *,
    fit_seed: int,
    training_step: int,
    policy: GlobalEikonalPolicyV42 = GlobalEikonalPolicyV42(),
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic scrambled-Sobol points over the full canonical cube.

    These points mint no geometry truth. They only choose where the SDF regularity
    assumption is checked. The cube boundary is avoided by the finite-difference guard.
    """

    policy.validate()
    seed = _deterministic_seed(fit_seed, training_step, "global-eikonal")
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=int(seed))
    u = engine.draw(int(policy.maximum_points_per_step)).to(dtype=torch.float32)
    guard = float(policy.interior_guard_normalized)
    points = -1.0 + guard + u * (2.0 - 2.0 * guard)
    return points.to(device=device, dtype=dtype).unsqueeze(0)


def finite_difference_global_eikonal_v42(
    *,
    model,
    scene_planes: torch.Tensor,
    fit_seed: int,
    training_step: int,
    policy: GlobalEikonalPolicyV42 = GlobalEikonalPolicyV42(),
) -> dict[str, torch.Tensor]:
    """Global first-order SDF regularity without importing teacher geometry authority."""

    policy.validate()
    planes = scene_planes.float()
    points = deterministic_global_points_v42(
        fit_seed=fit_seed,
        training_step=training_step,
        policy=policy,
        device=planes.device,
        dtype=torch.float32,
    )
    eps = float(policy.finite_difference_epsilon_normalized)
    derivatives: list[torch.Tensor] = []
    with torch.autocast(device_type=planes.device.type, enabled=False):
        for axis in range(3):
            delta = torch.zeros_like(points)
            delta[..., axis] = eps
            plus = model.query(planes, points + delta)["sdf"].float()
            minus = model.query(planes, points - delta)["sdf"].float()
            derivatives.append((plus - minus) / (2.0 * eps))
    gradient = torch.stack(derivatives, dim=-1)
    norm = torch.linalg.vector_norm(gradient, dim=-1)
    error = torch.abs(norm - 1.0)
    total = torch.mean((norm - 1.0) ** 2)
    ordered = torch.sort(norm.reshape(-1)).values
    p95_index = max(0, int(math.ceil(0.95 * int(norm.numel()))) - 1)
    p99_index = max(0, int(math.ceil(0.99 * int(norm.numel()))) - 1)
    return {
        "total": total,
        "sample_count": torch.as_tensor(norm.numel(), dtype=torch.int64, device=planes.device),
        "gradient_norm_mean": norm.mean(),
        "gradient_norm_p95": ordered[p95_index],
        "gradient_norm_p99": ordered[p99_index],
        "gradient_norm_max": torch.amax(norm),
        "gradient_norm_abs_error_mean": error.mean(),
    }


def ray_directional_lipschitz_v42(
    sdf_samples: torch.Tensor,
    ray_points_normalized: torch.Tensor,
    *,
    policy: RayLipschitzPolicyV42 = RayLipschitzPolicyV42(),
) -> dict[str, torch.Tensor]:
    """Penalize adjacent ray samples that violate the SDF 1-Lipschitz contract.

    Unlike a sign-change-count loss, this does not assume how many legitimate surfaces
    a foreground ray may intersect. It only suppresses sub-sample oscillation by bounding
    directional finite differences on the exact source-ray lattice already used by V4.
    """

    policy.validate()
    if sdf_samples.ndim != 3:
        raise ValueError("sdf_samples must be [B,R,D]")
    if ray_points_normalized.shape != sdf_samples.shape + (3,):
        raise ValueError("ray_points_normalized must be [B,R,D,3]")
    if sdf_samples.shape[-1] < 2:
        raise ValueError("at least two depth samples are required")
    if not torch.isfinite(sdf_samples).all() or not torch.isfinite(ray_points_normalized).all():
        raise ValueError("ray Lipschitz inputs must be finite")

    ds = torch.abs(sdf_samples[..., 1:] - sdf_samples[..., :-1]).float()
    dx = torch.linalg.vector_norm(
        ray_points_normalized[..., 1:, :].float() - ray_points_normalized[..., :-1, :].float(),
        dim=-1,
    ).clamp_min(1e-12)
    limit = float(policy.maximum_directional_lipschitz) * dx
    excess = torch.relu(ds - limit)
    if bool(policy.squared_excess):
        total = torch.mean(excess ** 2)
    else:
        total = torch.mean(excess)
    slope = ds / dx
    p95_index = max(0, int(math.ceil(0.95 * int(slope.numel()))) - 1)
    return {
        "total": total,
        "violating_fraction": (excess > 0.0).float().mean(),
        "maximum_excess": torch.amax(excess),
        "directional_slope_mean": slope.mean(),
        "directional_slope_p95": torch.sort(slope.reshape(-1)).values[p95_index],
    }


def sign_transition_diagnostic_v42(sdf_samples: torch.Tensor) -> dict[str, torch.Tensor]:
    """Detached diagnostic only; never a training authority or loss."""

    if sdf_samples.ndim != 3 or sdf_samples.shape[-1] < 2:
        raise ValueError("sdf_samples must be [B,R,D>=2]")
    sign = sdf_samples.detach() <= 0.0
    transitions = (sign[..., 1:] != sign[..., :-1]).sum(dim=-1)
    return {
        "transition_mean": transitions.float().mean(),
        "transition_p95": torch.quantile(transitions.float().reshape(-1), 0.95),
        "transition_max": torch.amax(transitions),
        "rays_gt2_fraction": (transitions > 2).float().mean(),
        "rays_gt4_fraction": (transitions > 4).float().mean(),
    }
