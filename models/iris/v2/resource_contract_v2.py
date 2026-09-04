from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import torch


@dataclass(frozen=True)
class IrisMaterializationEstimateV2:
    batch_size: int
    q_count: int
    depth_count: int
    view_count: int
    descriptor_dim: int
    hidden_dim: int
    spatial_neighbors: int
    element_bytes: int
    descriptor_bytes: int
    view_token_bytes: int
    pooled_bytes: int
    neighbor_hidden_bytes: int
    neighbor_point_bytes: int
    spatial_pair_bytes: int
    spatial_message_bytes: int
    spatial_logit_bytes: int
    forward_live_lower_bound_bytes: int
    schema: str = "RealSaS.IRISMaterializationEstimate.v2"

    @property
    def forward_live_lower_bound_gib(self) -> float:
        return float(self.forward_live_lower_bound_bytes) / float(2**30)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_current_iris_materialization_v2(
    *,
    batch_size: int,
    q_count: int,
    depth_count: int,
    descriptor_dim: int,
    hidden_dim: int,
    spatial_neighbors: int,
    view_count: int = 8,
    element_bytes: int = 4,
) -> IrisMaterializationEstimateV2:
    values = (batch_size, q_count, depth_count, descriptor_dim, hidden_dim, spatial_neighbors, view_count, element_bytes)
    if min(map(int, values)) < 1:
        raise ValueError("IRIS materialization estimate dimensions must be positive")
    B,Q,D,C,H,K,V,E = map(int, values)
    descriptor = B*Q*D*V*C*E
    view_tokens = B*Q*D*V*H*E
    pooled = B*Q*D*H*E
    neighbor_hidden = B*Q*K*D*H*E
    neighbor_point = B*Q*K*D*3*E
    spatial_pair = B*Q*K*D*(2*H+4)*E
    spatial_message = B*Q*K*D*H*E
    spatial_logit = B*Q*K*D*E
    # This is deliberately a lower bound, not an empirical peak estimate. In the
    # current forward(), sampled descriptors and encoded view tokens remain live
    # while EvidenceField materializes neighbor/pair/message tensors. Parameters,
    # native/foundation feature maps, allocator overhead and backward activations
    # are intentionally omitted. Exceeding device memory here is therefore a proof
    # of infeasibility for the current implementation, while passing is not a fit
    # guarantee.
    live = sum((descriptor, view_tokens, pooled, neighbor_hidden, neighbor_point, spatial_pair, spatial_message, spatial_logit))
    return IrisMaterializationEstimateV2(B,Q,D,V,C,H,K,E,descriptor,view_tokens,pooled,neighbor_hidden,neighbor_point,spatial_pair,spatial_message,spatial_logit,live)


def estimate_apparatus_materialization_v2(apparatus, domain, *, element_bytes: int = 4) -> IrisMaterializationEstimateV2:
    if not hasattr(apparatus, "learner"):
        raise TypeError("IRIS resource contract requires production apparatus")
    learner = apparatus.learner
    B,Q,D = map(int, domain.depth_values.shape)
    return estimate_current_iris_materialization_v2(
        batch_size=B,
        q_count=Q,
        depth_count=D,
        descriptor_dim=int(learner.sampler.descriptor_dim),
        hidden_dim=int(learner.field.hidden_dim),
        spatial_neighbors=int(learner.field.spatial_neighbors),
        view_count=8,
        element_bytes=int(element_bytes),
    )


def require_cuda_forward_live_set_v2(apparatus, domain, *, free_bytes: int | None = None) -> IrisMaterializationEstimateV2:
    """Fail before optimizer execution when current forward is provably too large.

    This does not claim that a PASS fits training; backward and other live tensors
    only increase memory. It prevents a known-impossible current materialization
    from being mistaken for an optimizer/model failure.
    """
    parameter = next(apparatus.learner.parameters())
    if parameter.device.type != "cuda":
        return estimate_apparatus_materialization_v2(apparatus, domain)
    if free_bytes is None:
        free_bytes, _ = torch.cuda.mem_get_info(parameter.device)
    estimate = estimate_apparatus_materialization_v2(apparatus, domain, element_bytes=4)
    if int(estimate.forward_live_lower_bound_bytes) > int(free_bytes):
        payload = {
            "status": "IRIS_RESOURCE_PREFLIGHT_FAIL_CURRENT_MATERIALIZATION",
            "device": str(parameter.device),
            "free_bytes": int(free_bytes),
            "forward_live_lower_bound_bytes": int(estimate.forward_live_lower_bound_bytes),
            "forward_live_lower_bound_gib": estimate.forward_live_lower_bound_gib,
            "note": "lower_bound_excludes_backward_parameters_feature_maps_and_allocator_overhead",
        }
        raise RuntimeError(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return estimate
