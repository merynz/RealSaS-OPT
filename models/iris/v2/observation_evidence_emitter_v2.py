from __future__ import annotations

from dataclasses import dataclass
import math
import torch

try:
    from compiler.realsas_compiler_core.types import ObservationEvidenceIR, ObservationSample
except ImportError:
    from realsas_compiler_core.types import ObservationEvidenceIR, ObservationSample

from .observation_contract_v2 import ObservationContractV2
from .q_domain_v2 import RayHypothesisDomainV2
from .ray_modes_v2 import RayModesV2
from .depth_output_head_v2 import DepthOutputV2


@dataclass(frozen=True)
class EmissionPolicyV2:
    support_probability_min: float = 0.05
    emit_all_views_for_group: bool = True
    schema_version: str = "RealSaS.IRISV2.EmissionPolicy.v1"


def _point_from_anchor(contract: ObservationContractV2, view: int, grid_xy, depth: float):
    cam = contract.cameras[view]
    return cam.point_for_grid_depth(grid_xy, depth)


def emit_observation_evidence_v2(
    contracts: tuple[ObservationContractV2, ...],
    domain: RayHypothesisDomainV2,
    modes: RayModesV2,
    refined_depth: torch.Tensor,
    output: DepthOutputV2,
    *,
    policy: EmissionPolicyV2 = EmissionPolicyV2(),
) -> tuple[ObservationEvidenceIR, ...]:
    B, Q, K = modes.mode_indices.shape
    if len(contracts) != B or refined_depth.shape != (B, Q, K):
        raise ValueError("emitter shape mismatch")
    results = []
    for b in range(B):
        samples = []
        groups = {}
        uncertainty = {}
        mode_ambiguity = {}
        anchor_raster = {}
        for q in range(Q):
            av = int(domain.anchor_view[b, q].item())
            grid = tuple(map(float, domain.anchor_grid[b, q].detach().cpu().tolist()))
            anchor_pixel = contracts[b].cameras[av].grid_to_pixel_center(grid)
            for m in range(K):
                if not bool(modes.mode_valid[b, q, m].item()):
                    continue
                depth = float(refined_depth[b, q, m].item())
                if not math.isfinite(depth):
                    continue
                point = _point_from_anchor(contracts[b], av, grid, depth)
                gid = f"Q{q:06d}:M{m:02d}"
                group_ids = []
                p_support = float(output.support_probability[b, q, m].item())
                sigma = float(torch.exp(output.log_sigma[b, q, m]).item())
                uncertainty[gid] = sigma
                mode_ambiguity[gid] = bool(modes.ambiguous[b, q].item())
                anchor_raster[gid] = {
                    "view_index": av,
                    "raster_xy": (float(anchor_pixel[0]), float(anchor_pixel[1])),
                    "anchor_grid_xy": grid,
                }
                for v, cam in enumerate(contracts[b].cameras):
                    target_grid = cam.project_grid(point)
                    in_frame = bool((abs(float(target_grid[0])) <= 1.0) and (abs(float(target_grid[1])) <= 1.0))
                    target_depth = float(cam.depth_for_point(point))
                    raster = cam.grid_to_pixel_center(target_grid)
                    support = bool(in_frame and p_support >= policy.support_probability_min)
                    oid = f"IRISV2:{b:03d}:{gid}:V{v}"
                    flags = ["Q_HYPOTHESIS", "ANALYTIC_REPROJECTION"]
                    if modes.ambiguous[b, q]:
                        flags.append("AMBIGUOUS_RAY")
                    if not in_frame:
                        flags.append("OUT_OF_FRAME")
                    if not support:
                        flags.append("UNSUPPORTED")
                    samples.append(ObservationSample(
                        observation_id=oid,
                        view_index=v,
                        raster_xy=(float(raster[0]), float(raster[1])),
                        ray_origin=tuple(map(float, cam.ray_origin_for_grid(target_grid))),
                        ray_forward=tuple(map(float, cam.forward)),
                        depth=target_depth,
                        support=support,
                        provenance_ref=f"IRIS_V2|{contracts[b].contract_hash}|{gid}",
                        validity_flags=tuple(flags),
                    ))
                    group_ids.append(oid)
                groups[gid] = group_ids
        results.append(ObservationEvidenceIR(
            tuple(samples),
            camera_model="KNOWN_ORTHOGRAPHIC_8VIEW",
            observation_frame="REALSAS_OBJECT_FRAME",
            evidence_version="RealSaS.ObservationEvidenceIR.v2.IRIS_REPROJECTION",
            metadata={
                "asset_id": contracts[b].asset_id,
                "observation_contract_hash": contracts[b].contract_hash,
                "camera_hashes": [c.camera_hash for c in contracts[b].cameras],
                "hypothesis_groups": groups,
                "hypothesis_anchor_raster": anchor_raster,
                "mode_sigma": uncertainty,
                "ray_ambiguous": mode_ambiguity,
                "raster_coordinate_system": "PIXEL_CENTER_XY",
                "resolution": 1024,
                "mechanical_authority": "FORWARD_DEPTH_SUPPORT_UNCERTAINTY_ONLY",
                "local_relation_authority": "OBSERVED_ANCHOR_RASTER_LOCALITY_ONLY",
                "learned_P_head": False,
                "learned_N_head": False,
                "full_3d_reconstruction_claim": False,
            },
        ))
    return tuple(results)
