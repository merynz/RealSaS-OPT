"""Materialize inference-equivalent TESSA conditioning from a sealed Truth139 IRIS model.

The cache contains no teacher topology.  It is derived solely from the student observation packet,
the sealed IRIS checkpoint, deterministic zero-surface extraction and inference-available IRIS
evidence.  Training can therefore reuse it across topology windows/epochs without repeatedly
querying a dense field or opening teacher mesh bytes on the conditioning side.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from models.iris.v3.zero_surface_decoder_v3 import dense_signed_grid_v3, extract_zero_surface_mesh_v3
from .tessa_bridge_v1 import Truth139TESSASourceBuilderV1
from .truth139_model_v1 import IRISV31Truth139V1


@dataclass(frozen=True)
class Truth139TESSAConditioningPolicyV1:
    zero_surface_resolution: int = 128
    query_chunk_size: int = 262144
    maximum_surface_witnesses: int = 8192

    def validate(self) -> None:
        if min(self.zero_surface_resolution, self.query_chunk_size, self.maximum_surface_witnesses) <= 0:
            raise ValueError("TRUTH139_TESSA_CONDITIONING_POLICY_INVALID")
        if self.zero_surface_resolution < 32:
            raise ValueError("TRUTH139_TESSA_ZERO_SURFACE_RESOLUTION_TOO_SMALL")


def spatially_stratified_witness_indices_v1(
    vertices: np.ndarray,
    *,
    maximum: int,
) -> np.ndarray:
    """Deterministic spatial-coverage sample without subject-specific semantics.

    Marching-cubes vertices are quantized to a fixed 10-bit lattice, Morton-like ordered by a
    stable interleaved bit key, and then sampled at evenly spaced ranks.  The policy depends only
    on geometry coordinates and a global maximum, not subject identity or morphology class.
    """
    value = np.asarray(vertices, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError("TRUTH139_TESSA_WITNESS_VERTICES_INVALID")
    if int(maximum) <= 0:
        raise ValueError("TRUTH139_TESSA_WITNESS_MAXIMUM_INVALID")
    if len(value) <= int(maximum):
        return np.arange(len(value), dtype=np.int64)
    if np.any(value < -1.000001) or np.any(value > 1.000001):
        raise ValueError("TRUTH139_TESSA_WITNESS_DOMAIN_DRIFT")

    q = np.clip(np.floor((value + 1.0) * 0.5 * 1023.0 + 0.5), 0, 1023).astype(np.uint32)
    key = np.zeros(len(q), dtype=np.uint64)
    for bit in range(10):
        key |= ((q[:, 0] >> bit) & 1).astype(np.uint64) << (3 * bit)
        key |= ((q[:, 1] >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        key |= ((q[:, 2] >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    order = np.argsort(key, kind="mergesort")
    rank = np.linspace(0, len(order) - 1, int(maximum), dtype=np.int64)
    chosen = order[rank]
    if len(np.unique(chosen)) != int(maximum):
        raise AssertionError("TRUTH139_TESSA_WITNESS_SAMPLE_DUPLICATE")
    return chosen.astype(np.int64)


@torch.no_grad()
def build_truth139_tessa_conditioning_v1(
    *,
    iris_model: IRISV31Truth139V1,
    carrier: torch.Tensor,
    source_builder: Truth139TESSASourceBuilderV1,
    policy: Truth139TESSAConditioningPolicyV1 = Truth139TESSAConditioningPolicyV1(),
) -> dict:
    policy.validate()
    if carrier.ndim != 5 or carrier.shape[0] != 1:
        raise ValueError("TRUTH139_TESSA_CONDITIONING_CARRIER_SHAPE")
    device = carrier.device

    def query(points: torch.Tensor) -> torch.Tensor:
        gg, _ = iris_model.query_states(carrier, points.to(device=device, dtype=carrier.dtype))
        return iris_model.geometry(gg)[..., 0].float()

    signed_grid = dense_signed_grid_v3(
        query,
        resolution=int(policy.zero_surface_resolution),
        bounds=(-1.0, 1.0),
        chunk_size=int(policy.query_chunk_size),
        device=device,
    )
    zero = extract_zero_surface_mesh_v3(signed_grid, bounds=(-1.0, 1.0), level=0.0)
    ids = spatially_stratified_witness_indices_v1(
        zero.vertices_normalized,
        maximum=int(policy.maximum_surface_witnesses),
    )
    witness = torch.from_numpy(zero.vertices_normalized[ids]).to(
        device=device, dtype=carrier.dtype
    )[None]
    scene = source_builder.scene_source(carrier).detach().float().cpu().contiguous()
    surface = source_builder.surface_source(
        iris_model=iris_model,
        carrier=carrier,
        witness_xyz_iris=witness,
    ).detach().float().cpu().contiguous()
    witness_cpu = witness.detach().float().cpu().contiguous()
    digest = hashlib.sha256(
        scene.numpy().tobytes() + surface.numpy().tobytes() + witness_cpu.numpy().tobytes()
    ).hexdigest()
    return {
        "schema": "RealSaS.Truth139TESSAConditioningCache.v1",
        "source_semantics": "SEALED_STUDENT_IRIS_ONLY__NO_TEACHER_TOPOLOGY",
        "scene_source": scene,
        "surface_source": surface,
        "witness_xyz_iris": witness_cpu,
        "surface_witness_count": int(surface.shape[1]),
        "zero_surface_vertex_count_before_sampling": int(len(zero.vertices_normalized)),
        "zero_surface_face_count": int(len(zero.faces)),
        "zero_surface_field_min": float(zero.field_min),
        "zero_surface_field_max": float(zero.field_max),
        "zero_surface_resolution": int(policy.zero_surface_resolution),
        "maximum_surface_witnesses": int(policy.maximum_surface_witnesses),
        "conditioning_tensor_sha256": digest,
        "teacher_topology_used": False,
        "semantic_part_taxonomy_used": False,
    }
