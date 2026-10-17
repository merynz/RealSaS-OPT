from __future__ import annotations

"""Differentiable model-produced surface witness for IRIS vNext."""

from dataclasses import dataclass
from typing import Any

import torch

from .flexicubes_v1 import (
    PinnedFlexiCubesV1,
    deterministic_surface_witness_v1,
    symmetric_surface_chamfer_v1,
)


@dataclass
class ExtractedSurfaceBatchV1:
    witness_xyz_iris: torch.Tensor
    witness_xyz_tessa: torch.Tensor
    witness_normal: torch.Tensor
    raw_parameters: torch.Tensor
    l_dev_mean: torch.Tensor
    extracted_vertex_counts: tuple[int, ...]
    extracted_face_counts: tuple[int, ...]


def extract_surface_batch_v1(
    model: Any,
    scene_planes: torch.Tensor,
    extractor: PinnedFlexiCubesV1,
    *,
    resolution: int,
    witness_count: int,
    training: bool,
    cube_chunk: int = 2048,
) -> ExtractedSurfaceBatchV1:
    """Extract one ragged mesh per subject after one shared scene forward.

    The field/H query over the common regular grid is batched across subjects.
    FlexiCubes itself is invoked once per subject because output topology is
    ragged; this is not the retired subject-serial model-forward trainer.
    """
    if scene_planes.ndim != 5:
        raise ValueError("VNEXT_EXTRACT_SCENE_PLANE_SHAPE_INVALID")
    batch = int(scene_planes.shape[0])
    if batch <= 0:
        raise ValueError("VNEXT_EXTRACT_BATCH_EMPTY")
    if cube_chunk <= 0:
        raise ValueError("VNEXT_EXTRACT_CUBE_CHUNK_INVALID")

    base_vertices, cubes = extractor.construct_iris_grid(int(resolution))
    base_vertices = base_vertices.to(scene_planes.device, dtype=scene_planes.dtype)
    cubes = cubes.to(scene_planes.device)
    grid = base_vertices.unsqueeze(0).expand(batch, -1, -1)
    evidence = model.query_evidence(scene_planes, grid)
    deformed_vertices = grid + evidence["extraction_delta"]
    h = evidence["H"]

    raw_chunks = []
    for start in range(0, int(cubes.shape[0]), int(cube_chunk)):
        cube = cubes[start : start + int(cube_chunk)]
        corner_h = h[:, cube]
        raw_chunks.append(model.scene_core.cube_extraction(corner_h))
    raw_parameters = torch.cat(raw_chunks, dim=1)
    if raw_parameters.shape != (batch, int(cubes.shape[0]), 21):
        raise RuntimeError("VNEXT_EXTRACT_PARAMETER_BATCH_SHAPE_DRIFT")

    witness_xyz = []
    witness_normal = []
    dev_rows = []
    vertex_counts = []
    face_counts = []
    for subject_index in range(batch):
        vertices, faces, l_dev = extractor.extract(
            vertices_iris=deformed_vertices[subject_index],
            field=evidence["field"][subject_index],
            cubes=cubes,
            raw_parameters=raw_parameters[subject_index],
            resolution=int(resolution),
            training=bool(training),
        )
        xyz, normal = deterministic_surface_witness_v1(
            vertices, faces, count=int(witness_count)
        )
        witness_xyz.append(xyz)
        witness_normal.append(normal)
        dev_rows.append(l_dev.mean() if l_dev.numel() else vertices.sum() * 0.0)
        vertex_counts.append(int(vertices.shape[0]))
        face_counts.append(int(faces.shape[0]))

    xyz_iris = torch.stack(witness_xyz, dim=0)
    normals = torch.stack(witness_normal, dim=0)
    # IRIS field coordinates are [-1,1].  TESSA's established topology grammar
    # encodes canonical XYZ in [-0.5,0.5]; make the conversion explicit.
    xyz_tessa = xyz_iris * 0.5
    return ExtractedSurfaceBatchV1(
        witness_xyz_iris=xyz_iris,
        witness_xyz_tessa=xyz_tessa,
        witness_normal=normals,
        raw_parameters=raw_parameters,
        l_dev_mean=torch.stack(dev_rows).mean(),
        extracted_vertex_counts=tuple(vertex_counts),
        extracted_face_counts=tuple(face_counts),
    )


def extracted_surface_loss_v1(
    extracted: ExtractedSurfaceBatchV1,
    teacher_surface_points_iris: torch.Tensor,
    *,
    l_dev_weight: float = 0.01,
    flexible_weight_l1: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Geometry supervision through the extracted surface, never 21-D labels."""
    if l_dev_weight < 0 or flexible_weight_l1 < 0:
        raise ValueError("VNEXT_EXTRACT_REGULARIZER_WEIGHT_NEGATIVE")
    chamfer = symmetric_surface_chamfer_v1(
        extracted.witness_xyz_iris, teacher_surface_points_iris
    )
    weight_l1 = extracted.raw_parameters.abs().mean()
    total = (
        chamfer
        + float(l_dev_weight) * extracted.l_dev_mean
        + float(flexible_weight_l1) * weight_l1
    )
    return {
        "total": total,
        "surface_chamfer": chamfer,
        "l_dev": extracted.l_dev_mean,
        "flexible_weight_l1": weight_l1,
    }
