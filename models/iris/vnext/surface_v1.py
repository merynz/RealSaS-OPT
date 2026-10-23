from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .flexicubes_v1 import deterministic_surface_witness_v1
from .model_v3 import IRISVNextEvidenceModelV3


class SurfaceExtractorV1(Protocol):
    def construct_iris_grid(self, resolution: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def extract(
        self,
        *,
        vertices_iris: torch.Tensor,
        field: torch.Tensor,
        cubes: torch.Tensor,
        raw_parameters: torch.Tensor,
        resolution: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class IRISSurfaceEvidenceV1:
    """Surface-addressable evidence from one frozen IRIS scene forward.

    The temporary extractor mesh is deliberately not retained. IRIS supplies
    evidence and surface witnesses only; TESSA/Compiler remain responsible for
    product Mesh decisions.
    """

    scene_planes: torch.Tensor
    surface_xyz_iris: torch.Tensor
    surface_xyz_tessa: torch.Tensor
    surface_normal: torch.Tensor
    surface_h: torch.Tensor
    mechanical: dict[str, torch.Tensor]
    contact_persistence_valid: torch.Tensor
    extracted_vertex_counts: tuple[int, ...]
    extracted_face_counts: tuple[int, ...]
    schema: str = "RealSaS.IRISSurfaceEvidence.v1"

    mesh_authority: bool = False
    extractor_faces_serialized: bool = False

    def validate(self, model: IRISVNextEvidenceModelV3) -> None:
        config = model.config
        if self.scene_planes.ndim != 5 or self.scene_planes.shape[1:3] != (
            3,
            config.plane_channels,
        ):
            raise RuntimeError("IRIS_SURFACE_SCENE_SHAPE_INVALID")
        batch = self.scene_planes.shape[0]
        witness_shape = self.surface_xyz_iris.shape
        if len(witness_shape) != 3 or witness_shape[0] != batch or witness_shape[-1] != 3:
            raise RuntimeError("IRIS_SURFACE_XYZ_SHAPE_INVALID")
        if self.surface_xyz_tessa.shape != witness_shape or self.surface_normal.shape != witness_shape:
            raise RuntimeError("IRIS_SURFACE_COORDINATE_SHAPE_DRIFT")
        if self.surface_h.shape != witness_shape[:2] + (config.local_feature_dim,):
            raise RuntimeError("IRIS_SURFACE_H_SHAPE_INVALID")
        if self.contact_persistence_valid.shape != witness_shape[:2] or self.contact_persistence_valid.dtype != torch.bool:
            raise RuntimeError("IRIS_SURFACE_CONTACT_VALIDITY_SHAPE_INVALID")
        if bool(self.contact_persistence_valid.any()):
            raise RuntimeError("IRIS_SURFACE_CONTACT_VALIDITY_MUST_ABSTAIN")
        if len(self.extracted_vertex_counts) != batch or len(self.extracted_face_counts) != batch:
            raise RuntimeError("IRIS_SURFACE_COUNT_BATCH_DRIFT")
        if any(value <= 0 for value in self.extracted_vertex_counts + self.extracted_face_counts):
            raise RuntimeError("IRIS_SURFACE_EMPTY_EXTRACTION")
        tensors = (
            self.scene_planes,
            self.surface_xyz_iris,
            self.surface_xyz_tessa,
            self.surface_normal,
            self.surface_h,
            *self.mechanical.values(),
        )
        if any(not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("IRIS_SURFACE_NUMERIC_INVALID")
        authorized_iris_max = 1.0 + float(config.maximum_extraction_delta)
        if self.surface_xyz_iris.numel() and float(self.surface_xyz_iris.abs().max()) > authorized_iris_max + 1.0e-6:
            raise RuntimeError("IRIS_SURFACE_EXCEEDS_AUTHORIZED_EXTRACTION_DELTA")
        if self.surface_xyz_tessa.numel() and float(self.surface_xyz_tessa.abs().max()) > 0.500001:
            raise RuntimeError("IRIS_SURFACE_TESSA_DOMAIN_DRIFT")
        required_mechanical = {
            "stretch_major",
            "stretch_minor",
            "tangent_direction_2theta",
            "bend_demand",
            "rigidity_probability",
            "contact_persistence_probability",
        }
        if set(self.mechanical) != required_mechanical:
            raise RuntimeError("IRIS_SURFACE_MECHANICAL_FIELDS_DRIFT")


@torch.no_grad()
def extract_surface_evidence_v1(
    model: IRISVNextEvidenceModelV3,
    scene_planes: torch.Tensor,
    extractor: SurfaceExtractorV1,
    *,
    resolution: int,
    witness_count: int,
    cube_chunk: int = 2048,
) -> IRISSurfaceEvidenceV1:
    """Extract Surface evidence without minting or serializing a product Mesh."""

    if not isinstance(model, IRISVNextEvidenceModelV3):
        raise TypeError("IRIS_SURFACE_V3_MODEL_REQUIRED")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_SURFACE_MODEL_NOT_FROZEN")
    if scene_planes.ndim != 5 or scene_planes.shape[1:3] != (
        3,
        model.config.plane_channels,
    ):
        raise RuntimeError("IRIS_SURFACE_SCENE_SHAPE_INVALID")
    if int(resolution) < 2 or int(witness_count) <= 0 or int(cube_chunk) <= 0:
        raise ValueError("IRIS_SURFACE_POLICY_INVALID")

    batch = int(scene_planes.shape[0])
    base_vertices, cubes = extractor.construct_iris_grid(int(resolution))
    if base_vertices.ndim != 2 or base_vertices.shape[-1] != 3:
        raise RuntimeError("IRIS_SURFACE_GRID_VERTEX_SHAPE_INVALID")
    if cubes.ndim != 2 or cubes.shape[-1] != 8 or cubes.shape[0] <= 0:
        raise RuntimeError("IRIS_SURFACE_GRID_CUBE_SHAPE_INVALID")
    base_vertices = base_vertices.to(scene_planes.device, dtype=scene_planes.dtype)
    cubes = cubes.to(scene_planes.device, dtype=torch.long)
    grid = base_vertices.unsqueeze(0).expand(batch, -1, -1)

    local = model.query_local(scene_planes, grid)
    deformed_vertices = grid + local["extraction_delta"]
    raw_chunks = []
    for start in range(0, int(cubes.shape[0]), int(cube_chunk)):
        cube = cubes[start : start + int(cube_chunk)]
        raw_chunks.append(model.scene_core.cube_extraction(local["H"][:, cube]))
    raw_parameters = torch.cat(raw_chunks, dim=1)
    if raw_parameters.shape != (batch, int(cubes.shape[0]), 21):
        raise RuntimeError("IRIS_SURFACE_PARAMETER_SHAPE_DRIFT")

    xyz_rows: list[torch.Tensor] = []
    normal_rows: list[torch.Tensor] = []
    vertex_counts: list[int] = []
    face_counts: list[int] = []
    for index in range(batch):
        vertices, faces, _ = extractor.extract(
            vertices_iris=deformed_vertices[index],
            field=local["field"][index],
            cubes=cubes,
            raw_parameters=raw_parameters[index],
            resolution=int(resolution),
            training=False,
        )
        xyz, normal = deterministic_surface_witness_v1(
            vertices,
            faces,
            count=int(witness_count),
        )
        xyz_rows.append(xyz)
        normal_rows.append(normal)
        vertex_counts.append(int(vertices.shape[0]))
        face_counts.append(int(faces.shape[0]))

    xyz_iris = torch.stack(xyz_rows, dim=0)
    normal = torch.stack(normal_rows, dim=0)
    authorized_iris_max = 1.0 + float(model.config.maximum_extraction_delta)
    if xyz_iris.numel() and float(xyz_iris.abs().max()) > authorized_iris_max + 1.0e-6:
        raise RuntimeError("IRIS_SURFACE_EXCEEDS_AUTHORIZED_EXTRACTION_DELTA")
    # TESSA consumes an exact [-0.5, 0.5] coordinate domain. Verify any IRIS
    # overshoot is model-authorized, then project only the structural coordinate.
    xyz_tessa = (xyz_iris * 0.5).clamp(-0.5, 0.5)

    witness = model.query_evidence(scene_planes, xyz_iris)
    mechanical = dict(witness["mechanical_demand"])
    evidence = IRISSurfaceEvidenceV1(
        scene_planes=scene_planes,
        surface_xyz_iris=xyz_iris,
        surface_xyz_tessa=xyz_tessa,
        surface_normal=normal,
        surface_h=witness["H"],
        mechanical=mechanical,
        # Contact probability is present, but product validity is not inferred
        # here. Unknown remains abstention until a typed validity source exists.
        contact_persistence_valid=torch.zeros(
            xyz_iris.shape[:2], device=xyz_iris.device, dtype=torch.bool
        ),
        extracted_vertex_counts=tuple(vertex_counts),
        extracted_face_counts=tuple(face_counts),
    )
    evidence.validate(model)
    return evidence
