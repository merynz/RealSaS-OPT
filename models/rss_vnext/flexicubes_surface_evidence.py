from __future__ import annotations

"""Differentiable IRIS→FlexiCubes→TESSA *surface witness*, never canonical mesh.

This module reuses our pinned NVIDIA FlexiCubes adapter and the SAME live
triplane H(x) as RSS.vNext. It is NOT a second independent reconstruction
network, a topology teacher, or an assignment of source material ownership.

Signed fields are only a conditional branch. A model may *hypothesize* a
signed surface at inference, but the 2-view input alone cannot qualify a
globally watertight physical source. Open/modular/UDF source geometry needs
a separate surface-patch route and this module must not silently close it.

Training FlexiCubes emits a differentiable four-triangle split per quad;
production extraction emits a discrete two-triangle split. TESSA's output
face list is independent until correspondence/posed consequences are proven.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.iris.vnext.scene_core_v1 import IRISVNextSceneCoreV1
from .semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec,
    RSSvNextSemanticRoleSchema,
    semantic_equivalent_signed_field,
)


@dataclass
class RSSvNextFlexiCubesSurfaceEvidence:
    xyz: Tensor                 # [1,V,3] dynamically extracted *witness* points
    faces: Tensor               # [F,3] witness connectivity only; NOT TESSA mesh
    local_features: Tensor      # [1,V,H] same scene H(x), not a new latent
    presentation_role_logits: Tensor  # [1,V,K]
    regularizer: Tensor         # differentiable L_dev from pinned extractor
    sample_grid_xyz: Tensor     # [1,G,3] shared local field query coordinates
    signed_field_grid: Tensor   # [1,G] optional semantic-equivalent signed field
    cube_parameters: Tensor     # [C,21] learned β12/α8/γ1
    source_policy: str
    layer_name: str
    evidence_authority: str = "UNQUALIFIED_FLEXICUBES_SURFACE_WITNESS"

    def validate(self, *, carrier_dim: int, role_count: int) -> None:
        if self.xyz.ndim != 3 or self.xyz.shape[0] != 1 or self.xyz.shape[-1] != 3:
            raise ValueError("RSS_VNEXT_FC_WITNESS_VERTEX_SHAPE")
        n = self.xyz.shape[1]
        if self.faces.ndim != 2 or self.faces.shape[-1] != 3 or not self.faces.numel():
            raise ValueError("RSS_VNEXT_FC_WITNESS_FACE_SHAPE")
        if int(self.faces.min()) < 0 or int(self.faces.max()) >= n:
            raise ValueError("RSS_VNEXT_FC_WITNESS_FACE_ADDRESS")
        if self.local_features.shape != (1, n, carrier_dim):
            raise ValueError("RSS_VNEXT_FC_WITNESS_CARRIER_ADDRESS")
        if self.presentation_role_logits.shape != (1, n, role_count):
            raise ValueError("RSS_VNEXT_FC_WITNESS_ROLE_ADDRESS")
        if self.cube_parameters.ndim != 2 or self.cube_parameters.shape[-1] != 21:
            raise ValueError("RSS_VNEXT_FC_WITNESS_FC_PARAMETERS")
        if self.sample_grid_xyz.ndim != 3 or self.signed_field_grid.shape != self.sample_grid_xyz.shape[:2]:
            raise ValueError("RSS_VNEXT_FC_WITNESS_FIELD_DOMAIN")
        if not all(bool(torch.isfinite(t).all()) for t in (
            self.xyz, self.local_features, self.presentation_role_logits,
            self.regularizer, self.signed_field_grid, self.cube_parameters,
        )):
            raise ValueError("RSS_VNEXT_FC_WITNESS_NONFINITE")
        if self.evidence_authority != "UNQUALIFIED_FLEXICUBES_SURFACE_WITNESS":
            raise ValueError("RSS_VNEXT_FC_WITNESS_AUTHORITY_ESCALATION")


class RSSvNextFlexiCubesEvidenceBridge(nn.Module):
    """Dense-field differentiable extractor with explicit policy and resource gates.

    Works with PinnedFlexiCubesV1 (verified external operator); the operator
    is injected, never silently downloaded or replaced by marching cubes.
    """

    def __init__(
        self,
        iris: IRISVNextSceneCoreV1,
        *,
        resolution: int = 16,
        max_cubes: int = 32768,
        max_output_vertices: int = 131072,
    ) -> None:
        super().__init__()
        if not 2 <= resolution <= 64 or max_cubes < 1 or max_output_vertices < 4:
            raise ValueError("RSS_VNEXT_FC_RESOURCE_CONFIG_INVALID")
        if resolution ** 3 > max_cubes:
            raise ValueError("RSS_VNEXT_FC_CUBE_BUDGET_EXCEEDED")
        self.iris = iris  # same instance shared by all neural experts
        self.resolution = resolution
        self.max_cubes = max_cubes
        self.max_output_vertices = max_output_vertices

    def forward(
        self,
        scene_triplanes: Tensor,
        *,
        extractor,
        role_head: nn.Module,
        schema: RSSvNextSemanticRoleSchema,
        layer: RSSvNextSemanticLayerSpec,
        field_policy: str,
    ) -> RSSvNextFlexiCubesSurfaceEvidence:
        """Surface evidence from existing Z/H, with no teacher geometry inputs.

        field_policy:
          SIGNED_HYPOTHESIS_ONLY: conditional candidate, not source proof
          QUALIFIED_SOURCE_SIGNED: requires caller-side source domain admission
        Other policies (OPEN/UDF/MODULAR) are rejected rather than fabricated.
        """
        if field_policy not in (
            "SIGNED_HYPOTHESIS_ONLY", "QUALIFIED_SOURCE_SIGNED",
        ):
            raise ValueError("RSS_VNEXT_FC_SIGNED_DOMAIN_NOT_PERMITTED")
        schema.validate()
        layer.validate(schema)
        if scene_triplanes.ndim != 5 or scene_triplanes.shape[0] != 1:
            raise ValueError("RSS_VNEXT_FC_ONE_SHARED_SCENE_REQUIRED")
        if not torch.is_floating_point(scene_triplanes) or not bool(torch.isfinite(scene_triplanes).all()):
            raise ValueError("RSS_VNEXT_FC_SHARED_SCENE_INVALID")
        if not hasattr(extractor, "construct_iris_grid") or not hasattr(extractor, "extract"):
            raise ValueError("RSS_VNEXT_FC_PINNED_ADAPTER_INTERFACE_REQUIRED")
        grid, cubes = extractor.construct_iris_grid(self.resolution)
        if grid.ndim != 2 or grid.shape[1] != 3 or cubes.ndim != 2 or cubes.shape[1] != 8:
            raise ValueError("RSS_VNEXT_FC_GRID_CONTRACT")
        if grid.device != scene_triplanes.device or cubes.device != scene_triplanes.device:
            raise ValueError("RSS_VNEXT_FC_DEVICE_DRIFT")
        if cubes.dtype != torch.long or cubes.numel() == 0 or cubes.shape[0] > self.max_cubes:
            raise ValueError("RSS_VNEXT_FC_CUBE_DOMAIN_INVALID")
        if int(cubes.min()) < 0 or int(cubes.max()) >= len(grid):
            raise ValueError("RSS_VNEXT_FC_GRID_CORNER_RANGE")

        grid_xyz = grid[None].to(dtype=scene_triplanes.dtype)
        observations = self.iris.query(scene_triplanes, grid_xyz)
        field = observations["field"]
        h = observations["H"]
        role_logits = role_head(h)
        if role_logits.shape != (1, grid.shape[0], schema.count):
            raise ValueError("RSS_VNEXT_FC_ROLE_HEAD_WIDTH")
        field_layer = semantic_equivalent_signed_field(
            field, role_logits, schema=schema, layer=layer,
            # Allow hypothesis computation, never SOURCE qualification by fiat:
            qualified_signed_domain=True,
        )
        if not bool(torch.isfinite(field_layer).all()):
            raise ValueError("RSS_VNEXT_FC_SIGNED_FIELD_NONFINITE")
        # Do NOT force SDF signs or silently draw an arbitrary zero level.
        if not bool((field_layer < 0).any()) or not bool((field_layer > 0).any()):
            raise ValueError("RSS_VNEXT_FC_NO_SIGN_CHANGE_ABSTAIN")

        # Preserve learned deformed voxel positions; coordinate magnitude is
        # bounded by the scene core's extraction_delta head.
        grid_delta = observations["extraction_delta"][0]
        if grid_delta.shape != grid.shape:
            raise ValueError("RSS_VNEXT_FC_GRID_DEFORMATION_SHAPE")
        free_room = (1 - grid_xyz[0].abs()).clamp_min(0)
        deformed_grid = grid + free_room * grid_delta

        # Cube-extraction parameters arise from 8 *original* corner features,
        # using the existing 21-way head; no source mesh/skin teacher enters.
        corners = h[:, cubes]  # [1,C,8,H]
        raw = self.iris.cube_extraction(corners)
        if raw.shape != (1, cubes.shape[0], 21):
            raise ValueError("RSS_VNEXT_FC_CUBE_HEAD_WIDTH")
        extracted_xyz, faces, regularizer = extractor.extract(
            vertices_iris=deformed_grid,
            field=field_layer[0],
            cubes=cubes,
            raw_parameters=raw[0],
            resolution=self.resolution,
            training=bool(self.training),
        )
        if extracted_xyz.ndim != 2 or extracted_xyz.shape[-1] != 3 or len(extracted_xyz) > self.max_output_vertices:
            raise ValueError("RSS_VNEXT_FC_OUTPUT_VERTEX_BUDGET")
        if faces.ndim != 2 or faces.shape[-1] != 3 or not faces.numel():
            raise ValueError("RSS_VNEXT_FC_EMPTY_SURFACE")
        # Sample H at the differentiably extracted surface. TESSA consumes
        # these tokens but never treats their face indices as its own output.
        local = self.iris.query(scene_triplanes, extracted_xyz[None])["H"]
        result = RSSvNextFlexiCubesSurfaceEvidence(
            xyz=extracted_xyz[None],
            faces=faces,
            local_features=local,
            presentation_role_logits=role_head(local),
            regularizer=regularizer,
            sample_grid_xyz=grid_xyz,
            signed_field_grid=field_layer,
            cube_parameters=raw[0],
            source_policy=field_policy,
            layer_name=layer.name,
        )
        result.validate(carrier_dim=h.shape[-1], role_count=schema.count)
        return result


__all__ = [
    "RSSvNextFlexiCubesSurfaceEvidence",
    "RSSvNextFlexiCubesEvidenceBridge",
]
