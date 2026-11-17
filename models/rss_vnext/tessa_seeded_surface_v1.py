from __future__ import annotations

"""One-surface geometry identity for StdGEN++ -> TESSA -> AXIS/MIRA.

This is a fail-closed topology ADDRESS and gradient bridge, not a replacement
high-resolution extractor, not a learned Topology-V3.2 network and not a
ProductRevision admission.  Geometry enters only ONCE, from IRIS semantic
extraction (e.g. pinned FlexiCubes). TESSA can keep its source vertex/face
identities, or subdivide the SAME indexed surface *before* AXIS/MIRA decode.
New vertices carry explicit barycentric parent support, preserving gradients
to the original shared H(x) field and excluding a second independently
generated visual/rig mesh. No post-skin remeshing is authorized.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TESSASeededSurfaceV1:
    xyz: Tensor                # [1,V,3] derived from ONE IRIS witness
    faces: Tensor              # [F,3] original or one-to-four subdivision
    vertex_component_ids: Tensor
    face_component_ids: Tensor
    # Each V row has exact parent pairs. (i,i) denotes original seed vertex.
    parent_seed_edges: tuple[tuple[int, int], ...]
    parent_seed_weights: tuple[tuple[float, float], ...]
    original_vertex_count: int
    original_face_count: int
    scene_sha256: str
    original_surface_sha256: str
    source_authority: str = "ONE_IRIS_SURFACE_TESSA_REFINEMENT_CANDIDATE"
    canonical_mesh_authority: bool = False

    def validate(self) -> None:
        n = self.xyz.shape[1] if self.xyz.ndim == 3 else -1
        if (self.xyz.ndim != 3 or self.xyz.shape[0] != 1
                or self.xyz.shape[2] != 3 or n < 3
                or not bool(torch.isfinite(self.xyz).all())
                or self.faces.ndim != 2 or self.faces.shape[1] != 3
                or self.faces.dtype != torch.long or not self.faces.numel()
                or bool((self.faces < 0).any())
                or int(self.faces.max()) >= n
                or self.vertex_component_ids.shape != (n,)
                or self.face_component_ids.shape != (len(self.faces),)
                or self.canonical_mesh_authority
                or self.source_authority != "ONE_IRIS_SURFACE_TESSA_REFINEMENT_CANDIDATE"
                or len(self.parent_seed_edges) != n
                or len(self.parent_seed_weights) != n):
            raise ValueError("TESSA_SEEDED_SURFACE_DOMAIN_INVALID")
        for (a, b), (wa, wb) in zip(
            self.parent_seed_edges, self.parent_seed_weights
        ):
            if (not 0 <= a < self.original_vertex_count
                    or not 0 <= b < self.original_vertex_count
                    or (wa, wb) not in ((1.0, 0.0), (0.5, 0.5))
                    or (a == b) != (wa == 1.0)):
                raise ValueError("TESSA_SEEDED_PARENT_VERTEX_SUPPORT_DRIFT")
        if (self.original_vertex_count < 3 or self.original_face_count < 1
                or any(type(s) is not str or len(s) != 64
                       or any(c not in "0123456789abcdef" for c in s)
                       for s in (self.scene_sha256, self.original_surface_sha256))):
            raise ValueError("TESSA_SEEDED_SCENE_OR_SURFACE_ID_INVALID")


def build_tessa_seeded_surface_v1(
    *,
    iris_surface_xyz: Tensor,
    iris_surface_faces: Tensor,
    component_ids_by_vertex: Tensor,
    scene_sha256: str,
    original_surface_sha256: str,
    one_to_four_refine: bool = False,
) -> TESSASeededSurfaceV1:
    """Return an exact indexed surface, without *any* independent TESSA mesh.

    Small structural subdivision is a topologically safe executable identity
    proof for seeded refinement, NOT StdGEN++ high-resolution sparse extraction.
    Its parent addresses are fixed before AXIS/MIRA and all positions are
    differentiable functions of the original IRIS vertex positions. Distinct
    semantic component boundaries are NEVER welded.
    """
    v = iris_surface_xyz.shape[1] if iris_surface_xyz.ndim == 3 else -1
    faces = iris_surface_faces
    if (iris_surface_xyz.ndim != 3 or iris_surface_xyz.shape[0] != 1
            or iris_surface_xyz.shape[-1] != 3 or v < 3
            or not bool(torch.isfinite(iris_surface_xyz).all())
            or faces.ndim != 2 or faces.shape[-1] != 3
            or faces.dtype != torch.long or not faces.numel()
            or bool((faces < 0).any()) or int(faces.max()) >= v
            or component_ids_by_vertex.shape != (v,)
            or component_ids_by_vertex.dtype != torch.long
            or faces.device != iris_surface_xyz.device
            or component_ids_by_vertex.device != faces.device
            or type(one_to_four_refine) is not bool):
        raise ValueError("TESSA_SEEDED_INPUT_DOMAIN_INVALID")
    if not bool((
        component_ids_by_vertex[faces]
        == component_ids_by_vertex[faces[:, 0:1]]
    ).all()):
        raise ValueError("TESSA_SEEDED_CROSS_COMPONENT_TRIANGLE")
    if bool(torch.any(
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    )):
        raise ValueError("TESSA_SEEDED_DUPLICATE_TRIANGLE_CORNER")
    tri = iris_surface_xyz[0, faces]
    area = torch.linalg.vector_norm(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1),
        dim=-1,
    )
    if not bool((area > 1e-11).all()):
        raise ValueError("TESSA_SEEDED_DEGENERATE_REST_TRIANGLE")
    parents = [(i, i) for i in range(v)]
    weights = [(1.0, 0.0) for _ in range(v)]
    if not one_to_four_refine:
        result = TESSASeededSurfaceV1(
            xyz=iris_surface_xyz, faces=faces,
            vertex_component_ids=component_ids_by_vertex,
            face_component_ids=component_ids_by_vertex[faces[:, 0]],
            parent_seed_edges=tuple(parents), parent_seed_weights=tuple(weights),
            original_vertex_count=v, original_face_count=len(faces),
            scene_sha256=scene_sha256,
            original_surface_sha256=original_surface_sha256,
        )
        result.validate()
        return result

    # Uniform one-to-four topology ensures no T-junctions at neighboring faces.
    # The high-resolution production path will use an adaptive SAME-surface
    # refinement that must prove analogous boundary and parent-address rules.
    face_indices = faces.detach().cpu().tolist()
    indexed_edges = sorted({
        tuple(sorted((a, b))) for a, b, c in face_indices
        for a, b in ((a, b), (b, c), (c, a))
    })
    mid_id = {edge: v + i for i, edge in enumerate(indexed_edges)}
    pairs = torch.tensor(indexed_edges, dtype=torch.long, device=faces.device)
    mid = 0.5 * (iris_surface_xyz[:, pairs[:, 0]] +
                 iris_surface_xyz[:, pairs[:, 1]])
    xyz = torch.cat((iris_surface_xyz, mid), dim=1)
    comp_mid = component_ids_by_vertex[pairs[:, 0]]
    vertex_components = torch.cat((component_ids_by_vertex, comp_mid))
    out_faces = []
    for a, b, c in face_indices:
        ab = mid_id[tuple(sorted((a, b)))]
        bc = mid_id[tuple(sorted((b, c)))]
        ca = mid_id[tuple(sorted((c, a)))]
        out_faces.extend((
            (a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca),
        ))
    final_faces = torch.tensor(
        out_faces, dtype=torch.long, device=faces.device,
    )
    parents.extend(indexed_edges)
    weights.extend([(0.5, 0.5) for _ in indexed_edges])
    result = TESSASeededSurfaceV1(
        xyz=xyz, faces=final_faces,
        vertex_component_ids=vertex_components,
        face_component_ids=vertex_components[final_faces[:, 0]],
        parent_seed_edges=tuple(parents), parent_seed_weights=tuple(weights),
        original_vertex_count=v, original_face_count=len(faces),
        scene_sha256=scene_sha256,
        original_surface_sha256=original_surface_sha256,
    )
    result.validate()
    return result


__all__ = [
    "TESSASeededSurfaceV1",
    "build_tessa_seeded_surface_v1",
]
