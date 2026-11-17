from __future__ import annotations

"""Join signed and open IRIS extraction into ONE pre-rig indexed TESSA seed.

The two extraction operators consume the SAME shared H(x), not independent
geometry models. Every vertex/face index is globally unique; even perfectly
coincident body/cloth coordinates are never welded by XYZ proximity.

The open-vs-closed component proposal remains unqualified. This is opt-in
research and never mints product admission or source supervision.
"""

from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import Tensor

from .layered_iris_seed_v1 import layer_namespace_digest_v1
from .native_layer_open_boundary_v1 import exact_open_boundary_edges_v1
from .open_udf_surface_v1 import (
    OpenUDFSurfaceCandidateV1, tensor_bytes_sha256_v1,
)
from .tessa_seeded_surface_v1 import (
    TESSASeededSurfaceV1, build_tessa_seeded_surface_v1,
)

SCHEMA = "RealSaS.RSSvNext.HybridSignedOpenIndexedSeed.v1"
SIGNED_KIND = "SIGNED_IRIS_H_CLOSED"
OPEN_KIND = "UDF_IRIS_H_OPEN"


@dataclass(frozen=True)
class HybridSignedOpenIRISSeedV1:
    seed: TESSASeededSurfaceV1
    signed_vertex_count: int
    signed_face_count: int
    open_vertex_count: int
    open_face_count: int
    component_surface_kinds: tuple[str,...]
    open_candidate_sample_sha256: str
    source_authority: str = "ONE_IRIS_H_HYBRID_GEOMETRIC_PROPOSAL"
    product_admission_claimed: bool = False

    def validate(self) -> None:
        self.seed.validate()
        v = self.signed_vertex_count + self.open_vertex_count
        f = self.signed_face_count + self.open_face_count
        comp = self.seed.vertex_component_ids
        kind = self.component_surface_kinds
        if (
            self.source_authority != "ONE_IRIS_H_HYBRID_GEOMETRIC_PROPOSAL"
            or self.product_admission_claimed
            or v != self.seed.original_vertex_count
            or f != self.seed.original_face_count
            or self.seed.xyz.shape[1] != v or len(self.seed.faces) != f
            or self.signed_vertex_count < 3 or self.open_vertex_count < 3
            or self.signed_face_count < 1 or self.open_face_count < 1
            or len(kind) != int(comp.max().item()) + 1
            or any(x not in {SIGNED_KIND,OPEN_KIND} for x in kind)
            or len(self.open_candidate_sample_sha256) != 64
            or self.seed.vertex_layer_ids is None
            or not bool((
                comp[:self.signed_vertex_count].max()
                < comp[self.signed_vertex_count:].min()
            ))
            or not bool((
                self.seed.faces[:self.signed_face_count]
                < self.signed_vertex_count
            ).all())
            or not bool((
                self.seed.faces[self.signed_face_count:]
                >= self.signed_vertex_count
            ).all())
        ):
            raise ValueError("RSS_VNEXT_HYBRID_ONE_M_BOUNDARY_DRIFT")
        if (
            any(kind[int(x)] != SIGNED_KIND for x in
                comp[:self.signed_vertex_count].unique().tolist())
            or any(kind[int(x)] != OPEN_KIND for x in
                comp[self.signed_vertex_count:].unique().tolist())
        ):
            raise ValueError("RSS_VNEXT_HYBRID_COMPONENT_FIELD_KIND_DRIFT")


def join_hybrid_iris_seed_v1(
    *,
    signed: TESSASeededSurfaceV1,
    open_udf: OpenUDFSurfaceCandidateV1,
    scene_triplanes: Tensor,
    open_layer_name: str,
    open_presentation_roles: tuple[str,...],
    signed_layer_name: str = "signed_solid",
    signed_presentation_roles: tuple[str,...] = ("primary_form",),
) -> HybridSignedOpenIRISSeedV1:
    """Merge disjoint index domains, preserving open-edge topology exactly.

    Accept ONLY native 1:1 signed extraction seeds, not already subdivided
    surfaces. This prevents a second topology authority after TESSA starts.
    """
    signed.validate()
    open_udf.validate()
    if (
        signed.original_vertex_count != signed.xyz.shape[1]
        or signed.original_face_count != len(signed.faces)
        or not torch.equal(signed.faces,signed.original_faces)
        or scene_triplanes.ndim != 5 or scene_triplanes.shape[0] != 1
        or not bool(torch.isfinite(scene_triplanes).all())
        or any(x.device != scene_triplanes.device for x in
               (signed.xyz,signed.faces,open_udf.xyz,open_udf.faces))
        or any(x.dtype != scene_triplanes.dtype
               for x in (signed.xyz,open_udf.xyz))
        or not isinstance(open_layer_name,str) or not open_layer_name
        or any(type(v) is not str or not v for v in
               (open_presentation_roles+signed_presentation_roles))
        or not open_presentation_roles or not signed_presentation_roles
    ):
        raise ValueError("RSS_VNEXT_HYBRID_SOURCE_FREE_DOMAIN")

    scene_hash=tensor_bytes_sha256_v1(scene_triplanes)
    if (signed.scene_sha256 != scene_hash
        or open_udf.scene_sha256 != scene_hash):
        raise ValueError("RSS_VNEXT_HYBRID_TWO_SCENES_FORBIDDEN")
    if len(exact_open_boundary_edges_v1(signed.faces)):
        raise ValueError("RSS_VNEXT_HYBRID_SIGNED_SHELL_NOT_CLOSED")
    # Already extracted separately tagged signed shells are admitted. The
    # new open layer receives an appended role namespace, never a guessed
    # label based on coincident geometric positions.
    if signed.vertex_layer_ids is not None:
        signed_names=signed.layer_names
        signed_roles=signed.layer_roles
        signed_ids=signed.vertex_layer_ids
    else:
        signed_names=(signed_layer_name,)
        signed_roles=(signed_presentation_roles,)
        signed_ids=torch.zeros(
            (signed.original_vertex_count,),device=signed.xyz.device,
            dtype=torch.long,
        )
    if (
        open_layer_name in signed_names
        or not 2 <= len(signed_names)+1 <= 16
        or len(set(r for roles in (signed_roles+(open_presentation_roles,))
                for r in roles))
           != sum(map(len,signed_roles+(open_presentation_roles,)))
    ):
        raise ValueError("RSS_VNEXT_HYBRID_LAYER_NAMESPACE_OVERLAP")

    cv=int(signed.vertex_component_ids.max())+1
    cu=int(open_udf.component_ids.max())+1
    if (
        not torch.equal(signed.vertex_component_ids.unique(),
                        torch.arange(cv,device=signed.xyz.device))
        or not torch.equal(open_udf.component_ids.unique(),
                           torch.arange(cu,device=signed.xyz.device))
        or signed.original_vertex_count+len(open_udf.xyz[0])>2_000_000
        or signed.original_face_count+len(open_udf.faces)>2_000_000
    ):
        raise ValueError("RSS_VNEXT_HYBRID_COMPONENT_OR_RESOURCE_POLICY")
    num_signed=signed.original_vertex_count
    combined_xyz=torch.cat((signed.xyz,open_udf.xyz),dim=1)
    combined_faces=torch.cat((
        signed.faces,open_udf.faces+num_signed,
    ),dim=0)
    comp=torch.cat((
        signed.vertex_component_ids,open_udf.component_ids+cv,
    ),dim=0)
    layer_ids=torch.cat((
        signed_ids,torch.full(
            (open_udf.xyz.shape[1],),len(signed_names),
            device=signed.xyz.device,dtype=torch.long,
        ),
    ))
    namespace=layer_namespace_digest_v1(
        signed_names+(open_layer_name,),
        signed_roles+(open_presentation_roles,),
    )
    surface_sha=sha256((
        tensor_bytes_sha256_v1(combined_xyz)
        +tensor_bytes_sha256_v1(combined_faces)
        +tensor_bytes_sha256_v1(comp)
    ).encode()).hexdigest()
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=combined_xyz,
        iris_surface_faces=combined_faces,
        component_ids_by_vertex=comp,
        scene_sha256=scene_hash,
        original_surface_sha256=surface_sha,
        original_vertex_layer_ids=layer_ids,
        layer_names=signed_names+(open_layer_name,),
        layer_roles=signed_roles+(open_presentation_roles,),
        layer_namespace_sha256=namespace,
    )
    result=HybridSignedOpenIRISSeedV1(
        seed=seed,signed_vertex_count=num_signed,
        signed_face_count=len(signed.faces),
        open_vertex_count=open_udf.xyz.shape[1],
        open_face_count=len(open_udf.faces),
        component_surface_kinds=(
            (SIGNED_KIND,)*cv+(OPEN_KIND,)*cu
        ),
        open_candidate_sample_sha256=open_udf.sampled_udf_sha256,
    )
    result.validate()
    return result


__all__ = [
    "SCHEMA","SIGNED_KIND","OPEN_KIND",
    "HybridSignedOpenIRISSeedV1","join_hybrid_iris_seed_v1",
]
