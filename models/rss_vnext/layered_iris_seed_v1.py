from __future__ import annotations

"""Semantic-decomposed, ONE-scene indexed IRIS→TESSA seed (signed branch).

Never weld coincident body/garment vertices. We join independent layer mesh
address spaces into ONE canonical indexed candidate before AXIS/MIRA. A
separate per-layer FC extraction is NOT a second learned geometry authority:
all layers query the same H(x), shared field and source-independent roles.

This is SIGNED semantic-equivalent surface extraction only. Open/UDF cloth
requires its own source-qualified patch topology and cannot enter through
this API. No SOURCE asset/pose information is allowed in the student forward.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

import torch
from torch import Tensor

from .flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence
from .semantic_geometry_prior import (
    RSSvNextSemanticRoleSchema, RSSvNextSemanticLayerSpec,
)
from .tessa_seeded_surface_v1 import (
    TESSASeededSurfaceV1, tessa_seed_from_flexicubes_v1,
    build_tessa_seeded_surface_v1,
)


LAYER_SCHEMA="RealSaS.RSSvNext.SemanticLayerIndexedSurface.v1"


def layer_namespace_digest_v1(
    names: tuple[str,...], role_groups: tuple[tuple[str,...],...],
) -> str:
    return sha256(json.dumps({
        "schema":LAYER_SCHEMA,
        "layers":[{"name":a,"keep_roles":list(b)}
                  for a,b in zip(names,role_groups)],
    },separators=(",",":"),sort_keys=True).encode()).hexdigest()


def _tensor_hash(t: Tensor) -> str:
    import numpy as np
    a=t.detach().contiguous().cpu().numpy()
    h=sha256()
    h.update(str((str(a.dtype),a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class LayeredIRISSeedV1:
    seed: TESSASeededSurfaceV1
    layer_names: tuple[str,...]
    layer_roles: tuple[tuple[str,...],...]
    original_vertex_layer_ids: Tensor  # [V0] integral, face-uniform
    layer_namespace_sha256: str
    source_authority: str="ONE_IRIS_H_SEMANTIC_DISJOINT_LAYER_PROPOSAL"

    def validate(self) -> None:
        self.seed.validate()
        n=self.seed.original_vertex_count
        ids=self.original_vertex_layer_ids
        if (
            not 2<=len(self.layer_names)<=16
            or len(self.layer_names)!=len(self.layer_roles)
            or len(set(self.layer_names))!=len(self.layer_names)
            or len(set(r for roles in self.layer_roles for r in roles))
                !=sum(len(roles) for roles in self.layer_roles)
            or ids.shape!=(n,) or ids.device!=self.seed.xyz.device
            or ids.dtype!=torch.long or bool((ids<0).any())
            or int(ids.max())>=len(self.layer_names)
            or not bool(torch.bincount(
                ids,minlength=len(self.layer_names)
            ).gt(0).all())
            or not bool((
                ids[self.seed.original_faces]
                ==ids[self.seed.original_faces[:,0:1]]
            ).all())
            or self.layer_namespace_sha256!=layer_namespace_digest_v1(
                self.layer_names,self.layer_roles,
            )
            or self.seed.original_vertex_layer_ids is None
            or not torch.equal(
                self.seed.original_vertex_layer_ids,ids,
            )
            or self.seed.layer_namespace_sha256!=self.layer_namespace_sha256
            or self.seed.layer_names!=self.layer_names
            or self.source_authority
                !="ONE_IRIS_H_SEMANTIC_DISJOINT_LAYER_PROPOSAL"
        ):
            raise ValueError("RSS_VNEXT_LAYERED_INDEXED_SEED_PROVENANCE_DRIFT")


def make_layered_iris_seed_v1(
    *,
    evidence: tuple[tuple[RSSvNextSemanticLayerSpec,
                          RSSvNextFlexiCubesSurfaceEvidence],...],
    schema: RSSvNextSemanticRoleSchema,
    scene_triplanes: Tensor,
) -> LayeredIRISSeedV1:
    """Assemble disjoint role shells without geometric union or face mixing.

    Each evidence must independently have valid signed extraction. Optional
    absent roles are NOT represented as invented geometry. When a claimed
    role cannot be extracted the upstream extractor ABSTAINS.
    """
    schema.validate()
    if (
        not isinstance(evidence,tuple) or not 2<=len(evidence)<=schema.count
        or scene_triplanes.ndim!=5 or scene_triplanes.shape[0]!=1
        or not bool(torch.isfinite(scene_triplanes).all())
    ):
        raise ValueError("RSS_VNEXT_LAYERED_SEED_INPUT_POLICY")
    used=set()
    seen=set()
    names=[]
    roles=[]
    xyz=[]
    faces=[]
    components=[]
    layer_ids=[]
    vertex_offset=0
    component_offset=0
    scene_hash=None
    for layer,witness in evidence:
        if not isinstance(layer,RSSvNextSemanticLayerSpec):
            raise ValueError("RSS_VNEXT_LAYERED_SPEC_TYPE")
        layer.validate(schema)
        if (
            layer.name in seen
            or any(r in used for r in layer.keep_roles)
            or not isinstance(witness,RSSvNextFlexiCubesSurfaceEvidence)
            or witness.layer_name!=layer.name
            or witness.source_policy!="SIGNED_HYPOTHESIS_ONLY"
        ):
            raise ValueError("RSS_VNEXT_LAYERED_OVERLAP_OR_WITNESS_DRIFT")
        witness.validate(
            carrier_dim=witness.local_features.shape[-1],
            role_count=schema.count,
        )
        if (
            witness.xyz.device!=scene_triplanes.device
            or witness.xyz.dtype!=scene_triplanes.dtype
            or witness.faces.device!=scene_triplanes.device
        ):
            raise ValueError("RSS_VNEXT_LAYERED_SCENE_DEVICE_DRIFT")
        per_layer=tessa_seed_from_flexicubes_v1(
            witness=witness,scene_triplanes=scene_triplanes,
        )
        if scene_hash is not None and scene_hash!=per_layer.scene_sha256:
            raise ValueError("RSS_VNEXT_LAYERED_SCENE_HASH_DRIFT")
        scene_hash=per_layer.scene_sha256
        v=per_layer.original_vertex_count
        xyz.append(per_layer.xyz)
        faces.append(per_layer.faces+vertex_offset)
        components.append(per_layer.vertex_component_ids+component_offset)
        layer_ids.append(torch.full(
            (v,),len(names),device=scene_triplanes.device,dtype=torch.long,
        ))
        vertex_offset+=v
        component_offset+=int(per_layer.vertex_component_ids.max())+1
        names.append(layer.name)
        roles.append(layer.keep_roles)
        seen.add(layer.name)
        used.update(layer.keep_roles)

    joined_xyz=torch.cat(xyz,dim=1)
    joined_faces=torch.cat(faces,dim=0)
    joined_components=torch.cat(components,dim=0)
    ids=torch.cat(layer_ids,dim=0)
    surface_hash=sha256((
        _tensor_hash(joined_xyz)
        +_tensor_hash(joined_faces)
        +_tensor_hash(joined_components)
    ).encode()).hexdigest()
    namespace=layer_namespace_digest_v1(tuple(names),tuple(roles))
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=joined_xyz,
        iris_surface_faces=joined_faces,
        component_ids_by_vertex=joined_components,
        scene_sha256=scene_hash,
        original_surface_sha256=surface_hash,
        original_vertex_layer_ids=ids,
        layer_names=tuple(names),
        layer_namespace_sha256=namespace,
    )
    result=LayeredIRISSeedV1(
        seed=seed,layer_names=tuple(names),
        layer_roles=tuple(roles),
        original_vertex_layer_ids=ids,
        layer_namespace_sha256=namespace,
    )
    result.validate()
    return result


__all__=[
    "LAYER_SCHEMA","LayeredIRISSeedV1",
    "layer_namespace_digest_v1","make_layered_iris_seed_v1",
]
