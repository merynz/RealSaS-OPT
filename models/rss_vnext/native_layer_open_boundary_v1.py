from __future__ import annotations

"""Source-native open-hem court for layered, indexed TESSA supervision.

A signed solid has no boundary edges. A real cape, skirt, or garment sheet may
have them; the correct target is not a hole-filled SDF surrogate. This module
derives **every** open edge from a *source-native* face graph, never from
student H(x), student geometry, an image mask, or generated views.

Only a proven, complete, indexed boundary is accepted as open-hem truth.
No material, cloth elasticity, pin, or rig inference is claimed.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import torch
from torch import Tensor

from .native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1, tensor_sha256_v1,
)

SCHEMA = "RealSaS.RSSvNext.NativeLayerOpenBoundary.v1"
DERIVATION = "COMPLETE_NATIVE_INDEXED_FACE_INCIDENCE_NOT_STUDENT"


def exact_open_boundary_edges_v1(faces: Tensor) -> Tensor:
    """Sorted unique undirected boundary edges of a manifold triangle graph.

    Nonmanifold edges, duplicate face addresses and illegal face indices are
    fatal: source orientation/geometry must be independently validated first.
    """
    if (
        not isinstance(faces, Tensor)
        or faces.ndim != 2 or faces.shape[-1] != 3
        or faces.dtype != torch.long or len(faces) < 1
        or bool((faces < 0).any())
        or bool(((faces[:,0] == faces[:,1]) |
                 (faces[:,1] == faces[:,2]) |
                 (faces[:,2] == faces[:,0])).any())
    ):
        raise ValueError("RSS_VNEXT_NATIVE_OPEN_FACE_DOMAIN")
    edges = torch.cat(
        (faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]),
        dim=0,
    ).sort(dim=-1).values
    unique, counts = torch.unique(edges,dim=0,return_counts=True)
    if bool((counts > 2).any()):
        raise ValueError("RSS_VNEXT_NATIVE_OPEN_NONMANIFOLD_EDGE")
    # Disallow duplicate triangles independently of winding: duplicate
    # double-sided surfaces must not accidentally claim closure.
    sorted_faces = faces.sort(dim=-1).values
    if torch.unique(sorted_faces, dim=0).shape[0] != len(faces):
        raise ValueError("RSS_VNEXT_NATIVE_OPEN_DUPLICATE_TRIANGLE")
    return unique[counts == 1]


@dataclass(frozen=True)
class NativeLayerOpenBoundaryV1:
    """Immutable source topology receipt; boundary position is re-derived."""

    boundary_edges: Tensor       # [E,2] sorted, exhaustive, exact source IDs
    boundary_vertex_ids: Tensor  # [K] sorted, exhaustive, exact source IDs
    open_layer_ids: Tensor       # [L] sorted source-native semantic layer IDs
    source_asset_sha256: str
    layer_namespace_sha256: str

    def tensors(self) -> dict[str, Tensor]:
        return {
            "boundary_edges": self.boundary_edges,
            "boundary_vertex_ids": self.boundary_vertex_ids,
            "open_layer_ids": self.open_layer_ids,
        }

    def receipt_sha256(self) -> str:
        payload = {
            "schema": SCHEMA,
            "source_asset_sha256": self.source_asset_sha256,
            "layer_namespace_sha256": self.layer_namespace_sha256,
            **{k: tensor_sha256_v1(v) for k,v in self.tensors().items()},
        }
        return sha256(json.dumps(
            payload, sort_keys=True, separators=(",",":"),
        ).encode()).hexdigest()

    def replay(self, native: NativeFaceLayerSamplesV1) -> tuple[Tensor, Tensor]:
        """Return source xyz[K,3] and layer[K], with no nearest-point relabel."""
        if not isinstance(native, NativeFaceLayerSamplesV1):
            raise ValueError("RSS_VNEXT_NATIVE_OPEN_FACE_WITNESS_REQUIRED")
        native.replay()  # Validates all original face/layer/component addresses.
        v, f, role = native.vertices, native.faces, native.face_layer_ids
        e, ids, layers = (
            self.boundary_edges, self.boundary_vertex_ids,
            self.open_layer_ids,
        )
        if (
            native.source_asset_sha256 != self.source_asset_sha256
            or native.layer_namespace_sha256 != self.layer_namespace_sha256
            or e.ndim != 2 or e.shape[-1] != 2 or not len(e)
            or e.dtype != torch.long
            or ids.ndim != 1 or ids.dtype != torch.long or len(ids) < 3
            or layers.ndim != 1 or layers.dtype != torch.long
            or not len(layers)
            or any(t.device != v.device for t in (e, ids, layers))
            or bool((e < 0).any()) or int(e.max()) >= len(v)
            or bool((ids < 0).any()) or int(ids.max()) >= len(v)
            or bool((layers < 0).any()) or int(layers.max()) > int(role.max())
            or bool((ids[1:] <= ids[:-1]).any())
            or bool((layers[1:] <= layers[:-1]).any())
        ):
            raise ValueError("RSS_VNEXT_NATIVE_OPEN_WITNESS_DOMAIN")
        actual = exact_open_boundary_edges_v1(f)
        if not bool(torch.equal(actual,e)):
            raise ValueError("RSS_VNEXT_NATIVE_OPEN_INCOMPLETE_EDGE_COURT")
        actual_ids = torch.unique(actual.reshape(-1),sorted=True)
        if not bool(torch.equal(ids,actual_ids)):
            raise ValueError("RSS_VNEXT_NATIVE_OPEN_VERTEX_ADDRESS_DRIFT")
        # Indexed vertex membership must be unique within a layer.
        vertex_roles = torch.full(
            (len(v),), -1, dtype=torch.long, device=v.device,
        )
        for i in range(3):
            corner = f[:,i]
            vertex_roles[corner] = role
        edge_roles = vertex_roles[actual]
        if not bool((edge_roles[:,0] == edge_roles[:,1]).all()):
            raise ValueError("RSS_VNEXT_NATIVE_OPEN_CROSS_LAYER_EDGE")
        actual_layers = torch.unique(edge_roles[:,0],sorted=True)
        if not bool(torch.equal(actual_layers,layers)):
            raise ValueError("RSS_VNEXT_NATIVE_OPEN_LAYER_COVERAGE_DRIFT")
        return v[ids], vertex_roles[ids]


def validate_native_open_boundary_v1(
    witness: NativeLayerOpenBoundaryV1 | None,
    native_faces: NativeFaceLayerSamplesV1 | None,
    *,
    source_boundary_xyz: Tensor | None,
    source_boundary_mask: Tensor | None,
    source_asset_sha256: str,
    layer_namespace_sha256: str,
    receipt: dict,
    tolerance: float = 2.e-5,
) -> tuple[Tensor,Tensor,Tensor]:
    if not isinstance(witness, NativeLayerOpenBoundaryV1):
        raise ValueError("RSS_VNEXT_NATIVE_OPEN_BOUNDARY_WITNESS_REQUIRED")
    if (
        witness.source_asset_sha256 != source_asset_sha256
        or witness.layer_namespace_sha256 != layer_namespace_sha256
        or receipt.get("source_native_open_boundary_schema") != SCHEMA
        or receipt.get("source_native_open_boundary_sha256")
            != witness.receipt_sha256()
        or receipt.get("source_native_open_boundary_derivation") != DERIVATION
    ):
        raise ValueError("RSS_VNEXT_NATIVE_OPEN_BOUNDARY_RECEIPT_DRIFT")
    xyz, roles = witness.replay(native_faces)
    if (
        source_boundary_xyz is None or source_boundary_mask is None
        or source_boundary_xyz.shape != xyz.shape
        or source_boundary_xyz.device != xyz.device
        or source_boundary_xyz.dtype != xyz.dtype
        or source_boundary_mask.shape != (len(xyz),)
        or source_boundary_mask.device != xyz.device
        or source_boundary_mask.dtype != torch.bool
        or not bool(source_boundary_mask.all())
        or not bool(torch.allclose(
            xyz,source_boundary_xyz,atol=tolerance,rtol=0,
        ))
    ):
        raise ValueError("RSS_VNEXT_NATIVE_OPEN_BOUNDARY_REPLAY_DRIFT")
    return xyz,roles,witness.open_layer_ids


__all__ = [
    "SCHEMA","DERIVATION","NativeLayerOpenBoundaryV1",
    "exact_open_boundary_edges_v1","validate_native_open_boundary_v1",
]
