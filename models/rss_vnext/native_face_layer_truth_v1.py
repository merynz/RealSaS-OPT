from __future__ import annotations

"""Native-face provenance for qualified layered TESSA source supervision.

This is a teacher-only court.  The source mesh and native face labels are
independent of student H(x), TESSA mesh, render guesses and generated views.
Source samples must replay from original triangle barycentric addresses.
Coincident body/cape positions remain distinct via FACE address, not XYZ.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

import torch
from torch import Tensor

SCHEMA = "RealSaS.RSSvNext.NativeFaceBarycentricLayerTruth.v1"


def tensor_sha256_v1(t: Tensor) -> str:
    c=t.detach().contiguous().cpu()
    header=json.dumps(
        {"shape":list(c.shape),"dtype":str(c.dtype)},
        separators=(",",":"),sort_keys=True,
    ).encode()
    return sha256(header+b"\x00"+c.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class NativeFaceLayerSamplesV1:
    vertices: Tensor                    # [N,3] native source mesh, NOT predictions
    faces: Tensor                       # [F,3] exact source indexed face IDs
    face_layer_ids: Tensor              # [F] material/part audited role mapping
    face_component_ids: Tensor          # [F] independently extracted connectivity
    sample_face_ids: Tensor             # [M] original face indices, never nearest
    sample_barycentric: Tensor          # [M,3] SOURCE triangle barycentric
    source_asset_sha256: str            # independently verified source bytes
    layer_namespace_sha256: str         # canonical semantic role mapping

    def tensors(self) -> dict[str,Tensor]:
        return {k:v for k,v in vars(self).items() if isinstance(v,Tensor)}

    def receipt_sha256(self) -> str:
        payload={k:tensor_sha256_v1(v) for k,v in self.tensors().items()}
        payload.update({
            "schema":SCHEMA,
            "source_asset_sha256":self.source_asset_sha256,
            "layer_namespace_sha256":self.layer_namespace_sha256,
        })
        return sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

    def replay(self) -> tuple[Tensor,Tensor,Tensor]:
        """Exact source XYZ and native layer/component targets by face address.

        Verifies native faces cannot mix layers/components on shared indexed
        vertices; disconnected shells with identical XYZ remain distinct.
        """
        v,f,l,c,ids,b = (
            self.vertices,self.faces,self.face_layer_ids,
            self.face_component_ids,self.sample_face_ids,
            self.sample_barycentric,
        )
        if (v.ndim!=2 or v.shape[-1]!=3 or len(v)<3
            or f.ndim!=2 or f.shape[-1]!=3 or len(f)<1
            or f.dtype!=torch.long or l.shape!=(len(f),)
            or c.shape!=(len(f),) or l.dtype!=torch.long
            or c.dtype!=torch.long or ids.ndim!=1
            or ids.dtype!=torch.long or not len(ids)
            or b.shape!=(len(ids),3) or not torch.is_floating_point(b)
            or any(t.device!=v.device for t in (f,l,c,ids,b))
            or not torch.is_floating_point(v)
            or not bool(torch.isfinite(v).all())
            or not bool(torch.isfinite(b).all())
            or bool((f<0).any()) or int(f.max())>=len(v)
            or bool((l<0).any()) or bool((c<0).any())
            or bool((ids<0).any()) or int(ids.max())>=len(f)
            or bool((b < -1.e-7).any()) or bool((b > 1+1.e-7).any())
            or not bool(torch.allclose(
                b.sum(-1),torch.ones_like(b[:,0]),atol=1e-6,rtol=0,
            ))
            or v.dtype!=b.dtype
            or not all(type(s) is str and len(s)==64 and all(
                ch in "0123456789abcdef" for ch in s
            ) for s in (self.source_asset_sha256,self.layer_namespace_sha256))
        ):
            raise ValueError("RSS_VNEXT_NATIVE_FACE_BARYCENTRIC_DOMAIN")
        if bool((
            (f[:,0]==f[:,1]) | (f[:,1]==f[:,2]) | (f[:,2]==f[:,0])
        ).any()):
            raise ValueError("RSS_VNEXT_NATIVE_FACE_INDEX_DEGENERATE")
        # Indexed face sharing is decisive; matching 3D positions are NOT.
        # A common indexed vertex may not silently become cloth and body.
        min_layer=torch.full((len(v),),2**62,device=v.device,dtype=torch.long)
        max_layer=torch.full((len(v),),-1,device=v.device,dtype=torch.long)
        min_comp=min_layer.clone()
        max_comp=max_layer.clone()
        flat=f.reshape(-1)
        for values,lo,hi in ((l,min_layer,max_layer),(c,min_comp,max_comp)):
            expanded=values.repeat_interleave(3)
            lo.scatter_reduce_(0,flat,expanded,reduce="amin",include_self=True)
            hi.scatter_reduce_(0,flat,expanded,reduce="amax",include_self=True)
        present=max_layer>=0
        if (not bool(present.all())
            or not bool(torch.equal(min_layer,max_layer))
            or not bool(torch.equal(min_comp,max_comp))):
            raise ValueError("RSS_VNEXT_NATIVE_FACE_CROSS_LAYER_OR_COMPONENT")
        tri=v[f[ids]]
        areas=torch.linalg.vector_norm(
            torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),
            dim=-1,
        )
        if not bool((areas>1e-11).all()):
            raise ValueError("RSS_VNEXT_NATIVE_FACE_SAMPLE_DEGENERATE")
        xyz=(tri*b.unsqueeze(-1)).sum(dim=1)
        return xyz,l[ids],c[ids]


def validate_native_face_samples_v1(
    witness: NativeFaceLayerSamplesV1 | None,
    *, source_xyz: Tensor, source_layer_ids: Tensor,
    source_component_ids: Tensor,source_asset_sha256: str,
    layer_namespace_sha256: str,receipt: dict,
    tolerance: float = 2.e-5,
) -> None:
    if not isinstance(witness,NativeFaceLayerSamplesV1):
        raise ValueError("RSS_VNEXT_LAYER_NATIVE_FACE_WITNESS_REQUIRED")
    if (
        witness.source_asset_sha256!=source_asset_sha256
        or witness.layer_namespace_sha256!=layer_namespace_sha256
        or receipt.get("source_native_face_witness_schema")!=SCHEMA
        or receipt.get("source_native_face_witness_sha256")
            !=witness.receipt_sha256()
        or receipt.get("source_native_face_witness_derivation")
            !="SOURCE_NATIVE_INDEXED_TRIANGLE_BARYCENTRIC_NOT_STUDENT"
    ):
        raise ValueError("RSS_VNEXT_LAYER_NATIVE_FACE_RECEIPT_DRIFT")
    reconstructed,layers,components=witness.replay()
    if (
        source_xyz.shape!=reconstructed.shape
        or source_xyz.device!=reconstructed.device
        or source_xyz.dtype!=reconstructed.dtype
        or not bool(torch.allclose(
            reconstructed,source_xyz,atol=tolerance,rtol=0,
        ))
        or not torch.equal(layers,source_layer_ids)
        or not torch.equal(components,source_component_ids)
    ):
        raise ValueError("RSS_VNEXT_LAYER_NATIVE_FACE_REPLAY_MISMATCH")


__all__=[
    "SCHEMA","NativeFaceLayerSamplesV1","tensor_sha256_v1",
    "validate_native_face_samples_v1",
]
