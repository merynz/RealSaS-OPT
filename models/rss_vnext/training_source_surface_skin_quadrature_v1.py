from __future__ import annotations

"""Deterministic source-surface skin quadrature for RSS.vNext training.

AniGen samples source triangles in proportion to area, then barycentrically
interpolates skin. In a signed source-truth training court we can instead use
a deterministic three-point degree-two quadrature with true source triangle
areas. This avoids tessellation-density bias and random sampling RNG while
preserving gradients through predicted per-vertex skin values.

SOURCE rest geometry and exact SOURCE faces are independent teacher-only
inputs; predicted H(x) mesh faces are never silently used to certify this
loss. A caller must separately qualify source face/vertex reindex receipts.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


SCHEMA="RealSaS.RSSvNext.SourceSurfaceSkinQuadrature.v1"
# Three symmetric points, each with 1/3 quadrature weight; degree-two exact.
_BARY=((2./3.,1./6.,1./6.),
       (1./6.,2./3.,1./6.),
       (1./6.,1./6.,2./3.))


@dataclass(frozen=True)
class RSSvNextSourceSurfaceSkinQuadratureV1:
    loss: Tensor
    qualified_source_face_count: int
    qualified_source_surface_area: float
    source_faces_used: Tensor
    teacher_only_source_geometry: bool=True
    eligible_for_product_qualification: bool=False


def source_surface_area_skin_loss_v1(
    source_rest_vertices:Tensor, source_faces:Tensor,
    predicted_vertex_skin:Tensor, source_vertex_skin:Tensor,
    *, source_face_mask:Tensor | None=None,
    faces_per_chunk:int=2048,
) -> RSSvNextSourceSurfaceSkinQuadratureV1:
    """Surface-integral SmoothL1, not a per-vertex/tessellation-count average.

    B=1 native canonical source mesh. Masks must be source-established,
    not student-generated visibility or an imputed source label. This
    function itself cannot qualify source rights, bone order or TESSA
    reindexing, which remain registry/Foundry admission responsibilities.
    """
    if (
        not all(isinstance(t,Tensor) for t in (
            source_rest_vertices,source_faces,
            predicted_vertex_skin,source_vertex_skin,
        ))
        or source_rest_vertices.ndim!=3
        or source_rest_vertices.shape[0]!=1
        or source_rest_vertices.shape[-1]!=3
        or source_faces.ndim!=2 or source_faces.shape[-1]!=3
        or source_faces.dtype!=torch.long
        or predicted_vertex_skin.ndim!=3
        or source_vertex_skin.shape!=predicted_vertex_skin.shape
        or predicted_vertex_skin.shape[0]!=1
        or predicted_vertex_skin.shape[1]!=source_rest_vertices.shape[1]
        or predicted_vertex_skin.shape[2]<2
        or predicted_vertex_skin.shape[2]>512
        or not torch.is_floating_point(predicted_vertex_skin)
        or not torch.is_floating_point(source_vertex_skin)
        or not torch.is_floating_point(source_rest_vertices)
        or type(faces_per_chunk) is not int
        or faces_per_chunk<1 or faces_per_chunk>16384
        or source_faces.shape[0]<1 or source_faces.shape[0]>262144
    ):
        raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_QUADRATURE_DOMAIN")
    tensors=(
        source_rest_vertices,source_faces,
        predicted_vertex_skin,source_vertex_skin,
    )
    if len({str(t.device) for t in tensors})!=1:
        raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_QUADRATURE_DEVICE")
    n=source_rest_vertices.shape[1]
    if (
        not bool(torch.isfinite(source_rest_vertices).all())
        or not bool(torch.isfinite(predicted_vertex_skin).all())
        or not bool(torch.isfinite(source_vertex_skin).all())
        or int(source_faces.min())<0
        or int(source_faces.max())>=n
        or bool((source_faces[:,0]==source_faces[:,1]).any())
        or bool((source_faces[:,0]==source_faces[:,2]).any())
        or bool((source_faces[:,1]==source_faces[:,2]).any())
    ):
        raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_QUADRATURE_NUMERIC_OR_INDEX")
    for skin in (predicted_vertex_skin,source_vertex_skin):
        if (
            bool((skin < -1e-7).any())
            or bool((skin > 1+1e-7).any())
            or not bool(torch.allclose(
                skin.sum(-1),torch.ones_like(skin[...,0]),
                atol=1e-4,rtol=0,
            ))
        ):
            raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_QUADRATURE_SIMPLEX")
    if source_face_mask is None:
        face_mask=torch.ones(
            (source_faces.shape[0],),dtype=torch.bool,device=source_faces.device,
        )
    else:
        if (
            not isinstance(source_face_mask,Tensor)
            or source_face_mask.shape!=(source_faces.shape[0],)
            or source_face_mask.dtype!=torch.bool
            or source_face_mask.device!=source_faces.device
        ):
            raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_FACE_MASK_DOMAIN")
        face_mask=source_face_mask
    if not bool(face_mask.any()):
        raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_NO_ELIGIBLE_FACES")
    selected=source_faces[face_mask]
    # Source geometry is not learnable from a skin loss, cannot let model
    # modify triangle areas to reduce an unrelated task objective.
    rest=source_rest_vertices.detach().to(dtype=torch.float32)
    bary=torch.tensor(_BARY,device=rest.device,dtype=torch.float32)
    a_loss=torch.zeros((),device=rest.device,dtype=torch.float32)
    area_total=torch.zeros((),device=rest.device,dtype=torch.float64)
    pred=predicted_vertex_skin.to(dtype=torch.float32)
    target=source_vertex_skin.detach().to(dtype=torch.float32)
    for off in range(0,selected.shape[0],faces_per_chunk):
        faces=selected[off:off+faces_per_chunk]
        verts=rest[0,faces]  # [F,3,3]
        areas=.5*torch.cross(
            verts[:,1]-verts[:,0],
            verts[:,2]-verts[:,0],
            dim=-1,
        ).norm(dim=-1)
        if (
            not bool(torch.isfinite(areas).all())
            or bool((areas<=1e-10).any())
        ):
            raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_DEGENERATE_SOURCE_FACE")
        pfaces=pred[0,faces]
        tfaces=target[0,faces]
        pquad=torch.einsum("qj,fjk->fqk",bary,pfaces)
        tquad=torch.einsum("qj,fjk->fqk",bary,tfaces)
        per_face=F.smooth_l1_loss(
            pquad,tquad,reduction="none",
        ).mean(dim=(-1,-2))
        a_loss=a_loss+(per_face*areas).sum()
        area_total=area_total+areas.double().sum()
    area_float=float(area_total.item())
    if not 0.<area_float<float("inf"):
        raise ValueError("RSS_VNEXT_SOURCE_SURFACE_SKIN_TOTAL_AREA_INVALID")
    return RSSvNextSourceSurfaceSkinQuadratureV1(
        loss=a_loss/area_total.to(dtype=a_loss.dtype),
        qualified_source_face_count=int(selected.shape[0]),
        qualified_source_surface_area=area_float,
        source_faces_used=selected.detach().clone(),
    )


__all__=["SCHEMA","RSSvNextSourceSurfaceSkinQuadratureV1",
         "source_surface_area_skin_loss_v1"]
