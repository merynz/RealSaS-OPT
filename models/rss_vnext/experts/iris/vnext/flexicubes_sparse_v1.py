from __future__ import annotations

"""Sparse, globally addressed adapter for EXACT pinned NVIDIA FlexiCubes.

Only the pinned operator's regular-grid ambiguity lookup depends on a dense
full-volume cube address; its dual vertices, crossing edges, QEF and triangle
weights already operate on the supplied indexed cube set. This adapter
replaces only that lookup with an indexed global-address equivalent.

Nonzero inactive cubes are not fabricated. The caller MUST supply the
exhaustive set of active sign-changing fine-grid cubes; coarse-only guesses
are not accepted as coverage proof. Training gradients remain inside the
original Apache-2.0 FlexiCubes implementation and the same IRIS H(x).
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .flexicubes_v1 import (
    PinnedFlexiCubesV1, split_flexicubes_parameters_v1,
    FLEXICUBES_COMMIT,
)

SPARSE_CASE_SCHEMA = "RealSaS.RSSvNext.SparsePinnedFlexiCubes.GlobalCase.v1"


def _resolution_v1(resolution: tuple[int,int,int]) -> tuple[int,int,int]:
    if (
        type(resolution) is not tuple or len(resolution)!=3
        or any(type(r) is not int or not 2 <= r <= 512
               for r in resolution)
        or math.prod(resolution)>32_000_000
    ):
        raise ValueError("RSS_VNEXT_SPARSE_FC_GLOBAL_GRID_DOMAIN")
    return resolution


@torch.no_grad()
def globally_addressed_case_ids_v1(
    operator, occ_fx8: Tensor, surf_cubes: Tensor,
    resolution: tuple[int,int,int], global_cube_ids: Tensor,
) -> Tensor:
    """Exact _get_case_id parity without an O(nx*ny*nz*5) dense long grid.

    Indexed lookup uses sorted global IDs of supplied cubes and the pinned
    check_table entries. Missing neighbors have the same default nonproblem
    state as dense upstream. All surface cubes must be present.
    """
    nx,ny,nz=_resolution_v1(resolution)
    count=int(global_cube_ids.numel())
    if (
        surf_cubes.shape!=(count,) or surf_cubes.dtype!=torch.bool
        or occ_fx8.shape!=(count,8)
        or global_cube_ids.dtype!=torch.long
        or global_cube_ids.ndim!=1 or count<1
        or global_cube_ids.device!=occ_fx8.device
        or surf_cubes.device!=occ_fx8.device
        or int(global_cube_ids.min())<0
        or int(global_cube_ids.max())>=nx*ny*nz
        or bool((global_cube_ids[1:]<=global_cube_ids[:-1]).any())
        or not bool(surf_cubes.any())
    ):
        raise ValueError("RSS_VNEXT_SPARSE_FC_CUBE_ADDRESS_DRIFT")
    corners=operator.cube_corners_idx.to(occ_fx8.device)
    raw_case_ids=(occ_fx8[surf_cubes]*corners[None]).sum(-1).long()
    problems=operator.check_table.to(occ_fx8.device)[raw_case_ids]
    is_problem=problems[:,0]==1
    if not bool(is_problem.any()):
        return raw_case_ids

    # To retain upstream parity, neighbor checks inspect the original
    # pinned check_table entries, never already-inverted case IDs.
    surface_global=global_cube_ids[surf_cubes]
    problem_global=surface_global[is_problem]
    problem_rows=problems[is_problem]
    dx=problem_rows[:,1].long()
    dy=problem_rows[:,2].long()
    dz=problem_rows[:,3].long()
    x=torch.div(problem_global,ny*nz,rounding_mode="floor")
    y=torch.div(problem_global,nz,rounding_mode="floor")%ny
    z=problem_global%nz
    x2,y2,z2=x+dx,y+dy,z+dz
    inside=(x2>=0)&(x2<nx)&(y2>=0)&(y2<ny)&(z2>=0)&(z2<nz)
    problem_positions=torch.nonzero(is_problem,as_tuple=False).flatten()
    if not bool(inside.any()):
        return raw_case_ids
    neighbor_global=(x2[inside]*ny+y2[inside])*nz+z2[inside]
    neighbor_addr=torch.searchsorted(surface_global,neighbor_global)
    within=neighbor_addr<len(surface_global)
    found=within & (
        surface_global[neighbor_addr.clamp_max(
            len(surface_global)-1,
        )]==neighbor_global
    )
    if not bool(found.any()):
        return raw_case_ids
    neighbor_cases=raw_case_ids[
        neighbor_addr.clamp_max(len(surface_global)-1)
    ]
    neighbor_problem=operator.check_table.to(
        occ_fx8.device,
    )[neighbor_cases][:,0]==1
    invert=found & neighbor_problem
    if not bool(invert.any()):
        return raw_case_ids
    result=raw_case_ids.clone()
    result[problem_positions[inside][invert]]=problem_rows[inside][invert,-1]
    return result


class PinnedSparseFlexiCubesV1:
    """Pinned source instance, isolated per-call ambiguity context.

    This wraps an EXACT verified upstream checkout; it does not vendor/patch
    the Git repository or substitute marching cubes. Construct a dedicated
    operator for each call to avoid cross-request mutable grid state.
    """

    def __init__(self, pinned: PinnedFlexiCubesV1) -> None:
        if not isinstance(pinned,PinnedFlexiCubesV1):
            raise ValueError("RSS_VNEXT_SPARSE_FC_PINNED_OPERATOR_REQUIRED")
        if pinned.identity.get("commit")!=FLEXICUBES_COMMIT:
            raise ValueError("RSS_VNEXT_SPARSE_FC_PINNED_SOURCE_DRIFT")
        self.pinned=pinned

    def extract(
        self, *,
        vertices_iris: Tensor,
        field: Tensor,
        cubes: Tensor,
        raw_parameters: Tensor,
        global_cube_ids: Tensor,
        resolution: tuple[int,int,int],
        training: bool,
    ) -> tuple[Tensor,Tensor,Tensor]:
        nx,ny,nz=_resolution_v1(resolution)
        n=vertices_iris.shape[0]
        c=cubes.shape[0]
        if (
            vertices_iris.ndim!=2 or vertices_iris.shape[-1]!=3
            or not 3<=n<=4_000_000
            or field.shape!=(n,)
            or cubes.shape!=(c,8) or cubes.dtype!=torch.long
            or not 1<=c<=2_000_000
            or raw_parameters.shape!=(c,21)
            or global_cube_ids.shape!=(c,)
            or any(x.device!=vertices_iris.device for x in (
                field,cubes,raw_parameters,global_cube_ids,
            ))
            or vertices_iris.device!=self.pinned.device
            or type(training) is not bool
            or bool((cubes<0).any()) or int(cubes.max())>=n
            or not bool(torch.isfinite(vertices_iris).all())
            or not bool(torch.isfinite(field).all())
            or not bool(torch.isfinite(raw_parameters).all())
        ):
            raise ValueError("RSS_VNEXT_SPARSE_FC_MESH_INPUT_INVALID")
        # SAFETY: require actual surface-cube closure. Each supplied cube
        # MUST change sign, and missing adjacent crossing cells are handled
        # by the separate exhaustive coverage certification before release.
        inside=field[cubes]<0
        if not bool((inside.any(-1) & ~inside.all(-1)).all()):
            raise ValueError("RSS_VNEXT_SPARSE_FC_NON_ACTIVE_CUBE_PROHIBITED")
        operator_type=type(self.pinned.operator)
        class SparseCasePinned(operator_type):
            @torch.no_grad()
            def _get_case_id(self, occ_fx8, surf_cubes, res):
                return globally_addressed_case_ids_v1(
                    self,occ_fx8,surf_cubes,tuple(res),
                    self._realsas_global_cube_ids,
                )
        operator=SparseCasePinned(
            device=str(self.pinned.device),
            weight_scale=self.pinned.weight_scale,
        )
        operator._realsas_global_cube_ids=global_cube_ids
        with torch.autocast(device_type=vertices_iris.device.type,enabled=False):
            params=split_flexicubes_parameters_v1(
                raw_parameters.float(),
                weight_scale=self.pinned.weight_scale,
            )
            xyz,faces,regularizer=operator(
                vertices_iris.float(),field.float(),cubes,
                resolution,beta_fx12=params.beta_raw,
                alpha_fx8=params.alpha_raw,
                gamma_f=params.gamma_raw,
                training=training,
            )
        if (
            xyz.ndim!=2 or xyz.shape[-1]!=3 or xyz.numel()==0
            or faces.ndim!=2 or faces.shape[-1]!=3 or faces.numel()==0
            or not bool(torch.isfinite(xyz).all())
            or not bool(torch.isfinite(regularizer).all())
        ):
            raise ValueError("RSS_VNEXT_SPARSE_FC_EMPTY_OR_INVALID_EXTRACT")
        return xyz,faces.long(),regularizer


__all__=[
    "SPARSE_CASE_SCHEMA","globally_addressed_case_ids_v1",
    "PinnedSparseFlexiCubesV1",
]
