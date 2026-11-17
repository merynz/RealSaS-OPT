from __future__ import annotations

"""Source-pinned MIT DCUDF2 backend for *opt-in* open surface extraction.

The official 2025 DCUDF2 code performs r-isosurface double-cover optimization
and optional graph cut. Open surface CUT MUST be enabled, and the original
code can silently fall back to a closed double cover when cut fails.

This adapter validates the output as a genuine indexed OPEN 2-manifold with
boundary loops, source-free same-scene H identity and bounded UDF residual.
It deliberately does NOT call a simple signed MC fallback or trim a closed
mesh to pretend an open garment exists.

The upstream runtime writes progress meshes into fixed relative paths and
requires CUDA. Therefore it is allowed ONLY in a dedicated isolated worker
whose cwd is already its unique workspace. It must not be called from a
multi-request process or change cwd in a shared application server.

All output remains unqualified, even if these preflight tests pass. Real GPU
execution, role-conditioned unsigned distance, tissue/self-intersection checks,
TESSA deformation and Compiler G2/G3/G3B are still separate gates.
"""

from dataclasses import dataclass
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Callable
import math

import numpy as np
import torch
from torch import Tensor

from .open_udf_surface_v1 import (
    AUTHORITY, OpenUDFSurfaceCandidateV1, _check_topology, _components,
    _orient_consistently, tensor_bytes_sha256_v1,
)

# Git blob hashes from exact MIT upstream source bytes. Detects a changed
# copy even if someone leaves a version string unchanged. It is NOT enough
# to assert full dependency licensing or GPU behavior.
UPSTREAM = "https://github.com/jjjkkyz/DCUDF2"
UPSTREAM_GIT_BLOBS = {
    "dcudf2/mesh_extraction.py":"bf1c4b09987ee8f81afdcc4edaac66475f3fdec8",
    "dcudf2/mesh_cut.py":"725bd77a9020442b9c3b4968e642dfcecd9d8ebb",
    "dcudf2/VectorAdam.py":"8e4f90f8db50383f81c3d80fdc72c0a08ed98940",
    "LICENSE":"8ba2a9a426fd73d26ce948129565ba878f953d03",
}
BACKEND_ID = "MIT_DCUDF2_UPSTREAM_SOURCE_PINNED_V1"


def _git_blob_sha1(raw: bytes) -> str:
    return sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()


def validate_dcudf2_checkout_v1(root: Path) -> None:
    """Require the exact official MIT source before importing its code."""
    if not isinstance(root,Path) or not root.is_dir():
        raise ValueError("RSS_VNEXT_DCUDF2_CHECKOUT_REQUIRED")
    for relative,expected in UPSTREAM_GIT_BLOBS.items():
        path=(root/relative).resolve()
        if (
            not path.is_file()
            or root.resolve() not in path.parents
            or _git_blob_sha1(path.read_bytes())!=expected
        ):
            raise ValueError("RSS_VNEXT_DCUDF2_PINNED_SOURCE_DRIFT:"+relative)


@dataclass(frozen=True)
class DCUDF2OpenPolicyV1:
    # Official code supports scalar resolution; not necessarily 256x256x384.
    resolution: int = 256
    threshold: float = .005
    max_iter: int = 400
    normal_step: int = 300
    laplacian_weight: float = 2000.
    max_batch: int = 100_000
    max_vertices: int = 1_000_000
    max_faces: int = 2_000_000
    max_residual_grid_steps: float = .6
    is_cut: bool = True
    watertight_separate: bool = False

    def validate(self) -> None:
        if (
            type(self.resolution) is not int
            or not 64<=self.resolution<=512
            or type(self.max_iter) is not int
            or not 100<=self.max_iter<=1000
            or type(self.normal_step) is not int
            or not 20<=self.normal_step<=self.max_iter
            or type(self.max_batch) is not int
            or not 2048<=self.max_batch<=250000
            or type(self.max_vertices) is not int
            or not 3<=self.max_vertices<=1_000_000
            or type(self.max_faces) is not int
            or not 1<=self.max_faces<=2_000_000
            or type(self.threshold) not in (int,float)
            or not 0<self.threshold<.05
            or type(self.laplacian_weight) not in (int,float)
            or not 0<self.laplacian_weight<=10000
            or type(self.max_residual_grid_steps) not in (int,float)
            or not 0<self.max_residual_grid_steps<=1
            or any(not math.isfinite(float(v)) for v in (
                self.threshold,self.laplacian_weight,
                self.max_residual_grid_steps
            ))
            or self.is_cut is not True
            or self.watertight_separate is not False
        ):
            raise ValueError("RSS_VNEXT_DCUDF2_OPEN_QUALITY_POLICY_INVALID")


@dataclass(frozen=True)
class DCUDF2OpenWitnessV1:
    surface: OpenUDFSurfaceCandidateV1
    backend_id: str
    pinned_source_git_blob_hashes: tuple[tuple[str,str],...]
    maximum_residual: float
    output_boundary_count: int
    open_surface_qualified: bool = False
    graphics_or_mechanics_qualified: bool = False

    def validate(self) -> None:
        self.surface.validate()
        if (
            self.backend_id!=BACKEND_ID
            or self.pinned_source_git_blob_hashes
                !=tuple(sorted(UPSTREAM_GIT_BLOBS.items()))
            or not math.isfinite(self.maximum_residual)
            or self.maximum_residual!=self.surface.field_max_residual
            or self.output_boundary_count!=len(self.surface.boundary_edges)
            or self.open_surface_qualified
            or self.graphics_or_mechanics_qualified
        ):
            raise ValueError("RSS_VNEXT_DCUDF2_WITNESS_DRIFT")


def _checked_queries(
    query: Callable[[Tensor],Tensor],
    x: Tensor, *, chunk: int,
) -> Tensor:
    values=[]
    for start in range(0,len(x),chunk):
        points=x[start:start+chunk]
        y=query(points)
        if (
            not isinstance(y,Tensor)
            or y.shape!=(len(points),)
            or y.device!=points.device
            or not bool(torch.isfinite(y).all())
            or bool((y<0).any())
        ):
            raise ValueError("RSS_VNEXT_DCUDF2_OPEN_UDF_QUERY_INVALID")
        values.append(y)
    return torch.cat(values)


def materialize_dcudf2_open_mesh_v1(
    *, optimized_vertices: np.ndarray,
    optimized_faces: np.ndarray,
    sampled_field: np.ndarray,
    scene_triplanes: Tensor,
    query_open_component_udf: Callable[[Tensor],Tensor],
    policy: DCUDF2OpenPolicyV1 = DCUDF2OpenPolicyV1(),
    source_role_conditioned: bool,
) -> DCUDF2OpenWitnessV1:
    """Validate exact output of pinned backend; no unqualified closed fallback.

    source_role_conditioned does NOT establish source role truth. It is the
    caller's explicit acknowledgement that this is an OPEN-COMPONENT-SPECIFIC
    UDF query, not the global nearest-surface UDF shared by all components.
    The learned role/field provenance must be bound by the caller's model
    receipt before Compiler admission. Nothing here claims that qualification.
    """
    policy.validate()
    if (
        source_role_conditioned is not True
        or not isinstance(scene_triplanes,Tensor)
        or scene_triplanes.ndim!=5 or scene_triplanes.shape[0]!=1
        or not bool(torch.isfinite(scene_triplanes).all())
        or not callable(query_open_component_udf)
    ):
        raise ValueError("RSS_VNEXT_DCUDF2_ROLE_CONDITIONED_H_REQUIRED")
    v=np.asarray(optimized_vertices)
    f=np.asarray(optimized_faces)
    volume=np.asarray(sampled_field)
    if (
        v.ndim!=2 or v.shape[-1]!=3
        or f.ndim!=2 or f.shape[-1]!=3
        or len(v)<3 or len(f)<1
        or len(v)>policy.max_vertices or len(f)>policy.max_faces
        or f.dtype.kind not in "iu"
        or volume.shape!=(policy.resolution,)*3
        or not np.isfinite(v).all()
        or not np.isfinite(volume).all()
        or (volume<0).any()
        or (f<0).any() or int(f.max())>=len(v)
    ):
        raise ValueError("RSS_VNEXT_DCUDF2_BACKEND_RESULT_DOMAIN")
    xyz=torch.as_tensor(
        v.copy(),device=scene_triplanes.device,dtype=scene_triplanes.dtype,
    )
    faces=torch.as_tensor(
        f.copy(),device=scene_triplanes.device,dtype=torch.long,
    )
    # A pure double-cover extraction is closed and MUST fail here.
    # The graph cut may be random or fail without raising; no silent success.
    face_list=_orient_consistently(faces.cpu().tolist())
    faces=torch.tensor(face_list,device=faces.device,dtype=torch.long)
    boundary=_check_topology(faces,len(xyz))
    tri=xyz[faces]
    area2=torch.linalg.vector_norm(torch.cross(
        tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1,
    ),dim=-1)
    if (
        not bool(torch.isfinite(area2).all())
        or bool((area2<=1.e-10).any())
    ):
        raise ValueError("RSS_VNEXT_DCUDF2_DEGENERATE_OPTIMIZED_FACE")
    # Reject disconnected orphan vertices; all indexed rows must belong
    # to at least one verified face and share its component namespace.
    if len(torch.unique(faces.reshape(-1)))!=len(xyz):
        raise ValueError("RSS_VNEXT_DCUDF2_ORPHAN_VERTEX")
    with torch.no_grad():
        residual=_checked_queries(
            query_open_component_udf,xyz,chunk=min(policy.max_batch,16384),
        )
        max_residual=float(residual.max().item())
    # Bounds in the grid's coordinate system, not arbitrary asset units.
    grid_step=2./(policy.resolution-1)
    if max_residual>policy.max_residual_grid_steps*grid_step:
        raise ValueError("RSS_VNEXT_DCUDF2_FIELD_RESIDUAL_TOO_HIGH")
    candidate=OpenUDFSurfaceCandidateV1(
        xyz=xyz.unsqueeze(0),
        faces=faces,
        component_ids=_components(len(xyz),faces),
        boundary_edges=boundary,
        scene_sha256=tensor_bytes_sha256_v1(scene_triplanes),
        sampled_udf_sha256=tensor_bytes_sha256_v1(
            torch.from_numpy(np.ascontiguousarray(volume)),
        ),
        extraction_resolution=(policy.resolution,)*3,
        field_max_residual=max_residual,
    )
    witness=DCUDF2OpenWitnessV1(
        surface=candidate,
        backend_id=BACKEND_ID,
        pinned_source_git_blob_hashes=tuple(sorted(UPSTREAM_GIT_BLOBS.items())),
        maximum_residual=max_residual,
        output_boundary_count=len(boundary),
    )
    witness.validate()
    return witness


def extract_dcudf2_open_on_isolated_gpu_worker_v1(
    *, source_checkout: Path,
    isolated_workspace: Path,
    scene_triplanes: Tensor,
    query_open_component_udf: Callable[[Tensor],Tensor],
    source_role_conditioned: bool,
    policy: DCUDF2OpenPolicyV1 = DCUDF2OpenPolicyV1(),
) -> DCUDF2OpenWitnessV1:
    """Run original MIT paper backend WITHOUT changing its algorithm.

    Must be invoked in a dedicated CUDA worker started with cwd set to the
    workspace. No implicit installation, download, system paths or fake
    fallback. Runtime GPU parity has not been independently demonstrated.
    """
    policy.validate()
    validate_dcudf2_checkout_v1(source_checkout)
    if (
        not isinstance(isolated_workspace,Path)
        or not isolated_workspace.is_dir()
        or Path.cwd().resolve()!=isolated_workspace.resolve()
        or source_role_conditioned is not True
        or not torch.cuda.is_available()
        or not scene_triplanes.is_cuda
    ):
        raise ValueError("RSS_VNEXT_DCUDF2_DEDICATED_CUDA_WORKER_REQUIRED")
    (isolated_workspace/"experiment"/"optimizer").mkdir(
        parents=True,exist_ok=True,
    )
    # Import strictly after verifying on-disk MIT code bytes. Import resolution
    # must point exactly at this checked source, not a different PYTHONPATH.
    module=import_module("dcudf2.mesh_extraction")
    if (
        Path(module.__file__).resolve()
        !=(source_checkout/"dcudf2"/"mesh_extraction.py").resolve()
    ):
        raise ValueError("RSS_VNEXT_DCUDF2_IMPORT_SHADOWED")
    # The backend may call query_func with large batches and auto-diff.
    # It is only allowed to query the selected component from SAME IRIS H.
    def strict_query(points: Tensor)->Tensor:
        if (
            not isinstance(points,Tensor)
            or points.ndim!=2 or points.shape[-1]!=3
            or points.device!=scene_triplanes.device
        ):
            raise ValueError("RSS_VNEXT_DCUDF2_QUERY_ADDRESS_DOMAIN")
        return _checked_queries(
            query_open_component_udf,points,chunk=policy.max_batch,
        )
    extractor=module.dcudf2(
        strict_query,policy.resolution,policy.threshold,
        max_iter=policy.max_iter,
        normal_step=policy.normal_step,
        laplacian_weight=policy.laplacian_weight,
        max_batch=policy.max_batch,
        is_cut=True,watertight_separate=False,
    )
    out=extractor.optimize()
    if out is None or extractor.u is None:
        raise ValueError("RSS_VNEXT_DCUDF2_OPTIMIZER_NO_OUTPUT")
    return materialize_dcudf2_open_mesh_v1(
        optimized_vertices=np.asarray(out.vertices),
        optimized_faces=np.asarray(out.faces),
        sampled_field=np.asarray(extractor.u),
        scene_triplanes=scene_triplanes,
        query_open_component_udf=strict_query,
        policy=policy,
        source_role_conditioned=source_role_conditioned,
    )


__all__=[
    "BACKEND_ID","UPSTREAM","UPSTREAM_GIT_BLOBS",
    "DCUDF2OpenPolicyV1","DCUDF2OpenWitnessV1",
    "validate_dcudf2_checkout_v1","materialize_dcudf2_open_mesh_v1",
    "extract_dcudf2_open_on_isolated_gpu_worker_v1",
]
