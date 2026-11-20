from __future__ import annotations

"""Source-verified, posed TESSA M* -> 2D Graphics/SLR training bridge.

Pose-space geometry controls first-hit visibility, silhouette, normals, depth.
Canonical/rest-surface positions control RGB and role sampling from ONE H(x):
material appearance does not 'swim' with world coordinates under LBS motion.
The source camera/pose/2D targets are LOSS-ONLY and are never observations.

CPU BUSTED raster is a bounded differentiable correctness reference, not
GPU throughput qualification. CUDA requires the compiled PyTorch3D backend.
This is opaque first-hit M* raster; NO hidden-texel, UV-atlas, XPBD, cloth
material or final ProductProof authority is minted here.
"""
from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.iris.vnext.flexicubes_v1 import vertex_normals_v1
from .realsas_flexicubes_mesh_render_v1 import (
    _camera_check, stage3_default_raster_backend_v1,
    RASTER_ID, PYTORCH3D_RASTER_ID,
)
from .busted_raster_backend_v1 import BustedTorchRasterV1
from .pytorch3d_raster_backend_v1 import PyTorch3DRasterV1

SCHEMA = "RealSaS.RSSvNext.NativeMstarPoseRasterTeacher.v1"
RENDER_SCHEMA = "RealSaS.RSSvNext.PosedIndexedMstarRaster.v1"
AUTHORITY = "QUALIFIED_SOURCE_NATIVE_POSE_RENDER_NOT_GENERATED"
VISIBLE_POLICY = "SOURCE_NATIVE_OPAQUE_ZBUFFER_FIRST_HIT"


@dataclass(frozen=True)
class NativeMstarPoseRasterTargetV1:
    pose_index: int
    cameras: Tensor                # [1,1|2|6,13], TRAINING TEACHER ONLY
    height: int
    width: int
    rgb_white: Tensor              # [1,R,3] source-3D rendered image
    alpha: Tensor                  # [1,R,1] first-hit geometric alpha
    rgb_mask: Tensor               # [1,R] source-qualified
    alpha_mask: Tensor             # [1,R] source-qualified
    source_3d_truth_binding_sha256: str
    role_indices: Tensor | None = None        # [1,R], -1 UNKNOWN
    role_mask: Tensor | None = None           # [1,R] bool
    normal_xyz: Tensor | None = None          # [1,R,3] world normal
    normal_mask: Tensor | None = None         # [1,R] bool
    depth: Tensor | None = None               # [1,R,1] camera-forward depth
    depth_mask: Tensor | None = None          # [1,R] bool
    source_authority: str = AUTHORITY
    visibility_policy: str = VISIBLE_POLICY
    generated_completion_used_as_truth: bool = False

    def validate(self, *, device: torch.device, role_count: int) -> None:
        if (
            type(self.pose_index) is not int or self.pose_index < 0
            or type(self.height) is not int or type(self.width) is not int
            or not 8 <= self.height <= 1024 or not 8 <= self.width <= 1024
            or self.source_authority != AUTHORITY
            or self.visibility_policy != VISIBLE_POLICY
            or self.generated_completion_used_as_truth is not False
            or not isinstance(self.source_3d_truth_binding_sha256, str)
            or len(self.source_3d_truth_binding_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in
                   self.source_3d_truth_binding_sha256)
            or not isinstance(self.cameras, Tensor)
            or self.cameras.device != device
        ):
            raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_TEACHER_DOMAIN")
        _camera_check(self.cameras)
        r = self.cameras.shape[1] * self.height * self.width
        values = (
            (self.rgb_white, (1, r, 3)),
            (self.alpha, (1, r, 1)),
            (self.rgb_mask, (1, r)),
            (self.alpha_mask, (1, r)),
        )
        if any(
            not isinstance(t, Tensor) or t.shape != shape or t.device != device
            for t, shape in values
        ) or (
            self.rgb_mask.dtype != torch.bool
            or self.alpha_mask.dtype != torch.bool
            or not bool(self.rgb_mask.any())
            or not bool(self.alpha_mask.any())
            or not torch.is_floating_point(self.rgb_white)
            or not torch.is_floating_point(self.alpha)
            or any(not bool(torch.isfinite(t).all())
                   for t in (self.rgb_white, self.alpha))
            or bool(((self.rgb_white < 0) | (self.rgb_white > 1)).any())
            or bool(((self.alpha < 0) | (self.alpha > 1)).any())
        ):
            raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_TARGET_INVALID")
        optional = (
            (self.role_indices, self.role_mask, (1, r), (1, r)),
            (self.normal_xyz, self.normal_mask, (1, r, 3), (1, r)),
            (self.depth, self.depth_mask, (1, r, 1), (1, r)),
        )
        for target, mask, tshape, mshape in optional:
            if (target is None) != (mask is None):
                raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_OPTIONAL_PAIR")
            if target is None:
                continue
            if (
                target.shape != tshape or mask.shape != mshape
                or mask.dtype != torch.bool
                or target.device != device or mask.device != device
                or not bool(mask.any())
            ):
                raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_OPTIONAL_SHAPE")
            if target is self.role_indices:
                if (
                    target.dtype != torch.long
                    or bool((target[mask] < 0).any())
                    or bool((target[mask] >= role_count).any())
                ):
                    raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_ROLE_DOMAIN")
            elif (
                not torch.is_floating_point(target)
                or not bool(torch.isfinite(target).all())
            ):
                raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_OPTIONAL_FINITE")
        # Source-rendered 2D truth may only supervise rendered surface samples.
        # It cannot assert the RGB/normal of fully transparent background.
        for mask in (self.role_mask, self.normal_mask, self.depth_mask):
            if mask is not None and bool((mask & (self.alpha[..., 0] == 0)).any()):
                raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_OCCLUDED_COLOR_TRUTH")


@dataclass(frozen=True)
class NativeMstarPoseRasterWeightsV1:
    rgb: float = 1.
    alpha: float = 1.
    role: float = 0.25
    normal: float = 0.25
    depth: float = 0.25

    def validate(self) -> None:
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
               for x in asdict(self).values()) or self.rgb <= 0 or self.alpha <= 0:
            raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_LOSS_POLICY")


@dataclass(frozen=True)
class NativeMstarPoseRasterV1:
    rgb_white: Tensor             # [1,R,3] in source top-left order
    alpha: Tensor                 # [1,R,1]
    roles: Tensor                 # [1,R,K]
    normal_xyz: Tensor            # [1,R,3]
    depth: Tensor                 # [1,R,1]
    raster_backend_id: str
    authority: str = "MODEL_POSED_MSTAR_RENDER_NOT_SOURCE_TRUTH"


def render_posed_indexed_mstar_v1(
    *, rest_xyz: Tensor, posed_xyz: Tensor, faces: Tensor,
    scene_triplanes: Tensor, iris: nn.Module,
    graphics_head: nn.Module, role_head: nn.Module,
    cameras: Tensor, height: int, width: int,
    raster_backend=None, query_chunk: int = 32768,
) -> NativeMstarPoseRasterV1:
    """Real indexed M* raster, NOT separately extracted FC or source mesh."""
    _camera_check(cameras)
    if (
        rest_xyz.ndim != 3 or posed_xyz.shape != rest_xyz.shape
        or rest_xyz.shape[0] != 1 or rest_xyz.shape[2] != 3
        or not isinstance(faces, Tensor) or faces.ndim != 2
        or faces.shape[1] != 3 or not 1 <= faces.shape[0] <= 250000
        or faces.dtype not in (torch.long, torch.int64)
        or not 3 <= rest_xyz.shape[1] <= 250000
        or int(faces.min()) < 0 or int(faces.max()) >= rest_xyz.shape[1]
        or not bool(torch.isfinite(rest_xyz).all())
        or not bool(torch.isfinite(posed_xyz).all())
        or not isinstance(scene_triplanes, Tensor)
        or scene_triplanes.ndim != 5
        or scene_triplanes.shape[:2] != (1, 3)
        or any(t.device != rest_xyz.device for t in
               (posed_xyz, faces, scene_triplanes, cameras))
        or type(height) is not int or type(width) is not int
        or not 8 <= height <= 1024 or not 8 <= width <= 1024
        or height * width * cameras.shape[1] > 6 * 1024 * 1024
        or type(query_chunk) is not int or not 128 <= query_chunk <= 262144
    ):
        raise ValueError("RSS_VNEXT_MSTAR_POSE_RASTER_INPUT_DOMAIN")
    if raster_backend is None:
        raster_backend = stage3_default_raster_backend_v1(rest_xyz.device)
    if raster_backend not in (BustedTorchRasterV1, PyTorch3DRasterV1):
        raise ValueError("RSS_VNEXT_MSTAR_POSE_RASTER_BACKEND_UNQUALIFIED")
    if (
        rest_xyz.device.type == "cuda"
        and raster_backend is not PyTorch3DRasterV1
    ):
        raise ValueError("RSS_VNEXT_MSTAR_POSE_RASTER_CUDA_COMPILED_ONLY")
    renderer_id = (
        PYTORCH3D_RASTER_ID if raster_backend is PyTorch3DRasterV1 else RASTER_ID
    )
    context = raster_backend.RasterizeCudaContext()
    tri = faces.to(dtype=torch.int32).contiguous()
    rest = rest_xyz[0].float()
    posed = posed_xyz[0].float()
    normals = vertex_normals_v1(posed, faces).float()
    per_view = []
    for vi in range(cameras.shape[1]):
        camera = cameras[0, vi].float()
        delta = posed - camera[None, :3]
        right = (delta @ camera[3:6]) / camera[12]
        up = (delta @ camera[6:9]) / camera[12]
        forward = delta @ camera[9:12]
        with torch.no_grad():
            zmin = forward.min() - 0.05
            zmax = forward.max() + 0.05
        if float((zmax - zmin).detach()) <= 1e-5:
            raise ValueError("RSS_VNEXT_MSTAR_POSE_RASTER_DEPTH_DEGENERATE")
        z = (forward - zmin) * 2. / (zmax - zmin) - 1.
        clip = torch.stack(
            (right, up, z, torch.ones_like(right)), dim=-1
        ).contiguous()[None]
        rast, _ = raster_backend.rasterize(
            context, clip, tri, resolution=(height, width),
        )
        # Interpolate CANONICAL material coordinates using POSED silhouette.
        # Rendering world-space H(posed_xyz) would cause texture swimming.
        rest_sample, _ = raster_backend.interpolate(rest[None], rast, tri)
        posed_sample, _ = raster_backend.interpolate(posed[None], rast, tri)
        normal_sample, _ = raster_backend.interpolate(normals[None], rast, tri)
        coverage = (rast[..., 3:4] > 0).to(dtype=rest.dtype)
        # Only visible first-hit pixels own material samples. A dense
        # H(x) query over millions of background pixels would waste GPU
        # activations (and backprop memory) without changing a single pixel.
        # index_copy remains differentiable through the sampled rest XYZ.
        positions = rest_sample.reshape(1, height * width, 3)
        visible_indices = (
            (rast[..., 3] > 0).reshape(-1)
            .nonzero(as_tuple=False).reshape(-1)
        )
        if not len(visible_indices):
            raise RuntimeError("RSS_VNEXT_MSTAR_POSE_RASTER_NO_VISIBLE_SOURCE_SURFACE")
        sampled = positions.index_select(1, visible_indices)
        colors, roles = [], []
        for start in range(0, sampled.shape[1], query_chunk):
            h = iris.query(
                scene_triplanes,
                sampled[:, start:start+query_chunk].to(scene_triplanes.dtype),
            )["H"]
            rgba_logits = graphics_head(h)
            role_logits = role_head(h)
            if rgba_logits.shape[-1] != 5 or role_logits.shape[-1] < 2:
                raise RuntimeError("RSS_VNEXT_MSTAR_POSE_GRAPHICS_HEAD_ABI")
            colors.append(torch.sigmoid(rgba_logits[..., :3].float()))
            roles.append(torch.softmax(role_logits.float(), dim=-1))
        visible_rgb = torch.cat(colors, dim=1)
        visible_role = torch.cat(roles, dim=1)
        # Explicit zero background; one predicted first-hit owner per pixel.
        color = visible_rgb.new_zeros((1, height*width, 3)).index_copy(
            1, visible_indices, visible_rgb,
        ).reshape(1, height, width, 3)
        role = visible_role.new_zeros(
            (1, height*width, visible_role.shape[-1])
        ).index_copy(
            1, visible_indices, visible_role,
        ).reshape(1, height, width, -1)
        rgba = torch.cat((color * coverage, coverage), dim=-1)
        smooth = raster_backend.antialias(rgba, rast, clip, tri).clamp(0., 1.)
        alpha = smooth[..., 3:4]
        white = (smooth[..., :3] + 1. - alpha).clamp(0., 1.)
        normal = F.normalize(normal_sample, dim=-1, eps=1e-8) * coverage
        depth = (
            (posed_sample - camera[None, None, None, :3])
            * camera[None, None, None, 9:12]
        ).sum(-1, keepdim=True) * coverage
        # Raster backends emit bottom-left; source PNG is top-left.
        per_view.append(tuple(
            t.flip(dims=(1,)).reshape(1, height*width, -1)
            for t in (white, alpha, role*coverage, normal, depth)
        ))
    groups = [
        torch.cat([x[i] for x in per_view], dim=1)
        for i in range(5)
    ]
    return NativeMstarPoseRasterV1(
        rgb_white=groups[0], alpha=groups[1], roles=groups[2],
        normal_xyz=groups[3], depth=groups[4],
        raster_backend_id=renderer_id,
    )


def qualified_product_mstar_pose_loss_v1(
    *, model, state, mesh, predicted_posed_xyz: Tensor | None,
    batch, seeded_teacher, target: NativeMstarPoseRasterTargetV1 | None,
    admission_key: str, trusted_registry: Mapping[str, Mapping[str, object]],
    dataset_manifest_sha256: str, loss_weights: NativeMstarPoseRasterWeightsV1,
    tensor_digest, axis_sequence_digest, seeded_mstar_address_digest,
) -> tuple[Tensor | None, dict[str, Tensor] | None, int]:
    """Fail closed on any unbound source-native pose/render/mesh teacher."""
    if target is None and not admission_key:
        return None, None, 0
    if (
        not isinstance(target, NativeMstarPoseRasterTargetV1)
        or not admission_key or not isinstance(trusted_registry, Mapping)
        or not isinstance(predicted_posed_xyz, Tensor)
        or seeded_teacher.native_pose_transport is None
        or seeded_teacher.source_pose_matrices is None
        or seeded_teacher.source_pose_mask is None
        or batch.axis_teacher_sequence is None
        or mesh.seeded_surface is None
    ):
        raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_NATIVE_POSE_REQUIRED")
    loss_weights.validate()
    target.validate(
        device=mesh.rest_vertices_iris.device,
        role_count=model.cfg.semantic_role_schema().count,
    )
    pose = target.pose_index
    source_matrices = seeded_teacher.source_pose_matrices
    source_mask = seeded_teacher.source_pose_mask
    if (
        source_matrices.ndim != 5 or source_matrices.shape[0] != 1
        or source_mask.ndim != 3 or source_mask.shape[0] != 1
        or source_mask.shape[1] != source_matrices.shape[1]
        or source_mask.shape[2] != mesh.rest_vertices_iris.shape[1]
        or pose >= source_matrices.shape[1]
        or predicted_posed_xyz.ndim != 4
        or predicted_posed_xyz.shape[:2] != source_matrices.shape[:2]
        or predicted_posed_xyz.shape[2:] != mesh.rest_vertices_iris.shape[1:]
        or not bool(source_mask[0, pose].all())
    ):
        raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_POSE_ADDRESS_INVALID")
    row = trusted_registry.get(admission_key)
    topo = seeded_mstar_address_digest(mesh.seeded_surface)
    expected = {
        "schema": SCHEMA,
        "admission_status": "QUALIFIED_SOURCE_NATIVE_MSTAR_POSE_RENDER",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_asset_sha256": seeded_teacher.source_asset_sha256,
        "source_qualification_key": batch.source_qualification_key,
        "observed_rgba_sha256": tensor_digest(batch.rgba),
        "original_camera_sha256": tensor_digest(batch.camera_features),
        "source_binding_sha256": batch.product_source_binding_sha256,
        "generation_checkpoint_sha256": batch.product_generation_checkpoint_sha256,
        "candidate_mstar_topology_address_sha256": topo,
        "axis_teacher_sequence_sha256": axis_sequence_digest(
            batch.axis_teacher_sequence
        ),
        "native_pose_transport_receipt_sha256":
            seeded_teacher.native_pose_transport.receipt_sha256(),
        "source_pose_matrices_sha256": tensor_digest(source_matrices),
        "source_pose_mask_sha256": tensor_digest(source_mask),
        "pose_index": pose,
        "render_height": target.height,
        "render_width": target.width,
        "source_camera_sha256": tensor_digest(target.cameras),
        "target_rgb_white_sha256": tensor_digest(target.rgb_white),
        "target_alpha_sha256": tensor_digest(target.alpha),
        "target_rgb_mask_sha256": tensor_digest(target.rgb_mask),
        "target_alpha_mask_sha256": tensor_digest(target.alpha_mask),
        "target_role_sha256": (
            tensor_digest(target.role_indices)
            if target.role_indices is not None else None
        ),
        "target_role_mask_sha256": (
            tensor_digest(target.role_mask)
            if target.role_mask is not None else None
        ),
        "target_normal_sha256": (
            tensor_digest(target.normal_xyz)
            if target.normal_xyz is not None else None
        ),
        "target_normal_mask_sha256": (
            tensor_digest(target.normal_mask)
            if target.normal_mask is not None else None
        ),
        "target_depth_sha256": (
            tensor_digest(target.depth)
            if target.depth is not None else None
        ),
        "target_depth_mask_sha256": (
            tensor_digest(target.depth_mask)
            if target.depth_mask is not None else None
        ),
        "source_3d_truth_binding_sha256":
            target.source_3d_truth_binding_sha256,
        "source_visibility_authority": VISIBLE_POLICY,
        "source_renderer_independent_of_student": True,
        "generated_six_views_are_not_source_gt": True,
        "source_native_pose_was_verified_before_renderer": True,
        "source_material_opacity_policy": "OPAQUE_SURFACE_ONLY",
        "source_truth_is_loss_only": True,
    }
    if row is None or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("RSS_VNEXT_MSTAR_POSE_RENDER_SOURCE_RECEIPT_DRIFT")
    raster = render_posed_indexed_mstar_v1(
        rest_xyz=mesh.rest_vertices_iris,
        posed_xyz=predicted_posed_xyz[:, pose],
        faces=mesh.faces, scene_triplanes=state.scene_triplanes,
        iris=model.iris, graphics_head=model.graphics,
        role_head=model.presentation_role_head,
        cameras=target.cameras,
        height=target.height, width=target.width,
    )
    if raster.authority != "MODEL_POSED_MSTAR_RENDER_NOT_SOURCE_TRUTH":
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_RASTER_AUTHORITY_DRIFT")
    def selected_mse(a: Tensor, b: Tensor, m: Tensor) -> Tensor:
        return F.mse_loss(a[m].float(), b[m].float())
    parts: dict[str, Tensor] = {
        "rgb": selected_mse(raster.rgb_white,target.rgb_white,target.rgb_mask),
        "alpha": selected_mse(
            raster.alpha[..., 0],target.alpha[..., 0],target.alpha_mask,
        ),
    }
    if target.role_indices is not None:
        parts["role"] = F.nll_loss(
            torch.log(raster.roles[target.role_mask].clamp_min(1e-8)),
            target.role_indices[target.role_mask],
        )
    if target.normal_xyz is not None:
        pred = F.normalize(
            raster.normal_xyz[target.normal_mask], dim=-1, eps=1e-8,
        )
        ref = F.normalize(
            target.normal_xyz[target.normal_mask], dim=-1, eps=1e-8,
        )
        parts["normal"] = (1. - (pred * ref).sum(-1)).mean()
    if target.depth is not None:
        parts["depth"] = selected_mse(
            raster.depth[..., 0], target.depth[..., 0], target.depth_mask,
        )
    coefficients = asdict(loss_weights)
    total = sum(
        (coefficients[name] * term for name, term in parts.items()),
        raster.rgb_white.new_zeros(()),
    )
    if not bool(torch.isfinite(total).all()):
        raise RuntimeError("RSS_VNEXT_MSTAR_POSE_RENDER_LOSS_NONFINITE")
    return total, parts, int(target.rgb_mask.sum().item())


__all__ = [
    "SCHEMA", "RENDER_SCHEMA", "AUTHORITY", "VISIBLE_POLICY",
    "NativeMstarPoseRasterTargetV1", "NativeMstarPoseRasterWeightsV1",
    "NativeMstarPoseRasterV1", "render_posed_indexed_mstar_v1",
    "qualified_product_mstar_pose_loss_v1",
]
