from __future__ import annotations

"""Qualified garment-vs-body posed clearance on the one indexed TESSA M*.

A source-native body-face/garment-vertex pair is admissible only if the
independent original source mesh and its exact source->M* face transport
support it. The source body winding must be certified outward by the
training launcher, not guessed from a Semantic role. No nearest STUDENT
vertex, unsigned SDF sign, material coefficient or XPBD contact is inferred.

This is a differentiable positional consequence loss under independently
authenticated native poses, NOT cloth dynamics, penetration-safe simulation,
contact/attachment, or the final ProductProof.
"""
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

SCHEMA = "RealSaS.RSSvNext.NativePosedBodyGarmentClearance.v1"
SOURCE_CLASS = "INDEPENDENT_SOURCE_BODY_FACE_TO_GARMENT_VERTEX"


@dataclass(frozen=True)
class SourceNativeLayerClearanceTeacherV1:
    source_asset_sha256: str
    source_qualification_key: str
    garment_vertex_ids: Tensor              # [K] exact TESSA M* vertex IDs
    body_face_ids: Tensor                   # [K] exact TESSA M* face IDs
    body_face_barycentric: Tensor           # [K,3] independently native-derived
    qualified_pose_pair_mask: Tensor        # [1,P,K] known source-only pair
    source_class: str = SOURCE_CLASS

    def validate(self, *, mesh, source_native_posed: Tensor,
                 source_pose_mask: Tensor) -> None:
        v = mesh.rest_vertices_iris.shape[1]
        f = mesh.faces.shape[0]
        p = source_native_posed.shape[1]
        k = self.garment_vertex_ids.numel()
        if (
            self.source_class != SOURCE_CLASS
            or not isinstance(self.source_asset_sha256, str)
            or len(self.source_asset_sha256) != 64
            or any(ch not in "0123456789abcdef"
                   for ch in self.source_asset_sha256)
            or type(self.source_qualification_key) is not str
            or not self.source_qualification_key
            or not 1 <= k <= 131072
            or self.garment_vertex_ids.shape != (k,)
            or self.body_face_ids.shape != (k,)
            or self.body_face_barycentric.shape != (k, 3)
            or self.qualified_pose_pair_mask.shape != (1, p, k)
            or self.garment_vertex_ids.dtype != torch.long
            or self.body_face_ids.dtype != torch.long
            or self.qualified_pose_pair_mask.dtype != torch.bool
            or not bool(self.qualified_pose_pair_mask.any())
            or source_native_posed.shape != (1, p, v, 3)
            or source_pose_mask.shape != (1, p, v)
            or mesh.vertex_component_ids.shape != (v,)
            or mesh.face_component_ids.shape != (f,)
            or any(t.device != mesh.rest_vertices_iris.device for t in (
                self.garment_vertex_ids, self.body_face_ids,
                self.body_face_barycentric, self.qualified_pose_pair_mask,
                source_native_posed, source_pose_mask, mesh.faces,
            ))
            or not torch.is_floating_point(self.body_face_barycentric)
            or not bool(torch.isfinite(self.body_face_barycentric).all())
            or not bool(torch.isfinite(source_native_posed).all())
            or bool((self.garment_vertex_ids < 0).any())
            or bool((self.garment_vertex_ids >= v).any())
            or bool((self.body_face_ids < 0).any())
            or bool((self.body_face_ids >= f).any())
            or bool((self.body_face_barycentric < -1e-6).any())
            or bool((self.body_face_barycentric > 1.+1e-6).any())
            or not bool(torch.allclose(
                self.body_face_barycentric.sum(-1),
                self.body_face_barycentric.new_ones((k,)),
                atol=1e-6, rtol=0,
            ))
        ):
            raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_TEACHER_DOMAIN")
        # Component membership is structural/native, NOT an IRIS semantic
        # 'body/garment' label. The receipt separately certifies which is body.
        body_faces = mesh.faces[self.body_face_ids]
        b_comp = mesh.face_component_ids[self.body_face_ids]
        garment_comp = mesh.vertex_component_ids[self.garment_vertex_ids]
        if (
            bool((b_comp == garment_comp).any())
            or not bool((
                mesh.vertex_component_ids[body_faces] == b_comp[:, None]
            ).all())
        ):
            raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_COMPONENT_COLLAPSE")
        support = source_pose_mask[0, :, self.garment_vertex_ids]
        for corner in range(3):
            support = support & source_pose_mask[0, :, body_faces[:, corner]]
        if bool((self.qualified_pose_pair_mask[0] & ~support).any()):
            raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_UNSUPPORTED_POSE")


def native_layer_clearance_objective_v1(
    *, mesh, predicted_posed: Tensor | None, source_native_posed: Tensor | None,
    source_pose_mask: Tensor | None, teacher: SourceNativeLayerClearanceTeacherV1 | None,
    admission_key: str, trusted_registry: Mapping[str, Mapping[str, object]],
    dataset_manifest_sha256: str, native_pose_transport_sha256: str,
    mstar_address_sha256: str, source_asset_sha256: str,
    source_qualification_key: str, tensor_digest,
) -> tuple[Tensor | None, dict[str, Tensor] | None, int]:
    if teacher is None and not admission_key:
        return None, None, 0
    if (
        not isinstance(teacher, SourceNativeLayerClearanceTeacherV1)
        or not isinstance(predicted_posed, Tensor)
        or not isinstance(source_native_posed, Tensor)
        or not isinstance(source_pose_mask, Tensor)
        or not isinstance(trusted_registry, Mapping)
        or type(admission_key) is not str or not admission_key
        or not isinstance(native_pose_transport_sha256, str)
        or not isinstance(mstar_address_sha256, str)
        or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
               for v in (native_pose_transport_sha256, mstar_address_sha256))
        or teacher.source_asset_sha256 != source_asset_sha256
        or teacher.source_qualification_key != source_qualification_key
    ):
        raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_POSE_AND_REGISTRY_REQUIRED")
    if predicted_posed.shape != source_native_posed.shape:
        raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_MSTAR_POSE_SHAPE")
    teacher.validate(
        mesh=mesh, source_native_posed=source_native_posed,
        source_pose_mask=source_pose_mask,
    )
    expected = {
        "schema": SCHEMA,
        "admission_status": "QUALIFIED_SOURCE_NATIVE_BODY_GARMENT_CLEARANCE",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_asset_sha256": teacher.source_asset_sha256,
        "source_qualification_key": teacher.source_qualification_key,
        "candidate_mstar_topology_address_sha256": mstar_address_sha256,
        "native_pose_transport_receipt_sha256": native_pose_transport_sha256,
        "source_candidate_posed_sha256": tensor_digest(source_native_posed),
        "source_pose_mask_sha256": tensor_digest(source_pose_mask),
        "garment_vertex_ids_sha256": tensor_digest(teacher.garment_vertex_ids),
        "body_face_ids_sha256": tensor_digest(teacher.body_face_ids),
        "body_face_barycentric_sha256":
            tensor_digest(teacher.body_face_barycentric),
        "qualified_pose_pair_mask_sha256":
            tensor_digest(teacher.qualified_pose_pair_mask),
        "source_native_body_face_winding_outward_certified": True,
        "source_body_shell_signed_solid_verified": True,
        "source_native_pair_correspondence_verified": True,
        "source_rights_independently_verified": True,
        "source_generated_views_as_clearance_truth": False,
        "student_nearest_pairing_forbidden": True,
        "physical_material_or_xpbd_authority": False,
    }
    row = trusted_registry.get(admission_key)
    if row is None or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_RECEIPT_DRIFT")

    ids = teacher.garment_vertex_ids
    body_faces = mesh.faces[teacher.body_face_ids]
    bary = teacher.body_face_barycentric.to(source_native_posed.dtype)
    source_tri = source_native_posed.detach()[0, :, body_faces, :]  # [P,K,3,3]
    source_body = (source_tri * bary[None, :, :, None]).sum(-2)
    source_garment = source_native_posed.detach()[0, :, ids, :]
    n = torch.cross(
        source_tri[..., 1, :] - source_tri[..., 0, :],
        source_tri[..., 2, :] - source_tri[..., 0, :],
        dim=-1,
    )
    n_norm = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    valid = teacher.qualified_pose_pair_mask[0]
    if bool((n_norm[..., 0][valid] < 1e-8).any()):
        raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_SOURCE_NORMAL_DEGENERATE")
    source_normal = (n / n_norm.clamp_min(1e-8)).detach()
    # Source signed gap, measured by independent source body outward winding.
    gap_source = (
        (source_garment - source_body) * source_normal
    ).sum(-1).detach()
    source_span = (
        source_native_posed[0, 0].max(dim=0).values
        - source_native_posed[0, 0].min(dim=0).values
    )
    scale = torch.linalg.vector_norm(source_span).clamp_min(1e-5).detach()
    if not bool(torch.isfinite(gap_source).all()) or bool(
        (gap_source[valid] <= 1e-5 * scale).any()
    ):
        raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_NOT_POSITIVE_SOURCE_GAP")

    pred_tri = predicted_posed[0, :, body_faces, :]
    pred_body = (pred_tri * bary[None, :, :, None]).sum(-2)
    pred_garment = predicted_posed[0, :, ids, :]
    gap_pred = ((pred_garment - pred_body) * source_normal).sum(-1)
    if not bool(torch.isfinite(gap_pred).all()):
        raise ValueError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_NONFINITE_STUDENT")

    # Only native-verified pairs/poses. A regression keeps known correct
    # clearance; a barrier prevents the learned garment from collapsing
    # into the body. Neither creates a cloth stiffness or contact material.
    ref = gap_source[valid] / scale
    pred = gap_pred[valid] / scale
    regression = F.smooth_l1_loss(pred, ref)
    barrier = F.relu(0.5*ref - pred).square().mean()
    total = regression + barrier
    if not bool(torch.isfinite(total)):
        raise RuntimeError("RSS_VNEXT_NATIVE_LAYER_CLEARANCE_LOSS_NONFINITE")
    return total, {
        "source_native_signed_gap":ref.detach().mean(),
        "student_signed_gap":pred.detach().mean(),
        "source_gap_regression":regression,
        "separation_barrier":barrier,
    }, int(valid.sum().item())


__all__ = [
    "SCHEMA", "SOURCE_CLASS", "SourceNativeLayerClearanceTeacherV1",
    "native_layer_clearance_objective_v1",
]
