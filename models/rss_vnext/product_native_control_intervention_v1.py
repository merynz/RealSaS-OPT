from __future__ import annotations

"""Source-native *control intervention* consequence for one TESSA M*.

This is not a learned material, attachment, bone-name class, or solver.
A trusted source-authored ONE-CONTROL pose pair must show which ORIGINAL
joints changed, which final-M* vertices responded, and which were unaffected.
We train the DIFFERENCE between two student AXIS/MIRA posed surfaces, so
global rest-position fit alone cannot substitute for functional isolation.
"""
from dataclasses import dataclass
from typing import Mapping
import math

import torch
from torch import Tensor
from torch.nn import functional as F

SCHEMA = "RealSaS.RSSvNext.NativeControlInterventionResponse.v1"
SCOPES = frozenset({"SOURCE_HOST_ONLY", "SOURCE_BRANCH_ONLY"})


def _sha(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class SourceNativeControlInterventionTeacherV1:
    source_asset_sha256: str
    source_qualification_key: str
    source_control_event_sha256: str
    source_scope: str
    baseline_pose_index: int
    intervention_pose_index: int
    changed_skin_joint_matrix_mask: Tensor  # [J], measured from source matrices
    responding_vertex_ids: Tensor     # [K], final TESSA M* addresses
    protected_vertex_ids: Tensor      # [Q], also final M*, disjoint from K

    def validate(
        self, *, source_posed: Tensor, source_pose_mask: Tensor,
        source_pose_matrices: Tensor,
    ) -> None:
        if (
            not _sha(self.source_asset_sha256)
            or not self.source_qualification_key
            or not _sha(self.source_control_event_sha256)
            or self.source_scope not in SCOPES
            or type(self.baseline_pose_index) is not int
            or type(self.intervention_pose_index) is not int
            or self.baseline_pose_index < 0
            or self.intervention_pose_index <= self.baseline_pose_index
            or source_posed.ndim != 4 or source_posed.shape[0] != 1
            or source_posed.shape[-1] != 3
            or source_pose_matrices.ndim != 5
            or source_pose_matrices.shape[0] != 1
            or source_pose_matrices.shape[-2:] != (4, 4)
        ):
            raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_SOURCE_DOMAIN")
        p, v = source_posed.shape[1:3]
        j = source_pose_matrices.shape[2]
        if (
            self.intervention_pose_index >= p
            or source_pose_matrices.shape[:2] != (1, p)
            or source_pose_mask.shape != (1, p, v)
            or source_pose_mask.dtype != torch.bool
            or self.changed_skin_joint_matrix_mask.shape != (j,)
            or self.changed_skin_joint_matrix_mask.dtype != torch.bool
            or self.responding_vertex_ids.ndim != 1
            or self.protected_vertex_ids.ndim != 1
            or self.responding_vertex_ids.dtype != torch.long
            or self.protected_vertex_ids.dtype != torch.long
            or not 1 <= self.responding_vertex_ids.numel() <= 65536
            or not 1 <= self.protected_vertex_ids.numel() <= 65536
            or self.responding_vertex_ids.numel()
               + self.protected_vertex_ids.numel() > 65536
            or any(t.device != source_posed.device for t in (
                source_pose_mask, source_pose_matrices,
                self.changed_skin_joint_matrix_mask, self.responding_vertex_ids,
                self.protected_vertex_ids,
            ))
            or not bool(torch.isfinite(source_posed).all())
            or not bool(torch.isfinite(source_pose_matrices).all())
        ):
            raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_SOURCE_SHAPE")
        ids = torch.cat(
            (self.responding_vertex_ids, self.protected_vertex_ids)
        )
        if (
            bool((ids < 0).any()) or bool((ids >= v).any())
            or torch.unique(ids).numel() != ids.numel()
            or not bool(source_pose_mask[0, self.baseline_pose_index, ids].all())
            or not bool(source_pose_mask[0, self.intervention_pose_index, ids].all())
        ):
            raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_VERTEX_SUPPORT")
        a = source_pose_matrices[0, self.baseline_pose_index]
        b = source_pose_matrices[0, self.intervention_pose_index]
        changed = (b - a).abs().amax(dim=(-1, -2)) > 1e-5
        if (
            not bool(changed.any())
            or not torch.equal(changed, self.changed_skin_joint_matrix_mask)
        ):
            raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_JOINT_WITNESS")
        source_delta = (
            source_posed[0, self.intervention_pose_index]
            - source_posed[0, self.baseline_pose_index]
        )
        span = (
            source_posed[0, self.baseline_pose_index].amax(dim=0)
            - source_posed[0, self.baseline_pose_index].amin(dim=0)
        )
        scale = torch.linalg.vector_norm(span)
        if not bool(torch.isfinite(scale)) or float(scale) <= 1e-6:
            raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_REST_SCALE")
        driven_motion = (
            torch.linalg.vector_norm(
                source_delta[self.responding_vertex_ids], dim=-1
            ) / scale
        )
        protected_motion = (
            torch.linalg.vector_norm(
                source_delta[self.protected_vertex_ids], dim=-1
            ) / scale
        )
        # Qualified source responders must move; source-protected anchors
        # must stay still in the SOURCE; never use student predictions here.
        if (
            not bool((driven_motion > 1e-4).all())
            or not bool((protected_motion < 1e-4).all())
        ):
            raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_SOURCE_RESPONSE_NOT_VERIFIED")


def qualified_native_control_intervention_loss_v1(
    *, predicted_posed: Tensor | None,
    source_posed: Tensor | None, source_pose_mask: Tensor | None,
    source_pose_matrices: Tensor | None,
    teacher: SourceNativeControlInterventionTeacherV1 | None,
    admission_key: str, trusted_registry: Mapping[str, Mapping[str, object]],
    dataset_manifest_sha256: str,
    mstar_address_sha256: str, native_pose_transport_sha256: str,
    source_asset_sha256: str, source_qualification_key: str,
    observed_rgba_sha256: str, generation_checkpoint_sha256: str,
    tensor_digest,
) -> tuple[Tensor | None, dict[str, Tensor] | None, int]:
    """Only a source-author-written held-out control intervention is a label.

    A single mesh, identical vertex addresses in before/after positions.
    The original CONTROL event is qualified externally. Matrix differences
    alone do NOT establish artist intent; registry requires both witnesses.
    """
    if teacher is None and not admission_key:
        return None, None, 0
    if (
        not isinstance(teacher, SourceNativeControlInterventionTeacherV1)
        or not isinstance(predicted_posed, Tensor)
        or not isinstance(source_posed, Tensor)
        or not isinstance(source_pose_mask, Tensor)
        or not isinstance(source_pose_matrices, Tensor)
        or not isinstance(trusted_registry, Mapping)
        or not admission_key
        or not all(_sha(v) for v in (
            dataset_manifest_sha256, mstar_address_sha256,
            native_pose_transport_sha256, source_asset_sha256,
            observed_rgba_sha256, generation_checkpoint_sha256,
        ))
        or source_asset_sha256 != teacher.source_asset_sha256
        or source_qualification_key != teacher.source_qualification_key
        or predicted_posed.shape != source_posed.shape
        or predicted_posed.device != source_posed.device
        or not bool(torch.isfinite(predicted_posed).all())
    ):
        raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_NATIVE_POSE_REQUIRED")
    teacher.validate(
        source_posed=source_posed, source_pose_mask=source_pose_mask,
        source_pose_matrices=source_pose_matrices,
    )
    expected = {
        "schema": SCHEMA,
        "admission_status": "QUALIFIED_SOURCE_NATIVE_ONE_CONTROL_INTERVENTION",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_asset_sha256": source_asset_sha256,
        "source_qualification_key": source_qualification_key,
        "observed_rgba_sha256": observed_rgba_sha256,
        "generation_checkpoint_sha256": generation_checkpoint_sha256,
        "candidate_mstar_topology_address_sha256": mstar_address_sha256,
        "native_pose_transport_receipt_sha256": native_pose_transport_sha256,
        "source_control_event_sha256": teacher.source_control_event_sha256,
        "source_scope": teacher.source_scope,
        "baseline_pose_index": teacher.baseline_pose_index,
        "intervention_pose_index": teacher.intervention_pose_index,
        "source_posed_sha256": tensor_digest(source_posed),
        "source_pose_mask_sha256": tensor_digest(source_pose_mask),
        "source_pose_matrices_sha256": tensor_digest(source_pose_matrices),
        "changed_skin_joint_matrix_mask_sha256":
            tensor_digest(teacher.changed_skin_joint_matrix_mask),
        "responding_vertex_ids_sha256":
            tensor_digest(teacher.responding_vertex_ids),
        "protected_vertex_ids_sha256":
            tensor_digest(teacher.protected_vertex_ids),
        "source_authored_exactly_one_control_changed": True,
        "source_response_observed_independently_of_student": True,
        "source_to_final_mstar_face_and_joint_transport_verified": True,
        "protected_not_a_negative_attachment_label": True,
        "student_generated_views_not_source_truth": True,
        "physical_material_or_xpbd_authority": False,
    }
    row = trusted_registry.get(admission_key)
    if row is None or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("RSS_VNEXT_CONTROL_INTERVENTION_RECEIPT_DRIFT")
    start = teacher.baseline_pose_index
    stop = teacher.intervention_pose_index
    scale = torch.linalg.vector_norm(
        source_posed.detach()[0, start].amax(0)
        - source_posed.detach()[0, start].amin(0)
    ).clamp_min(1e-5)
    source_delta = (
        source_posed.detach()[0, stop]
        - source_posed.detach()[0, start]
    ) / scale
    pred_delta = (
        predicted_posed[0, stop] - predicted_posed[0, start]
    ) / scale
    driven_ids = teacher.responding_vertex_ids
    protected_ids = teacher.protected_vertex_ids
    actual_driven = pred_delta[driven_ids]
    actual_protected = pred_delta[protected_ids]
    source_driven = source_delta[driven_ids]
    source_protected = source_delta[protected_ids]
    driven = F.smooth_l1_loss(actual_driven, source_driven)
    isolation = F.smooth_l1_loss(actual_protected, source_protected)
    # Beyond absolute pose error, explicitly contrast the moving chain's
    # transfer against its inactive counterpart; no invented rigid pin.
    contrast = F.smooth_l1_loss(
        actual_driven.mean(0) - actual_protected.mean(0),
        source_driven.mean(0) - source_protected.mean(0),
    )
    # Source protected vertices are measured static; this extra small,
    # explicit penalty prevents the shared skin from bleeding movement
    # into the inactive surface while preserving authored source deltas.
    bleed = (actual_protected.square()).mean()
    loss = driven + isolation + contrast + .25*bleed
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("RSS_VNEXT_CONTROL_INTERVENTION_LOSS_NONFINITE")
    return loss, {
        "responding_delta_error": driven,
        "protected_delta_error": isolation,
        "relative_response_contrast": contrast,
        "protected_motion_leakage": bleed,
    }, int(driven_ids.numel() + protected_ids.numel())


__all__ = [
    "SCHEMA", "SCOPES", "SourceNativeControlInterventionTeacherV1",
    "qualified_native_control_intervention_loss_v1",
]
