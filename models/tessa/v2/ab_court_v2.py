from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal


VariantId = Literal["DIRECT_IRIS_SURFACE", "TESSA_STRUCTURALIZED"]


@dataclass(frozen=True)
class StructuralizationVariantEvidenceV2:
    """Common downstream evidence for one exact mesh candidate.

    Values in this contract are produced by existing independent Compiler/source
    courts.  This module does not re-run or replace those authorities; it only
    makes the direct-surface vs structuralized comparison impossible to cherry-pick.
    """

    variant_id: VariantId
    candidate_binding_hash: str
    vertex_count: int
    face_count: int

    compiler_static_qualified: bool
    compiler_multiview_qualified: bool
    compiler_dynamic_qualified: bool
    support_binding_qualified: bool

    source_surface_p95_distance: float
    source_surface_max_distance: float
    minimum_multiview_mask_iou: float
    maximum_multiview_depth_abs_error: float
    minimum_multiview_normal_cosine: float

    geometric_component_count: int
    boundary_edge_count: int
    open_boundary_chain_count: int
    nonmanifold_edge_count: int
    zero_area_face_count: int

    g3_maximum_condition_number: float
    g3_minimum_edge_ratio: float
    g3_maximum_edge_ratio: float

    evidence_complete: bool = True
    evidence_provenance: dict[str, str] = field(default_factory=dict)
    schema: str = "RealSaS.StructuralizationVariantEvidence.v2"

    def validate(self) -> None:
        if self.variant_id not in {"DIRECT_IRIS_SURFACE", "TESSA_STRUCTURALIZED"}:
            raise ValueError("TESSA_AB_VARIANT_ID_INVALID")
        if len(self.candidate_binding_hash) != 64:
            raise ValueError("TESSA_AB_CANDIDATE_BINDING_HASH_INVALID")
        if self.vertex_count <= 0 or self.face_count <= 0:
            raise ValueError("TESSA_AB_MESH_COUNT_NONPOSITIVE")
        if self.geometric_component_count <= 0:
            raise ValueError("TESSA_AB_COMPONENT_COUNT_NONPOSITIVE")
        integer_nonnegative = (
            self.boundary_edge_count,
            self.open_boundary_chain_count,
            self.nonmanifold_edge_count,
            self.zero_area_face_count,
        )
        if min(integer_nonnegative) < 0:
            raise ValueError("TESSA_AB_TOPOLOGY_COUNT_NEGATIVE")

        nonnegative = (
            self.source_surface_p95_distance,
            self.source_surface_max_distance,
            self.maximum_multiview_depth_abs_error,
            self.g3_maximum_condition_number,
            self.g3_minimum_edge_ratio,
            self.g3_maximum_edge_ratio,
        )
        if any(not math.isfinite(float(x)) or float(x) < 0.0 for x in nonnegative):
            raise ValueError("TESSA_AB_NONNEGATIVE_METRIC_INVALID")
        if self.source_surface_p95_distance > self.source_surface_max_distance:
            raise ValueError("TESSA_AB_SURFACE_DISTANCE_ORDER_INVALID")
        if not (0.0 <= float(self.minimum_multiview_mask_iou) <= 1.0):
            raise ValueError("TESSA_AB_MASK_IOU_INVALID")
        if not (-1.0 <= float(self.minimum_multiview_normal_cosine) <= 1.0):
            raise ValueError("TESSA_AB_NORMAL_COSINE_INVALID")
        if self.g3_minimum_edge_ratio > self.g3_maximum_edge_ratio:
            raise ValueError("TESSA_AB_G3_EDGE_RATIO_ORDER_INVALID")
        if not self.evidence_complete:
            raise ValueError("TESSA_AB_VARIANT_EVIDENCE_INCOMPLETE")
        required_provenance = {
            "candidate",
            "static_qualification",
            "multiview_qualification",
            "dynamic_qualification",
            "source_fidelity",
        }
        if set(self.evidence_provenance) != required_provenance:
            raise ValueError("TESSA_AB_EVIDENCE_PROVENANCE_INCOMPLETE")
        if any(len(str(value)) != 64 for value in self.evidence_provenance.values()):
            raise ValueError("TESSA_AB_EVIDENCE_PROVENANCE_HASH_INVALID")


@dataclass(frozen=True)
class StructuralizationABPolicyV2:
    """Preregistered policy for deciding whether TESSA earns a mandatory role."""

    minimum_face_reduction_fraction: float = 0.25
    maximum_relative_surface_p95_increase: float = 0.05
    maximum_relative_surface_max_increase: float = 0.05
    maximum_mask_iou_drop: float = 0.01
    maximum_depth_error_increase_fraction: float = 0.05
    maximum_normal_cosine_drop: float = 0.01
    maximum_condition_number_increase_fraction: float = 0.05
    maximum_edge_ratio_envelope_expansion_fraction: float = 0.05
    require_no_new_nonmanifold_edges: bool = True
    require_no_new_zero_area_faces: bool = True
    require_no_new_open_boundary_chains: bool = True
    require_same_component_count: bool = True

    def validate(self) -> None:
        bounded = (
            self.minimum_face_reduction_fraction,
            self.maximum_relative_surface_p95_increase,
            self.maximum_relative_surface_max_increase,
            self.maximum_mask_iou_drop,
            self.maximum_depth_error_increase_fraction,
            self.maximum_normal_cosine_drop,
            self.maximum_condition_number_increase_fraction,
            self.maximum_edge_ratio_envelope_expansion_fraction,
        )
        if any(not math.isfinite(float(x)) or not (0.0 <= float(x) <= 1.0) for x in bounded):
            raise ValueError("TESSA_AB_POLICY_RANGE_INVALID")


@dataclass(frozen=True)
class StructuralizationABReceiptV2:
    status: Literal[
        "TESSA_PROMOTABLE",
        "DIRECT_SURFACE_PREFERRED",
        "INCONCLUSIVE",
    ]
    direct: StructuralizationVariantEvidenceV2
    tessa: StructuralizationVariantEvidenceV2
    policy: StructuralizationABPolicyV2
    checks: dict[str, bool]
    deltas: dict[str, float]
    blockers: tuple[str, ...]
    tessa_mandatory_authorized: bool
    training_authorized: bool = False
    scientific_pass_claimed: bool = False
    schema: str = "RealSaS.StructuralizationABCourt.v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_increase(new: float, base: float) -> float:
    if base <= 1.0e-12:
        return 0.0 if new <= 1.0e-12 else float("inf")
    return (new - base) / base


def evaluate_structuralization_ab_v2(
    direct: StructuralizationVariantEvidenceV2,
    tessa: StructuralizationVariantEvidenceV2,
    *,
    policy: StructuralizationABPolicyV2 = StructuralizationABPolicyV2(),
) -> StructuralizationABReceiptV2:
    """Compare exact downstream courts; never use training loss as promotion evidence."""

    policy.validate()
    direct.validate()
    tessa.validate()
    if direct.variant_id != "DIRECT_IRIS_SURFACE":
        raise ValueError("TESSA_AB_DIRECT_VARIANT_ID_REQUIRED")
    if tessa.variant_id != "TESSA_STRUCTURALIZED":
        raise ValueError("TESSA_AB_TESSA_VARIANT_ID_REQUIRED")

    direct_all_qualified = all(
        (
            direct.compiler_static_qualified,
            direct.compiler_multiview_qualified,
            direct.compiler_dynamic_qualified,
            direct.support_binding_qualified,
        )
    )
    tessa_all_qualified = all(
        (
            tessa.compiler_static_qualified,
            tessa.compiler_multiview_qualified,
            tessa.compiler_dynamic_qualified,
            tessa.support_binding_qualified,
        )
    )

    face_reduction = 1.0 - float(tessa.face_count) / float(direct.face_count)
    p95_increase = _relative_increase(
        tessa.source_surface_p95_distance,
        direct.source_surface_p95_distance,
    )
    max_increase = _relative_increase(
        tessa.source_surface_max_distance,
        direct.source_surface_max_distance,
    )
    depth_increase = _relative_increase(
        tessa.maximum_multiview_depth_abs_error,
        direct.maximum_multiview_depth_abs_error,
    )
    condition_increase = _relative_increase(
        tessa.g3_maximum_condition_number,
        direct.g3_maximum_condition_number,
    )
    direct_edge_width = max(
        abs(1.0 - direct.g3_minimum_edge_ratio),
        abs(direct.g3_maximum_edge_ratio - 1.0),
    )
    tessa_edge_width = max(
        abs(1.0 - tessa.g3_minimum_edge_ratio),
        abs(tessa.g3_maximum_edge_ratio - 1.0),
    )
    edge_envelope_increase = _relative_increase(tessa_edge_width, direct_edge_width)
    mask_drop = direct.minimum_multiview_mask_iou - tessa.minimum_multiview_mask_iou
    normal_drop = direct.minimum_multiview_normal_cosine - tessa.minimum_multiview_normal_cosine

    checks = {
        "direct_all_compiler_courts_pass": direct_all_qualified,
        "tessa_all_compiler_courts_pass": tessa_all_qualified,
        "face_reduction_sufficient": face_reduction >= policy.minimum_face_reduction_fraction,
        "surface_p95_preserved": p95_increase <= policy.maximum_relative_surface_p95_increase,
        "surface_max_preserved": max_increase <= policy.maximum_relative_surface_max_increase,
        "multiview_mask_preserved": mask_drop <= policy.maximum_mask_iou_drop,
        "multiview_depth_preserved": depth_increase <= policy.maximum_depth_error_increase_fraction,
        "multiview_normal_preserved": normal_drop <= policy.maximum_normal_cosine_drop,
        "g3_conditioning_preserved": condition_increase <= policy.maximum_condition_number_increase_fraction,
        "g3_edge_envelope_preserved": edge_envelope_increase
        <= policy.maximum_edge_ratio_envelope_expansion_fraction,
        "component_count_preserved": (
            not policy.require_same_component_count
            or tessa.geometric_component_count == direct.geometric_component_count
        ),
        "no_new_nonmanifold_edges": (
            not policy.require_no_new_nonmanifold_edges
            or tessa.nonmanifold_edge_count <= direct.nonmanifold_edge_count
        ),
        "no_new_zero_area_faces": (
            not policy.require_no_new_zero_area_faces
            or tessa.zero_area_face_count <= direct.zero_area_face_count
        ),
        "no_new_open_boundary_chains": (
            not policy.require_no_new_open_boundary_chains
            or tessa.open_boundary_chain_count <= direct.open_boundary_chain_count
        ),
    }
    deltas = {
        "face_reduction_fraction": face_reduction,
        "surface_p95_relative_increase": p95_increase,
        "surface_max_relative_increase": max_increase,
        "mask_iou_drop": mask_drop,
        "depth_error_relative_increase": depth_increase,
        "normal_cosine_drop": normal_drop,
        "g3_condition_number_relative_increase": condition_increase,
        "g3_edge_envelope_relative_increase": edge_envelope_increase,
    }

    blockers = tuple(name for name, passed in checks.items() if not passed)
    tessa_promotable = not blockers

    if tessa_promotable:
        status: Literal["TESSA_PROMOTABLE", "DIRECT_SURFACE_PREFERRED", "INCONCLUSIVE"] = (
            "TESSA_PROMOTABLE"
        )
    elif direct_all_qualified and not tessa_all_qualified:
        status = "DIRECT_SURFACE_PREFERRED"
    elif direct_all_qualified and tessa_all_qualified and not checks["face_reduction_sufficient"]:
        # If TESSA adds no structural economy while both paths qualify, there is
        # no demonstrated reason to make a second learned layer mandatory.
        status = "DIRECT_SURFACE_PREFERRED"
    else:
        status = "INCONCLUSIVE"

    return StructuralizationABReceiptV2(
        status=status,
        direct=direct,
        tessa=tessa,
        policy=policy,
        checks=checks,
        deltas=deltas,
        blockers=blockers,
        tessa_mandatory_authorized=bool(tessa_promotable),
        training_authorized=False,
        scientific_pass_claimed=False,
    )
