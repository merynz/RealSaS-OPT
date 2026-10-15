from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

SCHEMA_VERSION = "RealSaS.FMatchedPhysicalCourt.v1"
ARMS = ("FAN_BASELINE", "F_A", "F_B", "F_R", "F_C")


@dataclass(frozen=True)
class FMatchedPhysicalPolicyV1:
    max_symmetric_surface_p95_pitch: float
    max_normal_p95_deg: float
    max_depth_p95_pitch: float
    min_silhouette_iou: float
    min_thin_structure_recall: float
    max_self_intersections: int
    max_degenerate_faces: int
    require_exact_topology: bool
    require_zero_false_bridges: bool
    require_zero_missing_sheets: bool
    require_zero_residual_false_negatives: bool
    provenance: str

    def validate(self) -> None:
        if self.max_symmetric_surface_p95_pitch < 0.0:
            raise ValueError("F_MATCHED_PHYSICAL_SURFACE_THRESHOLD_INVALID")
        if self.max_normal_p95_deg < 0.0 or self.max_normal_p95_deg > 180.0:
            raise ValueError("F_MATCHED_PHYSICAL_NORMAL_THRESHOLD_INVALID")
        if self.max_depth_p95_pitch < 0.0:
            raise ValueError("F_MATCHED_PHYSICAL_DEPTH_THRESHOLD_INVALID")
        if not (0.0 <= self.min_silhouette_iou <= 1.0):
            raise ValueError("F_MATCHED_PHYSICAL_SILHOUETTE_THRESHOLD_INVALID")
        if not (0.0 <= self.min_thin_structure_recall <= 1.0):
            raise ValueError("F_MATCHED_PHYSICAL_THIN_THRESHOLD_INVALID")
        if self.max_self_intersections < 0 or self.max_degenerate_faces < 0:
            raise ValueError("F_MATCHED_PHYSICAL_COUNT_THRESHOLD_INVALID")
        if not self.provenance:
            raise ValueError("F_MATCHED_PHYSICAL_POLICY_PROVENANCE_MISSING")


@dataclass(frozen=True)
class FPhysicalCandidateMeasurementV1:
    subject_key: str
    arm: str
    teacher_bundle_sha256: str
    extraction_truth_sha256: str
    canonical_camera_contract_sha256: str
    extraction_scale_manifest_sha256: str
    measurement_protocol_sha256: str
    exact_topology: bool
    symmetric_surface_p95_pitch: float
    normal_p95_deg: float
    depth_p95_pitch: float
    silhouette_iou: float
    false_bridge_count: int
    missing_sheet_count: int
    self_intersection_count: int
    degenerate_face_count: int
    thin_structure_recall: float
    residual_required_entity_count: int
    residual_activated_entity_count: int
    residual_false_negative_count: int
    explicit_variable_payload_count: int
    candidate_vertex_count: int
    candidate_face_count: int
    peak_working_bytes: int

    def validate(self) -> None:
        if not self.subject_key:
            raise ValueError("F_MATCHED_PHYSICAL_SUBJECT_MISSING")
        if self.arm not in ARMS:
            raise ValueError("F_MATCHED_PHYSICAL_ARM_INVALID")
        for name in (
            "teacher_bundle_sha256",
            "extraction_truth_sha256",
            "canonical_camera_contract_sha256",
            "extraction_scale_manifest_sha256",
            "measurement_protocol_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"F_MATCHED_PHYSICAL_SHA_INVALID:{name}")
        for name in (
            "symmetric_surface_p95_pitch",
            "normal_p95_deg",
            "depth_p95_pitch",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"F_MATCHED_PHYSICAL_NEGATIVE_METRIC:{name}")
        for name in ("silhouette_iou", "thin_structure_recall"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"F_MATCHED_PHYSICAL_UNIT_METRIC_INVALID:{name}")
        for name in (
            "false_bridge_count",
            "missing_sheet_count",
            "self_intersection_count",
            "degenerate_face_count",
            "residual_required_entity_count",
            "residual_activated_entity_count",
            "residual_false_negative_count",
            "explicit_variable_payload_count",
            "candidate_vertex_count",
            "candidate_face_count",
            "peak_working_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_MATCHED_PHYSICAL_NEGATIVE_COUNT:{name}")
        if self.residual_activated_entity_count > self.residual_required_entity_count:
            raise ValueError("F_MATCHED_PHYSICAL_RESIDUAL_ACTIVATION_OVERCLAIM")
        expected_false_negative = (
            self.residual_required_entity_count - self.residual_activated_entity_count
        )
        if self.residual_false_negative_count != expected_false_negative:
            raise ValueError("F_MATCHED_PHYSICAL_RESIDUAL_FALSE_NEGATIVE_MISMATCH")


@dataclass(frozen=True)
class FPhysicalArmDecisionV1:
    arm: str
    admitted: bool
    failure_reasons: tuple[str, ...]
    explicit_variable_payload_count: int
    peak_working_bytes: int
    candidate_face_count: int


@dataclass(frozen=True)
class FMatchedPhysicalCourtReportV1:
    schema_version: str
    subject_key: str
    decisions: tuple[FPhysicalArmDecisionV1, ...]
    admitted_arms: tuple[str, ...]
    lowest_payload_admitted_arm: str | None

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_MATCHED_PHYSICAL_REPORT_SCHEMA_INVALID")
        if not self.subject_key:
            raise ValueError("F_MATCHED_PHYSICAL_REPORT_SUBJECT_MISSING")
        arms = [row.arm for row in self.decisions]
        if len(arms) != len(set(arms)):
            raise ValueError("F_MATCHED_PHYSICAL_REPORT_DUPLICATE_ARM")
        admitted = tuple(row.arm for row in self.decisions if row.admitted)
        if admitted != self.admitted_arms:
            raise ValueError("F_MATCHED_PHYSICAL_REPORT_ADMITTED_MISMATCH")
        if self.lowest_payload_admitted_arm is not None and self.lowest_payload_admitted_arm not in admitted:
            raise ValueError("F_MATCHED_PHYSICAL_REPORT_LOWEST_PAYLOAD_NOT_ADMITTED")


def _matched_authority_key(row: FPhysicalCandidateMeasurementV1) -> tuple[str, ...]:
    return (
        row.subject_key,
        row.teacher_bundle_sha256,
        row.extraction_truth_sha256,
        row.canonical_camera_contract_sha256,
        row.extraction_scale_manifest_sha256,
        row.measurement_protocol_sha256,
    )


def _failure_reasons(
    row: FPhysicalCandidateMeasurementV1,
    policy: FMatchedPhysicalPolicyV1,
) -> tuple[str, ...]:
    failures: list[str] = []
    if policy.require_exact_topology and not row.exact_topology:
        failures.append("TOPOLOGY_NOT_EXACT")
    if row.symmetric_surface_p95_pitch > policy.max_symmetric_surface_p95_pitch:
        failures.append("SURFACE_ERROR")
    if row.normal_p95_deg > policy.max_normal_p95_deg:
        failures.append("NORMAL_ERROR")
    if row.depth_p95_pitch > policy.max_depth_p95_pitch:
        failures.append("DEPTH_ERROR")
    if row.silhouette_iou < policy.min_silhouette_iou:
        failures.append("SILHOUETTE_ERROR")
    if row.thin_structure_recall < policy.min_thin_structure_recall:
        failures.append("THIN_STRUCTURE_LOSS")
    if policy.require_zero_false_bridges and row.false_bridge_count:
        failures.append("FALSE_BRIDGE")
    if policy.require_zero_missing_sheets and row.missing_sheet_count:
        failures.append("MISSING_SHEET")
    if row.self_intersection_count > policy.max_self_intersections:
        failures.append("SELF_INTERSECTION")
    if row.degenerate_face_count > policy.max_degenerate_faces:
        failures.append("DEGENERATE_FACE")
    if policy.require_zero_residual_false_negatives and row.residual_false_negative_count:
        failures.append("RESIDUAL_FALSE_NEGATIVE")
    return tuple(failures)


def evaluate_f_matched_physical_court_v1(
    measurements: Sequence[FPhysicalCandidateMeasurementV1],
    *,
    policy: FMatchedPhysicalPolicyV1,
) -> FMatchedPhysicalCourtReportV1:
    """Evaluate matched F arms without allowing complexity to buy a false PASS.

    Every arm must use the exact same subject, teacher bundle, extraction truth,
    canonical cameras, scale manifest and measurement protocol. Quality gates are
    preregistered through `policy`; this module intentionally does not invent
    numerical thresholds.

    Selection is lexicographic only *after* admission: an arm with lower explicit
    payload, fewer faces or lower memory cannot outrank an arm that fails topology,
    geometry, visible projection, thin-structure, or residual-recall requirements.
    """
    policy.validate()
    if not measurements:
        raise ValueError("F_MATCHED_PHYSICAL_MEASUREMENTS_EMPTY")
    rows = tuple(measurements)
    for row in rows:
        row.validate()
    authority_keys = {_matched_authority_key(row) for row in rows}
    if len(authority_keys) != 1:
        raise ValueError("F_MATCHED_PHYSICAL_AUTHORITY_NOT_MATCHED")
    arms = [row.arm for row in rows]
    if len(arms) != len(set(arms)):
        raise ValueError("F_MATCHED_PHYSICAL_DUPLICATE_ARM")

    decisions: list[FPhysicalArmDecisionV1] = []
    for row in rows:
        failures = _failure_reasons(row, policy)
        decisions.append(
            FPhysicalArmDecisionV1(
                arm=row.arm,
                admitted=not failures,
                failure_reasons=failures,
                explicit_variable_payload_count=row.explicit_variable_payload_count,
                peak_working_bytes=row.peak_working_bytes,
                candidate_face_count=row.candidate_face_count,
            )
        )

    admitted_rows = [row for row in decisions if row.admitted]
    admitted_arms = tuple(row.arm for row in decisions if row.admitted)
    lowest_payload = None
    if admitted_rows:
        lowest_payload = min(
            admitted_rows,
            key=lambda row: (
                row.explicit_variable_payload_count,
                row.peak_working_bytes,
                row.candidate_face_count,
                ARMS.index(row.arm),
            ),
        ).arm

    report = FMatchedPhysicalCourtReportV1(
        schema_version=SCHEMA_VERSION,
        subject_key=rows[0].subject_key,
        decisions=tuple(decisions),
        admitted_arms=admitted_arms,
        lowest_payload_admitted_arm=lowest_payload,
    )
    report.validate()
    return report


__all__ = [
    "ARMS",
    "FMatchedPhysicalCourtReportV1",
    "FMatchedPhysicalPolicyV1",
    "FPhysicalArmDecisionV1",
    "FPhysicalCandidateMeasurementV1",
    "SCHEMA_VERSION",
    "evaluate_f_matched_physical_court_v1",
]
