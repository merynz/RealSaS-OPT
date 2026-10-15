from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "RealSaS.FScalarAdmission.v1"

GLOBAL_METRIC_SIGN = "GLOBAL_METRIC_SIGN"
LOCAL_ORIENTED_SHEET_SIGN = "LOCAL_ORIENTED_SHEET_SIGN"
UNSIGNED_ONLY = "UNSIGNED_ONLY"
ABSTAIN = "ABSTAIN"
SIGN_CLASSES = {
    GLOBAL_METRIC_SIGN,
    LOCAL_ORIENTED_SHEET_SIGN,
    UNSIGNED_ONLY,
    ABSTAIN,
}

F_A = "F_A"
F_B = "F_B"
F_R = "F_R"
F_C = "F_C"
ARMS = {F_A, F_B, F_R, F_C}


@dataclass(frozen=True)
class FScalarEvidenceV1:
    sign_class: str
    cell_xyz: tuple[int, int, int]
    local_sheet_id: int | None
    orientation_provenance: str
    validity_artifact_sha256: str
    source_component_sign_authority: bool

    def validate(self) -> None:
        if self.sign_class not in SIGN_CLASSES:
            raise ValueError("F_SCALAR_ADMISSION_SIGN_CLASS_INVALID")
        if len(self.cell_xyz) != 3:
            raise ValueError("F_SCALAR_ADMISSION_CELL_INVALID")
        if len(self.validity_artifact_sha256) != 64:
            raise ValueError("F_SCALAR_ADMISSION_VALIDITY_SHA_INVALID")
        if not self.orientation_provenance:
            raise ValueError("F_SCALAR_ADMISSION_PROVENANCE_MISSING")
        if self.sign_class == GLOBAL_METRIC_SIGN:
            if not self.source_component_sign_authority:
                raise ValueError("F_SCALAR_ADMISSION_GLOBAL_SIGN_WITHOUT_AUTHORITY")
            if self.local_sheet_id is not None:
                raise ValueError("F_SCALAR_ADMISSION_GLOBAL_SIGN_MUST_NOT_REQUIRE_LOCAL_SHEET")
        elif self.sign_class == LOCAL_ORIENTED_SHEET_SIGN:
            if self.local_sheet_id is None or self.local_sheet_id < 0:
                raise ValueError("F_SCALAR_ADMISSION_LOCAL_SIGN_REQUIRES_SHEET")
        elif self.sign_class in {UNSIGNED_ONLY, ABSTAIN}:
            if self.local_sheet_id is not None and self.local_sheet_id < 0:
                raise ValueError("F_SCALAR_ADMISSION_LOCAL_SHEET_INVALID")


@dataclass(frozen=True)
class FScalarAdmissionDecisionV1:
    arm: str
    admitted: bool
    reason: str


def admit_scalar_evidence_for_arm_v1(
    arm: str,
    evidence: FScalarEvidenceV1,
) -> FScalarAdmissionDecisionV1:
    """Fail-closed admission of field evidence to an extraction arm.

    F-A models a vanilla single signed scalar field such as ordinary
    DMC/FlexiCubes occupancy. It therefore requires globally qualified sign.

    F-B/F-R may additionally consume a local oriented level-set only when the
    evidence is explicitly conditioned on one qualified cell-supported sheet.
    They must not reinterpret an unsigned field as signed truth.

    F-C is the explicit upper-capacity topology reference and can be driven from
    qualified crossing/strand/incidence truth without scalar sign. `ABSTAIN`
    still remains abstention rather than fabricated evidence.
    """
    if arm not in ARMS:
        raise ValueError("F_SCALAR_ADMISSION_ARM_INVALID")
    evidence.validate()

    if evidence.sign_class == ABSTAIN:
        return FScalarAdmissionDecisionV1(arm, False, "EVIDENCE_ABSTAINS")

    if arm == F_A:
        if evidence.sign_class != GLOBAL_METRIC_SIGN:
            return FScalarAdmissionDecisionV1(
                arm,
                False,
                "F_A_REQUIRES_GLOBAL_METRIC_SIGN",
            )
        return FScalarAdmissionDecisionV1(arm, True, "GLOBAL_SIGN_QUALIFIED")

    if arm in {F_B, F_R}:
        if evidence.sign_class == GLOBAL_METRIC_SIGN:
            return FScalarAdmissionDecisionV1(arm, True, "GLOBAL_SIGN_QUALIFIED")
        if evidence.sign_class == LOCAL_ORIENTED_SHEET_SIGN:
            return FScalarAdmissionDecisionV1(arm, True, "LOCAL_SHEET_SIGN_QUALIFIED")
        return FScalarAdmissionDecisionV1(
            arm,
            False,
            "F_B_F_R_REQUIRE_GLOBAL_OR_LOCAL_SHEET_SIGN",
        )

    # F-C can use explicit qualified topology evidence without a scalar sign.
    if evidence.sign_class in {
        GLOBAL_METRIC_SIGN,
        LOCAL_ORIENTED_SHEET_SIGN,
        UNSIGNED_ONLY,
    }:
        return FScalarAdmissionDecisionV1(arm, True, "EXPLICIT_TOPOLOGY_REFERENCE")
    raise AssertionError("unreachable sign class")


__all__ = [
    "ABSTAIN",
    "ARMS",
    "F_A",
    "F_B",
    "F_C",
    "F_R",
    "FScalarAdmissionDecisionV1",
    "FScalarEvidenceV1",
    "GLOBAL_METRIC_SIGN",
    "LOCAL_ORIENTED_SHEET_SIGN",
    "SCHEMA_VERSION",
    "SIGN_CLASSES",
    "UNSIGNED_ONLY",
    "admit_scalar_evidence_for_arm_v1",
]
