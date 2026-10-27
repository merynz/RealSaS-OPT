from __future__ import annotations

"""Fail-closed availability court for Motion Truth Foundry families.

A kinematic parameter that can be perturbed is not evidence that the same axis is
an authored physical degree of freedom.  Likewise, the presence of a source clip
reference is not a measured temporal response.  This module keeps those domains
separate before any IRIS/TESSA target or Compiler consumer is allowed to bind.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MotionTruthFamilyAvailabilityV1:
    family: str
    status: str
    reason: str
    teacher_measured: bool
    student_target_allowed: bool
    compiler_consequence_allowed: bool
    schema_version: str = "RealSaS.MotionTruthFamilyAvailability.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionTruthAvailabilityReportV1:
    subject_key: str
    kinematic_response_diagnostic: MotionTruthFamilyAvailabilityV1
    admissible_motion_subspace: MotionTruthFamilyAvailabilityV1
    physical_impulse_response: MotionTruthFamilyAvailabilityV1
    temporal_response: MotionTruthFamilyAvailabilityV1
    secondary_motion_consumer: MotionTruthFamilyAvailabilityV1
    schema_version: str = "RealSaS.MotionTruthAvailabilityReport.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _family(
    family: str,
    status: str,
    reason: str,
    *,
    measured: bool = False,
    student: bool = False,
    consequence: bool = False,
) -> MotionTruthFamilyAvailabilityV1:
    return MotionTruthFamilyAvailabilityV1(
        family=family,
        status=status,
        reason=reason,
        teacher_measured=measured,
        student_target_allowed=student,
        compiler_consequence_allowed=consequence,
    )


def evaluate_motion_truth_availability_v1(
    teacher_scene: Mapping[str, Any],
) -> MotionTruthAvailabilityReportV1:
    """Classify only what the current teacher package actually measured.

    This is intentionally stricter than capability discovery.  A family becomes
    a model target or Compiler authority only after the package contains that
    family's measured response and its own later qualification/observability
    gates.  This function never mints such qualification.
    """
    if str(teacher_scene.get("schema") or "") != "RealSaS.TeacherSceneIR.v1":
        raise ValueError("MOTION_TRUTH_TEACHER_SCENE_SCHEMA_INVALID")
    subject = str(teacher_scene.get("subject_key") or "")
    if not subject:
        raise ValueError("MOTION_TRUTH_SUBJECT_REQUIRED")
    if teacher_scene.get("qualification_minted") is not False:
        raise ValueError("MOTION_TRUTH_INPUT_AUTHORITY_BOUNDARY_INVALID")
    if teacher_scene.get("student_input_allowed") is not False:
        raise ValueError("MOTION_TRUTH_PRIVILEGED_TEACHER_INPUT_EXPECTED")

    capabilities = dict(teacher_scene.get("constraint_capabilities") or {})
    coordinate = dict(teacher_scene.get("coordinate_contract") or {})
    calibration = dict(teacher_scene.get("calibration") or {})

    local_rotation = str(capabilities.get("local_native_rotation_parameter") or "UNKNOWN")
    physical_dof = str(capabilities.get("authored_physical_dof_or_joint_limit") or "UNKNOWN")
    impulse = str(capabilities.get("physical_impulse") or "UNKNOWN")
    time_unit = str(coordinate.get("time_unit") or "UNKNOWN")
    actions = tuple(calibration.get("actions") or ())

    if local_rotation == "MEASURED_CANDIDATE" and actions:
        kinematic = _family(
            "KINEMATIC_RESPONSE_DIAGNOSTIC",
            "MEASURED_DIAGNOSTIC_ONLY",
            "ISOLATED_NATIVE_ROTATION_PARAMETER_RESPONSE__NOT_AUTHORED_DOF",
            measured=True,
        )
    else:
        kinematic = _family(
            "KINEMATIC_RESPONSE_DIAGNOSTIC",
            "UNAVAILABLE",
            "NO_MEASURED_KINEMATIC_PARAMETER_RESPONSE",
        )

    if physical_dof in {"MEASURED", "QUALIFIED_MEASURED"}:
        # The current schema still needs actual Jacobian/residual arrays. A
        # capability flag alone cannot mint the family.
        admissible = _family(
            "ADMISSIBLE_MOTION_SUBSPACE",
            "UNAVAILABLE",
            "PHYSICAL_DOF_CAPABILITY_DECLARED_BUT_SUBSPACE_ARRAYS_NOT_PRODUCED",
        )
    else:
        admissible = _family(
            "ADMISSIBLE_MOTION_SUBSPACE",
            "UNAVAILABLE",
            "AUTHORED_PHYSICAL_DOF_OR_JOINT_LIMIT_NOT_MEASURED",
        )

    if impulse in {"MEASURED", "QUALIFIED_MEASURED"}:
        impulse_family = _family(
            "PHYSICAL_IMPULSE_RESPONSE",
            "UNAVAILABLE",
            "IMPULSE_CAPABILITY_DECLARED_BUT_RESPONSE_ARRAYS_NOT_PRODUCED",
        )
    else:
        impulse_family = _family(
            "PHYSICAL_IMPULSE_RESPONSE",
            "UNAVAILABLE",
            "AUTHORED_PHYSICAL_IMPULSE_SOLVER_UNAVAILABLE",
        )

    dynamic_time_measured = time_unit not in {
        "UNKNOWN",
        "NO_DYNAMIC_TIME_IN_STATIC_CALIBRATION",
    }
    if dynamic_time_measured:
        temporal = _family(
            "TEMPORAL_RESPONSE",
            "UNAVAILABLE",
            "DYNAMIC_TIME_DECLARED_BUT_TEMPORAL_RESPONSE_ARRAYS_NOT_PRODUCED",
        )
    else:
        temporal = _family(
            "TEMPORAL_RESPONSE",
            "UNAVAILABLE",
            "STATIC_CALIBRATION_HAS_NO_DYNAMIC_TIME_RESPONSE",
        )

    secondary_ready = (
        admissible.teacher_measured
        and temporal.teacher_measured
        and impulse_family.teacher_measured
    )
    secondary = _family(
        "SECONDARY_MOTION_CONSUMER",
        "MEASURED_CANDIDATE" if secondary_ready else "UNAVAILABLE",
        (
            "REQUIRED_MOTION_FAMILIES_PRESENT"
            if secondary_ready
            else "REQUIRES_QUALIFIED_ADMISSIBLE_AND_TEMPORAL_OR_SOLVER_RESPONSE"
        ),
        measured=secondary_ready,
        student=False,
        consequence=False,
    )

    return MotionTruthAvailabilityReportV1(
        subject_key=subject,
        kinematic_response_diagnostic=kinematic,
        admissible_motion_subspace=admissible,
        physical_impulse_response=impulse_family,
        temporal_response=temporal,
        secondary_motion_consumer=secondary,
    )
