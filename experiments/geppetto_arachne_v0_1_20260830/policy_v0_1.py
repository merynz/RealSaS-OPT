from __future__ import annotations

from dataclasses import dataclass


# Post-audit clean-C0 consumer capacity. This is a model/training profile bound,
# not a statement that larger rigs are invalid product inputs.
GEPPETTO_C0_MAX_CONTROLS = 160
GEPPETTO_OVERFLOW_POLICY = "ABSTAIN_NO_TRUNCATION"

# Projection policy frozen after the complete FIT audit.
SKELETON_PARENT_PROJECTION_POLICY = "NEAREST_DEFORM_ANCESTOR"
MULTI_ROOT_PROJECTION_POLICY = "PRESERVE_FOREST__NO_SYNTHETIC_SUPER_ROOT"
ZERO_LENGTH_DEFORM_POLICY = "FAIL_IF_ENCOUNTERED_AFTER_FREEZE"

# Arachne clean-C0 teacher authority: projected deform columns only. We do not
# transport or renormalize helper/non-deform mass. Assets carrying such mass
# are excluded from clean C0 rather than semantically rewritten.
ARACHNE_NONDEFORM_SKIN_POLICY = "EXCLUDE_FROM_C0__NO_TRANSPORT__NO_RENORMALIZATION"


@dataclass(frozen=True)
class ConsumerC0AdmissionV1:
    eligible: bool
    reason: str
    structural_overflow: bool = False
    nondeform_skin_mass_present: bool = False


def classify_geppetto_c0_v1(*, deform_control_count: int) -> ConsumerC0AdmissionV1:
    k = int(deform_control_count)
    if k < 1:
        return ConsumerC0AdmissionV1(False, "NO_DEFORM_CONTROLS")
    if k > GEPPETTO_C0_MAX_CONTROLS:
        return ConsumerC0AdmissionV1(
            False,
            f"STRUCTURAL_OVERFLOW_{k}_GT_{GEPPETTO_C0_MAX_CONTROLS}",
            structural_overflow=True,
        )
    return ConsumerC0AdmissionV1(True, "C0_ELIGIBLE")


def classify_arachne_c0_v1(
    *,
    deform_control_count: int,
    nondeform_skin_mass_total: float,
) -> ConsumerC0AdmissionV1:
    g = classify_geppetto_c0_v1(deform_control_count=deform_control_count)
    if not g.eligible:
        return g
    mass = float(nondeform_skin_mass_total)
    if mass < 0.0:
        raise ValueError("nondeform_skin_mass_total must be nonnegative")
    if mass > 0.0:
        return ConsumerC0AdmissionV1(
            False,
            "NONDEFORM_SKIN_MASS_REQUIRES_UNFROZEN_SEMANTIC_TRANSPORT",
            nondeform_skin_mass_present=True,
        )
    return ConsumerC0AdmissionV1(True, "C0_ELIGIBLE")
