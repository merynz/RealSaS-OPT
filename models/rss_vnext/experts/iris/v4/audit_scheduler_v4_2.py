from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditComputeBudgetV42:
    """Apparatus-only compute guards; never product-quality thresholds."""

    r128_exact_stage13_face_cap: int = 250_000
    r256_exact_stage13_face_cap: int = 1_000_000
    r512_exact_stage13_face_cap: int = 2_000_000

    def validate(self) -> None:
        if min(
            int(self.r128_exact_stage13_face_cap),
            int(self.r256_exact_stage13_face_cap),
            int(self.r512_exact_stage13_face_cap),
        ) <= 0:
            raise ValueError("audit face caps must be positive")
        if not (
            self.r128_exact_stage13_face_cap
            <= self.r256_exact_stage13_face_cap
            <= self.r512_exact_stage13_face_cap
        ):
            raise ValueError("audit face caps must be nondecreasing with resolution")


def exact_stage13_compute_admissible_v42(
    *,
    resolution: int,
    face_count: int,
    budget: AuditComputeBudgetV42 = AuditComputeBudgetV42(),
) -> tuple[bool, str]:
    """Return whether the exact unchanged Stage13 evaluator may be scheduled.

    A false result says only that the apparatus refuses an expensive exact audit at this
    resolution. It is not a geometry FAIL and may never be interpreted as a product gate.
    """

    budget.validate()
    r = int(resolution)
    n = int(face_count)
    if n < 0:
        raise ValueError("face_count must be non-negative")
    caps = {
        128: int(budget.r128_exact_stage13_face_cap),
        256: int(budget.r256_exact_stage13_face_cap),
        512: int(budget.r512_exact_stage13_face_cap),
    }
    if r not in caps:
        raise ValueError("supported audit resolutions are 128, 256, and 512")
    cap = caps[r]
    if n > cap:
        return False, f"COMPUTE_GUARD_FACE_CAP:R{r}:{n}>{cap}"
    return True, "COMPUTE_ADMISSIBLE"


def r512_promotion_audit_required_v42(*, r256_stage13_pass: bool) -> bool:
    """R512 is mandatory only after the R256 screen remains promotion-viable."""

    return bool(r256_stage13_pass)
