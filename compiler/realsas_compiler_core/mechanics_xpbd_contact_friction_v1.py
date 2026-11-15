from __future__ import annotations

"""Coulomb tangential position corrections for explicitly declared XPBD contacts.

The particle is first projected out of a plane/sphere contact. A bounded
position-level tangential correction opposes its displacement relative to
the collider over the same fixed substep. There is no learned material truth.
"""

import math
import numpy as np

from .types import QualificationError


def apply_xpbd_coulomb_friction_v1(
    position: np.ndarray,
    previous_position: np.ndarray,
    normal: np.ndarray,
    normal_projection: float,
    *,
    static_friction: float,
    dynamic_friction: float,
    collider_step_displacement: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Apply static sticking or bounded dynamic sliding, preserving normal.

    The available Coulomb budget is mu * accumulated nonnegative normal
    projection over this substep. The previous position refers to the beginning
    of that SAME physics substep, not the beginning of the animation clip.
    For joint-attached colliders, motion must be relative to collider movement.
    """
    p=np.asarray(position,dtype=np.float64)
    start=np.asarray(previous_position,dtype=np.float64)
    n=np.asarray(normal,dtype=np.float64)
    moving=np.zeros(3,dtype=np.float64) if collider_step_displacement is None else np.asarray(
        collider_step_displacement,dtype=np.float64,
    )
    if (p.shape!=(3,) or start.shape!=(3,) or n.shape!=(3,)
            or moving.shape!=(3,)
            or not all(np.isfinite(x).all() for x in (p,start,n,moving))):
        raise QualificationError("XPBD_FRICTION_VECTOR_INVALID")
    try:
        correction=float(normal_projection)
        mu_static=float(static_friction)
        mu_dynamic=float(dynamic_friction)
    except (TypeError,ValueError) as exc:
        raise QualificationError("XPBD_FRICTION_POLICY_INVALID") from exc
    if (not all(math.isfinite(x) for x in (correction,mu_static,mu_dynamic))
            or correction<0. or mu_static<0. or mu_dynamic<0.
            or mu_dynamic>mu_static or mu_static>100.):
        raise QualificationError("XPBD_FRICTION_POLICY_INVALID")
    length=float(np.linalg.norm(n))
    if abs(length-1.)>1e-6:
        raise QualificationError("XPBD_FRICTION_NORMAL_NOT_UNIT")
    if correction==0. or mu_static==0.:
        return p.copy(),False
    relative=p-start-moving
    tangent=relative-float(relative@n)*n
    magnitude=float(np.linalg.norm(tangent))
    if magnitude<=1e-14:
        return p.copy(),False
    if magnitude<=mu_static*correction:
        correction_length=magnitude
    else:
        correction_length=min(magnitude,mu_dynamic*correction)
    if correction_length<=0.:
        return p.copy(),False
    adjusted=p-correction_length*(tangent/magnitude)
    if not np.isfinite(adjusted).all():
        raise QualificationError("XPBD_FRICTION_NONFINITE")
    return adjusted,True
