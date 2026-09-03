from __future__ import annotations

"""Compatibility target type for the promoted Geppetto V2 training lane.

The original R6 file mixed Geppetto and SkinField teacher targets and imported the
older V1 conditioning package. Current Geppetto V2 loss only requires this typed
anonymous teacher target; V2 target projection lives in ``training_targets_v2``.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeppettoTeacherTargetV1:
    positions_normalized: np.ndarray
    parent_indices: np.ndarray
    root_mask: np.ndarray
    valid: bool = True
    schema_version: str = "RealSaS.GeppettoTeacherTarget.v1"


__all__ = ["GeppettoTeacherTargetV1"]
