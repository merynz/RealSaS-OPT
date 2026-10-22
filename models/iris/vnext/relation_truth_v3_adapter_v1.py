from __future__ import annotations

"""Pure adapter from tie-safe all-hit evidence arrays to Relation Truth V3.

No ray tracing or new labels happen here. V2 evidence is authoritative about its
own validity: equal-depth front ties and tied frontmost visibility explicitly
ABSTAIN. The adapter never reconstructs global visibility from serialization
rank. Static ownership/cut/attachment and projected boundaries abstain.
"""

from typing import Mapping

import numpy as np

from .boundary_causality_v1 import EvidenceAuthority
from .graphics_truth_v2 import QualifiedProbabilityTargetV2
from .graphics_truth_v3 import GraphicsPairTruthRowV3, RayStackVisibilityTruthV1


def _abstain(label: str) -> QualifiedProbabilityTargetV2:
    return QualifiedProbabilityTargetV2.abstain(provenance=label)


def _qualified(value: float, valid: bool, provenance: str, authority: EvidenceAuthority) -> QualifiedProbabilityTargetV2:
    if not valid:
        if float(value) != 0.0:
            raise ValueError("RELATION_TRUTH_V3_INVALID_TARGET_MUST_BE_ZERO")
        return _abstain(provenance + ":ABSTAIN")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError("RELATION_TRUTH_V3_PROBABILITY_OUT_OF_RANGE")
    return QualifiedProbabilityTargetV2(float(value), authority, provenance, True)


def all_hit_pair_truth_v3(
    arrays: Mapping[str, np.ndarray], *, view_index: int, provenance_prefix: str,
) -> tuple[GraphicsPairTruthRowV3, ...]:
    required = ("face_a", "face_b", "front_probability", "front_valid", "occlusion_positive_support")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError("RELATION_TRUTH_V3_PAIR_ARRAY_MISSING:" + ",".join(missing))
    face_a = np.asarray(arrays["face_a"], dtype=np.int64)
    face_b = np.asarray(arrays["face_b"], dtype=np.int64)
    front = np.asarray(arrays["front_probability"], dtype=np.float64)
    front_valid = np.asarray(arrays["front_valid"], dtype=np.bool_)
    occlusion = np.asarray(arrays["occlusion_positive_support"], dtype=np.float64)
    count = len(face_a)
    if any(len(value) != count for value in (face_b, front, front_valid, occlusion)):
        raise ValueError("RELATION_TRUTH_V3_PAIR_ARRAY_CARDINALITY_DRIFT")
    if np.any(~front_valid & (front != 0.0)):
        raise ValueError("RELATION_TRUTH_V3_INVALID_FRONT_NONZERO")
    if np.any(occlusion != 1.0):
        raise ValueError("RELATION_TRUTH_V3_OCCLUSION_NOT_POSITIVE_SUPPORT_ONLY")
    rows = []
    for index in range(count):
        provenance = f"{provenance_prefix}:view={view_index}:pair_row={index}"
        rows.append(GraphicsPairTruthRowV3(
            view_index=int(view_index), face_a=int(face_a[index]), face_b=int(face_b[index]),
            same_drawable_affinity=_abstain(provenance + ":same_drawable:UNKNOWN"),
            cut_probability=_abstain(provenance + ":cut:UNKNOWN"),
            attachment_affinity=_abstain(provenance + ":attachment:UNKNOWN"),
            front_probability=_qualified(float(front[index]), bool(front_valid[index]), provenance + ":tie_safe_complete_stack_depth_order", EvidenceAuthority.DETERMINISTIC_DERIVED),
            occlusion_probability=QualifiedProbabilityTargetV2(float(occlusion[index]), EvidenceAuthority.QUALIFIED_DERIVED, provenance + ":positive_same_ray_support_only", True),
            projected_alpha_boundary_probability=_abstain(provenance + ":alpha_boundary:SEPARATE_RGBA_AUTHORITY"),
            projected_appearance_boundary_probability=_abstain(provenance + ":appearance_boundary:SEPARATE_RGBA_AUTHORITY"),
        ))
    return tuple(rows)


def all_hit_stack_visibility_truth_v3(
    arrays: Mapping[str, np.ndarray], *, view_index: int, provenance_prefix: str,
) -> tuple[RayStackVisibilityTruthV1, ...]:
    required = ("stack_offsets", "stack_face", "stack_depth", "stack_visibility_probability", "stack_visibility_valid", "stack_tied_minimum")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError("RELATION_TRUTH_V3_STACK_ARRAY_MISSING:" + ",".join(missing))
    offsets = np.asarray(arrays["stack_offsets"], dtype=np.int64)
    faces = np.asarray(arrays["stack_face"], dtype=np.int64)
    depths = np.asarray(arrays["stack_depth"], dtype=np.float64)
    values = np.asarray(arrays["stack_visibility_probability"], dtype=np.float64)
    valid = np.asarray(arrays["stack_visibility_valid"], dtype=np.bool_)
    tied = np.asarray(arrays["stack_tied_minimum"], dtype=np.bool_)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("RELATION_TRUTH_V3_STACK_OFFSETS_INVALID")
    if int(offsets[-1]) != len(faces) or any(len(x) != len(faces) for x in (depths, values, valid, tied)):
        raise ValueError("RELATION_TRUTH_V3_STACK_CSR_CARDINALITY_DRIFT")
    if np.any(~valid & (values != 0.0)):
        raise ValueError("RELATION_TRUTH_V3_INVALID_VISIBILITY_NONZERO")
    rows = []
    for ray_index in range(len(offsets) - 1):
        start, end = int(offsets[ray_index]), int(offsets[ray_index + 1])
        if start == end:
            continue
        provenance = f"{provenance_prefix}:view={view_index}:ray={ray_index}:complete_all_hit_csr_v2"
        visibility = tuple(
            _qualified(float(values[i]), bool(valid[i]), provenance + f":rank={i-start}" + (":TIED_MINIMUM" if tied[i] else ""), EvidenceAuthority.DETERMINISTIC_DERIVED)
            for i in range(start, end)
        )
        rows.append(RayStackVisibilityTruthV1(
            view_index=int(view_index), ray_index=int(ray_index),
            face_ids=tuple(int(v) for v in faces[start:end]), depths=tuple(float(v) for v in depths[start:end]),
            visibility=visibility, complete_stack=True, provenance=provenance,
        ))
    return tuple(rows)
