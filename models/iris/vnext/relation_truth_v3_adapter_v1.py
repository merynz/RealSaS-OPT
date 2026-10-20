from __future__ import annotations

"""Pure adapter from all-hit evidence arrays to Relation Truth V3.

No ray tracing or new labels happen here.  The adapter consumes the already
materialized complete-stack arrays. Historical pairwise visibility columns are
intentionally ignored: global visibility is reconstructed only from CSR stack
rank. Static ownership/cut/attachment and projected boundaries abstain.
"""

from typing import Mapping

import numpy as np

from .boundary_causality_v1 import EvidenceAuthority
from .graphics_truth_v2 import QualifiedProbabilityTargetV2
from .graphics_truth_v3 import GraphicsPairTruthRowV3, RayStackVisibilityTruthV1


def _abstain(label: str) -> QualifiedProbabilityTargetV2:
    return QualifiedProbabilityTargetV2.abstain(provenance=label)


def all_hit_pair_truth_v3(
    arrays: Mapping[str, np.ndarray],
    *,
    view_index: int,
    provenance_prefix: str,
) -> tuple[GraphicsPairTruthRowV3, ...]:
    required = ("face_a", "face_b", "front_probability", "occlusion_positive_support")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError("RELATION_TRUTH_V3_PAIR_ARRAY_MISSING:" + ",".join(missing))
    face_a = np.asarray(arrays["face_a"], dtype=np.int64)
    face_b = np.asarray(arrays["face_b"], dtype=np.int64)
    front = np.asarray(arrays["front_probability"], dtype=np.float64)
    occlusion = np.asarray(arrays["occlusion_positive_support"], dtype=np.float64)
    count = len(face_a)
    if any(len(value) != count for value in (face_b, front, occlusion)):
        raise ValueError("RELATION_TRUTH_V3_PAIR_ARRAY_CARDINALITY_DRIFT")
    rows = []
    for index in range(count):
        provenance = f"{provenance_prefix}:view={view_index}:pair_row={index}"
        rows.append(
            GraphicsPairTruthRowV3(
                view_index=int(view_index),
                face_a=int(face_a[index]),
                face_b=int(face_b[index]),
                same_drawable_affinity=_abstain(provenance + ":same_drawable:UNKNOWN"),
                cut_probability=_abstain(provenance + ":cut:UNKNOWN"),
                attachment_affinity=_abstain(provenance + ":attachment:UNKNOWN"),
                front_probability=QualifiedProbabilityTargetV2(
                    float(front[index]), EvidenceAuthority.DETERMINISTIC_DERIVED,
                    provenance + ":complete_stack_depth_order", True,
                ),
                occlusion_probability=QualifiedProbabilityTargetV2(
                    float(occlusion[index]), EvidenceAuthority.QUALIFIED_DERIVED,
                    provenance + ":positive_same_ray_support_only", True,
                ),
                projected_alpha_boundary_probability=_abstain(provenance + ":alpha_boundary:SEPARATE_RGBA_AUTHORITY"),
                projected_appearance_boundary_probability=_abstain(provenance + ":appearance_boundary:SEPARATE_RGBA_AUTHORITY"),
            )
        )
    return tuple(rows)


def all_hit_stack_visibility_truth_v3(
    arrays: Mapping[str, np.ndarray],
    *,
    view_index: int,
    provenance_prefix: str,
) -> tuple[RayStackVisibilityTruthV1, ...]:
    required = ("stack_offsets", "stack_face", "stack_depth")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError("RELATION_TRUTH_V3_STACK_ARRAY_MISSING:" + ",".join(missing))
    offsets = np.asarray(arrays["stack_offsets"], dtype=np.int64)
    faces = np.asarray(arrays["stack_face"], dtype=np.int64)
    depths = np.asarray(arrays["stack_depth"], dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("RELATION_TRUTH_V3_STACK_OFFSETS_INVALID")
    if int(offsets[-1]) != len(faces) or len(faces) != len(depths):
        raise ValueError("RELATION_TRUTH_V3_STACK_CSR_CARDINALITY_DRIFT")
    rows = []
    for ray_index in range(len(offsets) - 1):
        start, end = int(offsets[ray_index]), int(offsets[ray_index + 1])
        if start == end:
            continue
        rows.append(
            RayStackVisibilityTruthV1.from_complete_depth_order(
                view_index=int(view_index),
                ray_index=int(ray_index),
                face_ids=tuple(int(v) for v in faces[start:end]),
                depths=tuple(float(v) for v in depths[start:end]),
                provenance=f"{provenance_prefix}:view={view_index}:ray={ray_index}:complete_all_hit_csr",
            )
        )
    return tuple(rows)
