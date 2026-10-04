"""TESSA V1 — Topology Estimation for Stable Surface Animation.

TESSA consumes observation-grounded GSA surface evidence and emits learned mesh
proposals.  It never owns canonical mesh truth; Compiler qualification does.
"""

from .conditioning_v1 import (
    TESSAConditioningV1,
    TESSA_SURFACE_FEATURE_DIM_V1,
    build_tessa_conditioning_v1,
    denormalize_tessa_xyz_v1,
)
from .contracts_v1 import (
    TESSAActionKindV1,
    TESSAMechanicalRewardPolicyV1,
    TESSAMeshProposalV1,
    TESSAScalingPolicyV1,
    TESSAVertexProposalV1,
)
from .mechanical_objective_v1 import (
    TESSATriangleMetricsV1,
    mechanical_consequence_loss_v1,
    triangle_deformation_metrics_v1,
)
from .model_v1 import (
    TESSAConfigV1,
    TESSAOutputV1,
    TESSASurfaceEncoderV1,
    TESSAV1,
    WindowedCausalSelfAttentionV1,
    tessa_attention_work_upper_bound_v1,
)
from .tokenization_v1 import (
    TESSATeacherSequenceV1,
    adjacent_sequence_token_upper_bound_v1,
    dequantize_tessa_xyz_v1,
    encode_adjacent_teacher_mesh_v1,
    quantize_tessa_xyz_v1,
)

__all__ = [
    "TESSAActionKindV1",
    "TESSAConditioningV1",
    "TESSAConfigV1",
    "TESSAMechanicalRewardPolicyV1",
    "TESSAMeshProposalV1",
    "TESSAOutputV1",
    "TESSAScalingPolicyV1",
    "TESSASurfaceEncoderV1",
    "TESSA_SURFACE_FEATURE_DIM_V1",
    "TESSATeacherSequenceV1",
    "TESSATriangleMetricsV1",
    "TESSAV1",
    "TESSAVertexProposalV1",
    "WindowedCausalSelfAttentionV1",
    "adjacent_sequence_token_upper_bound_v1",
    "build_tessa_conditioning_v1",
    "denormalize_tessa_xyz_v1",
    "dequantize_tessa_xyz_v1",
    "encode_adjacent_teacher_mesh_v1",
    "mechanical_consequence_loss_v1",
    "quantize_tessa_xyz_v1",
    "tessa_attention_work_upper_bound_v1",
    "triangle_deformation_metrics_v1",
]
