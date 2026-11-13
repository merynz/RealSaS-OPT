"""TESSA V1 — Topological Evidence for Surface Structure Approximation.

TESSA consumes observation-grounded IRIS/GSA surface evidence and emits learned
mesh/topology proposals. It never owns canonical mesh truth; Compiler
qualification, repair, and adoption remain the only path to mechanical authority.
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
from .proposal_v1 import assemble_tessa_proposal_v1
from .sequence_codec_v1 import TESSADecodedMeshV1, decode_tessa_asset_sequence_v1
from .tokenization_v1 import (
    TESSATeacherSequenceV1,
    adjacent_sequence_token_upper_bound_v1,
    dequantize_tessa_xyz_v1,
    encode_adjacent_teacher_mesh_v1,
    quantize_tessa_xyz_v1,
)
from .training_data_v1 import (
    TESSATeacherAssetSequenceV1,
    TESSATrainingWindowV1,
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
    connected_face_components_v1,
    deterministic_face_charts_v1,
)

__all__ = [
    "TESSAActionKindV1",
    "TESSAConditioningV1",
    "TESSAConfigV1",
    "TESSADecodedMeshV1",
    "TESSAMechanicalRewardPolicyV1",
    "TESSAMeshProposalV1",
    "TESSAOutputV1",
    "TESSAScalingPolicyV1",
    "TESSASurfaceEncoderV1",
    "TESSA_SURFACE_FEATURE_DIM_V1",
    "TESSATeacherAssetSequenceV1",
    "TESSATeacherSequenceV1",
    "TESSATrainingWindowV1",
    "TESSATriangleMetricsV1",
    "TESSAV1",
    "TESSAVertexProposalV1",
    "WindowedCausalSelfAttentionV1",
    "adjacent_sequence_token_upper_bound_v1",
    "assemble_tessa_proposal_v1",
    "build_teacher_asset_sequence_v1",
    "build_truncated_teacher_windows_v1",
    "build_tessa_conditioning_v1",
    "connected_face_components_v1",
    "decode_tessa_asset_sequence_v1",
    "denormalize_tessa_xyz_v1",
    "dequantize_tessa_xyz_v1",
    "deterministic_face_charts_v1",
    "encode_adjacent_teacher_mesh_v1",
    "mechanical_consequence_loss_v1",
    "quantize_tessa_xyz_v1",
    "tessa_attention_work_upper_bound_v1",
    "triangle_deformation_metrics_v1",
]
