"""Proof-side diagnostic and measurement services behind current Compiler authority."""

from .failure_signatures import derive_failure_signatures, no_owner_attribution
from .motion_bake import (
    QualificationOwnedMotionFrameIR,
    QualificationOwnedMotionBakeIR,
    bind_qualification_owned_motion_bake,
    assert_motion_bake_binding,
    deploy_frame_dicts,
)
from .motion_frame_metrics import (
    RESTORED_V05_POLICY_V1,
    measure_motion_bake_geometry,
    evaluate_motion_bake_metrics,
)
from .directional_motion_evaluator import (
    DirectionalMotionEvaluatorPolicyV1,
    EVALUATOR_SEMANTIC_VERSION,
    evaluate_clip_to_qualification_bake,
    make_qualified_motion_bake_provider,
)
from .causal_attribution import (
    ControlledInterventionEvidenceV1,
    attribute_signature_owner_v1,
    build_controlled_owner_attribution_v1,
    proof_probe_fingerprint_v1,
)
from .repair_loop import (
    BoundedRepairOperationV1,
    RepairDirectiveV1,
    RepairApplicationRecordV1,
    RepairReproofEvidenceV1,
    build_bounded_repair_directives_v1,
    validate_repair_application_v1,
    evaluate_repair_effect_v1,
)

__all__ = [
    "derive_failure_signatures",
    "no_owner_attribution",
    "QualificationOwnedMotionFrameIR",
    "QualificationOwnedMotionBakeIR",
    "bind_qualification_owned_motion_bake",
    "assert_motion_bake_binding",
    "deploy_frame_dicts",
    "RESTORED_V05_POLICY_V1",
    "measure_motion_bake_geometry",
    "evaluate_motion_bake_metrics",
    "DirectionalMotionEvaluatorPolicyV1",
    "EVALUATOR_SEMANTIC_VERSION",
    "evaluate_clip_to_qualification_bake",
    "make_qualified_motion_bake_provider",
    "ControlledInterventionEvidenceV1",
    "proof_probe_fingerprint_v1",
    "attribute_signature_owner_v1",
    "build_controlled_owner_attribution_v1",
    "BoundedRepairOperationV1",
    "RepairDirectiveV1",
    "RepairApplicationRecordV1",
    "RepairReproofEvidenceV1",
    "build_bounded_repair_directives_v1",
    "validate_repair_application_v1",
    "evaluate_repair_effect_v1",
]
