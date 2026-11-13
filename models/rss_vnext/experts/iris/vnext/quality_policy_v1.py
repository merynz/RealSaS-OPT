from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


OFFICIAL_NUMERIC_QUALITY_THRESHOLDS_PRESENT = False
HISTORICAL_COMPARABLE_NORMALIZED_RESIDUAL_REFERENCE_P95 = 0.05
SEALED_SHARED_PARAMETER_PREFIXES = ("scene_core.",)

# These IDs are the actual stage identifiers emitted by the V8 curriculum.
# Do not introduce aliases here: a renamed stage must fail the integration court
# until this contract and the trainer are changed together.
M1_SEAL_STAGE = "M1_MECHANICAL_DEMAND"
DOWNSTREAM_SPECIALIST_STAGES = (
    "A1_APPEARANCE_EVIDENCE",
    "R1_ROLE_FREE_RELATIONS",
    "T1_MECHANICAL_STRUCTURE",
    "J1_JOINT_FINETUNE",
)


class QualityStatus(str, Enum):
    MEASUREMENT_ONLY = "MEASUREMENT_ONLY"
    OPTIMIZATION_DEBT = "OPTIMIZATION_DEBT"
    ELIGIBLE_FOR_COURT = "ELIGIBLE_FOR_COURT"
    COURT_PASS = "COURT_PASS"


@dataclass(frozen=True)
class CarrierSealPolicyV1:
    seal_after_stage: str = M1_SEAL_STAGE
    sealed_parameter_prefixes: tuple[str, ...] = SEALED_SHARED_PARAMETER_PREFIXES
    downstream_stages: tuple[str, ...] = DOWNSTREAM_SPECIALIST_STAGES
    reopen_requires_explicit_court: bool = True
    structural_freeze_not_loss_balancing: bool = True

    def is_sealed_parameter(self, parameter_name: str) -> bool:
        return any(parameter_name.startswith(prefix) for prefix in self.sealed_parameter_prefixes)

    def assert_known_downstream_stage(self, stage_name: str) -> None:
        if stage_name.startswith(("A1_", "R1_", "T1_", "J1_")) and stage_name not in self.downstream_stages:
            raise RuntimeError(f"UNKNOWN_POST_M1_STAGE_ID:{stage_name}")

    def assert_downstream_parameter_allowed(
        self,
        *,
        stage_name: str,
        parameter_name: str,
        reopening_court_id: Optional[str] = None,
    ) -> None:
        self.assert_known_downstream_stage(stage_name)
        if stage_name not in self.downstream_stages:
            return
        if not self.is_sealed_parameter(parameter_name):
            return
        if self.reopen_requires_explicit_court and not reopening_court_id:
            raise RuntimeError(
                "SEALED_SHARED_CARRIER_REOPEN_REQUIRES_EXPLICIT_COURT:"
                f"{stage_name}:{parameter_name}"
            )

    def assert_stage_definition(
        self,
        *,
        stage_name: str,
        scene_lr_scale: float,
        reopening_court_id: Optional[str] = None,
    ) -> None:
        self.assert_known_downstream_stage(stage_name)
        if stage_name not in self.downstream_stages:
            return
        if float(scene_lr_scale) == 0.0:
            return
        if reopening_court_id:
            return
        raise RuntimeError(
            "POST_M1_STAGE_REOPENS_SEALED_SCENE_CORE_WITHOUT_COURT:"
            f"{stage_name}:scene_lr_scale={scene_lr_scale}"
        )

    def assert_optimizer_configuration(
        self,
        *,
        stage_name: str,
        group_lrs: Mapping[str, float],
        reopening_court_id: Optional[str] = None,
    ) -> None:
        self.assert_known_downstream_stage(stage_name)
        if stage_name not in self.downstream_stages:
            return
        if "scene_core" not in group_lrs:
            raise RuntimeError(f"SEALED_SCENE_CORE_OPTIMIZER_GROUP_MISSING:{stage_name}")
        scene_lr = float(group_lrs["scene_core"])
        if scene_lr == 0.0:
            return
        if reopening_court_id:
            return
        raise RuntimeError(
            "SEALED_SHARED_CARRIER_OPTIMIZER_LR_NONZERO_WITHOUT_COURT:"
            f"{stage_name}:lr={scene_lr}"
        )


@dataclass(frozen=True)
class ConvergencePolicyV1:
    preserve_best_heldout_checkpoint: bool = True
    require_non_regression: bool = True
    require_per_head_optimizer_groups: bool = True
    require_gradient_norm_telemetry: bool = True
    require_update_norm_telemetry: bool = True
    require_class_balance_audit: bool = True
    require_trivial_negative_audit: bool = True
    adaptive_extension_requires_preregistered_rule: bool = True
    unidentifiable_target_must_abstain: bool = True
    historical_comparable_reference_p95: float = (
        HISTORICAL_COMPARABLE_NORMALIZED_RESIDUAL_REFERENCE_P95
    )


@dataclass(frozen=True)
class MetricCourtV1:
    metric_name: str
    threshold: Optional[float]
    threshold_preregistered: bool
    target_authoritative: bool
    heldout: bool
    lower_is_better: bool = True

    def status(self, value: float) -> QualityStatus:
        if not self.target_authoritative or not self.heldout:
            return QualityStatus.MEASUREMENT_ONLY
        if self.threshold is None or not self.threshold_preregistered:
            return QualityStatus.ELIGIBLE_FOR_COURT

        passed = value <= self.threshold if self.lower_is_better else value >= self.threshold
        return QualityStatus.COURT_PASS if passed else QualityStatus.OPTIMIZATION_DEBT


def historical_reference_status(
    *,
    value_p95: float,
    comparable_normalized_residual: bool,
) -> QualityStatus:
    """Planning-only classifier; it can never produce COURT_PASS.

    The ~0.05 reference is historical engineering context, not a sealed scientific
    threshold. Heterogeneous metrics must not be forced onto this scale.
    """

    if not comparable_normalized_residual:
        return QualityStatus.MEASUREMENT_ONLY
    if value_p95 > HISTORICAL_COMPARABLE_NORMALIZED_RESIDUAL_REFERENCE_P95:
        return QualityStatus.OPTIMIZATION_DEBT
    return QualityStatus.ELIGIBLE_FOR_COURT


def quality_policy_v1() -> tuple[CarrierSealPolicyV1, ConvergencePolicyV1]:
    return CarrierSealPolicyV1(), ConvergencePolicyV1()
