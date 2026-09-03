from __future__ import annotations

"""Typed proof-provider authority for current directional motion evaluation.

ProofEngine must not accept arbitrary callables as qualification-owned frame
authority. This wrapper binds the exact directional joint/view binding, evaluator
policy and evaluator semantic version into one immutable provider hash. The
provider remains subordinate to current Compiler product/proof authority.
"""

from dataclasses import dataclass

from compiler.realsas_compiler_core.directional_binding import (
    DirectionalJointViewBindingSetIR,
    assert_directional_binding_for_product,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from .directional_motion_evaluator import (
    DirectionalMotionEvaluatorPolicyV1,
    EVALUATOR_SEMANTIC_VERSION,
    evaluate_clip_to_qualification_bake,
)


PROVIDER_SCHEMA_VERSION = "RealSaS.QualifiedDirectionalMotionBakeProvider.v1"


@dataclass(frozen=True)
class QualifiedDirectionalMotionBakeProviderV1:
    binding: DirectionalJointViewBindingSetIR
    policy: DirectionalMotionEvaluatorPolicyV1
    provider_hash: str
    schema_version: str = PROVIDER_SCHEMA_VERSION

    @property
    def source_product_state_hash(self) -> str:
        return self.binding.source_product_state_hash

    @property
    def directional_binding_set_hash(self) -> str:
        return self.binding.binding_set_hash

    @property
    def evaluator_policy_hash(self) -> str:
        return self.policy.policy_hash

    @property
    def evaluator_semantic_version(self) -> str:
        return EVALUATOR_SEMANTIC_VERSION

    def expected_provider_hash(self) -> str:
        return content_sha256({
            "schema_version": self.schema_version,
            "source_product_state_hash": self.source_product_state_hash,
            "directional_binding_set_hash": self.directional_binding_set_hash,
            "evaluator_policy_hash": self.evaluator_policy_hash,
            "evaluator_semantic_version": self.evaluator_semantic_version,
        })

    def assert_for_product(self, product) -> None:
        if self.schema_version != PROVIDER_SCHEMA_VERSION:
            raise QualificationError("MOTION_PROVIDER_SCHEMA_MISMATCH")
        if self.provider_hash != self.expected_provider_hash():
            raise QualificationError("MOTION_PROVIDER_HASH_MISMATCH")
        if self.source_product_state_hash != product.product_state_hash:
            raise QualificationError("STALE_MOTION_PROVIDER_PRODUCT")
        assert_directional_binding_for_product(product, self.binding)
        self.policy.validate()

    def __call__(self, product, plan, clip):
        self.assert_for_product(product)
        return evaluate_clip_to_qualification_bake(product, plan, clip, self.binding, policy=self.policy)


def make_qualified_directional_motion_provider(
    binding: DirectionalJointViewBindingSetIR,
    *,
    policy: DirectionalMotionEvaluatorPolicyV1 = DirectionalMotionEvaluatorPolicyV1(),
) -> QualifiedDirectionalMotionBakeProviderV1:
    policy.validate()
    provisional = QualifiedDirectionalMotionBakeProviderV1(binding, policy, "")
    return QualifiedDirectionalMotionBakeProviderV1(binding, policy, provisional.expected_provider_hash())
