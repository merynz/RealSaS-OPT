from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "RealSaS.FGeometryBackendContract.v1"
CANDIDATE = "CANDIDATE"
WITNESS = "WITNESS"
ROLES = {CANDIDATE, WITNESS}
CANDIDATE_ARMS = {"F_A", "F_B", "F_R", "F_C"}
WITNESS_ARMS = {"WITNESS_TETRA"}


@dataclass(frozen=True)
class FGeometryBackendV1:
    backend_id: str
    role: str
    extraction_family: str
    implementation_ref: str
    implementation_sha256: str
    deterministic_given_inputs: bool
    supports_open_surface: bool
    supports_nonmanifold_surface: bool
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_GEOMETRY_BACKEND_SCHEMA_INVALID")
        if not self.backend_id or not self.extraction_family or not self.implementation_ref:
            raise ValueError("F_GEOMETRY_BACKEND_IDENTITY_MISSING")
        if self.role not in ROLES:
            raise ValueError("F_GEOMETRY_BACKEND_ROLE_INVALID")
        if len(self.implementation_sha256) != 64:
            raise ValueError("F_GEOMETRY_BACKEND_IMPLEMENTATION_SHA_INVALID")
        if not self.provenance:
            raise ValueError("F_GEOMETRY_BACKEND_PROVENANCE_MISSING")


@dataclass(frozen=True)
class FGeometryBackendRunBindingV1:
    arm: str
    backend: FGeometryBackendV1
    teacher_bundle_sha256: str
    extraction_truth_sha256: str
    measurement_protocol_sha256: str

    def validate(self) -> None:
        self.backend.validate()
        if self.arm in CANDIDATE_ARMS:
            if self.backend.role != CANDIDATE:
                raise ValueError("F_GEOMETRY_BACKEND_CANDIDATE_ARM_REQUIRES_CANDIDATE_ROLE")
        elif self.arm in WITNESS_ARMS:
            if self.backend.role != WITNESS:
                raise ValueError("F_GEOMETRY_BACKEND_WITNESS_ARM_REQUIRES_WITNESS_ROLE")
        else:
            raise ValueError("F_GEOMETRY_BACKEND_ARM_INVALID")
        for name in (
            "teacher_bundle_sha256",
            "extraction_truth_sha256",
            "measurement_protocol_sha256",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"F_GEOMETRY_BACKEND_RUN_SHA_INVALID:{name}")


@dataclass(frozen=True)
class FGeometryWitnessComparisonV1:
    candidate_arm: str
    witness_arm: str
    candidate_measurement_protocol_sha256: str
    witness_measurement_protocol_sha256: str
    containment_disagrees: bool
    topology_disagrees: bool
    geometry_disagrees: bool
    disagreement_requires_abstention: bool = True

    def validate(self) -> None:
        if self.candidate_arm not in CANDIDATE_ARMS:
            raise ValueError("F_GEOMETRY_WITNESS_CANDIDATE_ARM_INVALID")
        if self.witness_arm not in WITNESS_ARMS:
            raise ValueError("F_GEOMETRY_WITNESS_ARM_INVALID")
        if len(self.candidate_measurement_protocol_sha256) != 64:
            raise ValueError("F_GEOMETRY_WITNESS_CANDIDATE_PROTOCOL_SHA_INVALID")
        if len(self.witness_measurement_protocol_sha256) != 64:
            raise ValueError("F_GEOMETRY_WITNESS_PROTOCOL_SHA_INVALID")
        if not self.disagreement_requires_abstention:
            raise ValueError("F_GEOMETRY_WITNESS_DISAGREEMENT_MUST_FAIL_CLOSED")

    @property
    def has_material_disagreement(self) -> bool:
        return bool(
            self.containment_disagrees
            or self.topology_disagrees
            or self.geometry_disagrees
        )

    @property
    def candidate_may_be_selected(self) -> bool:
        return not self.has_material_disagreement


__all__ = [
    "CANDIDATE",
    "CANDIDATE_ARMS",
    "FGeometryBackendRunBindingV1",
    "FGeometryBackendV1",
    "FGeometryWitnessComparisonV1",
    "ROLES",
    "SCHEMA_VERSION",
    "WITNESS",
    "WITNESS_ARMS",
]
