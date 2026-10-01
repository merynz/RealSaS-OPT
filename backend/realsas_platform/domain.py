from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Mapping

Json = dict[str, Any]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def require_sha256(value: str, label: str) -> str:
    value = str(value)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label}: expected lowercase sha256 hex")
    return value


@dataclass(frozen=True)
class ArtifactInputIdentity:
    role: str
    ordinal: int
    artifact_type: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        if not self.role or not self.artifact_type or self.ordinal < 0:
            raise ValueError("invalid artifact input identity")
        require_sha256(self.semantic_sha256, "artifact input semantic_sha256")


@dataclass(frozen=True)
class ArtifactSemanticDescriptor:
    artifact_type: str
    schema_version: str
    producer_contract: str
    implementation_sha256: str
    policy_sha256: str
    inputs: tuple[ArtifactInputIdentity, ...] = ()
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_type or not self.schema_version or not self.producer_contract:
            raise ValueError("artifact type/schema/producer contract are required")
        require_sha256(self.implementation_sha256, "implementation_sha256")
        require_sha256(self.policy_sha256, "policy_sha256")
        ordinals = [x.ordinal for x in self.inputs]
        if ordinals != list(range(len(ordinals))):
            raise ValueError("artifact inputs must use contiguous ordered ordinals from zero")

    def payload(self) -> Json:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "producer_contract": self.producer_contract,
            "implementation_sha256": self.implementation_sha256,
            "policy_sha256": self.policy_sha256,
            "inputs": [asdict(x) for x in self.inputs],
            "semantic_parameters": dict(self.semantic_parameters),
        }

    @property
    def semantic_sha256(self) -> str:
        return sha256_json(self.payload())


@dataclass(frozen=True)
class ProductRoleBinding:
    role: str
    artifact_type: str
    artifact_semantic_sha256: str

    def __post_init__(self) -> None:
        if not self.role or not self.artifact_type:
            raise ValueError("product role and artifact type are required")
        require_sha256(self.artifact_semantic_sha256, "artifact_semantic_sha256")


@dataclass(frozen=True)
class ProductRevisionManifest:
    subject_id: str
    roles: tuple[ProductRoleBinding, ...]
    contract_version: str = "RealSaS.ProductRevisionManifest.v1"

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id is required")
        names = [x.role for x in self.roles]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("product roles must be unique and lexicographically sorted")

    def payload(self) -> Json:
        return {"contract_version": self.contract_version, "subject_id": self.subject_id, "roles": [asdict(x) for x in self.roles]}

    @property
    def manifest_sha256(self) -> str:
        return sha256_json(self.payload())


@dataclass(frozen=True)
class RenderRequestSpec:
    product_revision_manifest_sha256: str
    motion_artifact_semantic_sha256: str
    view_spec: Mapping[str, Any]
    render_settings: Mapping[str, Any]
    contract_version: str = "RealSaS.RenderRequest.v1"

    def __post_init__(self) -> None:
        require_sha256(self.product_revision_manifest_sha256, "product_revision_manifest_sha256")
        require_sha256(self.motion_artifact_semantic_sha256, "motion_artifact_semantic_sha256")

    def payload(self) -> Json:
        return {
            "contract_version": self.contract_version,
            "product_revision_manifest_sha256": self.product_revision_manifest_sha256,
            "motion_artifact_semantic_sha256": self.motion_artifact_semantic_sha256,
            "view_spec": dict(self.view_spec),
            "render_settings": dict(self.render_settings),
        }

    @property
    def semantic_sha256(self) -> str:
        return sha256_json(self.payload())
