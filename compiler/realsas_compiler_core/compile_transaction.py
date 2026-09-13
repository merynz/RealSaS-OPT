from __future__ import annotations

"""Fail-closed current compile transaction identity for RealSaS.

This module deliberately restores only the valuable historical orchestrator
semantics: immutable attempts, exact artifact identity, frozen per-stage policy /
implementation bindings, terminal FAIL/ABSTAIN, and explicit child attempts.
It does not restore the historical monolithic pipeline or any old semantic owner.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

from .hashing import content_sha256
from .types import QualificationError, RuntimePackageIR

Json = dict[str, Any]
_ALLOWED_STAGE_STATUS = {"PASS", "FAIL", "ABSTAIN"}
_ALLOWED_TX_STATUS = {"OPEN", "PASS", "FAIL", "ABSTAIN"}
DEFAULT_STAGE_ORDER = (
    "OBSERVATION",
    "SURFACE",
    "SKELETON",
    "SKIN",
    "DIRECTIONAL_MESH",
    "MESH_SKIN",
    "COMPONENTS",
    "MOTION",
    "PROOF",
    "RUNTIME_EXPORT",
)


def _hash_without(value: Any, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


def _require_hash(name: str, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise QualificationError(f"COMPILE_TRANSACTION_MISSING_{name}")
    return value


@dataclass(frozen=True)
class CompileStageContractIR:
    stage_id: str
    policy_hash: str
    implementation_binding_hash: str
    dependencies: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.CompileStageContractIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompileRequestIR:
    request_id: str
    source_observation_hash: str
    source_camera_bundle_hash: str
    compiler_semantic_version: str
    policy_bundle_hash: str
    stage_contracts: tuple[CompileStageContractIR, ...]
    required_final_slots: tuple[str, ...]
    request_hash: str
    required_view_indices: tuple[int, ...] = tuple(range(8))
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.CompileRequestIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompileArtifactBindingIR:
    slot_id: str
    authority_class: str
    artifact_schema: str
    artifact_id: str
    content_hash: str
    producer_stage: str
    binding_hash: str
    source_product_state_hash: str = ""
    view_index: int | None = None
    component_id: str = ""
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.CompileArtifactBindingIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompileStageResultIR:
    stage_id: str
    status: str
    request_hash: str
    parent_stage_result_hash: str
    consumed_binding_hashes: tuple[str, ...]
    produced_binding_hashes: tuple[str, ...]
    policy_hash: str
    implementation_binding_hash: str
    diagnostics_hash: str
    blockers: tuple[str, ...]
    stage_result_hash: str
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.CompileStageResultIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompileTransactionIR:
    transaction_id: str
    request: CompileRequestIR
    attempt_id: str
    parent_transaction_hash: str | None
    stage_results: tuple[CompileStageResultIR, ...]
    artifact_bindings: tuple[CompileArtifactBindingIR, ...]
    status: str
    terminal_stage_id: str
    transaction_hash: str
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.CompileTransactionIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompileResultIR:
    transaction_hash: str
    request_hash: str
    status: str
    product_state_hash: str
    proof_bundle_hash: str
    runtime_package_hash: str
    output_binding_hashes: tuple[str, ...]
    blockers: tuple[str, ...]
    result_hash: str
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.CompileResultIR.v1"

    def to_dict(self) -> Json:
        return asdict(self)


def request_hash(value: CompileRequestIR) -> str:
    return _hash_without(value, "request_hash")


def artifact_binding_hash(value: CompileArtifactBindingIR) -> str:
    return _hash_without(value, "binding_hash")


def stage_result_hash(value: CompileStageResultIR) -> str:
    return _hash_without(value, "stage_result_hash")


def transaction_hash(value: CompileTransactionIR) -> str:
    return _hash_without(value, "transaction_hash")


def compile_result_hash(value: CompileResultIR) -> str:
    return _hash_without(value, "result_hash")


def make_stage_contract(
    stage_id: str,
    *,
    policy_hash: str,
    implementation_binding_hash: str,
    dependencies: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CompileStageContractIR:
    stage_id = str(stage_id or "").strip().upper()
    if not stage_id:
        raise QualificationError("COMPILE_STAGE_CONTRACT_MISSING_STAGE_ID")
    return CompileStageContractIR(
        stage_id=stage_id,
        policy_hash=_require_hash("STAGE_POLICY_HASH", policy_hash),
        implementation_binding_hash=_require_hash(
            "STAGE_IMPLEMENTATION_BINDING_HASH", implementation_binding_hash
        ),
        dependencies=tuple(str(x).strip().upper() for x in dependencies if str(x).strip()),
        metadata=dict(metadata or {}),
    )


def make_compile_request(
    *,
    request_id: str,
    source_observation_hash: str,
    source_camera_bundle_hash: str,
    compiler_semantic_version: str,
    policy_bundle_hash: str,
    stage_contracts: Iterable[CompileStageContractIR],
    required_final_slots: Iterable[str] = ("PRODUCT", "PROOF_BUNDLE", "RUNTIME_PACKAGE"),
    required_view_indices: Iterable[int] = range(8),
    metadata: Mapping[str, Any] | None = None,
) -> CompileRequestIR:
    contracts = tuple(stage_contracts)
    if not contracts:
        raise QualificationError("COMPILE_REQUEST_REQUIRES_STAGE_CONTRACTS")
    ids = tuple(c.stage_id for c in contracts)
    if len(ids) != len(set(ids)):
        raise QualificationError("COMPILE_REQUEST_DUPLICATE_STAGE_ID")
    seen: set[str] = set()
    for contract in contracts:
        if any(dep not in seen for dep in contract.dependencies):
            raise QualificationError("COMPILE_REQUEST_STAGE_DEPENDENCY_NOT_EARLIER")
        seen.add(contract.stage_id)
    views = tuple(int(v) for v in required_view_indices)
    if views != tuple(range(8)):
        raise QualificationError("COMPILE_REQUEST_REQUIRES_EXACT_VIEWS_0_TO_7")
    value = CompileRequestIR(
        request_id=str(request_id or "").strip(),
        source_observation_hash=_require_hash("SOURCE_OBSERVATION_HASH", source_observation_hash),
        source_camera_bundle_hash=_require_hash("SOURCE_CAMERA_BUNDLE_HASH", source_camera_bundle_hash),
        compiler_semantic_version=str(compiler_semantic_version or "").strip(),
        policy_bundle_hash=_require_hash("POLICY_BUNDLE_HASH", policy_bundle_hash),
        stage_contracts=contracts,
        required_final_slots=tuple(str(x).strip() for x in required_final_slots if str(x).strip()),
        request_hash="",
        required_view_indices=views,
        metadata=dict(metadata or {}),
    )
    if not value.request_id:
        raise QualificationError("COMPILE_REQUEST_MISSING_REQUEST_ID")
    if not value.compiler_semantic_version:
        raise QualificationError("COMPILE_REQUEST_MISSING_COMPILER_SEMANTIC_VERSION")
    if not value.required_final_slots:
        raise QualificationError("COMPILE_REQUEST_REQUIRES_FINAL_SLOTS")
    value = replace(value, request_hash=request_hash(value))
    validate_compile_request(value)
    return value


def validate_compile_request(value: CompileRequestIR) -> None:
    if value.request_hash != request_hash(value):
        raise QualificationError("COMPILE_REQUEST_HASH_MISMATCH")
    if tuple(value.required_view_indices) != tuple(range(8)):
        raise QualificationError("COMPILE_REQUEST_REQUIRES_EXACT_VIEWS_0_TO_7")
    if not value.stage_contracts:
        raise QualificationError("COMPILE_REQUEST_REQUIRES_STAGE_CONTRACTS")
    ids = [c.stage_id for c in value.stage_contracts]
    if len(ids) != len(set(ids)):
        raise QualificationError("COMPILE_REQUEST_DUPLICATE_STAGE_ID")
    seen: set[str] = set()
    for contract in value.stage_contracts:
        _require_hash("STAGE_POLICY_HASH", contract.policy_hash)
        _require_hash("STAGE_IMPLEMENTATION_BINDING_HASH", contract.implementation_binding_hash)
        if any(dep not in seen for dep in contract.dependencies):
            raise QualificationError("COMPILE_REQUEST_STAGE_DEPENDENCY_NOT_EARLIER")
        seen.add(contract.stage_id)


def make_artifact_binding(
    *,
    slot_id: str,
    authority_class: str,
    artifact_schema: str,
    artifact_id: str,
    content_hash: str,
    producer_stage: str,
    source_product_state_hash: str = "",
    view_index: int | None = None,
    component_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CompileArtifactBindingIR:
    slot_id = str(slot_id or "").strip()
    artifact_id = str(artifact_id or "").strip()
    if not slot_id or not artifact_id:
        raise QualificationError("COMPILE_ARTIFACT_BINDING_MISSING_ID")
    if artifact_id.lower() in {"latest", "current", "newest"}:
        raise QualificationError("COMPILE_ARTIFACT_LATEST_ALIAS_FORBIDDEN")
    if view_index is not None and int(view_index) not in range(8):
        raise QualificationError("COMPILE_ARTIFACT_INVALID_VIEW_INDEX")
    value = CompileArtifactBindingIR(
        slot_id=slot_id,
        authority_class=str(authority_class or "").strip(),
        artifact_schema=str(artifact_schema or "").strip(),
        artifact_id=artifact_id,
        content_hash=_require_hash("ARTIFACT_CONTENT_HASH", content_hash),
        producer_stage=str(producer_stage or "").strip().upper(),
        binding_hash="",
        source_product_state_hash=str(source_product_state_hash or "").strip(),
        view_index=None if view_index is None else int(view_index),
        component_id=str(component_id or "").strip(),
        metadata=dict(metadata or {}),
    )
    if not value.authority_class or not value.artifact_schema:
        raise QualificationError("COMPILE_ARTIFACT_BINDING_MISSING_AUTHORITY_OR_SCHEMA")
    value = replace(value, binding_hash=artifact_binding_hash(value))
    validate_artifact_binding(value)
    return value


def validate_artifact_binding(value: CompileArtifactBindingIR) -> None:
    if value.binding_hash != artifact_binding_hash(value):
        raise QualificationError("COMPILE_ARTIFACT_BINDING_HASH_MISMATCH")
    if not value.slot_id or not value.artifact_id or not value.content_hash:
        raise QualificationError("COMPILE_ARTIFACT_BINDING_INCOMPLETE")
    if value.artifact_id.lower() in {"latest", "current", "newest"}:
        raise QualificationError("COMPILE_ARTIFACT_LATEST_ALIAS_FORBIDDEN")
    if value.view_index is not None and value.view_index not in range(8):
        raise QualificationError("COMPILE_ARTIFACT_INVALID_VIEW_INDEX")


def _source_bindings(request: CompileRequestIR) -> tuple[CompileArtifactBindingIR, ...]:
    return (
        make_artifact_binding(
            slot_id="SOURCE_OBSERVATION",
            authority_class="EXACT_INPUT_EVIDENCE",
            artifact_schema="RealSaS.ObservationBundleHash.v1",
            artifact_id=f"OBS:{request.source_observation_hash[:24]}",
            content_hash=request.source_observation_hash,
            producer_stage="INPUT",
        ),
        make_artifact_binding(
            slot_id="SOURCE_CAMERAS",
            authority_class="EXACT_INPUT_EVIDENCE",
            artifact_schema="RealSaS.CameraBundleHash.v1",
            artifact_id=f"CAM:{request.source_camera_bundle_hash[:24]}",
            content_hash=request.source_camera_bundle_hash,
            producer_stage="INPUT",
        ),
    )


def start_compile_transaction(
    request: CompileRequestIR,
    *,
    transaction_id: str,
    attempt_id: str,
    parent_transaction_hash: str | None = None,
    carry_bindings: Iterable[CompileArtifactBindingIR] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CompileTransactionIR:
    validate_compile_request(request)
    transaction_id = str(transaction_id or "").strip()
    attempt_id = str(attempt_id or "").strip()
    if not transaction_id or not attempt_id:
        raise QualificationError("COMPILE_TRANSACTION_REQUIRES_IDS")
    bindings = list(_source_bindings(request))
    for binding in carry_bindings:
        validate_artifact_binding(binding)
        if binding.slot_id in {"SOURCE_OBSERVATION", "SOURCE_CAMERAS"}:
            raise QualificationError("COMPILE_TRANSACTION_SOURCE_BINDING_OVERRIDE_FORBIDDEN")
        bindings.append(binding)
    _validate_unique_slots(bindings)
    value = CompileTransactionIR(
        transaction_id=transaction_id,
        request=request,
        attempt_id=attempt_id,
        parent_transaction_hash=parent_transaction_hash,
        stage_results=(),
        artifact_bindings=tuple(bindings),
        status="OPEN",
        terminal_stage_id="",
        transaction_hash="",
        metadata=dict(metadata or {}),
    )
    return replace(value, transaction_hash=transaction_hash(value))


def fork_compile_transaction(
    parent: CompileTransactionIR,
    *,
    transaction_id: str,
    attempt_id: str,
    carry_slots: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CompileTransactionIR:
    validate_compile_transaction(parent)
    by_slot = {b.slot_id: b for b in parent.artifact_bindings}
    carried: list[CompileArtifactBindingIR] = []
    for slot in carry_slots:
        slot = str(slot)
        if slot in {"SOURCE_OBSERVATION", "SOURCE_CAMERAS"}:
            continue
        try:
            carried.append(by_slot[slot])
        except KeyError as exc:
            raise QualificationError(f"COMPILE_FORK_UNKNOWN_CARRY_SLOT:{slot}") from exc
    return start_compile_transaction(
        parent.request,
        transaction_id=transaction_id,
        attempt_id=attempt_id,
        parent_transaction_hash=parent.transaction_hash,
        carry_bindings=carried,
        metadata={"forked_from_attempt_id": parent.attempt_id, **dict(metadata or {})},
    )


def _validate_unique_slots(bindings: Iterable[CompileArtifactBindingIR]) -> None:
    seen: dict[str, str] = {}
    for binding in bindings:
        validate_artifact_binding(binding)
        previous = seen.get(binding.slot_id)
        if previous is not None:
            if previous != binding.binding_hash:
                raise QualificationError(f"COMPILE_ARTIFACT_SLOT_REBIND_FORBIDDEN:{binding.slot_id}")
            raise QualificationError(f"COMPILE_ARTIFACT_DUPLICATE_SLOT:{binding.slot_id}")
        seen[binding.slot_id] = binding.binding_hash


def _contract_for(request: CompileRequestIR, stage_id: str) -> CompileStageContractIR:
    for contract in request.stage_contracts:
        if contract.stage_id == stage_id:
            return contract
    raise QualificationError(f"COMPILE_STAGE_NOT_IN_REQUEST:{stage_id}")


def append_compile_stage(
    transaction: CompileTransactionIR,
    *,
    stage_id: str,
    status: str,
    consumed_slots: Iterable[str],
    produced_bindings: Iterable[CompileArtifactBindingIR] = (),
    diagnostics: Mapping[str, Any] | None = None,
    blockers: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CompileTransactionIR:
    validate_compile_transaction(transaction)
    if transaction.status != "OPEN":
        raise QualificationError("COMPILE_TRANSACTION_IS_TERMINAL")
    index = len(transaction.stage_results)
    if index >= len(transaction.request.stage_contracts):
        raise QualificationError("COMPILE_TRANSACTION_HAS_NO_NEXT_STAGE")
    expected = transaction.request.stage_contracts[index]
    stage_id = str(stage_id or "").strip().upper()
    if stage_id != expected.stage_id:
        raise QualificationError(f"COMPILE_STAGE_ORDER_MISMATCH:{stage_id}!={expected.stage_id}")
    status = str(status or "").strip().upper()
    if status not in _ALLOWED_STAGE_STATUS:
        raise QualificationError("COMPILE_STAGE_INVALID_STATUS")
    existing_by_slot = {b.slot_id: b for b in transaction.artifact_bindings}
    consumed: list[str] = []
    for slot in consumed_slots:
        slot = str(slot)
        try:
            consumed.append(existing_by_slot[slot].binding_hash)
        except KeyError as exc:
            raise QualificationError(f"COMPILE_STAGE_UNKNOWN_INPUT_SLOT:{slot}") from exc
    produced = tuple(produced_bindings)
    if status == "PASS" and not produced:
        raise QualificationError("COMPILE_STAGE_PASS_REQUIRES_EXACT_OUTPUT_BINDING")
    next_bindings = list(transaction.artifact_bindings)
    for binding in produced:
        validate_artifact_binding(binding)
        if binding.producer_stage != stage_id:
            raise QualificationError("COMPILE_ARTIFACT_PRODUCER_STAGE_MISMATCH")
        if binding.slot_id in existing_by_slot:
            raise QualificationError(f"COMPILE_ARTIFACT_SLOT_REBIND_FORBIDDEN:{binding.slot_id}")
        next_bindings.append(binding)
        existing_by_slot[binding.slot_id] = binding
    _validate_unique_slots(next_bindings)
    blocker_tuple = tuple(str(x) for x in blockers if str(x))
    if status in {"FAIL", "ABSTAIN"} and not blocker_tuple:
        raise QualificationError("COMPILE_TERMINAL_STAGE_REQUIRES_BLOCKER")
    parent_stage_hash = transaction.stage_results[-1].stage_result_hash if transaction.stage_results else ""
    diagnostics_hash = content_sha256(dict(diagnostics or {}))
    stage = CompileStageResultIR(
        stage_id=stage_id,
        status=status,
        request_hash=transaction.request.request_hash,
        parent_stage_result_hash=parent_stage_hash,
        consumed_binding_hashes=tuple(consumed),
        produced_binding_hashes=tuple(b.binding_hash for b in produced),
        policy_hash=expected.policy_hash,
        implementation_binding_hash=expected.implementation_binding_hash,
        diagnostics_hash=diagnostics_hash,
        blockers=blocker_tuple,
        stage_result_hash="",
        metadata=dict(metadata or {}),
    )
    stage = replace(stage, stage_result_hash=stage_result_hash(stage))
    results = transaction.stage_results + (stage,)
    if status in {"FAIL", "ABSTAIN"}:
        tx_status = status
        terminal_stage = stage_id
    elif len(results) == len(transaction.request.stage_contracts):
        tx_status = "PASS"
        terminal_stage = stage_id
    else:
        tx_status = "OPEN"
        terminal_stage = ""
    value = replace(
        transaction,
        stage_results=results,
        artifact_bindings=tuple(next_bindings),
        status=tx_status,
        terminal_stage_id=terminal_stage,
        transaction_hash="",
    )
    value = replace(value, transaction_hash=transaction_hash(value))
    validate_compile_transaction(value)
    return value


def validate_compile_transaction(value: CompileTransactionIR) -> None:
    validate_compile_request(value.request)
    if value.status not in _ALLOWED_TX_STATUS:
        raise QualificationError("COMPILE_TRANSACTION_INVALID_STATUS")
    if value.transaction_hash != transaction_hash(value):
        raise QualificationError("COMPILE_TRANSACTION_HASH_MISMATCH")
    _validate_unique_slots(value.artifact_bindings)
    bindings = {b.binding_hash: b for b in value.artifact_bindings}
    if len(value.stage_results) > len(value.request.stage_contracts):
        raise QualificationError("COMPILE_TRANSACTION_TOO_MANY_STAGE_RESULTS")
    prior_hash = ""
    saw_terminal = False
    for index, stage in enumerate(value.stage_results):
        expected = value.request.stage_contracts[index]
        if stage.stage_id != expected.stage_id:
            raise QualificationError("COMPILE_TRANSACTION_STAGE_ORDER_MISMATCH")
        if stage.status not in _ALLOWED_STAGE_STATUS:
            raise QualificationError("COMPILE_STAGE_INVALID_STATUS")
        if stage.request_hash != value.request.request_hash:
            raise QualificationError("COMPILE_STAGE_REQUEST_HASH_MISMATCH")
        if stage.parent_stage_result_hash != prior_hash:
            raise QualificationError("COMPILE_STAGE_PARENT_HASH_MISMATCH")
        if stage.policy_hash != expected.policy_hash:
            raise QualificationError("COMPILE_STAGE_POLICY_MUTATION_FORBIDDEN")
        if stage.implementation_binding_hash != expected.implementation_binding_hash:
            raise QualificationError("COMPILE_STAGE_IMPLEMENTATION_MUTATION_FORBIDDEN")
        if stage.stage_result_hash != stage_result_hash(stage):
            raise QualificationError("COMPILE_STAGE_RESULT_HASH_MISMATCH")
        if any(h not in bindings for h in (*stage.consumed_binding_hashes, *stage.produced_binding_hashes)):
            raise QualificationError("COMPILE_STAGE_REFERENCES_UNKNOWN_BINDING")
        if stage.status == "PASS" and not stage.produced_binding_hashes:
            raise QualificationError("COMPILE_STAGE_PASS_REQUIRES_EXACT_OUTPUT_BINDING")
        if stage.status in {"FAIL", "ABSTAIN"}:
            if not stage.blockers:
                raise QualificationError("COMPILE_TERMINAL_STAGE_REQUIRES_BLOCKER")
            if index != len(value.stage_results) - 1:
                raise QualificationError("COMPILE_TRANSACTION_HAS_STAGE_AFTER_TERMINAL")
            saw_terminal = True
        prior_hash = stage.stage_result_hash
    if saw_terminal:
        expected_status = value.stage_results[-1].status
        if value.status != expected_status or value.terminal_stage_id != value.stage_results[-1].stage_id:
            raise QualificationError("COMPILE_TRANSACTION_TERMINAL_STATUS_MISMATCH")
    elif len(value.stage_results) == len(value.request.stage_contracts):
        if value.status != "PASS":
            raise QualificationError("COMPILE_TRANSACTION_COMPLETE_BUT_NOT_PASS")
    elif value.status != "OPEN":
        raise QualificationError("COMPILE_TRANSACTION_INCOMPLETE_BUT_NOT_OPEN")


def bind_current_product_v3(product: Any) -> tuple[CompileArtifactBindingIR, ...]:
    from .v4 import validate_product_ontology

    validate_product_ontology(product)
    bindings: list[CompileArtifactBindingIR] = [
        make_artifact_binding(
            slot_id="S",
            authority_class="QUALIFIED_GEOMETRY",
            artifact_schema=product.mechanical_state.surface.schema_version,
            artifact_id=f"S:{product.mechanical_state.surface.geometry_lineage_hash[:24]}",
            content_hash=product.mechanical_state.surface.geometry_lineage_hash,
            producer_stage="SURFACE",
        ),
        make_artifact_binding(
            slot_id="G",
            authority_class="QUALIFIED_SKELETON",
            artifact_schema=product.mechanical_state.skeleton.schema_version,
            artifact_id=f"G:{product.mechanical_state.skeleton.skeleton_lineage_hash[:24]}",
            content_hash=product.mechanical_state.skeleton.skeleton_lineage_hash,
            producer_stage="SKELETON",
        ),
        make_artifact_binding(
            slot_id="W",
            authority_class="QUALIFIED_SKIN",
            artifact_schema=product.mechanical_state.skin.schema_version,
            artifact_id=f"W:{product.mechanical_state.skin.skin_lineage_hash[:24]}",
            content_hash=product.mechanical_state.skin.skin_lineage_hash,
            producer_stage="SKIN",
        ),
        make_artifact_binding(
            slot_id="DIRECTIONAL_VISUAL",
            authority_class="CANONICAL_DIRECTIONAL_VISUAL_STATE",
            artifact_schema=product.directional_renderables.schema_version,
            artifact_id=f"DV:{product.directional_visual_state_hash[:24]}",
            content_hash=product.directional_visual_state_hash,
            producer_stage="COMPONENTS",
            source_product_state_hash=product.product_state_hash,
        ),
        make_artifact_binding(
            slot_id="MOTION_STATE",
            authority_class="CANONICAL_MOTION_STATE",
            artifact_schema=product.motion_state.schema_version,
            artifact_id=f"MOTION:{product.motion_state_hash[:24]}",
            content_hash=product.motion_state_hash,
            producer_stage="MOTION",
            source_product_state_hash=product.product_state_hash,
        ),
        make_artifact_binding(
            slot_id="PRODUCT",
            authority_class="CANONICAL_PRODUCT_STATE",
            artifact_schema=product.schema_version,
            artifact_id=product.product_lineage_id,
            content_hash=product.product_state_hash,
            producer_stage="MOTION",
            source_product_state_hash=product.product_state_hash,
        ),
    ]
    for direction in product.directional_renderables.directions:
        for component in direction.components:
            suffix = f"view={direction.view_index}:component={component.component_id}"
            bindings.extend((
                make_artifact_binding(
                    slot_id=f"M:{suffix}",
                    authority_class="QUALIFIED_EDITABLE_MESH",
                    artifact_schema=component.mesh.schema_version,
                    artifact_id=f"M:{component.mesh.mesh_lineage_hash[:24]}",
                    content_hash=component.mesh.mesh_lineage_hash,
                    producer_stage="DIRECTIONAL_MESH",
                    source_product_state_hash=product.product_state_hash,
                    view_index=direction.view_index,
                    component_id=component.component_id,
                ),
                make_artifact_binding(
                    slot_id=f"B:{suffix}",
                    authority_class="QUALIFIED_MESH_SKIN",
                    artifact_schema=component.mesh_skin.schema_version,
                    artifact_id=f"B:{component.mesh_skin.mesh_skin_lineage_hash[:24]}",
                    content_hash=component.mesh_skin.mesh_skin_lineage_hash,
                    producer_stage="MESH_SKIN",
                    source_product_state_hash=product.product_state_hash,
                    view_index=direction.view_index,
                    component_id=component.component_id,
                    metadata={"mesh_binding_hash": component.mesh_skin.mesh_binding_hash},
                ),
                make_artifact_binding(
                    slot_id=f"COMPONENT:{suffix}",
                    authority_class="CANONICAL_RENDERABLE_COMPONENT",
                    artifact_schema=component.schema_version,
                    artifact_id=f"COMP:{component.component_state_hash[:24]}",
                    content_hash=component.component_state_hash,
                    producer_stage="COMPONENTS",
                    source_product_state_hash=product.product_state_hash,
                    view_index=direction.view_index,
                    component_id=component.component_id,
                ),
            ))
    _validate_unique_slots(bindings)
    return tuple(bindings)


def bind_product_proof_bundle(proof_bundle: Any) -> CompileArtifactBindingIR:
    from .v4 import proof_bundle_hash

    if proof_bundle.proof_bundle_hash != proof_bundle_hash(proof_bundle):
        raise QualificationError("COMPILE_PROOF_BUNDLE_HASH_MISMATCH")
    return make_artifact_binding(
        slot_id="PROOF_BUNDLE",
        authority_class="DERIVED_PRODUCT_PROOF_BUNDLE",
        artifact_schema=proof_bundle.schema_version,
        artifact_id=f"PROOF:{proof_bundle.proof_bundle_hash[:24]}",
        content_hash=proof_bundle.proof_bundle_hash,
        producer_stage="PROOF",
        source_product_state_hash=proof_bundle.source_product_state_hash,
    )


def bind_runtime_package(runtime: RuntimePackageIR) -> CompileArtifactBindingIR:
    content_hash = content_sha256(runtime.to_dict())
    return make_artifact_binding(
        slot_id="RUNTIME_PACKAGE",
        authority_class="RUNTIME_PROJECTION",
        artifact_schema=runtime.schema_version,
        artifact_id=f"RUNTIME:{content_hash[:24]}",
        content_hash=content_hash,
        producer_stage="RUNTIME_EXPORT",
        source_product_state_hash=runtime.source_product_state_hash,
        metadata={
            "source_proof_hash": runtime.source_proof_hash,
            "abi_schema_version": runtime.abi_schema_version,
        },
    )


def seal_compile_result(transaction: CompileTransactionIR) -> CompileResultIR:
    validate_compile_transaction(transaction)
    blockers: list[str] = []
    if transaction.status != "PASS":
        blockers.append("transaction_not_pass")
    by_slot = {b.slot_id: b for b in transaction.artifact_bindings}
    for slot in transaction.request.required_final_slots:
        if slot not in by_slot:
            blockers.append(f"missing_required_final_slot:{slot}")
    product = by_slot.get("PRODUCT")
    proof = by_slot.get("PROOF_BUNDLE")
    runtime = by_slot.get("RUNTIME_PACKAGE")
    product_hash = product.content_hash if product else ""
    proof_hash = proof.content_hash if proof else ""
    runtime_hash = runtime.content_hash if runtime else ""
    if product and product.source_product_state_hash not in {"", product_hash}:
        blockers.append("product_self_identity_mismatch")
    if proof and proof.source_product_state_hash != product_hash:
        blockers.append("proof_product_identity_mismatch")
    if runtime and runtime.source_product_state_hash != product_hash:
        blockers.append("runtime_product_identity_mismatch")
    if runtime and str(runtime.metadata.get("source_proof_hash", "")) != proof_hash:
        blockers.append("runtime_proof_identity_mismatch")
    status = "PASS" if not blockers else "FAIL"
    value = CompileResultIR(
        transaction_hash=transaction.transaction_hash,
        request_hash=transaction.request.request_hash,
        status=status,
        product_state_hash=product_hash,
        proof_bundle_hash=proof_hash,
        runtime_package_hash=runtime_hash,
        output_binding_hashes=tuple(sorted(b.binding_hash for b in transaction.artifact_bindings)),
        blockers=tuple(blockers),
        result_hash="",
        metadata={
            "exact_identity_chain_required": True,
            "latest_path_selection_forbidden": True,
            "hidden_retriangulation_forbidden": True,
            "implicit_repair_credit_forbidden": True,
        },
    )
    value = replace(value, result_hash=compile_result_hash(value))
    if value.status != "PASS":
        raise QualificationError("COMPILE_RESULT_SEAL_FAILED:" + ",".join(value.blockers))
    return value
