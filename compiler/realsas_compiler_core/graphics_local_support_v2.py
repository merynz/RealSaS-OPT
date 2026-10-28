from __future__ import annotations

"""Executable sparse support for ray-local Graphics evidence.

`scope_hash` is provenance, not a spatial predicate.  This module binds a
qualified/declared ray record to one exact raster sample so a runtime consumer
can know *where* a local precedence rule is allowed to have a consequence.

The carrier is deliberately sparse and non-interpolating.  Missing samples are
UNKNOWN, never false and never permission to extrapolate a nearby rule.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

from .graphics_v2 import CompiledRayGraphicsV2
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationError(code)


def _hash(value: str, code: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), code)
    return value


@dataclass(frozen=True)
class ExactRasterRaySupportV2:
    record_hash: str
    camera_set_binding_hash: str
    sampling_contract_hash: str
    view_index: int
    pixel_x: int
    pixel_y: int
    resolution: int
    support_witness_hash: str
    support_kind: str = "EXACT_HALF_INTEGER_RASTER_SAMPLE"
    schema_version: str = "RealSaS.ExactRasterRaySupport.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)

    def validate(self) -> None:
        for value, code in (
            (self.record_hash, "GRAPHICS_V2_SUPPORT_RECORD_HASH_REQUIRED"),
            (self.camera_set_binding_hash, "GRAPHICS_V2_SUPPORT_CAMERA_HASH_REQUIRED"),
            (self.sampling_contract_hash, "GRAPHICS_V2_SUPPORT_SAMPLING_HASH_REQUIRED"),
            (self.support_witness_hash, "GRAPHICS_V2_SUPPORT_WITNESS_HASH_REQUIRED"),
        ):
            _hash(value, code)
        _require(type(self.view_index) is int and self.view_index >= 0, "GRAPHICS_V2_SUPPORT_VIEW_INVALID")
        _require(type(self.resolution) is int and self.resolution > 0, "GRAPHICS_V2_SUPPORT_RESOLUTION_INVALID")
        _require(type(self.pixel_x) is int and 0 <= self.pixel_x < self.resolution, "GRAPHICS_V2_SUPPORT_PIXEL_X_INVALID")
        _require(type(self.pixel_y) is int and 0 <= self.pixel_y < self.resolution, "GRAPHICS_V2_SUPPORT_PIXEL_Y_INVALID")
        _require(self.support_kind == "EXACT_HALF_INTEGER_RASTER_SAMPLE", "GRAPHICS_V2_SUPPORT_KIND_UNSUPPORTED")
        body = self.to_dict()
        body.pop("support_witness_hash")
        _require(content_sha256(body) == self.support_witness_hash, "GRAPHICS_V2_SUPPORT_WITNESS_DRIFT")


@dataclass(frozen=True)
class RuntimeLocalOrderRuleV2:
    record_hash: str
    support_witness_hash: str
    view_index: int
    pixel_x: int
    pixel_y: int
    resolution: int
    order_constraints: tuple[tuple[int, int], ...]
    rule_hash: str
    schema_version: str = "RealSaS.RuntimeLocalOrderRule.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompiledLocalOrderSupportV2:
    compiled_ray_graphics_hash: str
    camera_set_binding_hash: str
    sampling_contract_hash: str
    rules: tuple[RuntimeLocalOrderRuleV2, ...]
    observed_sample_count: int
    consequence_sample_count: int
    coverage_policy: str
    interpolation_authorized: bool
    unobserved_support_status: str
    product_activation_allowed: bool
    support_binding_hash: str
    schema_version: str = "RealSaS.CompiledLocalOrderSupport.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def make_exact_raster_support_v2(
    *,
    record_hash: str,
    camera_set_binding_hash: str,
    sampling_contract_hash: str,
    view_index: int,
    pixel_x: int,
    pixel_y: int,
    resolution: int,
    metadata: Json | None = None,
) -> ExactRasterRaySupportV2:
    body = {
        "record_hash": record_hash,
        "camera_set_binding_hash": camera_set_binding_hash,
        "sampling_contract_hash": sampling_contract_hash,
        "view_index": view_index,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "resolution": resolution,
        "support_kind": "EXACT_HALF_INTEGER_RASTER_SAMPLE",
        "schema_version": "RealSaS.ExactRasterRaySupport.v2",
        "metadata": dict(metadata or {}),
    }
    value = ExactRasterRaySupportV2(**body, support_witness_hash=content_sha256(body))
    value.validate()
    return value


def bind_compiled_ray_support_v2(
    *,
    compiled: CompiledRayGraphicsV2,
    supports: Iterable[ExactRasterRaySupportV2],
    camera_set_binding_hash: str,
    sampling_contract_hash: str,
) -> CompiledLocalOrderSupportV2:
    """Bind every measured ray to one exact runtime-evaluable raster sample.

    This function does not infer support from `ray_key` or `scope_hash`.  It also
    does not densify/interpolate sparse rules.  Every compiled ray must have one
    explicit support witness so absence remains distinguishable from false.
    """
    _hash(camera_set_binding_hash, "GRAPHICS_V2_SUPPORT_CAMERA_HASH_REQUIRED")
    _hash(sampling_contract_hash, "GRAPHICS_V2_SUPPORT_SAMPLING_HASH_REQUIRED")
    _hash(compiled.graph_hash, "GRAPHICS_V2_COMPILED_GRAPH_HASH_REQUIRED")

    ray_by_hash = {str(ray["record_hash"]): ray for ray in compiled.rays}
    _require(len(ray_by_hash) == len(compiled.rays), "GRAPHICS_V2_SUPPORT_DUPLICATE_COMPILED_RECORD")

    support_by_record: dict[str, ExactRasterRaySupportV2] = {}
    sample_owner: dict[tuple[int, int, int, int], str] = {}
    for support in supports:
        support.validate()
        _require(support.camera_set_binding_hash == camera_set_binding_hash, "GRAPHICS_V2_SUPPORT_CAMERA_BINDING_DRIFT")
        _require(support.sampling_contract_hash == sampling_contract_hash, "GRAPHICS_V2_SUPPORT_SAMPLING_BINDING_DRIFT")
        _require(support.record_hash in ray_by_hash, "GRAPHICS_V2_SUPPORT_RECORD_UNKNOWN")
        _require(support.record_hash not in support_by_record, "GRAPHICS_V2_SUPPORT_RECORD_DUPLICATE")
        ray = ray_by_hash[support.record_hash]
        _require(int(ray["view_index"]) == support.view_index, "GRAPHICS_V2_SUPPORT_VIEW_BINDING_DRIFT")
        sample = (support.view_index, support.pixel_x, support.pixel_y, support.resolution)
        previous = sample_owner.get(sample)
        _require(previous is None or previous == support.record_hash, "GRAPHICS_V2_SUPPORT_SAMPLE_CONFLICT")
        sample_owner[sample] = support.record_hash
        support_by_record[support.record_hash] = support

    _require(set(support_by_record) == set(ray_by_hash), "GRAPHICS_V2_EXECUTABLE_SUPPORT_REQUIRED")

    rules: list[RuntimeLocalOrderRuleV2] = []
    consequence_samples = 0
    for record_hash, ray in sorted(ray_by_hash.items()):
        support = support_by_record[record_hash]
        constraints = tuple(tuple(map(int, edge)) for edge in ray.get("order_constraints") or ())
        if not constraints:
            continue
        consequence_samples += 1
        body = {
            "record_hash": record_hash,
            "support_witness_hash": support.support_witness_hash,
            "view_index": support.view_index,
            "pixel_x": support.pixel_x,
            "pixel_y": support.pixel_y,
            "resolution": support.resolution,
            "order_constraints": constraints,
            "schema_version": "RealSaS.RuntimeLocalOrderRule.v2",
            "metadata": {"support_kind": support.support_kind},
        }
        rules.append(RuntimeLocalOrderRuleV2(**body, rule_hash=content_sha256(body)))

    payload = {
        "compiled_ray_graphics_hash": compiled.graph_hash,
        "camera_set_binding_hash": camera_set_binding_hash,
        "sampling_contract_hash": sampling_contract_hash,
        "rules": tuple(rule.to_dict() for rule in rules),
        "observed_sample_count": len(support_by_record),
        "consequence_sample_count": consequence_samples,
        "coverage_policy": "EXACT_SAMPLES_ONLY__NO_EXTRAPOLATION",
        "interpolation_authorized": False,
        "unobserved_support_status": "UNKNOWN",
        "product_activation_allowed": False,
        "schema_version": "RealSaS.CompiledLocalOrderSupport.v2",
        "metadata": {
            "scope_hash_is_provenance_not_spatial_authority": True,
            "ray_key_parsing_as_authority_forbidden": True,
        },
    }
    # Hash the unchanged portable representation, but keep the typed rules at
    # the Python consumer boundary. Serializing here used to return dictionaries
    # despite the declared RuntimeLocalOrderRuleV2 contract.
    return CompiledLocalOrderSupportV2(
        **{**payload, "rules": tuple(rules)},
        support_binding_hash=content_sha256(payload),
    )
