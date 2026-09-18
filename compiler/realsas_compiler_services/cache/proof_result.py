from __future__ import annotations

"""Fail-closed content-addressed cache for current product proof + owned motion bakes.

The cache is an optimization only.  A HIT reconstructs typed IR and re-runs the
same structural/hash validators consumed by export.  Cache identity binds the
exact product, evaluator/provider identity, thresholds/fixtures, and source bytes
of the proof implementation.  Any uncertainty becomes a MISS.
"""

from dataclasses import dataclass, is_dataclass, asdict
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping
import json
import math
import zlib

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.proof_engine import evaluate_product_proof
from compiler.realsas_compiler_core.v4 import require_current_proof_bundle
from compiler.realsas_compiler_core.v4_types import DomainProofReportIR, ProductProofBundleIR
from compiler.realsas_compiler_services.proof.directional_motion_provider import QualifiedDirectionalMotionBakeProviderV1
from compiler.realsas_compiler_services.proof.motion_bake import (
    QualificationOwnedMotionBakeIR,
    QualificationOwnedMotionFrameIR,
    assert_motion_bake_binding,
)
from .content_addressed import ContentAddressedStageCache, sha256_file

PROOF_RESULT_CACHE_SCHEMA = "RealSaS.ProofResultCache.v1"
PROOF_RESULT_CACHE_STAGE = "product-proof-v1"

# Conservative dependency set. A source-byte change invalidates all proof cache
# entries even when it would not change semantics. False misses are acceptable;
# stale proof reuse is not.
_PROOF_SOURCE_MODULES = (
    "compiler.realsas_compiler_core.proof_engine",
    "compiler.realsas_compiler_core.v4",
    "compiler.realsas_compiler_core.deformation",
    "compiler.realsas_compiler_core.mesh.quality",
    "compiler.realsas_compiler_core.product_external_render",
    "compiler.realsas_compiler_services.proof.failure_signatures",
    "compiler.realsas_compiler_services.proof.motion_bake",
    "compiler.realsas_compiler_services.proof.motion_frame_metrics",
    "compiler.realsas_compiler_services.proof.directional_motion_provider",
    "compiler.realsas_compiler_services.proof.directional_motion_evaluator",
)


@dataclass(frozen=True)
class CachedProofResult:
    proof_bundle: ProductProofBundleIR
    motion_bakes: tuple[QualificationOwnedMotionBakeIR, ...]
    cache_hit: bool
    cache_key: str
    cache_reason: str
    artifact_sha256: str


def _source_fingerprint() -> tuple[str, dict[str, str]]:
    rows: dict[str, str] = {}
    for name in _PROOF_SOURCE_MODULES:
        module = import_module(name)
        source = getattr(module, "__file__", None)
        if not source:
            raise RuntimeError(f"PROOF_CACHE_SOURCE_IDENTITY_UNAVAILABLE:{name}")
        path = Path(source).resolve()
        if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        if not path.is_file():
            raise RuntimeError(f"PROOF_CACHE_SOURCE_FILE_MISSING:{name}")
        rows[name] = sha256_file(path)
    producer = content_sha256({
        "schema": PROOF_RESULT_CACHE_SCHEMA,
        "source_sha256": rows,
    })
    return producer, rows


def _exact_identity(value: Any) -> Any:
    """Canonical identity without repr-based fallbacks or truncated ndarray text."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("PROOF_CACHE_NONFINITE_INPUT")
        return value
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        payload = value.tobytes(order="C")
        return {
            "array_dtype": str(value.dtype),
            "array_shape": [int(x) for x in value.shape],
            "array_sha256": sha256(payload).hexdigest(),
            "array_bytes": len(payload),
        }
    if hasattr(value, "to_dict"):
        return _exact_identity(value.to_dict())
    if is_dataclass(value):
        return _exact_identity(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _exact_identity(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_exact_identity(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_exact_identity(v) for v in value), key=lambda row: json.dumps(row, sort_keys=True))
    raise TypeError(f"PROOF_CACHE_UNSUPPORTED_INPUT_IDENTITY:{type(value).__name__}")


def _provider_identity(provider) -> dict[str, Any] | None:
    if provider is None:
        return None
    if not isinstance(provider, QualifiedDirectionalMotionBakeProviderV1):
        # Preserve proof engine fail-closed semantics instead of inventing an
        # identity for arbitrary callables.
        raise TypeError("PROOF_CACHE_REQUIRES_TYPED_MOTION_PROVIDER")
    return {
        "schema_version": provider.schema_version,
        "provider_hash": provider.provider_hash,
        "source_product_state_hash": provider.source_product_state_hash,
        "directional_binding_set_hash": provider.directional_binding_set_hash,
        "evaluator_policy_hash": provider.evaluator_policy_hash,
        "evaluator_semantic_version": provider.evaluator_semantic_version,
    }


def _frame_from_dict(row: Mapping[str, Any]) -> QualificationOwnedMotionFrameIR:
    meshes = tuple(
        (str(mid), tuple((float(p[0]), float(p[1])) for p in points))
        for mid, points in row.get("mesh_vertices_by_id", ())
    )
    orders = tuple(
        (str(view), tuple(map(str, order)))
        for view, order in row.get("render_order_by_view", ())
    )
    return QualificationOwnedMotionFrameIR(
        time_seconds=float(row["time_seconds"]),
        mesh_vertices_by_id=meshes,
        render_order_by_view=orders,
        semantic_sha256=str(row["semantic_sha256"]),
        metadata=dict(row.get("metadata") or {}),
        schema_version=str(row.get("schema_version") or "RealSaS.QualificationOwnedMotionFrameIR.v1"),
    )


def _bake_from_dict(row: Mapping[str, Any]) -> QualificationOwnedMotionBakeIR:
    rest = tuple(
        (str(mid), tuple((float(p[0]), float(p[1])) for p in points))
        for mid, points in row.get("rest_mesh_vertices_by_id", ())
    )
    triangles = tuple(
        (str(mid), tuple(tuple(map(int, tri)) for tri in tris))
        for mid, tris in row.get("triangles_by_mesh_id", ())
    )
    return QualificationOwnedMotionBakeIR(
        source_product_state_hash=str(row["source_product_state_hash"]),
        proof_plan_hash=str(row["proof_plan_hash"]),
        clip_id=str(row["clip_id"]),
        duration_seconds=float(row["duration_seconds"]),
        fps=float(row["fps"]),
        loop=bool(row["loop"]),
        evaluator_semantic_version=str(row["evaluator_semantic_version"]),
        evaluator_binding_hash=str(row["evaluator_binding_hash"]),
        sampling_policy=str(row["sampling_policy"]),
        rest_mesh_vertices_by_id=rest,
        triangles_by_mesh_id=triangles,
        frames=tuple(_frame_from_dict(frame) for frame in row.get("frames", ())),
        bake_hash=str(row["bake_hash"]),
        metadata=dict(row.get("metadata") or {}),
        schema_version=str(row.get("schema_version") or "RealSaS.QualificationOwnedMotionBakeIR.v1"),
    )


def _domain_from_dict(row: Mapping[str, Any]) -> DomainProofReportIR:
    return DomainProofReportIR(
        source_product_state_hash=str(row["source_product_state_hash"]),
        proof_domain=str(row["proof_domain"]),
        proof_plan_hash=str(row["proof_plan_hash"]),
        measurement_report_hash=str(row["measurement_report_hash"]),
        status=str(row["status"]),
        failure_signatures=tuple(dict(x) for x in row.get("failure_signatures", ())),
        owner_attribution=tuple(dict(x) for x in row.get("owner_attribution", ())),
        domain_proof_hash=str(row["domain_proof_hash"]),
        schema_version=str(row.get("schema_version") or "RealSaS.DomainProofReportIR.v1"),
        metadata=dict(row.get("metadata") or {}),
    )


def _bundle_from_dict(row: Mapping[str, Any]) -> ProductProofBundleIR:
    return ProductProofBundleIR(
        source_product_state_hash=str(row["source_product_state_hash"]),
        required_domains=tuple(map(str, row.get("required_domains", ()))),
        domain_reports=tuple(_domain_from_dict(x) for x in row.get("domain_reports", ())),
        overall_status=str(row["overall_status"]),
        proof_bundle_hash=str(row["proof_bundle_hash"]),
        schema_version=str(row.get("schema_version") or "RealSaS.ProductProofBundleIR.v1"),
        metadata=dict(row.get("metadata") or {}),
    )


def _validate_restored(product, bundle: ProductProofBundleIR, bakes: tuple[QualificationOwnedMotionBakeIR, ...]) -> None:
    require_current_proof_bundle(product, bundle, require_pass=False)
    motion_reports = tuple(row for row in bundle.domain_reports if row.proof_domain == "MOTION")
    if not motion_reports:
        if bakes:
            raise ValueError("PROOF_CACHE_UNEXPECTED_MOTION_BAKES")
        return
    if len(motion_reports) != 1:
        raise ValueError("PROOF_CACHE_MOTION_REPORT_CARDINALITY")
    report = motion_reports[0]
    expected = dict((report.metadata or {}).get("qualification_owned_bake_hashes") or {})
    by_clip = {bake.clip_id: bake for bake in bakes}
    if len(by_clip) != len(bakes) or set(by_clip) != set(expected):
        raise ValueError("PROOF_CACHE_MOTION_BAKE_SET_MISMATCH")
    for clip_id, bake in by_clip.items():
        assert_motion_bake_binding(
            bake,
            source_product_state_hash=product.product_state_hash,
            proof_plan_hash=report.proof_plan_hash,
        )
        if expected[clip_id] != bake.bake_hash:
            raise ValueError("PROOF_CACHE_MOTION_BAKE_HASH_MISMATCH")


def _encode(bundle: ProductProofBundleIR, bakes: tuple[QualificationOwnedMotionBakeIR, ...]) -> bytes:
    payload = {
        "schema": PROOF_RESULT_CACHE_SCHEMA,
        "proof_bundle": bundle.to_dict(),
        "motion_bakes": [bake.to_dict() for bake in sorted(bakes, key=lambda row: row.clip_id)],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return zlib.compress(raw, level=1)


def _decode(payload: bytes) -> tuple[ProductProofBundleIR, tuple[QualificationOwnedMotionBakeIR, ...]]:
    try:
        value = json.loads(zlib.decompress(payload).decode("utf-8"))
    except Exception as exc:
        raise ValueError("PROOF_CACHE_PAYLOAD_DECODE_FAILED") from exc
    if not isinstance(value, dict) or value.get("schema") != PROOF_RESULT_CACHE_SCHEMA:
        raise ValueError("PROOF_CACHE_PAYLOAD_SCHEMA_MISMATCH")
    bundle = _bundle_from_dict(value["proof_bundle"])
    bakes = tuple(_bake_from_dict(row) for row in value.get("motion_bakes", ()))
    return bundle, bakes


def evaluate_product_proof_cached(
    product,
    *,
    cache_root: str | Path,
    deformation_fixture: dict | None = None,
    motion_bake_provider=None,
    motion_policy: dict | None = None,
    artifacts_out: dict | None = None,
) -> CachedProofResult:
    producer, source_hashes = _source_fingerprint()
    inputs = {
        "product_state_hash": str(product.product_state_hash),
        "deformation_fixture": _exact_identity(deformation_fixture),
        "motion_provider": _provider_identity(motion_bake_provider),
        "motion_policy": _exact_identity(dict(motion_policy or {})),
    }
    cache = ContentAddressedStageCache(
        cache_root,
        stage=PROOF_RESULT_CACHE_STAGE,
        producer_fingerprint=producer,
    )
    key = cache.key(inputs)

    with TemporaryDirectory(prefix="realsas_proof_cache_") as temp_dir:
        artifact_path = Path(temp_dir) / "proof_result.zlib"
        restored = cache.restore(key, artifact_path)
        if restored.hit:
            try:
                bundle, bakes = _decode(artifact_path.read_bytes())
                _validate_restored(product, bundle, bakes)
            except Exception:
                # Content cache verified the bytes, but semantic validation still
                # owns authority. Never return a semantically stale/corrupt HIT.
                bundle = None
            else:
                if artifacts_out is not None:
                    artifacts_out.setdefault("motion_bakes", {}).update({b.clip_id: b for b in bakes})
                return CachedProofResult(
                    bundle,
                    bakes,
                    True,
                    key,
                    restored.reason,
                    str(restored.artifact_sha256),
                )

        local_artifacts: dict[str, Any] = {}
        bundle = evaluate_product_proof(
            product,
            deformation_fixture=deformation_fixture,
            motion_bake_provider=motion_bake_provider,
            motion_policy=motion_policy,
            artifacts_out=local_artifacts,
        )
        bakes = tuple(sorted((local_artifacts.get("motion_bakes") or {}).values(), key=lambda row: row.clip_id))
        _validate_restored(product, bundle, bakes)
        artifact_path.write_bytes(_encode(bundle, bakes))
        stored = cache.store(
            key,
            artifact_path,
            metadata={
                "schema": PROOF_RESULT_CACHE_SCHEMA,
                "product_state_hash": str(product.product_state_hash),
                "proof_bundle_hash": bundle.proof_bundle_hash,
                "overall_status": bundle.overall_status,
                "motion_bake_hashes": {b.clip_id: b.bake_hash for b in bakes},
                "proof_source_sha256": source_hashes,
                "inputs_identity": inputs,
            },
        )
        if artifacts_out is not None:
            artifacts_out.setdefault("motion_bakes", {}).update({b.clip_id: b for b in bakes})
        return CachedProofResult(
            bundle,
            bakes,
            False,
            key,
            "MISS_EVALUATED_AND_STORED",
            stored.artifact_sha256,
        )
