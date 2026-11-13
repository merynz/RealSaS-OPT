from __future__ import annotations

"""One RSS.vNext checkpoint -> immutable multi-consumer Compiler evidence CAS.

This is the real *candidate* replacement executor for product.shape. Unlike
historical Shape v2 it NEVER asks for separate IRIS/TESSA model releases,
eight-view DINO ShapeInput, a teacher mesh or a fallback topology grid.

One original front/back canonical source and ONE unified RSS checkpoint are
mandatory. ABSTAIN means the neural mesh was unavailable, NOT that Semantic,
Graphics, source observations and Mechanics priors can be thrown away.

This executor is callable on its own before the registered Go DAG is migrated.
It grants neither CanonicalMesh/ProductRevision nor training qualification.
"""

from hashlib import sha256
import json
import os
import re
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Callable

from compiler.realsas_compiler_services.platform_worker.model_inputs import (
    materialize_model_releases,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    ARTIFACT_TYPE as RSS_SOURCE_ARTIFACT,
    SCHEMA_VERSION as RSS_SOURCE_VERSION,
    validate_rss_vnext_source_ref_v1,
)

EXECUTOR_CONTRACT = "RealSaS.ProductShapeInference.v3"
MODEL_FAMILY = "RSSVNEXT"
MODEL_ARCHITECTURE = "RSS.vNext.UnifiedCharacterCompiler.v1"
CHECKPOINT_ARTIFACT_TYPE = "RealSaS.RSSvNextUnifiedInferenceCheckpoint"
CHECKPOINT_SCHEMA_VERSION = "v1"
FANOUT_SCHEMA = "RealSaS.RSSvNext.ProductEvidenceFanout.v1"
_RSS_RECEIPT_FILES = (
    ("original_source", "original_two_view_source.npz",
     "RealSaS.RSSvNextOriginalSource", "v1",
     "ORIGINAL_OBSERVED_RGBA_AND_CAMERAS_NO_RIGHTS_AUTHORITY"),
    ("normalized_model_source", "observed_source_tensors.npz",
     "RealSaS.RSSvNextModelSourceTensors", "v1",
     "DETERMINISTIC_TWO_VIEW_SOURCE_NORMALIZATION_ONLY"),
    ("neural_evidence", "neural_evidence.json",
     "RealSaS.RSSvNextInferenceEvidence", "v1",
     "LEARNED_INFERENCE_CANDIDATES_ONLY"),
    ("neural_scene", "rss_vnext_neural_scene.npz",
     "RealSaS.RSSvNextNeuralSceneCarrier", "v1",
     "PERSISTED_SHARED_EXPERT_CARRIER_NO_PRODUCT_AUTHORITY"),
    ("inference_receipt", "inference_receipt.json",
     "RealSaS.RSSvNextInferenceReceipt", "v2",
     "PINNED_SOURCE_AND_UNIFIED_CHECKPOINT_PROVENANCE"),
    ("unified_model_config", "rss_vnext_unified_model_config.json",
     "RealSaS.RSSvNextUnifiedModelConfig", "v1",
     "EXACT_CODEC_AND_SPECIALIST_ARCHITECTURE_NO_PRODUCT_AUTHORITY"),
    ("embedded_mira_decoder", "rss_vnext_embedded_mira_decoder.pt",
     "RealSaS.RSSvNextEmbeddedMIRADecoder", "v1",
     "EXACT_UNIFIED_MODEL_SUBMODULE_WEIGHTS_NOT_SKIN_AUTHORITY"),
)


def _hex64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _one_canonical_source(request: dict[str, Any]) -> dict[str, Any]:
    # Do not accept any legacy ShapeInput or eight-source artifact. This
    # executor cannot silently choose a front/back pair from historical data.
    refs = list(request.get("input_artifacts") or ())
    matches = [
        dict(ref) for ref in refs
        if ref.get("artifact_type") == RSS_SOURCE_ARTIFACT
        and ref.get("schema_version") == RSS_SOURCE_VERSION
    ]
    # Go binds the goal's original source and the byte-identical result of
    # product.input as separate *semantic* artifacts. Both may appear here
    # with different artifact IDs despite identical CAS bytes. This is a valid
    # producer-consumer edge only if their exact immutable bytes agree.
    # Standalone compiler invocations supply just one.
    if len(matches) not in (1, 2):
        raise RuntimeError("RSS_VNEXT_SHAPE_TWO_VIEW_SOURCE_CARDINALITY")
    if len(matches) == 2 and any(
        row.get(key) != matches[0].get(key)
        for row in matches[1:] for key in (
            "content_sha256", "size_bytes", "storage_key",
        )
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_GO_INPUT_SOURCE_BYTES_DRIFT")
    if any(
        ref.get("artifact_type") in {
            "RealSaS.SourceObservation", "RealSaS.ShapeInput",
            "RealSaS.ShapeProposal",
        }
        for ref in refs
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_LEGACY_SOURCE_FORBIDDEN")
    return matches[0]


def _inference_device() -> str:
    import torch

    raw = os.environ.get("REALSAS_INFERENCE_DEVICE", "cpu").strip() or "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RSS_VNEXT_SHAPE_CUDA_UNAVAILABLE")
    return str(device)


def _production_vision_factory():
    from models.rss_vnext.unified_character_model import RSSvNextFrozenDINOv2

    raw = os.environ.get("REALSAS_RSS_DINOV2_SOURCE", "").strip()
    if not raw:
        root = os.environ.get("REALSAS_DEPLOY_ROOT", "").strip()
        if not root:
            raise RuntimeError("RSS_VNEXT_SHAPE_LOCAL_DINOV2_REQUIRED")
        raw = str(Path(root) / "foundations" / "dinov2" / "rss_vnext_pretrained")
    source = Path(raw).expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError("RSS_VNEXT_SHAPE_LOCAL_DINOV2_NOT_FOUND")
    return RSSvNextFrozenDINOv2(str(source))


def _output_ref(
    filename: str,
    *,
    role: str,
    artifact_type: str,
    schema_version: str,
    authority: str,
    directory: Path,
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    data = (directory / filename).read_bytes()
    sealed = put_bytes(data)
    if (
        sealed.get("content_sha256") != sha256(data).hexdigest()
        or sealed.get("size_bytes") != len(data)
        or not sealed.get("storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_CAS_WRITE_RECEIPT_DRIFT")
    return {
        "role": role,
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": authority,
    }


def execute_rss_vnext_shape_capability_v3(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
    # A Python-only test seam, never read from request or Platform metadata.
    # If set, uses the exact inference pipeline with product_mode=False; this
    # is explicitly forbidden as a production inference qualification.
    test_vision_encoder_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    meta = dict(request.get("capability_metadata") or {})
    if (
        str(request.get("kind") or "") != "MODULE"
        or meta.get("executor_contract") != EXECUTOR_CONTRACT
        or str(request.get("capability_id") or "") != "product.shape"
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_CAPABILITY_CONTRACT_DRIFT")
    execution_id = str(request.get("execution_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", execution_id):
        raise RuntimeError("RSS_VNEXT_SHAPE_EXECUTION_ID_INVALID")
    if dict(request.get("goal_parameters") or {}).get("product_mode") is False:
        raise RuntimeError("RSS_VNEXT_SHAPE_CALLER_CANNOT_DISABLE_PRODUCT_MODE")
    max_tokens = meta.get("maximum_new_tokens")
    if type(max_tokens) is not int or not 2 <= max_tokens <= 131072:
        raise RuntimeError("RSS_VNEXT_SHAPE_TOPOLOGY_BUDGET_INVALID")

    source_ref = _one_canonical_source(request)
    raw_source = read_object(source_ref)
    source = validate_rss_vnext_source_ref_v1(source_ref, raw_source)

    # Reject deprecated independent IRIS/TESSA releases BEFORE materializing
    # any checkpoint. There must be exactly one unified model identity.
    declared_models = list(request.get("model_releases") or ())
    if (
        len(declared_models) != 1
        or declared_models[0].get("family") != MODEL_FAMILY
        or declared_models[0].get("architecture_id") != MODEL_ARCHITECTURE
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_ONE_UNIFIED_MODEL_RELEASE_REQUIRED")
    manifest = {"run_id": "rss-vnext-" + execution_id}
    released = materialize_model_releases(
        manifest=manifest, models=request.get("model_releases") or (),
        read_object=read_object,
    )
    if (
        len(released) != 1
        or released[0].get("family") != MODEL_FAMILY
        or released[0].get("architecture_id") != MODEL_ARCHITECTURE
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_ONE_UNIFIED_MODEL_RELEASE_REQUIRED")
    release = released[0]
    checkpoint = dict(release.get("checkpoint") or {})
    expected_checkpoint_sha = checkpoint.get("content_sha256")
    if (
        checkpoint.get("artifact_type") != CHECKPOINT_ARTIFACT_TYPE
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or not _hex64(expected_checkpoint_sha)
        or not Path(checkpoint.get("path") or "").is_file()
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_CHECKPOINT_RELEASE_ABI_DRIFT")
    if test_vision_encoder_factory is None:
        vision = _production_vision_factory()
        product_mode = True
    else:
        vision = test_vision_encoder_factory()
        product_mode = False

    from models.rss_vnext.inference_pipeline_v1 import (
        run_canonical_two_view_to_compiler_proposal_v1,
        read_compiler_proposal_receipt_v1,
    )

    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="rss-vnext-product-shape-") as temp:
        root = Path(temp)
        original_receipt = run_canonical_two_view_to_compiler_proposal_v1(
            source_observation=source,
            checkpoint_path=checkpoint["path"],
            expected_checkpoint_sha256=expected_checkpoint_sha,
            vision_encoder=vision,
            output_directory=root,
            max_topology_tokens=max_tokens,
            generate_controls=True,
            product_mode=product_mode,
        )
        replayed, evidence = read_compiler_proposal_receipt_v1(
            root, expected_checkpoint_sha256=expected_checkpoint_sha,
        )
        if original_receipt != replayed:
            raise RuntimeError("RSS_VNEXT_SHAPE_INFERENCE_RECEIPT_DRIFT")
        if evidence.mesh_status not in {"ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE"}:
            raise RuntimeError("RSS_VNEXT_SHAPE_CANDIDATE_STATUS_INVALID")
        if (
            replayed.get("canonical_source_binding_sha256") != source.source_binding_sha256
            or replayed.get("source_rights_verified") is not False
            or replayed.get("product_revision_minted") is not False
        ):
            raise RuntimeError("RSS_VNEXT_SHAPE_ORIGINAL_SOURCE_AUTHORITY_DRIFT")
        refs = [
            _output_ref(
                name, role=role, artifact_type=artifact, schema_version=version,
                authority=authority, directory=root, put_bytes=put_bytes,
            )
            for role, name, artifact, version, authority in _RSS_RECEIPT_FILES
        ]
    # The index is smaller than neural scene, independently addressable and
    # binds all artifacts. Mesh/Graphics/Mechanics can load their OWN CAS inputs
    # without requiring prior consumers or dropping higher-dimensional H(x).
    index = {
        "schema": FANOUT_SCHEMA,
        "source_artifact_sha256": source_ref["content_sha256"],
        "source_binding_sha256": source.source_binding_sha256,
        "source_rights_claim_sha256": source.source_rights_claim_sha256,
        "unified_model_release_behavior_sha256": release["behavior_identity_sha256"],
        "unified_checkpoint_sha256": expected_checkpoint_sha,
        "unified_model_config_sha256": replayed["unified_model_config_sha256"],
        "evidence_hash": evidence.evidence_hash,
        "mesh_status": evidence.mesh_status,
        "original_view_count": 2,
        "original_source_rights_verified": False,
        "learned_evidence_is_not_source_truth": True,
        "canonical_mesh_authority": False,
        "product_revision_minted": False,
        "topology_abstain_retains_shared_carrier": True,
        "children": [
            {k: item[k] for k in (
                "role", "artifact_type", "schema_version", "content_sha256",
                "size_bytes",
            )}
            for item in refs
        ],
    }
    data = json.dumps(
        index, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    # Producer runs the SAME cheap structural court as the independent
    # consumers; no misleading index can be emitted even if model succeeds.
    from .rss_vnext_product_fanout_v1 import (
        validate_rss_vnext_fanout_index_v1,
        bind_rss_vnext_fanout_refs_v1,
    )
    validated_index = validate_rss_vnext_fanout_index_v1(
        data,
        expected_source_content_sha256=source_ref["content_sha256"],
        expected_source_binding_sha256=source.source_binding_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha,
        expected_model_release_behavior_sha256=release["behavior_identity_sha256"],
    )
    bind_rss_vnext_fanout_refs_v1(validated_index, input_artifacts=refs)
    index_sealed = put_bytes(data)
    if (
        index_sealed.get("content_sha256") != sha256(data).hexdigest()
        or index_sealed.get("size_bytes") != len(data)
    ):
        raise RuntimeError("RSS_VNEXT_SHAPE_FANOUT_CAS_WRITE_DRIFT")
    refs.append({
        "role": "rss_evidence_fanout",
        "artifact_type": "RealSaS.RSSvNextProductEvidenceFanout",
        "schema_version": "v1",
        "storage_key": index_sealed["storage_key"],
        "content_sha256": index_sealed["content_sha256"],
        "size_bytes": index_sealed["size_bytes"],
        "authority_class": "INDEPENDENT_NEURAL_EVIDENCE_CAS_INDEX_ONLY",
    })
    return {
        "capability_id": "product.shape",
        "status": "PASS",
        "outputs": refs,
        "failure": None,
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductShapeDiagnostics.v1",
            "inference_checkpoint_sha256": expected_checkpoint_sha,
            "original_source_sha256": source_ref["content_sha256"],
            "neural_evidence_hash": evidence.evidence_hash,
            "mesh_status": evidence.mesh_status,
            "mesh_qualified": False,
            "product_revision_minted": False,
            "legacy_iris_tessa_model_releases_required": False,
            "unified_inference_count": 1,
            "original_observed_view_count": 2,
            "product_mode_test_only": not product_mode,
            "elapsed_seconds": float(perf_counter() - started),
        },
    }


__all__ = [
    "EXECUTOR_CONTRACT", "FANOUT_SCHEMA",
    "execute_rss_vnext_shape_capability_v3",
]
