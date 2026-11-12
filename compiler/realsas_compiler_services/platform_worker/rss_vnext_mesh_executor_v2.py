from __future__ import annotations

"""RSS.vNext Candidate Mesh Compiler: actual Source -> Shape CAS consumer.

This is a DIFFERENT contract from legacy eight-view ProductMeshCompile.v1.
It replays TESSA using the identical unified checkpoint config; compiles
indexed surface geometry, presentation, original rays and semantic/affordance
candidates from the actual model proposal. Typed learned directed relation
scores are available for explicitly requested face pairs.

ABSTAIN is an evidence-retaining outcome, never a fabricated fallback mesh.
Source-grounded CanonicalMesh, local draw order, executable joint/attachment,
and ProductRevision require downstream, separate courts and are NEVER minted
here. Mesh consumes a derived view; full H(x) remains untouched in CAS.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    validate_rss_vnext_source_evidence_binding_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
    compile_rss_vnext_surface_candidates_v1,
)
from compiler.realsas_compiler_core.rss_vnext_face_relation_schedule_v1 import (
    compile_rss_vnext_sparse_face_relations_v1,
)
from compiler.realsas_compiler_core.rss_vnext_cross_expert_capability_court_v1 import (
    compile_rss_vnext_cross_expert_capability_court_v1,
)
from compiler.realsas_compiler_core.rss_vnext_canonical_mesh_candidate_bridge_v1 import (
    build_rss_vnext_canonical_mesh_candidate_bridge_v1,
)
from compiler.realsas_compiler_core.rss_vnext_relation_epistemic_court_v1 import (
    adjudicate_rss_vnext_relations_v1,
    learned_face_relation_assertions_v1,
    derive_candidate_topology_relations_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_evidence_continuity_v1 import (
    ARTIFACT_TYPE as CONTINUITY_ARTIFACT_TYPE,
    ROLE as CONTINUITY_ROLE,
    seal_rss_vnext_evidence_continuity_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    ARTIFACT_TYPE as SOURCE_ARTIFACT_TYPE,
    validate_rss_vnext_source_ref_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_product_fanout_v1 import (
    INDEX_ARTIFACT_TYPE,
    bind_rss_vnext_fanout_refs_v1,
    read_rss_vnext_neural_evidence_v1,
    read_rss_vnext_unified_model_config_v1,
    validate_rss_vnext_fanout_index_v1,
)

EXECUTOR_CONTRACT = "RealSaS.ProductMeshCompile.v2"
PARENT_BINDING_SCHEMA = "RealSaS.RSSvNext.ShapeParentBinding.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024


def _digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _read_cas(ref: dict[str, Any], read_object: Callable[[dict[str, Any]], bytes]) -> bytes:
    data = read_object(ref)
    if (
        type(data) is not bytes
        or type(ref.get("size_bytes")) is not int
        or ref["size_bytes"] != len(data)
        or ref.get("content_sha256") != sha256(data).hexdigest()
    ):
        raise RuntimeError("RSS_VNEXT_MESH_PARENT_CAS_BYTES_DRIFT")
    return data


def _parents(request: dict[str, Any]):
    refs = list(request.get("input_artifacts") or ())
    if len(refs) != 9 or any(type(x) is not dict for x in refs):
        raise RuntimeError("RSS_VNEXT_MESH_EXPECTS_SOURCE_INDEX_AND_SEVEN_CHILDREN")
    sources = [x for x in refs if x.get("artifact_type") == SOURCE_ARTIFACT_TYPE]
    indexes = [x for x in refs if x.get("artifact_type") == INDEX_ARTIFACT_TYPE]
    if len(sources) != 1 or len(indexes) != 1:
        raise RuntimeError("RSS_VNEXT_MESH_SOURCE_OR_INDEX_CARDINALITY")
    if (
        sources[0].get("schema_version") != "v1"
        or indexes[0].get("schema_version") != "v1"
    ):
        raise RuntimeError("RSS_VNEXT_MESH_PARENT_ABI_DRIFT")
    forbidden = {
        "RealSaS.ShapeInput", "RealSaS.ShapeProposal",
        "RealSaS.SourceObservation", "RealSaS.IRISSceneEvidence",
    }
    if any(x.get("artifact_type") in forbidden for x in refs):
        raise RuntimeError("RSS_VNEXT_MESH_LEGACY_PARENT_FORBIDDEN")
    return sources[0], indexes[0], refs


def _read_receipt(
    index: Mapping[str, Any], children: Mapping[str, dict[str, Any]],
    read_object: Callable[[dict[str, Any]], bytes],
) -> None:
    receipt = json.loads(_read_cas(children["inference_receipt"], read_object))
    if type(receipt) is not dict or receipt.get("schema") != (
        "RealSaS.RSSvNext.SourceToCompilerProposalReceipt.v2"
    ):
        raise RuntimeError("RSS_VNEXT_MESH_INFERENCE_RECEIPT_SCHEMA_DRIFT")
    claimed = dict(receipt)
    signature = claimed.pop("receipt_hash", None)
    if signature != content_sha256(claimed):
        raise RuntimeError("RSS_VNEXT_MESH_INFERENCE_RECEIPT_HASH_DRIFT")
    bindings = {
        "checkpoint_sha256": index["unified_checkpoint_sha256"],
        "unified_model_config_sha256": index["unified_model_config_sha256"],
        "unified_model_config_file_sha256": children["unified_model_config"]["content_sha256"],
        "neural_scene_file_sha256": children["neural_scene"]["content_sha256"],
        "embedded_mira_decoder_file_sha256": children["embedded_mira_decoder"]["content_sha256"],
        "evidence_file_sha256": children["neural_evidence"]["content_sha256"],
        "source_archive_sha256": children["normalized_model_source"]["content_sha256"],
        "canonical_source_file_sha256": children["original_source"]["content_sha256"],
        "canonical_source_binding_sha256": index["source_binding_sha256"],
        "source_rights_claim_sha256": index["source_rights_claim_sha256"],
        "evidence_hash": index["evidence_hash"],
        "status": index["mesh_status"],
    }
    if any(receipt.get(k) != v for k, v in bindings.items()):
        raise RuntimeError("RSS_VNEXT_MESH_INFERENCE_RECEIPT_PARENT_DRIFT")
    if (
        receipt.get("source_rights_verified") is not False
        or receipt.get("product_revision_minted") is not False
        or receipt.get("neural_scene_product_authority") is not False
        or receipt.get("embedded_mira_decoder_schema") != "RealSaS.RSSvNext.EmbeddedMIRADecoder.v1"
        or receipt.get("embedded_mira_independent_release_required") is not False
        or receipt.get("embedded_mira_discrete_runtime_parity_qualified") is not False
        or receipt.get("observed_view_count") != 2
    ):
        raise RuntimeError("RSS_VNEXT_MESH_INFERENCE_RECEIPT_PREMATURE_AUTHORITY")


def _store_json(
    payload: dict[str, Any], *,
    role: str, artifact_type: str,
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf8")
    if not 1 <= len(data) <= _MAX_JSON_BYTES:
        raise RuntimeError("RSS_VNEXT_MESH_COMPILER_OUTPUT_BUDGET")
    sealed = put_bytes(data)
    if (
        sealed.get("content_sha256") != sha256(data).hexdigest()
        or sealed.get("size_bytes") != len(data)
    ):
        raise RuntimeError("RSS_VNEXT_MESH_CAS_WRITE_DRIFT")
    return {
        "role": role,
        "artifact_type": artifact_type,
        "schema_version": "v1",
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": "CALCULATED_NEURAL_CANDIDATE_NOT_CANONICAL_PRODUCT",
    }


def execute_rss_vnext_mesh_capability_v2(
    request: dict[str, Any], *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if (
        request.get("kind") != "MODULE"
        or request.get("capability_id") != "product.mesh"
        or metadata.get("executor_contract") != EXECUTOR_CONTRACT
    ):
        raise RuntimeError("RSS_VNEXT_MESH_EXECUTOR_CONTRACT_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("RSS_VNEXT_MESH_OWN_MODEL_RELEASE_FORBIDDEN")
    binding = request.get("shape_parent_binding")
    if (
        type(binding) is not dict
        or set(binding) != {
            "schema", "source_content_sha256", "checkpoint_sha256",
            "model_release_behavior_sha256",
        }
        or binding["schema"] != PARENT_BINDING_SCHEMA
        or not all(_digest(binding[k]) for k in (
            "source_content_sha256", "checkpoint_sha256",
            "model_release_behavior_sha256",
        ))
    ):
        raise RuntimeError("RSS_VNEXT_MESH_ATTEMPT_SHAPE_PARENT_BINDING_REQUIRED")

    source_ref, index_ref, refs = _parents(request)
    original = _read_cas(source_ref, read_object)
    source = validate_rss_vnext_source_ref_v1(source_ref, original)
    if binding["source_content_sha256"] != source_ref["content_sha256"]:
        raise RuntimeError("RSS_VNEXT_MESH_ATTEMPT_SOURCE_BINDING_DRIFT")
    raw_index = _read_cas(index_ref, read_object)
    index = validate_rss_vnext_fanout_index_v1(
        raw_index,
        expected_source_content_sha256=source_ref["content_sha256"],
        expected_source_binding_sha256=source.source_binding_sha256,
        expected_checkpoint_sha256=binding["checkpoint_sha256"],
        expected_model_release_behavior_sha256=binding["model_release_behavior_sha256"],
    )
    if index["source_rights_claim_sha256"] != source.source_rights_claim_sha256:
        raise RuntimeError("RSS_VNEXT_MESH_SOURCE_RIGHTS_CLAIM_DRIFT")
    children = bind_rss_vnext_fanout_refs_v1(
        index, input_artifacts=refs,
    )
    _read_receipt(index, children, read_object)
    cfg = read_rss_vnext_unified_model_config_v1(
        index, child_refs=children, read_object=read_object,
    )
    evidence = read_rss_vnext_neural_evidence_v1(
        index, child_refs=children, read_object=read_object,
    )
    # Exact source CAS bytes must match the original tensors and cameras
    # claimed by the neural evidence. Hash-sealed proposal/index records are
    # not independently observed source authority, even on TESSA ABSTAIN.
    source_evidence_court = validate_rss_vnext_source_evidence_binding_v1(
        source=source, evidence=evidence,
        expected_source_binding_sha256=index["source_binding_sha256"],
    )
    # A failed learned TESSA topology is NOT a failure to preserve evidence.
    # Do not allocate huge shared triplanes merely to report mesh abstention.
    if evidence.mesh_status == "ABSTAIN":
        surface = RSSvNextComputedCandidateSurfaceV1(
            source_inference_evidence_hash=evidence.evidence_hash,
            neural_scene_carrier_sha256=children["neural_scene"]["content_sha256"],
            unified_model_config_sha256=index["unified_model_config_sha256"],
            candidate_mesh_sha256=None, topology_token_sequence_sha256=None,
            state="ABSTAIN_NO_VALID_TOPOLOGY", reason=evidence.reason,
            faces=(),
        )
        relation = None
        pair_list = list((request.get("goal_parameters") or {}).get(
            "requested_face_pairs", (),
        ))
        if pair_list:
            raise RuntimeError("RSS_VNEXT_MESH_ABSTAIN_FACE_RELATIONS_FORBIDDEN")
    else:
        from compiler.realsas_compiler_core.rss_vnext_expert_fanout_v1 import (
            open_rss_vnext_expert_fanout_v1,
        )
        # Full H(x) is loaded once ONLY if decoded mesh exists. A source
        # graphics/mechanics consumer can still address that CAS separately.
        carrier = _read_cas(children["neural_scene"], read_object)
        fanout = open_rss_vnext_expert_fanout_v1(
            evidence, carrier,
            expected_checkpoint_sha256=binding["checkpoint_sha256"],
            expected_model_config_sha256=index["unified_model_config_sha256"],
            model_config=cfg,
        )
        surface = compile_rss_vnext_surface_candidates_v1(fanout)
        pair_list = (request.get("goal_parameters") or {}).get(
            "requested_face_pairs", (),
        )
        relation = compile_rss_vnext_sparse_face_relations_v1(
            fanout, surface, requested_face_pairs=pair_list,
        )

    # A candidate mesh is a DERIVED VIEW; the original seven model/source
    # children must remain separately addressable by every later expert.
    output = [
        _store_json(
            asdict(surface),
            role="rss_computed_surface_candidate",
            artifact_type="RealSaS.RSSvNextComputedCandidateSurface",
            put_bytes=put_bytes,
        )
    ]
    if relation is not None:
        output.append(_store_json(
            asdict(relation),
            role="rss_sparse_face_relation_candidates",
            artifact_type="RealSaS.RSSvNextSparseCandidateFaceRelations",
            put_bytes=put_bytes,
        ))
    # Keep all three learned relation channels independently typed. The
    # actual scores remain UNQUALIFIED, not boolean contact, cut, z-buffer
    # order or source truth; no averaging or implicit binary complement.
    # Exact indexed mesh facts are Compiler-derived, not mandatory source
    # targets or model logits; keep optional learned relation channels distinct.
    requested_pairs = (request.get("goal_parameters") or {}).get(
        "requested_face_pairs", (),
    )
    compiler_primitives = derive_candidate_topology_relations_v1(
        evidence=evidence, surface=surface, requested_face_pairs=requested_pairs,
    )
    learned_predictions = (
        learned_face_relation_assertions_v1(
            relation, inference_evidence_hash=evidence.evidence_hash,
            candidate_mesh_sha256=surface.candidate_mesh_sha256,
        ) if relation is not None else ()
    )
    relation_epistemic = adjudicate_rss_vnext_relations_v1(
        (*compiler_primitives, *learned_predictions),
        inference_evidence_hash=evidence.evidence_hash,
        candidate_mesh_sha256=surface.candidate_mesh_sha256,
    )
    output.append(_store_json(
        asdict(relation_epistemic), role="rss_relation_epistemic_court",
        artifact_type="RealSaS.RSSvNextRelationEpistemicCourt",
        put_bytes=put_bytes,
    ))
    # Reconcile specialist proposals against exact topology/source address
    # ownership. This court cannot mint a rig, drawable or source appearance.
    court = compile_rss_vnext_cross_expert_capability_court_v1(
        evidence=evidence, surface=surface, model_config=cfg, relation=relation,
    )
    output.append(_store_json(
        asdict(court), role="rss_cross_expert_capability_court",
        artifact_type="RealSaS.RSSvNextCrossExpertCapabilityCourt",
        put_bytes=put_bytes,
    ))
    # TESSA indexed topology cannot receive CanonicalMesh authority without
    # independently addressable source-support correspondences. Emit the
    # readiness/ABSTAIN decision, never synthesize source nodes from H(x).
    bridge = build_rss_vnext_canonical_mesh_candidate_bridge_v1(
        evidence=evidence, computed_surface=surface, model_config=cfg,
        allow_unqualified_neural_basis=True,
    )
    output.append(_store_json(
        asdict(bridge), role="rss_canonical_mesh_candidate_bridge",
        artifact_type="RealSaS.RSSvNextCanonicalMeshCandidateBridge",
        put_bytes=put_bytes,
    ))
    # Transport the exact input CAS references, not a lossy summary, through
    # product.mesh. Even ABSTAIN must retain H(x), both source views, TESSA
    # context, the MIRA decoder and every independent expert readout.
    continuity = seal_rss_vnext_evidence_continuity_v1(
        index=index, source_ref=source_ref, index_ref=index_ref,
        child_refs=children, derived_refs=tuple(output),
    )
    output.append(_store_json(
        continuity, role=CONTINUITY_ROLE,
        artifact_type=CONTINUITY_ARTIFACT_TYPE, put_bytes=put_bytes,
    ))
    return {
        "capability_id": "product.mesh",
        "status": "PASS",
        "outputs": output,
        "failure": None,
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductMeshCandidateDiagnostics.v1",
            "topology_state": surface.state,
            "computed_face_count": len(surface.faces),
            "candidate_mesh_hash": surface.candidate_mesh_sha256,
            "cross_expert_capability_state": court.state,
            "canonical_candidate_bridge_state": bridge.state,
            "relation_epistemic_state": relation_epistemic.state,
            "independent_executable_capabilities_qualified": False,
            "neural_scene_preserved_parent_cas": children["neural_scene"]["content_sha256"],
            "neural_scene_replaced_by_surface": False,
            "evidence_continuity_hash": continuity["continuity_hash"],
            "all_seven_neural_expert_children_preserved_by_reference": True,
            "source_surface_qualified": False,
            "original_source_tensor_identity_court_hash": source_evidence_court.court_hash,
            "original_source_tensor_identity_state": source_evidence_court.state,
            "original_source_tensor_identity_is_not_source_fidelity": True,
            "canonical_mesh_authority": False,
            "source_visibility_authority": False,
            "executable_rig_skin_qualified": False,
            "product_revision_minted": False,
            "tessa_replayed_from_exact_unified_checkpoint_config": relation is not None,
            "independent_rss_model_checkpoint_loaded_by_mesh": False,
        },
    }


__all__ = ["EXECUTOR_CONTRACT", "PARENT_BINDING_SCHEMA", "execute_rss_vnext_mesh_capability_v2"]
