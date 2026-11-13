from __future__ import annotations

"""RSS.vNext product.rig v3: one unified Shape inference -> CanonicalRig.v2.

Consumes only exact CAS parents from Shape v3 and statically admitted Mesh v2.
Reuses the existing AXIS V6 causal grammar/Compiler qualification and the
exact embedded MIRA decoder binding. No separate AXIS or MIRA ModelRelease
can run, no parent re-selection or control/constraint synthesis is performed.

The presence of a CanonicalRig is not skin, attachment, deformation,
source appearance or ProductRevision qualification.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Callable

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.rss_vnext_canonical_rig_lowering_v1 import (
    lower_rss_vnext_rig_program_to_canonical_v1,
)
from compiler.realsas_compiler_core.rss_vnext_neural_mesh_proposal_basis_v1 import (
    build_rss_vnext_neural_mesh_candidate_basis_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    validate_rss_vnext_source_evidence_binding_v1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_product_fanout_v1 import (
    bind_rss_vnext_fanout_refs_v1,
    read_rss_vnext_embedded_mira_decoder_v1,
    read_rss_vnext_neural_evidence_v1,
    read_rss_vnext_unified_model_config_v1,
    validate_rss_vnext_fanout_index_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    validate_rss_vnext_source_ref_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256

EXECUTOR_CONTRACT = "RealSaS.ProductRigInference.v3"


def _one(request, kind: str, version: str) -> dict[str, Any]:
    found = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == kind
        and row.get("schema_version") == version
    ]
    if len(found) != 1:
        raise RuntimeError("RSS_VNEXT_RIG_PARENT_CARDINALITY:" + kind)
    return found[0]


def _cas(ref, reader: Callable, *, limit: int = 256 * 1024 * 1024) -> bytes:
    if (
        type(ref) is not dict or type(ref.get("size_bytes")) is not int
        or ref["size_bytes"] < 0 or ref["size_bytes"] > limit
    ):
        raise RuntimeError("RSS_VNEXT_RIG_CAS_RESOURCE_LIMIT")
    data = reader(ref)
    if (
        type(data) is not bytes or len(data) != ref["size_bytes"]
        or sha256(data).hexdigest() != ref.get("content_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RIG_CAS_BYTES_OR_HASH_DRIFT")
    return data


def _json(request, kind, version, reader):
    ref = _one(request, kind, version)
    try:
        value = json.loads(_cas(ref, reader))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RSS_VNEXT_RIG_PARENT_JSON_INVALID:" + kind) from exc
    if type(value) is not dict:
        raise RuntimeError("RSS_VNEXT_RIG_PARENT_JSON_OBJECT_REQUIRED:" + kind)
    return value


def _seal(data: Any, *, role: str, kind: str, version: str, put_bytes: Callable) -> dict:
    raw = (json.dumps(
        data.to_dict() if hasattr(data, "to_dict") else data,
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")
    ref = put_bytes(raw)
    if (
        type(ref) is not dict or ref.get("content_sha256") != sha256(raw).hexdigest()
        or ref.get("size_bytes") != len(raw) or not ref.get("storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_RIG_OUTPUT_CAS_DRIFT")
    return {
        "role":role, "artifact_type":kind, "schema_version":version,
        "storage_key":ref["storage_key"], "content_sha256":ref["content_sha256"],
        "size_bytes":ref["size_bytes"],
        "authority_class":"COMPILER_CANONICAL_RIG_NOT_PRODUCT_REVISION",
    }


def execute_rss_vnext_rig_capability_v3(
    request: dict[str, Any], *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    meta = request.get("capability_metadata")
    if (
        type(request) is not dict or request.get("kind") != "MODULE"
        or request.get("capability_id") != "product.rig"
        or type(meta) is not dict
        or meta.get("executor_contract") != EXECUTOR_CONTRACT
        or meta.get("required_models") not in (None, [])
        or request.get("model_releases")
    ):
        raise RuntimeError("RSS_VNEXT_RIG_MODEL_OR_EXECUTOR_CONTRACT_DRIFT")

    source_ref = _one(request, "RealSaS.RSSvNextTwoViewSource", "v1")
    original = _cas(source_ref,read_object)
    source = validate_rss_vnext_source_ref_v1(source_ref,original)
    index_payload = _cas(
        _one(request, "RealSaS.RSSvNextProductEvidenceFanout", "v1"),
        read_object,
    )
    # Only the Shape producer (already ModelRelease pinned by Platform) can
    # manufacture this sealed CAS index. It independently binds original
    # source, neural checkpoint, decoder and every specialist child.
    try:
        claimed = json.loads(index_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RSS_VNEXT_RIG_FANOUT_INDEX_JSON_INVALID") from exc
    if type(claimed) is not dict:
        raise RuntimeError("RSS_VNEXT_RIG_FANOUT_INDEX_OBJECT_REQUIRED")
    index = validate_rss_vnext_fanout_index_v1(
        index_payload,
        expected_source_content_sha256=source_ref["content_sha256"],
        expected_source_binding_sha256=source.source_binding_sha256,
        expected_checkpoint_sha256=claimed.get("unified_checkpoint_sha256"),
        expected_model_release_behavior_sha256=claimed.get(
            "unified_model_release_behavior_sha256",
        ),
    )
    children = bind_rss_vnext_fanout_refs_v1(
        index, input_artifacts=list(request.get("input_artifacts") or ()),
    )
    cfg = read_rss_vnext_unified_model_config_v1(
        index, child_refs=children, read_object=read_object,
    )
    evidence = read_rss_vnext_neural_evidence_v1(
        index, child_refs=children, read_object=read_object,
    )
    validate_rss_vnext_source_evidence_binding_v1(
        source=source, evidence=evidence,
        expected_source_binding_sha256=index["source_binding_sha256"],
    )
    if evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE":
        raise RuntimeError("RSS_VNEXT_RIG_NEURAL_TOPOLOGY_ABSTAIN")

    surface_payload = _json(
        request, "RealSaS.RSSvNextComputedCandidateSurface", "v1", read_object,
    )
    surface = RSSvNextComputedCandidateSurfaceV1(**surface_payload)
    basis = build_rss_vnext_neural_mesh_candidate_basis_v1(
        evidence=evidence, computed_surface=surface, model_config=cfg,
    )
    bridge = _json(
        request, "RealSaS.RSSvNextCanonicalMeshCandidateBridge", "v1",
        read_object,
    )
    if (
        bridge.get("state") != "UNQUALIFIED_NEURAL_BASIS_CANONICAL_CANDIDATE"
        or bridge.get("neural_basis_lineage_sha256") != basis.basis_hash
        or bridge.get("inference_evidence_hash") != evidence.evidence_hash
        or bridge.get("candidate", {}).get("candidate_lineage_hash")
        != basis.candidate.candidate_lineage_hash
    ):
        raise RuntimeError("RSS_VNEXT_RIG_TESSA_BRIDGE_PROVENANCE_DRIFT")

    canonical = canonical_mesh_from_dict_v1(_json(
        request, "RealSaS.CanonicalMeshIR", "v1", read_object,
    ))
    if (
        evidence.evidence_hash not in canonical.proposal_evidence_refs
        or basis.basis_hash not in canonical.proposal_evidence_refs
        or canonical.source_candidate_lineage_hash
        != basis.candidate.candidate_lineage_hash
    ):
        raise RuntimeError("RSS_VNEXT_RIG_CANONICAL_MESH_SOURCE_QUALIFICATION_DRIFT")

    # This is a pinned submodule of the single, already selected RSS ModelRelease
    # and never a separate MIRA checkpoint. The loader verifies the actual
    # embedded decoder CAS, its model configuration and parent checkpoint hash.
    _, codec = read_rss_vnext_embedded_mira_decoder_v1(
        index, child_refs=children, read_object=read_object, model_config=cfg,
    )
    result = lower_rss_vnext_rig_program_to_canonical_v1(
        canonical_mesh=canonical, neural_mesh_candidate=basis.candidate,
        evidence=evidence, model_config=cfg, codec_binding=codec,
    )
    receipt = {
        "schema":"RealSaS.RSSvNext.CanonicalRigCompileReceipt.v1",
        "source_binding_sha256":source.source_binding_sha256,
        "source_inference_evidence_hash":evidence.evidence_hash,
        "unified_checkpoint_sha256":index["unified_checkpoint_sha256"],
        "embedded_mira_decoder_sha256":codec.checkpoint_hash,
        "canonical_mesh_lineage_hash":canonical.mesh_lineage_hash,
        "canonical_rig_lineage_hash":result.canonical_rig.rig_lineage_hash,
        "skin_code_proposal_hash":result.skin_code_proposal.proposal_lineage_hash,
        "same_unified_attempt_inference":True,
        "separate_axis_mira_inference":False,
        "mechanics_graphics_skin_motion_qualified":False,
        "source_rig_truth_claimed":False,
        "product_revision_minted":False,
    }
    receipt["receipt_hash"] = content_sha256(receipt)
    outputs = [
        _seal(result.canonical_rig,role="canonical_rig",
              kind="RealSaS.CanonicalRigIR",version="v2",put_bytes=put_bytes),
        _seal(result.skin_code_proposal,role="skin_code_proposal",
              kind="RealSaS.SkinCodeProposalIR",version="v1",put_bytes=put_bytes),
        _seal(result.qualified_program,role="qualified_rig_program",
              kind="RealSaS.QualifiedRigProgramIR",version="v1",put_bytes=put_bytes),
        _seal(receipt,role="rss_rig_rebind_receipt",
              kind="RealSaS.RSSvNextCanonicalRigCompileReceipt",version="v1",
              put_bytes=put_bytes),
    ]
    return {
        "capability_id":"product.rig", "status":"PASS",
        "outputs":outputs, "failure":None,
        "diagnostics":{
            "schema":"RealSaS.RSSvNext.ProductRigCompileDiagnostics.v3",
            "mesh_lineage_hash":canonical.mesh_lineage_hash,
            "rig_lineage_hash":result.canonical_rig.rig_lineage_hash,
            "skin_code_proposal_hash":result.skin_code_proposal.proposal_lineage_hash,
            "joint_count":len(result.canonical_rig.joints),
            "control_count":len(result.canonical_rig.controls),
            "constraint_count":len(result.canonical_rig.constraints),
            "attachment_anchor_count":len(result.canonical_rig.attachment_anchors),
            "separate_axis_mira_inference":False,
            "source_3d_joint_teacher_claimed":False,
            "product_revision_minted":False,
        },
    }


__all__ = ["EXECUTOR_CONTRACT", "execute_rss_vnext_rig_capability_v3"]
