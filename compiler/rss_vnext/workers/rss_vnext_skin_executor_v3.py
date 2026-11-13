from __future__ import annotations

"""RSS.vNext Skin v3: same unified checkpoint's FSQ codes and dense replay.

The Shape expert archive contains exact discrete generated weights and the
embedded MIRA submodule checkpoint. Rig has already bound its causal AXIS
program and SkinCodeProposal to CanonicalMesh. This worker uses the existing
MIRA carrier/simplex court, independently replays the embedded decoder, and
mints CanonicalSkin.v2 only when every vertex/joint retains its original index.

No separate neural worker, nearest-surface transfer, code search, weight
repair, invented influence labels, or ProductRevision authority is allowed.
"""

from hashlib import sha256
import json
from typing import Any, Callable

import torch

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import mint_canonical_skin_v2
from compiler.realsas_compiler_core.carrier_skin_v1 import qualify_carrier_skin_v1
from compiler.realsas_compiler_core.skin_code_ir_v1 import validate_skin_code_proposal_v1
from compiler.realsas_compiler_services.platform_worker.skin_executor_v2 import (
    _skin_code_from_dict, _qualification_policy,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_rig_executor_v3 import (
    _one, _cas, _json, _seal,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_source_codec_v1 import (
    validate_rss_vnext_source_ref_v1,
)
from compiler.realsas_compiler_services.platform_worker.rss_vnext_product_fanout_v1 import (
    bind_rss_vnext_fanout_refs_v1, read_rss_vnext_discrete_skin_replay_v1,
    read_rss_vnext_neural_evidence_v1, read_rss_vnext_unified_model_config_v1,
    validate_rss_vnext_fanout_index_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    validate_rss_vnext_source_evidence_binding_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from models.mira.conditioning_v6 import (
    build_mira_conditioning_v6, carrier_skin_proposal_from_decoded_v6,
)

EXECUTOR_CONTRACT = "RealSaS.ProductSkinInference.v3"


def execute_rss_vnext_skin_capability_v3(
    request: dict[str, Any], *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    meta = request.get("capability_metadata")
    if (
        type(request) is not dict or request.get("kind") != "MODULE"
        or request.get("capability_id") != "product.skin"
        or type(meta) is not dict
        or meta.get("executor_contract") != EXECUTOR_CONTRACT
        or meta.get("required_models") not in (None, [])
        or request.get("model_releases")
        or meta.get("compiler_code_search_allowed") is not False
        or meta.get("compiler_code_repair_allowed") is not False
        or meta.get("post_hoc_skin_transfer_allowed") is not False
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_UNIFIED_EXECUTOR_OR_MODEL_CONTRACT_DRIFT")

    mesh = canonical_mesh_from_dict_v1(_json(
        request, "RealSaS.CanonicalMeshIR", "v1", read_object,
    ))
    rig = canonical_rig_from_dict_v2(_json(
        request, "RealSaS.CanonicalRigIR", "v2", read_object,
    ))
    code = _skin_code_from_dict(_json(
        request, "RealSaS.SkinCodeProposalIR", "v1", read_object,
    ))
    if mesh.mesh_lineage_hash != rig.mesh_lineage_hash:
        raise RuntimeError("RSS_VNEXT_SKIN_CANONICAL_MESH_RIG_PARENT_DRIFT")
    validate_skin_code_proposal_v1(
        code,
        expected_proposal_joint_ids=tuple(j.source_proposal_joint_id for j in rig.joints),
        expected_rig_program_proposal_hash=rig.proposal_evidence_hash,
    )
    if (
        code.carrier_evidence_hash != mesh.carrier_evidence_hash
        or code.carrier_topology_hash != mesh.carrier_topology_hash
        or code.carrier_geometry_hash != mesh.carrier_geometry_hash
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_CARRIER_BINDING_DRIFT")

    source_ref = _one(request, "RealSaS.RSSvNextTwoViewSource", "v1")
    source = validate_rss_vnext_source_ref_v1(source_ref, _cas(source_ref, read_object))
    index_ref = _one(request, "RealSaS.RSSvNextProductEvidenceFanout", "v1")
    index_payload = _cas(index_ref, read_object)
    try:
        claimed = json.loads(index_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RSS_VNEXT_SKIN_FANOUT_JSON_INVALID") from exc
    if type(claimed) is not dict:
        raise RuntimeError("RSS_VNEXT_SKIN_FANOUT_OBJECT_REQUIRED")
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
    evidence = read_rss_vnext_neural_evidence_v1(
        index, child_refs=children, read_object=read_object,
    )
    validate_rss_vnext_source_evidence_binding_v1(
        source=source, evidence=evidence,
        expected_source_binding_sha256=index["source_binding_sha256"],
    )
    if (
        evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh is None or evidence.dynamic is None
        or evidence.evidence_hash not in mesh.proposal_evidence_refs
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_SOURCE_EVIDENCE_OR_TOPOLOGY_ABSTAIN")
    cfg = read_rss_vnext_unified_model_config_v1(
        index, child_refs=children, read_object=read_object,
    )
    replay = read_rss_vnext_discrete_skin_replay_v1(
        index, child_refs=children, read_object=read_object,
    )
    if (
        replay.inference_evidence_hash != evidence.evidence_hash
        or replay.checkpoint_sha256 != index["unified_checkpoint_sha256"]
        or replay.embedded_decoder_sha256 != code.codec_checkpoint_hash
        or replay.vertex_count != len(mesh.vertices)
        or replay.joint_count != len(rig.joints)
        or replay.skin_code_count != len(rig.joints) * cfg.skin_tokens_per_joint
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_EMBEDDED_DECODER_OR_JOINT_PARITY_DRIFT")
    if (
        code.codebook_size != cfg.axis_config().skin_vocab_size
        or code.codes_per_joint != cfg.skin_tokens_per_joint
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_CODE_VOCAB_OR_CARDINALITY_DRIFT")

    # CanonicalMesh and AXIS decode preserve TESSA vertex index and causal
    # joint order. Reordering by semantic region, proximity or Blender labels
    # destroys exact SkinTokens identity, even if resulting motion looks close.
    for i, vertex in enumerate(mesh.vertices):
        if vertex.source_candidate_vertex_id != f"RSSV:{i:08d}":
            raise RuntimeError("RSS_VNEXT_SKIN_TESSA_VERTEX_ORDER_DRIFT")
    for i, joint in enumerate(rig.joints):
        if joint.source_proposal_joint_id != f"P:J:{i:04d}":
            raise RuntimeError("RSS_VNEXT_SKIN_AXIS_JOINT_ORDER_DRIFT")
    dense = evidence.dynamic.get("discrete_skin_weights")
    if not isinstance(dense, list) or len(dense) != len(mesh.vertices):
        raise RuntimeError("RSS_VNEXT_SKIN_DENSE_REPLAY_MISSING")
    device = torch.device("cpu")
    weights = torch.as_tensor(dense,device=device,dtype=torch.float32)
    if tuple(weights.shape) != (len(mesh.vertices),len(rig.joints)):
        raise RuntimeError("RSS_VNEXT_SKIN_DENSE_REPLAY_SHAPE_DRIFT")
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("RSS_VNEXT_SKIN_DENSE_REPLAY_NONFINITE")
    conditioning = build_mira_conditioning_v6(mesh,rig,device=device)
    dense_proposal = carrier_skin_proposal_from_decoded_v6(
        weights,conditioning,
        model_provenance=(
            "RSSVNEXT:" + index["unified_checkpoint_sha256"] + ":"
            + replay.embedded_decoder_sha256
        ),
    )
    policy = _qualification_policy(meta)
    qualified = qualify_carrier_skin_v1(
        conditioning.carrier,conditioning.skeleton,dense_proposal,
        max_simplex_repair_l1=policy["max_simplex_repair_l1"],
        max_total_correction_l1=policy["max_total_correction_l1"],
        negative_tolerance=policy["negative_tolerance"],
        max_influences=policy["max_influences"],
    )
    skin = mint_canonical_skin_v2(
        mesh,rig,qualified_skin=qualified,
        skin_code_proposal_hash=code.proposal_lineage_hash,
        skin_codec_architecture_id=code.codec_architecture_id,
        skin_codec_checkpoint_hash=code.codec_checkpoint_hash,
    )
    receipt = {
        "schema":"RealSaS.RSSvNext.CanonicalSkinReplayReceipt.v1",
        "source_inference_evidence_hash":evidence.evidence_hash,
        "unified_checkpoint_sha256":index["unified_checkpoint_sha256"],
        "embedded_decoder_sha256":replay.embedded_decoder_sha256,
        "discrete_skin_max_abs_replay_error":replay.dense_weight_max_abs_replay_error,
        "canonical_mesh_lineage_hash":mesh.mesh_lineage_hash,
        "deform_structure_hash":rig.deform_structure_hash,
        "canonical_skin_lineage_hash":skin.skin_lineage_hash,
        "separate_mira_release_loaded":False,
        "compiler_code_search_or_repair":False,
        "animation_deformation_stress_qualified":False,
        "source_skin_authority_claimed":False,
        "product_revision_minted":False,
    }
    receipt["receipt_hash"] = content_sha256(receipt)
    return {
        "capability_id":"product.skin","status":"PASS","failure":None,
        "outputs":[
            _seal(skin,role="canonical_skin",
                  kind="RealSaS.CanonicalSkinIR",version="v2",put_bytes=put_bytes),
            _seal(dense_proposal,role="exact_discrete_skin_dense_proposal",
                  kind="RealSaS.CarrierSkinProposalIR",version="v1",put_bytes=put_bytes),
            _seal(qualified,role="qualified_carrier_skin",
                  kind="RealSaS.QualifiedCarrierSkinIR",version="v1",put_bytes=put_bytes),
            _seal(receipt,role="rss_skin_replay_receipt",
                  kind="RealSaS.RSSvNextCanonicalSkinReplayReceipt",
                  version="v1",put_bytes=put_bytes),
        ],
        "diagnostics":{
            "schema":"RealSaS.RSSvNext.ProductSkinDiagnostics.v3",
            "mesh_lineage_hash":mesh.mesh_lineage_hash,
            "deform_structure_hash":rig.deform_structure_hash,
            "skin_lineage_hash":skin.skin_lineage_hash,
            "vertex_count":len(mesh.vertices),
            "joint_count":len(rig.joints),
            "discrete_skin_max_abs_replay_error":replay.dense_weight_max_abs_replay_error,
            "simplex_max_abs_residual_before":qualified.qualification_report.get(
                "simplex_max_abs_residual_before",
            ),
            "separate_mira_model_release_loaded":False,
            "compiler_code_repair_performed":False,
            "product_revision_minted":False,
        },
    }


__all__ = ["EXECUTOR_CONTRACT","execute_rss_vnext_skin_capability_v3"]
