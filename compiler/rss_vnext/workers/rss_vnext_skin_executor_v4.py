from __future__ import annotations

"""One product.skin role: calibrated V3 or raw two-image exact MIRA replay.

Raw Skin consumes a source-qualified CanonicalMesh, exact CanonicalRig and
causal AXIS SkinCode from the single Shape checkpoint; independently reopens
the pinned MIRA codec and replays every discrete SkinToken into dense weights.
No nearest-neighbor transfer, independent model release, code repair, teacher
skin, unverified source contact or ProductRevision promotion is allowed.
"""

from hashlib import sha256
import json
from typing import Any, Callable

import numpy as np
import torch

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    canonical_rig_from_dict_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    mint_canonical_skin_v2,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import (
    qualify_carrier_skin_v1,
)
from compiler.realsas_compiler_core.rss_vnext_discrete_skin_replay_v1 import (
    replay_rss_vnext_discrete_skin_candidate_v1,
)
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    decode_uncalibrated_two_image_source_v1,
)
from compiler.realsas_compiler_core.skin_code_ir_v1 import (
    validate_skin_code_proposal_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.platform_worker.skin_executor_v2 import (
    _skin_code_from_dict, _qualification_policy,
)
from compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 import (
    RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT, RAW_CONFIG, RAW_DECODER,
    SCENE_REQUIRED, MESH_KEYS, AXIS_MIRA_KEYS,
    _archive, _validate_candidate, _validate_raw_axis_mira,
)
from compiler.rss_vnext.workers.rss_vnext_rig_executor_v4 import (
    _bytes, _json, CANONICAL_MESH, STATIC_QUALIFICATION,
)
from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from compiler.rss_vnext.workers.rss_vnext_rig_executor_v3 import _seal
from compiler.rss_vnext.workers.rss_vnext_skin_executor_v3 import (
    execute_rss_vnext_skin_capability_v3,
)
from models.mira.conditioning_v6 import (
    build_mira_conditioning_v6, carrier_skin_proposal_from_decoded_v6,
)

EXECUTOR_CONTRACT = "RealSaS.ProductSkinInference.v4"
RIG_TYPE = "RealSaS.CanonicalRigIR"
SKIN_CODE_TYPE = "RealSaS.SkinCodeProposalIR"
RIG_PROGRAM_TYPE = "RealSaS.QualifiedRigProgramIR"
RIG_RECEIPT_TYPE = "RealSaS.RSSvNextCanonicalRigCompileReceipt"
EXPECTED = {
    RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT, RAW_CONFIG, RAW_DECODER,
    CANONICAL_MESH, STATIC_QUALIFICATION, RIG_TYPE, SKIN_CODE_TYPE,
    RIG_PROGRAM_TYPE, RIG_RECEIPT_TYPE,
}
MAX_MODEL_CONFIG_BYTES = 128 * 1024
MAX_EMBEDDED_DECODER_BYTES = 512 * 1024 * 1024


def _raw_parent_set(request: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    refs = request.get("input_artifacts") or ()
    if type(refs) not in (list, tuple) or any(type(r) is not dict for r in refs):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_PARENT_LIST_INVALID")
    raw = [r for r in refs if r.get("artifact_type") == RAW_SOURCE]
    if not raw:
        return None
    if len(raw) not in (1, 2) or len(refs) != len(raw) + len(EXPECTED) - 1:
        raise RuntimeError("RSS_VNEXT_SKIN_V4_EXACT_PARENT_CARDINALITY")
    if any(
        r.get(k) != raw[0].get(k)
        for r in raw[1:]
        for k in ("content_sha256", "size_bytes", "storage_key")
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_SOURCE_DOUBLE_BIND_DRIFT")
    parents = {}
    for r in refs:
        kind = r.get("artifact_type")
        if kind not in EXPECTED or r.get("schema_version") != (
            "v2" if kind == RIG_TYPE else "v1"
        ):
            raise RuntimeError("RSS_VNEXT_SKIN_V4_UNEXPECTED_PARENT_OR_ABI")
        if kind in parents:
            if kind != RAW_SOURCE:
                raise RuntimeError("RSS_VNEXT_SKIN_V4_DUPLICATE_PARENT")
            continue
        parents[kind] = r
    if set(parents) != EXPECTED:
        raise RuntimeError("RSS_VNEXT_SKIN_V4_MISSING_CANONICAL_OR_SCENE_PARENT")
    return parents


def execute_rss_vnext_skin_capability_v4(
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
        raise RuntimeError("RSS_VNEXT_SKIN_V4_ROLE_OR_MODEL_CONTRACT")
    refs = _raw_parent_set(request)
    if refs is None:
        calibrated = {
            **request,
            "capability_metadata": {
                **meta, "executor_contract": "RealSaS.ProductSkinInference.v3",
            },
        }
        return execute_rss_vnext_skin_capability_v3(
            calibrated, read_object=read_object, put_bytes=put_bytes,
        )

    source_ref = refs[RAW_SOURCE]
    original = decode_uncalibrated_two_image_source_v1(
        _bytes(source_ref, read_object, limit=70 * 1024 * 1024),
    )
    receipt = _json(refs[RAW_RECEIPT], read_object)
    scene_ref = refs[RAW_SCENE]
    if (
        receipt.get("source_content_sha256") != source_ref["content_sha256"]
        or receipt.get("source_binding_sha256") != original.source_binding_sha256
        or receipt.get("scene_content_sha256") != scene_ref["content_sha256"]
        or receipt.get("unified_model_config_content_sha256")
        != refs[RAW_CONFIG]["content_sha256"]
        or receipt.get("embedded_mira_decoder_content_sha256")
        != refs[RAW_DECODER]["content_sha256"]
        or receipt.get("generated_views_content_sha256")
        != refs[RAW_VIEWS]["content_sha256"]
        or receipt.get("original_camera_calibrated") is not False
        or receipt.get("source_ray_evidence_qualified") is not False
        or receipt.get("product_revision_minted") is not False
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_SOURCE_AND_CHECKPOINT_BINDING_DRIFT")
    scene = _archive(
        _bytes(scene_ref, read_object),
        required=SCENE_REQUIRED, allowed=SCENE_REQUIRED | MESH_KEYS | AXIS_MIRA_KEYS,
    )
    geometry = _validate_candidate(scene, receipt)
    controls = _validate_raw_axis_mira(scene, receipt, geometry)
    if (
        geometry["mesh_state"] != "UNQUALIFIED_MESH_CANDIDATE"
        or controls["control_state"] != "UNQUALIFIED_AXISMIRA_CONTROL_CANDIDATE"
        or not controls["discrete_skin_present"]
        or controls["control_event_count"] < 1
        or not geometry["component_id_continuity"]
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_EXACT_DISCRETE_PROPOSAL_REQUIRED")

    mesh = canonical_mesh_from_dict_v1(_json(refs[CANONICAL_MESH], read_object))
    rig = canonical_rig_from_dict_v2(_json(refs[RIG_TYPE], read_object))
    code = _skin_code_from_dict(_json(refs[SKIN_CODE_TYPE], read_object))
    expected, qualification, _ = admit_raw_two_view_mesh_v1(
        source=original, scene=scene, receipt=receipt,
        scene_digest=scene_ref["content_sha256"],
    )
    if (
        expected is None or qualification is None or expected != mesh
        or _json(refs[STATIC_QUALIFICATION], read_object) != qualification.to_dict()
        or rig.mesh_lineage_hash != mesh.mesh_lineage_hash
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_SOURCE_QUALIFIED_MESH_OR_RIG_DRIFT")
    validate_skin_code_proposal_v1(
        code,
        expected_proposal_joint_ids=tuple(
            j.source_proposal_joint_id for j in rig.joints
        ),
        expected_rig_program_proposal_hash=rig.proposal_evidence_hash,
    )
    if (
        code.carrier_evidence_hash != mesh.carrier_evidence_hash
        or code.carrier_topology_hash != mesh.carrier_topology_hash
        or code.carrier_geometry_hash != mesh.carrier_geometry_hash
        or code.codec_checkpoint_hash != refs[RAW_DECODER]["content_sha256"]
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_RIG_SKINCODE_PARENT_MISMATCH")

    from models.rss_vnext.model_config_archive_v1 import (
        decode_rss_vnext_unified_model_config_v1,
    )
    from models.rss_vnext.embedded_mira_decoder_v1 import (
        load_embedded_mira_decoder_v1,
    )
    config_blob = _bytes(refs[RAW_CONFIG], read_object, limit=MAX_MODEL_CONFIG_BYTES)
    try:
        config_claim = json.loads(config_blob.decode("utf-8"))
        cfg_sha = config_claim["unified_model_config_sha256"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("RSS_VNEXT_SKIN_V4_PINNED_MODEL_CONFIG_INVALID") from exc
    cfg = decode_rss_vnext_unified_model_config_v1(
        config_blob, expected_checkpoint_sha256=receipt["checkpoint_sha256"],
        expected_config_sha256=cfg_sha,
    )
    decoder, codec = load_embedded_mira_decoder_v1(
        _bytes(refs[RAW_DECODER], read_object, limit=MAX_EMBEDDED_DECODER_BYTES),
        expected_parent_unified_checkpoint_sha256=receipt["checkpoint_sha256"],
        expected_subcheckpoint_sha256=refs[RAW_DECODER]["content_sha256"],
        expected_model_config_sha256=cfg_sha,
        model_config=cfg,
    )
    if (
        code.codec_architecture_id != codec.architecture_id
        or code.codebook_size != codec.codebook_size
        or code.codes_per_joint != cfg.skin_tokens_per_joint
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_MIRA_CODEC_IDENTITY_DRIFT")
    n = len(mesh.vertices)
    j = len(rig.joints)
    if not 1 <= j <= cfg.control_max_seq or n != geometry["candidate_vertices"]:
        raise RuntimeError("RSS_VNEXT_SKIN_V4_CANONICAL_DOMAIN_DRIFT")
    for idx, vertex in enumerate(mesh.vertices):
        if vertex.source_candidate_vertex_id != f"RSSV:{idx:08d}":
            raise RuntimeError("RSS_VNEXT_SKIN_V4_VERTEX_REORDER_FORBIDDEN")
    for idx, joint in enumerate(rig.joints):
        if joint.source_proposal_joint_id != f"P:J:{idx:04d}":
            raise RuntimeError("RSS_VNEXT_SKIN_V4_JOINT_REORDER_FORBIDDEN")
    dense = scene["axis_mira_discrete_skin_weights"]
    if dense.shape != (1, n, j):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_SKIN_WEIGHTS_DOMAIN_MISMATCH")
    if scene["axis_mira_joint_valid_mask"].shape != (1, j):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_JOINT_VALIDITY_DOMAIN_MISMATCH")
    if not bool(np.all(scene["axis_mira_joint_valid_mask"] == 1)):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_INVALID_PADDED_JOINT_PRESENT")

    soft = scene["axis_mira_soft_skin_weights"][0].astype(np.float64)
    hard = dense[0].astype(np.float64)
    discrepancy = np.abs(soft - hard)
    dynamic = {
        "vertices": scene["axis_mira_dynamic_vertices"][0].tolist(),
        "weights": soft.tolist(),
        "displacement": scene["axis_mira_dynamic_displacement"][0].tolist(),
        "readiness_logits": scene["axis_mira_readiness_logits"][0].tolist(),
        "authority": "UNQUALIFIED_DYNAMIC_MESH_PROPOSAL",
        "discrete_skin_weights": hard.tolist(),
        "skin_joint_hidden_states": scene["axis_mira_joint_hidden_states"][0].tolist(),
        "soft_to_discrete_max_abs_difference": float(discrepancy.max()),
        "soft_to_discrete_mean_abs_difference": float(discrepancy.mean()),
        "soft_to_discrete_parity_qualified": False,
    }
    proposal_mesh = {
        "local_features": scene["tessa_candidate_local_features"][0].tolist(),
        "rest_xyz": scene["tessa_candidate_rest_vertices"][0].tolist(),
        "normals_xyz": scene["tessa_candidate_normals"][0].tolist(),
    }
    replay = replay_rss_vnext_discrete_skin_candidate_v1(
        mesh=proposal_mesh, dynamic=dynamic,
        control_events=tuple(receipt["axis_mira_control_events"]),
        model_config=cfg, decoder=decoder,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        inference_evidence_hash=content_sha256({
            "source": original.source_binding_sha256,
            "scene": scene_ref["content_sha256"],
            "events": receipt["axis_mira_control_events_sha256"],
        }),
        embedded_decoder_sha256=refs[RAW_DECODER]["content_sha256"],
    )
    if (
        replay.vertex_count != n or replay.joint_count != j
        or replay.skin_code_count != j * cfg.skin_tokens_per_joint
        or replay.dense_weight_max_abs_replay_error > 5e-5
    ):
        raise RuntimeError("RSS_VNEXT_SKIN_V4_DISCRETE_FSQTOKENS_REPLAY_DRIFT")
    conditioning = build_mira_conditioning_v6(mesh, rig, device=torch.device("cpu"))
    dense_proposal = carrier_skin_proposal_from_decoded_v6(
        torch.as_tensor(hard, dtype=torch.float32),
        conditioning,
        model_provenance=(
            "RSSVNEXT:" + receipt["checkpoint_sha256"] + ":"
            + replay.embedded_decoder_sha256
        ),
    )
    policy = _qualification_policy(meta)
    qualified = qualify_carrier_skin_v1(
        conditioning.carrier, conditioning.skeleton, dense_proposal,
        max_simplex_repair_l1=policy["max_simplex_repair_l1"],
        max_total_correction_l1=policy["max_total_correction_l1"],
        negative_tolerance=policy["negative_tolerance"],
        max_influences=policy["max_influences"],
    )
    skin = mint_canonical_skin_v2(
        mesh, rig, qualified_skin=qualified,
        skin_code_proposal_hash=code.proposal_lineage_hash,
        skin_codec_architecture_id=code.codec_architecture_id,
        skin_codec_checkpoint_hash=code.codec_checkpoint_hash,
    )
    witness = {
        "schema": "RealSaS.RSSvNext.RawCanonicalSkinReplayReceipt.v1",
        "original_source_content_sha256": source_ref["content_sha256"],
        "unified_checkpoint_sha256": receipt["checkpoint_sha256"],
        "model_config_content_sha256": refs[RAW_CONFIG]["content_sha256"],
        "embedded_decoder_content_sha256": refs[RAW_DECODER]["content_sha256"],
        "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
        "canonical_rig_lineage_hash": rig.rig_lineage_hash,
        "deform_structure_hash": rig.deform_structure_hash,
        "canonical_skin_lineage_hash": skin.skin_lineage_hash,
        "discrete_skin_max_abs_replay_error": replay.dense_weight_max_abs_replay_error,
        "soft_to_discrete_max_abs_difference": replay.soft_to_discrete_max_abs_difference,
        "independent_mira_release_loaded": False,
        "nearest_neighbor_transfer": False,
        "compiler_code_repair_performed": False,
        "source_skin_truth_claimed": False,
        "contact_and_pose_qualified": False,
        "product_revision_minted": False,
    }
    witness["receipt_hash"] = content_sha256(witness)
    return {
        "capability_id": "product.skin", "status": "PASS", "failure": None,
        "outputs": [
            _seal(skin, role="canonical_skin",
                  kind="RealSaS.CanonicalSkinIR", version="v2", put_bytes=put_bytes),
            _seal(dense_proposal, role="exact_discrete_skin_dense_proposal",
                  kind="RealSaS.CarrierSkinProposalIR", version="v1", put_bytes=put_bytes),
            _seal(qualified, role="qualified_carrier_skin",
                  kind="RealSaS.QualifiedCarrierSkinIR", version="v1", put_bytes=put_bytes),
            _seal(witness, role="rss_skin_replay_receipt",
                  kind="RealSaS.RSSvNextCanonicalSkinReplayReceipt", version="v1",
                  put_bytes=put_bytes),
        ],
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductSkinDiagnostics.v4",
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "deform_structure_hash": rig.deform_structure_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "vertex_count": n, "joint_count": j,
            "discrete_skin_max_abs_replay_error": replay.dense_weight_max_abs_replay_error,
            "separate_mira_model_release_loaded": False,
            "compiler_code_repair_performed": False,
            "contact_and_pose_qualified": False,
            "product_revision_minted": False,
        },
    }


__all__ = ["EXECUTOR_CONTRACT", "execute_rss_vnext_skin_capability_v4"]
