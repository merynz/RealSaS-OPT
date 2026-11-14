from __future__ import annotations

"""One product.rig role: calibrated V3 replay or raw original-alpha CanonicalRig.

The raw path consumes the SAME checkpoint's TESSA / AXIS / MIRA outputs. It
independently replays the admitted canonical mesh from real source alpha,
decodes causal AXIS events with the exact embedded MIRA codec and mints a
canonical rig. No source joint teacher, original camera, independently
selected model, constraint synthesis or ProductRevision is introduced.
"""

from hashlib import sha256
import json
from typing import Any, Callable

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1, validate_rig_constraint_operator_v1,
)
from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    qualify_rig_program_v1,
)
from compiler.realsas_compiler_core.skin_code_ir_v1 import (
    validate_skin_code_proposal_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    mint_canonical_rig_v2,
)
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    decode_uncalibrated_two_image_source_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.rss_vnext_mesh_executor_v3 import (
    RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT,
    RAW_CONFIG, RAW_DECODER,
    SCENE_REQUIRED, MESH_KEYS, AXIS_MIRA_KEYS, RAW_GRAPHICS_KEYS,
    _archive, _validate_candidate, _validate_raw_axis_mira,
)
from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    admit_raw_two_view_mesh_v1,
)
from .rss_vnext_rig_executor_v3 import (
    execute_rss_vnext_rig_capability_v3, _seal,
)

EXECUTOR_CONTRACT = "RealSaS.ProductRigInference.v4"
CANONICAL_MESH = "RealSaS.CanonicalMeshIR"
STATIC_QUALIFICATION = "RealSaS.StaticCanonicalMeshQualificationIR"
ALLOWED = {
    RAW_SOURCE, RAW_SCENE, RAW_VIEWS, RAW_RECEIPT, RAW_CONFIG,
    RAW_DECODER, CANONICAL_MESH, STATIC_QUALIFICATION,
}
MAX_CONFIG = 128 * 1024
MAX_DECODER = 512 * 1024 * 1024


def _bytes(ref: dict, reader: Callable, *, limit: int = 256 * 1024 * 1024) -> bytes:
    if (
        type(ref) is not dict
        or type(ref.get("size_bytes")) is not int
        or not 1 <= ref["size_bytes"] <= limit
    ):
        raise RuntimeError("RSS_VNEXT_RIG_V4_CAS_BUDGET_OR_REF_INVALID")
    blob = reader(ref)
    if (
        type(blob) is not bytes or len(blob) != ref["size_bytes"]
        or sha256(blob).hexdigest() != ref.get("content_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RIG_V4_CAS_BYTES_OR_SHA_DRIFT")
    return blob


def _json(ref, reader: Callable) -> dict:
    try:
        value = json.loads(_bytes(ref, reader).decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_RIG_V4_JSON_PARENT_INVALID") from exc
    if type(value) is not dict:
        raise RuntimeError("RSS_VNEXT_RIG_V4_OBJECT_PARENT_REQUIRED")
    return value


def execute_rss_vnext_rig_capability_v4(
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
        raise RuntimeError("RSS_VNEXT_RIG_V4_ROLE_OR_MODEL_CONTRACT")
    refs = request.get("input_artifacts") or ()
    if type(refs) not in (list, tuple) or any(type(r) is not dict for r in refs):
        raise RuntimeError("RSS_VNEXT_RIG_V4_PARENT_LIST_INVALID")
    sources = [r for r in refs if r.get("artifact_type") == RAW_SOURCE]
    if not sources:
        legacy = {
            **request, "capability_metadata": {
                **meta, "executor_contract": "RealSaS.ProductRigInference.v3",
            },
        }
        return execute_rss_vnext_rig_capability_v3(
            legacy, read_object=read_object, put_bytes=put_bytes,
        )
    if (
        len(sources) not in (1, 2) or len(refs) != len(sources) + 7
        or {r.get("artifact_type") for r in refs} != ALLOWED
        or any(r.get("schema_version") != (
            "v2" if r["artifact_type"] == "RealSaS.CanonicalRigIR" else "v1"
        ) for r in refs)
        or any(
            r.get(k) != sources[0].get(k) for r in sources[1:]
            for k in ("content_sha256", "size_bytes", "storage_key")
        )
    ):
        raise RuntimeError("RSS_VNEXT_RIG_V4_EXACT_RAW_PARENTS_REQUIRED")

    def one(kind: str) -> dict:
        found = [r for r in refs if r.get("artifact_type") == kind]
        if len(found) != 1:
            raise RuntimeError("RSS_VNEXT_RIG_V4_DUPLICATE_OR_MISSING_CHILD")
        return found[0]

    source_ref = sources[0]
    source = decode_uncalibrated_two_image_source_v1(
        _bytes(source_ref, read_object, limit=70 * 1024 * 1024),
    )
    scene_ref = one(RAW_SCENE)
    config_ref, decoder_ref = one(RAW_CONFIG), one(RAW_DECODER)
    receipt = _json(one(RAW_RECEIPT), read_object)
    if (
        receipt.get("source_content_sha256") != source_ref["content_sha256"]
        or receipt.get("source_binding_sha256") != source.source_binding_sha256
        or receipt.get("scene_content_sha256") != scene_ref["content_sha256"]
        or receipt.get("unified_model_config_content_sha256") != config_ref["content_sha256"]
        or receipt.get("embedded_mira_decoder_content_sha256") != decoder_ref["content_sha256"]
        or receipt.get("original_camera_calibrated") is not False
        or receipt.get("source_ray_evidence_qualified") is not False
        or receipt.get("canonical_mesh_qualified") is not False
        or receipt.get("product_revision_minted") is not False
    ):
        raise RuntimeError("RSS_VNEXT_RIG_V4_SOURCE_OR_MODEL_PARENT_DRIFT")
    scene = _archive(
        _bytes(scene_ref, read_object),
        required=SCENE_REQUIRED, allowed=SCENE_REQUIRED | MESH_KEYS | AXIS_MIRA_KEYS | RAW_GRAPHICS_KEYS,
    )
    measured = _validate_candidate(scene, receipt)
    raw_anchor = receipt.get("raw_model_socket_face_candidate")
    if raw_anchor is not None:
        from .rss_vnext_raw_anchor_face_evidence_v1 import (
            validate_raw_neural_socket_faces_v1,
        )
        socket_face_replay = validate_raw_neural_socket_faces_v1(
            raw_anchor, scene=scene, construction_receipt=receipt,
        )
    else:
        # Older test/research receipts remain readable, but MUST NOT claim
        # any learned socket-face evidence or product attachment authority.
        socket_face_replay = {
            "state": "NO_MODEL_SOCKET_FACE_EVIDENCE",
            "candidate_count": 0, "abstain_count": 0,
            "attachment_qualified": False,
        }
    controls = _validate_raw_axis_mira(scene, receipt, measured)
    if (
        measured["mesh_state"] != "UNQUALIFIED_MESH_CANDIDATE"
        or controls["control_state"] != "UNQUALIFIED_AXISMIRA_CONTROL_CANDIDATE"
        or controls["control_event_count"] < 1
    ):
        raise RuntimeError("RSS_VNEXT_RIG_V4_NO_LEARNED_CAUSAL_PROGRAM")

    expected, static, _ = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt,
        scene_digest=scene_ref["content_sha256"],
    )
    canonical = canonical_mesh_from_dict_v1(_json(one(CANONICAL_MESH), read_object))
    if (
        expected is None or static is None
        or canonical != expected
        or canonical.mesh_lineage_hash != expected.mesh_lineage_hash
        or _json(one(STATIC_QUALIFICATION), read_object) != static.to_dict()
        or not measured["component_id_continuity"]
    ):
        raise RuntimeError("RSS_VNEXT_RIG_V4_CANONICAL_MESH_NOT_REPRODUCIBLE")

    from models.rss_vnext.model_config_archive_v1 import (
        decode_rss_vnext_unified_model_config_v1,
    )
    from models.rss_vnext.embedded_mira_decoder_v1 import (
        load_embedded_mira_decoder_v1,
    )
    config_blob = _bytes(config_ref, read_object, limit=MAX_CONFIG)
    try:
        config_claim = json.loads(config_blob.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_RIG_V4_CONFIG_NOT_JSON") from exc
    cfg = decode_rss_vnext_unified_model_config_v1(
        config_blob,
        expected_checkpoint_sha256=receipt["checkpoint_sha256"],
        expected_config_sha256=config_claim["unified_model_config_sha256"],
    )
    _, codec = load_embedded_mira_decoder_v1(
        _bytes(decoder_ref, read_object, limit=MAX_DECODER),
        expected_parent_unified_checkpoint_sha256=receipt["checkpoint_sha256"],
        expected_subcheckpoint_sha256=decoder_ref["content_sha256"],
        expected_model_config_sha256=config_claim["unified_model_config_sha256"],
        model_config=cfg,
    )
    from models.rss_vnext.candidate_rig_court_v1 import _events
    from models.axis.rig_program_decode_v6 import decode_axis_v6_events
    program, codes = decode_axis_v6_events(
        _events(tuple(receipt["axis_mira_control_events"])),
        config=cfg.axis_config(),
        normalization_center=(0., 0., 0.),
        normalization_scale=1.,
        mesh_lineage_hash=canonical.mesh_lineage_hash,
        feature_evidence_binding_hash=canonical.source_evidence_binding_hash,
        carrier_evidence_hash=canonical.carrier_evidence_hash,
        carrier_topology_hash=canonical.carrier_topology_hash,
        carrier_geometry_hash=canonical.carrier_geometry_hash,
        model_provenance="RSS.vNext.UnifiedInference:" + receipt["checkpoint_sha256"],
        codec_binding=codec,
    )
    for constraint in program.constraints:
        if constraint.operator_id not in RIG_CONSTRAINT_OPERATOR_IDS_V1:
            raise QualificationError("RSS_VNEXT_RIG_V4_NONEXECUTABLE_OPERATOR")
        validate_rig_constraint_operator_v1(
            operator_id=constraint.operator_id,
            input_control_count=len(constraint.input_proposal_control_ids),
            output_joint_count=len(constraint.output_proposal_joint_ids),
            has_chain=constraint.proposal_chain_id is not None,
            parameters=constraint.parameters,
        )
    qualified = qualify_rig_program_v1(program)
    validate_skin_code_proposal_v1(
        codes, expected_proposal_joint_ids=tuple(
            row.proposal_joint_id for row in program.joints
        ),
        expected_rig_program_proposal_hash=program.proposal_lineage_hash,
    )
    rig = mint_canonical_rig_v2(canonical, qualified_program=qualified)
    provenance = {
        "schema": "RealSaS.RSSvNext.RawCanonicalRigCompileReceipt.v1",
        "original_source_content_sha256": source_ref["content_sha256"],
        "source_binding_sha256": source.source_binding_sha256,
        "unified_checkpoint_sha256": receipt["checkpoint_sha256"],
        "embedded_mira_decoder_sha256": codec.checkpoint_hash,
        "canonical_mesh_lineage_hash": canonical.mesh_lineage_hash,
        "canonical_rig_lineage_hash": rig.rig_lineage_hash,
        "skin_code_proposal_hash": codes.proposal_lineage_hash,
        "axis_event_count": controls["control_event_count"],
        "model_socket_face_replay": socket_face_replay,
        "source_camera_observed": False,
        "source_3d_joint_teacher_claimed": False,
        "product_revision_minted": False,
    }
    provenance["receipt_hash"] = content_sha256(provenance)
    outputs = [
        _seal(rig, role="canonical_rig",
              kind="RealSaS.CanonicalRigIR", version="v2", put_bytes=put_bytes),
        _seal(codes, role="skin_code_proposal",
              kind="RealSaS.SkinCodeProposalIR", version="v1", put_bytes=put_bytes),
        _seal(qualified, role="qualified_rig_program",
              kind="RealSaS.QualifiedRigProgramIR", version="v1", put_bytes=put_bytes),
        _seal(provenance, role="rss_rig_rebind_receipt",
              kind="RealSaS.RSSvNextCanonicalRigCompileReceipt", version="v1",
              put_bytes=put_bytes),
    ]
    return {
        "capability_id": "product.rig", "status": "PASS",
        "outputs": outputs, "failure": None,
        "diagnostics": {
            "schema": "RealSaS.RSSvNext.ProductRigCompileDiagnostics.v4",
            "mesh_lineage_hash": canonical.mesh_lineage_hash,
            "rig_lineage_hash": rig.rig_lineage_hash,
            "skin_code_proposal_hash": codes.proposal_lineage_hash,
            "joint_count": len(rig.joints),
            "control_count": len(rig.controls),
            "constraint_count": len(rig.constraints),
            "attachment_anchor_count": len(rig.attachment_anchors),
            "neural_socket_face_candidate_count": socket_face_replay["candidate_count"],
            "neural_socket_face_abstain_count": socket_face_replay["abstain_count"],
            "neural_socket_face_attachment_qualified": False,
            "same_unified_checkpoint": True,
            "original_camera_observed": False,
            "source_3d_joint_teacher_claimed": False,
            "product_revision_minted": False,
        },
    }


__all__ = ["EXECUTOR_CONTRACT", "execute_rss_vnext_rig_capability_v4"]
