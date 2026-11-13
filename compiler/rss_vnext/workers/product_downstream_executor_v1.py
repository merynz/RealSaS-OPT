from __future__ import annotations

"""CAS-native downstream product consumers.

All model releases registered on the upstream Attempt remain visible in the\nPlatform execution envelope; these downstream workers neither load an extra\ncheckpoint nor treat inherited model release refs as a request for inference.\n\nThese are executable MODULE boundaries, not a substitute for RSS.vNext source
qualification. All inputs are typed, content-addressed, and independently
reopened; a successful compilation of a graphics/motion *candidate* cannot
authorize ProductRevision. Proof deliberately abstains until the source,
runtime, pose/appearance and edit-consequence gates are implemented.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from typing import Any, Callable, Mapping


GRAPHICS_CONTRACT = "RealSaS.ProductGraphicsCompile.v1"
MOTION_CONTRACT = "RealSaS.ProductMotionCompile.v2"
PROOF_CONTRACT = "RealSaS.ProductProofCompile.v1"


def _require(request: dict, expected_id: str, contract: str) -> None:
    meta = request.get("capability_metadata")
    if (
        type(request) is not dict
        or request.get("kind") != "MODULE"
        or request.get("capability_id") != expected_id
        or type(meta) is not dict
        or meta.get("executor_contract") != contract
        or meta.get("required_models", []) != []
    ):
        raise RuntimeError("PRODUCT_DOWNSTREAM_CONTRACT_OR_MODEL_DRIFT:" + expected_id)


def _read(
    request: dict, read_object: Callable, kind: str, version: str,
) -> dict[str, Any]:
    refs = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == kind
        and row.get("schema_version") == version
    ]
    if len(refs) != 1:
        raise RuntimeError("PRODUCT_DOWNSTREAM_INPUT_CARDINALITY:" + kind)
    ref = refs[0]
    raw = read_object(ref)
    if (
        type(raw) is not bytes or type(ref.get("size_bytes")) is not int
        or len(raw) != ref["size_bytes"]
        or sha256(raw).hexdigest() != ref.get("content_sha256")
    ):
        raise RuntimeError("PRODUCT_DOWNSTREAM_INPUT_CAS_DRIFT:" + kind)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("PRODUCT_DOWNSTREAM_INPUT_JSON_INVALID:" + kind) from exc
    if type(value) is not dict:
        raise RuntimeError("PRODUCT_DOWNSTREAM_INPUT_OBJECT_REQUIRED:" + kind)
    return value


def _seal(
    value: Mapping[str, Any] | object, *, role: str, kind: str,
    version: str, put_bytes: Callable,
) -> dict[str, Any]:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ref = put_bytes(raw)
    if (
        type(ref) is not dict or ref.get("content_sha256") != sha256(raw).hexdigest()
        or ref.get("size_bytes") != len(raw)
        or type(ref.get("storage_key")) is not str
    ):
        raise RuntimeError("PRODUCT_DOWNSTREAM_OUTPUT_CAS_DRIFT:" + kind)
    return {
        "role":role, "artifact_type":kind, "schema_version":version,
        "storage_key":ref["storage_key"],
        "content_sha256":ref["content_sha256"],
        "size_bytes":ref["size_bytes"],
        "authority_class":"COMPILER_CANDIDATE_NOT_PRODUCT_REVISION",
    }


def _canonical_parents(request, read_object):
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_from_dict_v1
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
    from compiler.realsas_compiler_core.canonical_skin_ir_v2 import canonical_skin_from_dict_v2
    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import canonical_mechanics_from_dict_v1

    mesh = canonical_mesh_from_dict_v1(
        _read(request,read_object,"RealSaS.CanonicalMeshIR","v1"))
    rig = canonical_rig_from_dict_v2(
        _read(request,read_object,"RealSaS.CanonicalRigIR","v2"))
    skin = canonical_skin_from_dict_v2(
        _read(request,read_object,"RealSaS.CanonicalSkinIR","v2"))
    mechanics = canonical_mechanics_from_dict_v1(
        _read(request,read_object,"RealSaS.CanonicalMechanicsIR","v1"))
    if (
        rig.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.deform_structure_hash != rig.deform_structure_hash
        or mechanics.mesh_lineage_hash != mesh.mesh_lineage_hash
        or mechanics.rig_lineage_hash != rig.rig_lineage_hash
        or mechanics.skin_lineage_hash != skin.skin_lineage_hash
    ):
        raise RuntimeError("PRODUCT_DOWNSTREAM_CANONICAL_PARENT_DRIFT")
    return mesh,rig,skin,mechanics


def _read_graphics_mechanics_intervention_v1(request, read_object, mesh, rig, skin, mechanics):
    """Strict ancestor CAS court, never a source-contact or secondary-motion pass."""
    refs = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == "RealSaS.MechanicsInterventionCourt"
        and row.get("schema_version") == "v1"
    ]
    requirement = (request.get("capability_metadata") or {}).get(
        "mechanics_intervention_required", False,
    )
    if type(requirement) is not bool or len(refs) > 1:
        raise RuntimeError("PRODUCT_GRAPHICS_MECHANICS_INTERVENTION_CARDINALITY")
    if not refs:
        if requirement:
            raise RuntimeError("PRODUCT_GRAPHICS_MECHANICS_INTERVENTION_REQUIRED")
        return None
    from compiler.realsas_compiler_core.hashing import content_sha256

    receipt = _read(
        request, read_object, "RealSaS.MechanicsInterventionCourt", "v1",
    )
    unsealed = dict(receipt)
    claimed_hash = unsealed.pop("court_hash", None)
    if (
        receipt.get("schema") != "RealSaS.MechanicsInterventionCourt.v1"
        or receipt.get("status") != "MEASURED_COMPILER_CONTROL_INTERVENTIONS_NOT_SOURCE_BEHAVIOR"
        or claimed_hash != content_sha256(unsealed)
        or receipt.get("canonical_mesh_lineage_hash") != mesh.mesh_lineage_hash
        or receipt.get("canonical_rig_lineage_hash") != rig.rig_lineage_hash
        or receipt.get("canonical_skin_lineage_hash") != skin.skin_lineage_hash
        or receipt.get("canonical_mechanics_lineage_hash") != mechanics.mechanics_lineage_hash
    ):
        raise RuntimeError("PRODUCT_GRAPHICS_MECHANICS_INTERVENTION_PARENT_OR_HASH_DRIFT")
    probes = receipt.get("probes")
    control_ids = [row.canonical_control_id for row in rig.controls]
    if (
        type(probes) is not list or len(probes) != len(control_ids)
        or any(type(row) is not dict or row.get("control_id") != control_ids[i]
               for i, row in enumerate(probes))
        or type(receipt.get("control_count")) is not int
        or receipt["control_count"] != len(control_ids)
        or type(receipt.get("anchor_count")) is not int
        or receipt["anchor_count"] != len(rig.attachment_anchors)
        or type(receipt.get("full_mesh_deformation_probe_count")) is not int
        or not 1 <= receipt["full_mesh_deformation_probe_count"] <= len(control_ids)
        or receipt.get("full_mesh_deformation_complete") is not (
            receipt["full_mesh_deformation_probe_count"] == len(control_ids)
        )
    ):
        raise RuntimeError("PRODUCT_GRAPHICS_MECHANICS_INTERVENTION_CONTROL_OR_SOCKET_COVERAGE")
    for claim in (
        "source_attachment_contact_qualified", "learned_secondary_motion_qualified",
        "collision_qualified", "source_pose_quality_qualified",
        "edit_consequence_qualified", "product_revision_minted",
    ):
        if receipt.get(claim) is not False:
            raise RuntimeError("PRODUCT_GRAPHICS_MECHANICS_INTERVENTION_FALSE_AUTHORITY:" + claim)
    return receipt


def _raw_source_rgb_candidate_from_cas_v1(
    request: dict, read_object: Callable, mesh,
) -> tuple[bytes, dict[str, Any]]:
    """Read original upload CAS and derive an explicitly unqualified
    image-plane color/face correspondence, not a rendered UV texture."""
    from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
        decode_uncalibrated_two_image_source_v1,
    )
    from compiler.realsas_compiler_services.platform_worker.rss_vnext_cas_parent_v1 import (
        read_rss_vnext_parent_cas_v1,
    )
    from compiler.rss_vnext.workers.rss_vnext_source_rgb_projection_candidate_v1 import (
        build_raw_source_rgb_image_plane_candidates_v1,
    )
    source_type = "RealSaS.RSSvNextUncalibratedTwoImageSource"
    refs = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == source_type
    ]
    if (
        len(refs) not in (1, 2)
        or any(ref.get("schema_version") != "v1" for ref in refs)
        or any(
            (r.get("content_sha256"), r.get("size_bytes"), r.get("storage_key"))
            != (refs[0].get("content_sha256"), refs[0].get("size_bytes"),
                refs[0].get("storage_key"))
            for r in refs[1:]
        )
    ):
        raise RuntimeError("PRODUCT_GRAPHICS_ORIGINAL_RGB_CAS_SOURCE_CARDINALITY")
    source = decode_uncalibrated_two_image_source_v1(
        read_rss_vnext_parent_cas_v1(
            refs[0], read_object, limit=70 * 1024 * 1024,
        )
    )
    return build_raw_source_rgb_image_plane_candidates_v1(source=source, mesh=mesh)


def _seal_binary_candidate_v1(
    body: bytes, *, role: str, kind: str, put_bytes: Callable,
) -> dict[str, Any]:
    from compiler.rss_vnext.workers.rss_vnext_source_rgb_projection_candidate_v1 import (
        MAX_ARTIFACT_BYTES,
    )
    if not body or len(body) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("PRODUCT_GRAPHICS_SOURCE_RGB_CAS_BUDGET")
    ref = put_bytes(body)
    if (
        type(ref) is not dict
        or ref.get("content_sha256") != sha256(body).hexdigest()
        or ref.get("size_bytes") != len(body)
        or type(ref.get("storage_key")) is not str
    ):
        raise RuntimeError("PRODUCT_GRAPHICS_SOURCE_RGB_CAS_WRITE_DRIFT")
    return {
        "role": role, "artifact_type": kind, "schema_version": "v1",
        "storage_key": ref["storage_key"],
        "content_sha256": ref["content_sha256"],
        "size_bytes": ref["size_bytes"],
        "authority_class": "COMPILER_CANDIDATE_NOT_PRODUCT_REVISION",
    }


def _read_dynamic_integrity_proof_v1(
    request: dict, read_object: Callable, mesh, rig, skin, mechanics,
):
    """Proof independently reruns moving LBS/collision from CAS parents.

    A rehashed JSON receipt alone is not collision evidence. Bad/missing
    witnesses must not be promoted to PASS by source claims or neural labels.
    """
    rows = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict
        and row.get("artifact_type") == "RealSaS.MechanicsDynamicIntegrityCourt"
        and row.get("schema_version") == "v1"
    ]
    if len(rows) > 1:
        raise RuntimeError("PRODUCT_PROOF_DYNAMIC_INTEGRITY_CARDINALITY")
    if not rows:
        return None
    from compiler.realsas_compiler_core.mechanics_dynamic_integrity_court_v1 import (
        validate_mechanics_dynamic_integrity_court_v1,
    )
    receipt = _read(
        request, read_object, "RealSaS.MechanicsDynamicIntegrityCourt", "v1",
    )
    return validate_mechanics_dynamic_integrity_court_v1(
        receipt, mesh=mesh, rig=rig, skin=skin, mechanics=mechanics,
    )


def _replay_raw_source_graphics_proof_v1(
    request: dict, read_object: Callable, *, mesh, rig, skin,
    mechanics, compiled, intervention,
) -> dict[str, Any] | None:
    """Independently replay original alpha + neural typed Graphics at Proof.

    A compiled candidate hash and even a newly rehashed receipt are insufficient:
    replay the two *original uploaded image bytes*, the one-checkpoint Shape
    scene, the TESSA source-alpha Mesh qualification, and anonymous typed
    face graph from CAS. The public product remains FAIL until source RGB,
    moving occlusion, renderer, contact and edit evidence are independently
    qualified; this court only seals the static candidate's actual lineage.
    """
    raw_type = "RealSaS.RSSvNextUncalibratedTwoImageSource"
    receipt_type = "RealSaS.RSSvNextGraphicsNativeCandidateReceipt"
    refs = request.get("input_artifacts") or ()
    raws = [
        r for r in refs if type(r) is dict and r.get("artifact_type") == raw_type
    ]
    supplied = [
        r for r in refs if type(r) is dict
        and r.get("artifact_type") == receipt_type
    ]
    if not raws:
        if supplied or compiled.metadata.get("origin") == (
            "RAW_TWO_IMAGE_UNIFIED_H_TYPED_GRAPHICS_CANDIDATES"
        ):
            raise RuntimeError("PRODUCT_PROOF_ORPHAN_RAW_GRAPHICS_RECEIPT")
        return None
    if (
        len(raws) not in (1, 2)
        or len(supplied) != 1
        or supplied[0].get("schema_version") != "v1"
        or any(r.get("schema_version") != "v1" for r in raws)
    ):
        raise RuntimeError("PRODUCT_PROOF_RAW_GRAPHICS_PARENT_CARDINALITY")
    from .rss_vnext_graphics_consumer_v1 import _raw_two_image_graphics_v1
    from compiler.realsas_compiler_core.graphics_v1 import (
        compiled_graphics_hash_v1,
    )
    from compiler.realsas_compiler_core.hashing import content_sha256
    # The caller already reopened all canonical parents and the persisted
    # Graphics/Runtime graph. This is a *second* compiler evaluation from
    # source CAS, never a second ModelRelease/inference invocation.
    replayed_graph, replayed_receipt = _raw_two_image_graphics_v1(
        request, canonical_mesh=mesh, canonical_rig=rig,
        canonical_skin=skin, read_object=read_object,
    )
    if intervention is not None:
        bound = replace(replayed_graph, metadata={
            **replayed_graph.metadata,
            "canonical_mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
            "mechanics_intervention_court_hash": intervention["court_hash"],
            "mechanical_socket_pose_probes_consumed": intervention["control_count"],
            "source_attachment_contact_qualified": False,
            "secondary_motion_qualified": False,
        }, graph_hash="")
        replayed_graph = replace(
            bound, graph_hash=compiled_graphics_hash_v1(bound),
        )
        replayed_receipt = {
            **replayed_receipt,
            "candidate_graph_hash": replayed_graph.graph_hash,
            "mechanics_intervention_court_hash": intervention["court_hash"],
        }
        replayed_receipt.pop("receipt_hash", None)
        replayed_receipt["receipt_hash"] = content_sha256(replayed_receipt)
    if (
        compiled.graph_hash != replayed_graph.graph_hash
        or content_sha256(compiled.to_dict()) != content_sha256(replayed_graph.to_dict())
    ):
        raise RuntimeError("PRODUCT_PROOF_RAW_GRAPHICS_SOURCE_REPLAY_MISMATCH")
    claimed_receipt = _read(
        request, read_object, receipt_type, "v1",
    )
    if (
        claimed_receipt.get("schema") !=
        "RealSaS.RSSvNext.GraphicsNativeCandidateReceipt.v1"
        or claimed_receipt.get("receipt_hash") != content_sha256({
            key: value for key, value in claimed_receipt.items()
            if key != "receipt_hash"
        })
        or content_sha256(claimed_receipt) != content_sha256(replayed_receipt)
        or claimed_receipt.get("candidate_graph_hash") != compiled.graph_hash
        or claimed_receipt.get("graphics_source_observation_qualified") is not False
        or claimed_receipt.get("source_hidden_surface_qualified") is not False
        or claimed_receipt.get("pose_conditioned_visibility_qualified") is not False
        or claimed_receipt.get("attachment_socket_authority") is not False
        or claimed_receipt.get("product_revision_minted") is not False
    ):
        raise RuntimeError("PRODUCT_PROOF_RAW_GRAPHICS_RECEIPT_REPLAY_MISMATCH")
    return {
        "original_two_image_source_binding_replayed": True,
        "original_alpha_static_mesh_qualification_recomputed": True,
        "same_checkpoint_typed_graphics_recomputed": True,
        "original_source_rgb_appearance_qualified": False,
        "source_camera_observed": False,
        "dynamic_source_visibility_qualified": False,
        "source_attachment_contact_qualified": False,
        "graphics_receipt_hash": replayed_receipt["receipt_hash"],
        "compiled_graphics_graph_hash": replayed_graph.graph_hash,
    }


def execute_graphics_capability_v1(
    request: dict[str, Any], *, read_object: Callable, put_bytes: Callable,
) -> dict[str, Any]:
    _require(request, "product.graphics", GRAPHICS_CONTRACT)
    mesh,rig,skin,mechanics = _canonical_parents(request,read_object)
    intervention = _read_graphics_mechanics_intervention_v1(
        request, read_object, mesh, rig, skin, mechanics,
    )

    from compiler.realsas_compiler_core.graphics_v1 import (
        GraphicsStaticRelationEvidenceV1, GraphicsViewRelationEvidenceV1,
        GraphicsEvidenceV1, graphics_evidence_hash_v1,
        compile_graphics_on_canonical_puppet_v1, compiled_graphics_hash_v1,
    )
    from compiler.realsas_compiler_core.graphics_runtime_v1 import build_runtime_graphics_v1

    # In the active one-trigger product route, Shape v3 already emitted the
    # shared Semantic/Graphics scene. A manually authored GraphicsEvidence
    # source must NOT be a hidden prerequisite for product.graphics.
    supplied_refs = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == "RealSaS.GraphicsEvidence"
        and row.get("schema_version") == "v1"
    ]
    if len(supplied_refs) > 1:
        raise RuntimeError("PRODUCT_GRAPHICS_AMBIGUOUS_EXTERNAL_EVIDENCE")
    native_receipt = None
    if supplied_refs:
        # Explicit legacy/test evidence remains a separate, inspectable
        # authority scope. It does not become source-observed appearance.
        supplied = _read(request,read_object,"RealSaS.GraphicsEvidence","v1")
        if supplied.get("schema_version") != "RealSaS.GraphicsEvidence.v1":
            raise RuntimeError("PRODUCT_GRAPHICS_EVIDENCE_SCHEMA_DRIFT")
        try:
            evidence = GraphicsEvidenceV1(
                mesh_binding_hash=supplied["mesh_binding_hash"],
                static_relations=tuple(
                    GraphicsStaticRelationEvidenceV1(**row)
                    for row in supplied["static_relations"]),
                view_relations=tuple(
                    GraphicsViewRelationEvidenceV1(**row)
                    for row in supplied["view_relations"]),
                evidence_hash=supplied["evidence_hash"],
                metadata=supplied.get("metadata") or {},
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("PRODUCT_GRAPHICS_EVIDENCE_DECODE_INVALID") from exc
        if (
            evidence.evidence_hash != graphics_evidence_hash_v1(evidence)
            or evidence.mesh_binding_hash != mesh.mesh_lineage_hash
        ):
            raise RuntimeError("PRODUCT_GRAPHICS_EVIDENCE_HASH_OR_MESH_DRIFT")
        compiled = compile_graphics_on_canonical_puppet_v1(
            mesh=mesh,rig=rig,skin=skin,
            source_graphics_binding_hash=evidence.evidence_hash,
            evidence=evidence,
        )
        source_kind = "EXPLICIT_GRAPHICS_EVIDENCE_NOT_RSS_SOURCE_TRUTH"
    else:
        from .rss_vnext_graphics_consumer_v1 import (
            compile_rss_vnext_shape_native_graphics_v1,
        )
        compiled, native_receipt = compile_rss_vnext_shape_native_graphics_v1(
            request, canonical_mesh=mesh, canonical_rig=rig,
            canonical_skin=skin, read_object=read_object,
        )
        source_kind = "SINGLE_RSSVNEXT_SHAPE_INFERENCE_REGION_CANDIDATES"

    if intervention is not None:
        from compiler.realsas_compiler_core.hashing import content_sha256

        # The compiler's Graphics result is content-addressed to the exact
        # non-rest solver/socket witness. Neither neural attachment affinity
        # nor a static raster can substitute for source contact authority.
        bound = replace(
            compiled,
            metadata={
                **compiled.metadata,
                "canonical_mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
                "mechanics_intervention_court_hash": intervention["court_hash"],
                "mechanical_socket_pose_probes_consumed": intervention["control_count"],
                "source_attachment_contact_qualified": False,
                "secondary_motion_qualified": False,
            },
            graph_hash="",
        )
        compiled = replace(bound, graph_hash=compiled_graphics_hash_v1(bound))
        if native_receipt is not None:
            native_receipt = {
                **native_receipt,
                "candidate_graph_hash": compiled.graph_hash,
                "mechanics_intervention_court_hash": intervention["court_hash"],
            }
            native_receipt.pop("receipt_hash", None)
            native_receipt["receipt_hash"] = content_sha256(native_receipt)
    raw_source_rgb = None
    raw_source_rgb_summary = None
    if compiled.metadata.get("origin") == "RAW_TWO_IMAGE_UNIFIED_H_TYPED_GRAPHICS_CANDIDATES":
        raw_source_rgb, raw_source_rgb_summary = _raw_source_rgb_candidate_from_cas_v1(
            request, read_object, mesh,
        )
    rendered = build_runtime_graphics_v1(compiled,face_count=len(mesh.faces))
    outputs = [
        _seal(compiled,role="compiled_graphics",kind="RealSaS.CompiledGraphics",
              version="v1",put_bytes=put_bytes),
        _seal(rendered,role="runtime_graphics_candidate",
              kind="RealSaS.RuntimeGraphics",version="v1",put_bytes=put_bytes),
    ]
    if native_receipt is not None:
        outputs.append(_seal(
            native_receipt,
            role="rss_graphics_native_candidate_receipt",
            kind="RealSaS.RSSvNextGraphicsNativeCandidateReceipt",
            version="v1", put_bytes=put_bytes,
        ))
    if raw_source_rgb is not None:
        outputs.append(_seal_binary_candidate_v1(
            raw_source_rgb,
            role="rss_original_rgb_fitted_face_candidates",
            kind="RealSaS.RSSvNextSourceRGBImagePlaneCandidates",
            put_bytes=put_bytes,
        ))
    return {
        "capability_id":"product.graphics","status":"PASS",
        "outputs":outputs,"failure":None,
        "diagnostics":{
            "schema":"RealSaS.ProductGraphicsDiagnostics.v1",
            "mesh_lineage_hash":mesh.mesh_lineage_hash,
            "mechanics_lineage_hash":mechanics.mechanics_lineage_hash,
            "graphics_graph_hash":compiled.graph_hash,
            "graphics_runtime_hash":rendered.runtime_graphics_hash,
            "source_appearance_qualified":False,
            "source_original_rgb_image_plane_candidate_present": raw_source_rgb is not None,
            "source_original_rgb_image_plane_sample_count": (
                raw_source_rgb_summary["sample_count"]
                if raw_source_rgb_summary is not None else 0
            ),
            "source_rgb_surface_ownership_qualified": False,
            "graphics_source_kind":source_kind,
            "mechanics_intervention_court_hash": (
                intervention["court_hash"] if intervention is not None else None
            ),
            "mechanical_socket_pose_probes_consumed": (
                intervention["control_count"] if intervention is not None else 0
            ),
            "mechanical_source_contact_qualified":False,
            "source_visibility_qualified":False,
            "pose_conditioned_occlusion_qualified":False,
            "attachment_socket_qualified":False,
            "product_revision_minted":False,
        }
    }


def execute_motion_capability_v2(
    request: dict[str, Any], *, read_object: Callable, put_bytes: Callable,
) -> dict[str, Any]:
    _require(request, "product.motion", MOTION_CONTRACT)
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import canonical_mechanics_from_dict_v1
    from compiler.realsas_compiler_core.control_motion_ir_v1 import (
        ControlMotionClipProposalIRV1, control_motion_proposal_from_dict_v1,
        qualify_control_motion_v1, seal_control_motion_proposal_v1,
    )
    from compiler.realsas_compiler_core.canonical_motion_ir_v2 import mint_canonical_motion_v2

    rig = canonical_rig_from_dict_v2(
        _read(request,read_object,"RealSaS.CanonicalRigIR","v2"))
    mechanics = canonical_mechanics_from_dict_v1(
        _read(request,read_object,"RealSaS.CanonicalMechanicsIR","v1"))
    proposed = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict
        and row.get("artifact_type") == "RealSaS.ControlMotionClipProposalIR"
    ]
    if len(proposed) == 1:
        proposal = control_motion_proposal_from_dict_v1(
            _read(request,read_object,"RealSaS.ControlMotionClipProposalIR","v1"))
        motion_source = "EXTERNAL_CONTROL_MOTION_PROPOSAL"
    elif not proposed and (
        (request.get("capability_metadata") or {}).get("default_rest_hold_policy")
        == "DETERMINISTIC_ZERO_TRACK_REST_HOLD_V1"
    ):
        # A control-ready puppet must be able to stand still. No invented
        # animation, breathing cycle, artist motion, learned pose or retargeting
        # is hidden behind a compelling-looking Idle clip.
        proposal = seal_control_motion_proposal_v1(ControlMotionClipProposalIRV1(
            proposal_clip_id="POLICY_REST_HOLD",
            display_name="Rest Hold",
            intent="REST_HOLD_POLICY",
            duration_seconds=1.0,
            loop=True,
            tracks=(),
            rig_lineage_hash=rig.rig_lineage_hash,
            control_program_hash=rig.control_program_hash,
            mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
            metadata={
                "representation_policy":"DETERMINISTIC_ZERO_TRACK_REST_HOLD_V1",
                "authored_motion_present":False,
                "animated_idle_quality_proven":False,
                "source_motion_authority":False,
            },
        ))
        motion_source = "COMPILER_POLICY_REST_HOLD_NOT_ANIMATED_IDLE"
    else:
        raise RuntimeError("PRODUCT_MOTION_SOURCE_AMBIGUOUS_OR_ABSENT")
    qualified = qualify_control_motion_v1(proposal,rig=rig,mechanics=mechanics)
    motion = mint_canonical_motion_v2(qualified)
    outputs = [
        _seal(qualified,role="qualified_control_motion",
              kind="RealSaS.QualifiedControlMotionIR",version="v1",put_bytes=put_bytes),
        _seal(motion,role="canonical_motion",
              kind="RealSaS.CanonicalMotionIR",version="v2",put_bytes=put_bytes),
    ]
    return {
        "capability_id":"product.motion","status":"PASS",
        "outputs":outputs,"failure":None,
        "diagnostics":{
            "schema":"RealSaS.ProductMotionDiagnostics.v2",
            "rig_lineage_hash":rig.rig_lineage_hash,
            "mechanics_lineage_hash":mechanics.mechanics_lineage_hash,
            "motion_lineage_hash":motion.motion_lineage_hash,
            "clip_count":1,
            "motion_source_kind":motion_source,
            "source_motion_independently_verified":False,
            "product_revision_minted":False,
        }
    }


def execute_proof_capability_v1(
    request: dict[str, Any], *, read_object: Callable, put_bytes: Callable,
) -> dict[str, Any]:
    _require(request, "product.proof", PROOF_CONTRACT)
    # A structurally valid canonical tuple is not independent source proof,
    # but even a permanently FAIL-closed Proof must not tolerate fraudulent
    # rehashed parent content in the final Engine execution envelope.
    from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
        canonical_motion_from_dict_v2,
    )
    from compiler.realsas_compiler_core.graphics_reopen_v1 import (
        reopen_compiled_graphics_v1,
    )
    from compiler.realsas_compiler_core.hashing import content_sha256

    mesh, rig, skin, mechanics = _canonical_parents(request, read_object)
    graphics_payload = _read(
        request, read_object, "RealSaS.CompiledGraphics", "v1",
    )
    motion = canonical_motion_from_dict_v2(
        _read(request, read_object, "RealSaS.CanonicalMotionIR", "v2")
    )
    compiled, rebuilt_runtime = reopen_compiled_graphics_v1(
        graphics_payload, face_count=len(mesh.faces),
    )
    if (
        compiled.canonical_mesh_binding_hash != mesh.mesh_lineage_hash
        or compiled.canonical_mesh_skin_binding_hash != skin.skin_lineage_hash
        or motion.rig_lineage_hash != rig.rig_lineage_hash
        or motion.control_program_hash != rig.control_program_hash
        or motion.mechanics_lineage_hash != mechanics.mechanics_lineage_hash
    ):
        raise RuntimeError("PRODUCT_PROOF_CANONICAL_PARENT_LINEAGE_DRIFT")
    metadata = request.get("capability_metadata") or {}
    require_runtime = metadata.get("runtime_graphics_required", False)
    if type(require_runtime) is not bool:
        raise RuntimeError("PRODUCT_PROOF_RUNTIME_GRAPHICS_POLICY_TYPE")
    runtime_refs = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == "RealSaS.RuntimeGraphics"
        and row.get("schema_version") == "v1"
    ]
    if len(runtime_refs) > 1 or (require_runtime and len(runtime_refs) != 1):
        raise RuntimeError("PRODUCT_PROOF_RUNTIME_GRAPHICS_CARDINALITY")
    if runtime_refs:
        reopened_runtime = _read(
            request, read_object, "RealSaS.RuntimeGraphics", "v1",
        )
        if (
            reopened_runtime.get("schema_version") != "RealSaS.RuntimeGraphics.v1"
            or reopened_runtime.get("runtime_graphics_hash")
            != rebuilt_runtime.runtime_graphics_hash
            or content_sha256(reopened_runtime) != content_sha256(rebuilt_runtime.to_dict())
        ):
            raise RuntimeError("PRODUCT_PROOF_RUNTIME_GRAPHICS_REPLAY_DRIFT")
    intervention = _read_graphics_mechanics_intervention_v1(
        request, read_object, mesh, rig, skin, mechanics,
    )
    if intervention is not None and (
        compiled.metadata.get("canonical_mechanics_lineage_hash")
        != mechanics.mechanics_lineage_hash
        or compiled.metadata.get("mechanics_intervention_court_hash")
        != intervention["court_hash"]
    ):
        raise RuntimeError("PRODUCT_PROOF_GRAPHICS_MECHANICS_WITNESS_DRIFT")
    raw_replay = _replay_raw_source_graphics_proof_v1(
        request, read_object, mesh=mesh, rig=rig, skin=skin,
        mechanics=mechanics, compiled=compiled,
        intervention=intervention,
    )
    dynamic_integrity = _read_dynamic_integrity_proof_v1(
        request, read_object, mesh, rig, skin, mechanics,
    )
    dynamic_required = metadata.get("dynamic_integrity_court_required", False)
    if type(dynamic_required) is not bool:
        raise RuntimeError("PRODUCT_PROOF_DYNAMIC_INTEGRITY_POLICY_TYPE")
    geometry_valid = (
        dynamic_integrity is not None
        and dynamic_integrity.complete_face_and_control_coverage
        and dynamic_integrity.no_new_self_intersections
        and dynamic_integrity.all_probe_vertices_finite
    )
    # This court has rechecked the whole present canonical graph and pose
    # authority. ProductPASS still requires independent source pixels/rays,
    # native parity, time-varying collision/contact and edit consequences.
    missing = (
        *(() if geometry_valid else ("DYNAMIC_COLLISION_AND_DEFORMATION_INTEGRITY",)),
        "SOURCE_APPEARANCE_AND_POSE_VISIBILITY",
        "NATIVE_GRAPHICS_RENDERER_PARITY",
        "CONTACT_AND_ATTACHMENT_BEHAVIOR",
        "EDIT_CONSEQUENCE_AND_RUNTIME_RSS_REOPEN",
        "SOURCE_AND_MODEL_RELEASE_ADMISSION",
    )
    diagnostics = {
        "schema": "RealSaS.ProductProofDiagnostics.v1",
        "parent_artifacts_replayed": True,
        "typed_canonical_mesh_rig_skin_reopened": True,
        "typed_motion_reopened": True,
        "graphics_runtime_face_order_recomputed": True,
        "persisted_graphics_runtime_replayed": bool(runtime_refs),
        "mechanics_intervention_court_hash": (
            intervention["court_hash"] if intervention is not None else None
        ),
        "dynamic_integrity_court_hash": (
            dynamic_integrity.court_hash if dynamic_integrity is not None else None
        ),
        "dynamic_integrity_complete": bool(
            dynamic_integrity and dynamic_integrity.complete_face_and_control_coverage
        ),
        "dynamic_collision_nonregression_qualified": bool(geometry_valid),
        "dynamic_integrity_required_by_release": dynamic_required,
        "dynamic_integrity_recomputed_in_proof": dynamic_integrity is not None,
        "graphics_graph_hash": compiled.graph_hash,
        "motion_lineage_hash": motion.motion_lineage_hash,
        "canonical_parent_binding_hash": content_sha256({
            "mesh": mesh.mesh_lineage_hash,
            "rig": rig.rig_lineage_hash,
            "skin": skin.skin_lineage_hash,
            "mechanics": mechanics.mechanics_lineage_hash,
            "graphics": compiled.graph_hash,
            "motion": motion.motion_lineage_hash,
        }),
        "raw_original_source_graphics_replay": raw_replay,
        "raw_two_image_replay_completed": raw_replay is not None,
        "source_visibility_independent_witness_present": False,
        "source_contact_independent_witness_present": False,
        "missing_independent_witnesses": list(missing),
        "product_pass_authority": False,
        "product_revision_minted": False,
    }
    return {
        "capability_id": "product.proof",
        "status": "FAIL",
        "outputs": [],
        "failure": {
            "code": "PRODUCT_PROOF_INDEPENDENT_WITNESSES_MISSING",
            "class": "QUALIFICATION",
            "reported_owner_module_id": "product.proof",
            "diagnostics": diagnostics,
        },
        "diagnostics": diagnostics,
    }


__all__ = [
    "GRAPHICS_CONTRACT","MOTION_CONTRACT","PROOF_CONTRACT",
    "execute_graphics_capability_v1","execute_motion_capability_v2",
    "execute_proof_capability_v1",
]
