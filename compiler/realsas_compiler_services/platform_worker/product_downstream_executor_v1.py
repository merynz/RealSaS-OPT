from __future__ import annotations

"""CAS-native downstream product consumers.

All model releases registered on the upstream Attempt remain visible in the\nPlatform execution envelope; these downstream workers neither load an extra\ncheckpoint nor treat inherited model release refs as a request for inference.\n\nThese are executable MODULE boundaries, not a substitute for RSS.vNext source
qualification. All inputs are typed, content-addressed, and independently
reopened; a successful compilation of a graphics/motion *candidate* cannot
authorize ProductRevision. Proof deliberately abstains until the source,
runtime, pose/appearance and edit-consequence gates are implemented.
"""

from dataclasses import asdict
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


def execute_graphics_capability_v1(
    request: dict[str, Any], *, read_object: Callable, put_bytes: Callable,
) -> dict[str, Any]:
    _require(request, "product.graphics", GRAPHICS_CONTRACT)
    mesh,rig,skin,mechanics = _canonical_parents(request,read_object)

    from compiler.realsas_compiler_core.graphics_v1 import (
        GraphicsStaticRelationEvidenceV1, GraphicsViewRelationEvidenceV1,
        GraphicsEvidenceV1, graphics_evidence_hash_v1,
        compile_graphics_on_canonical_puppet_v1,
    )
    from compiler.realsas_compiler_core.graphics_runtime_v1 import build_runtime_graphics_v1

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
    rendered = build_runtime_graphics_v1(compiled,face_count=len(mesh.faces))
    outputs = [
        _seal(compiled,role="compiled_graphics",kind="RealSaS.CompiledGraphics",
              version="v1",put_bytes=put_bytes),
        _seal(rendered,role="runtime_graphics_candidate",
              kind="RealSaS.RuntimeGraphics",version="v1",put_bytes=put_bytes),
    ]
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
    _require(request,"product.proof",PROOF_CONTRACT)
    # The proof role is registered so the DAG is honest and schedulable, but
    # candidate Graphics and Motion do not prove source rays, GPU compositing,
    # collision, animation edit consequence, or portable RSS runtime replay.
    # Not even a syntactically valid graph may mint ProductRevision here.
    mechanics = _read(request,read_object,"RealSaS.CanonicalMechanicsIR","v1")
    graphics = _read(request,read_object,"RealSaS.CompiledGraphics","v1")
    motion = _read(request,read_object,"RealSaS.CanonicalMotionIR","v2")
    if (
        mechanics.get("schema_version") != "RealSaS.CanonicalMechanicsIR.v1"
        or graphics.get("schema_version") != "RealSaS.CompiledGraphics.v1"
        or motion.get("schema_version") != "RealSaS.CanonicalMotionIR.v2"
    ):
        raise RuntimeError("PRODUCT_PROOF_PARENT_SCHEMA_INVALID")
    if (
        graphics.get("canonical_mesh_binding_hash") != mechanics.get("mesh_lineage_hash")
        or motion.get("mechanics_lineage_hash") != mechanics.get("mechanics_lineage_hash")
        or graphics.get("canonical_mesh_skin_binding_hash") != mechanics.get("skin_lineage_hash")
    ):
        raise RuntimeError("PRODUCT_PROOF_PARENT_LINEAGE_DRIFT")
    missing = (
        "SOURCE_APPEARANCE_AND_POSE_VISIBILITY",
        "NATIVE_GRAPHICS_RENDERER_PARITY",
        "CONTACT_AND_ATTACHMENT_BEHAVIOR",
        "EDIT_CONSEQUENCE_AND_RUNTIME_RSS_REOPEN",
        "SOURCE_AND_MODEL_RELEASE_ADMISSION",
    )
    # The Go capability orchestrator commits any non-PASS MODULE result as
    # terminal FAIL, unlike its special STAGE external-evidence WAIT protocol.
    # Return an explicit qualification failure: there is no registered source
    # signal that could safely unblock a suspended product.proof MODULE job.
    diagnostics = {
        "schema":"RealSaS.ProductProofDiagnostics.v1",
        "parent_artifacts_replayed":True,
        "missing_independent_witnesses":list(missing),
        "product_pass_authority":False,
        "product_revision_minted":False,
    }
    return {
        "capability_id":"product.proof",
        "status":"FAIL",
        "outputs":[],
        "failure":{
            "code":"PRODUCT_PROOF_INDEPENDENT_WITNESSES_MISSING",
            "class":"QUALIFICATION",
            "reported_owner_module_id":"product.proof",
            "diagnostics":diagnostics,
        },
        "diagnostics":diagnostics,
    }


__all__ = [
    "GRAPHICS_CONTRACT","MOTION_CONTRACT","PROOF_CONTRACT",
    "execute_graphics_capability_v1","execute_motion_capability_v2",
    "execute_proof_capability_v1",
]
