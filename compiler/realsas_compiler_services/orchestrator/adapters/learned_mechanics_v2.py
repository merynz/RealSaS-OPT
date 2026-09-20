from __future__ import annotations

"""V2 mainline stages 26-33: Geppetto/Arachne execution receipts and Compiler qualification."""

import json

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    model_checkpoint_seal_from_dict, model_fit_execution_from_dict,
    model_fit_preregistration_from_dict, rigging_surface_qualification_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict, qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import (
    QualificationError, SkeletonProposalEdge, SkeletonProposalIR, SkeletonProposalJoint,
    SkinInfluenceProposal, SkinProposalIR,
)
from compiler.realsas_compiler_services.orchestrator.adapters.model_execution_common_v2 import (
    build_fit_execution, build_fit_preregistration, seal_model_checkpoint,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    resolved_path, sha256_file, stage_output_payload, write_ir,
)


def _execution_json(execution,*,expected_schema:str):
    if execution.proposal_path is None or execution.proposal_sha256 is None:
        raise QualificationError("MODEL_PROPOSAL_EXECUTION_REF_MISSING")
    path=resolved_path(execution.proposal_path)
    if not path.is_file() or sha256_file(path)!=execution.proposal_sha256:
        raise QualificationError("MODEL_PROPOSAL_BYTES_DRIFT")
    payload=json.loads(path.read_text(encoding="utf-8"))
    actual=str(payload.get("schema") or payload.get("schema_version") or "")
    if actual!=expected_schema:
        raise QualificationError(f"MODEL_PROPOSAL_SCHEMA_DRIFT:{actual}!={expected_schema}")
    return payload


def _skeleton_proposal(payload:dict)->SkeletonProposalIR:
    joints=tuple(SkeletonProposalJoint(
        proposal_id=str(row["proposal_id"]),position=tuple(map(float,row["position"])),
        root_score=float(row.get("root_score",0.0)),confidence=float(row.get("confidence",1.0)),
        support_surface_ids=tuple(map(str,row.get("support_surface_ids") or ())),
        metadata=dict(row.get("metadata") or {}),
    ) for row in payload.get("joints") or ())
    edges=tuple(SkeletonProposalEdge(
        edge_id=str(row["edge_id"]),parent_proposal_id=str(row["parent_proposal_id"]),
        child_proposal_id=str(row["child_proposal_id"]),score=float(row["score"]),
        confidence=float(row.get("confidence",1.0)),hard_required=bool(row.get("hard_required",False)),
        hard_forbidden=bool(row.get("hard_forbidden",False)),reason=str(row.get("reason") or ""),
        metadata=dict(row.get("metadata") or {}),
    ) for row in payload.get("edges") or ())
    return SkeletonProposalIR(
        joints,edges,str(payload["surface_binding_hash"]),str(payload.get("model_provenance") or ""),
        schema_version=str(payload.get("schema_version") or "RealSaS.SkeletonProposalIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _skin_proposal(payload:dict)->SkinProposalIR:
    influences=tuple(SkinInfluenceProposal(
        str(row["surface_id"]),str(row["canonical_joint_id"]),float(row["weight"])
    ) for row in payload.get("influences") or ())
    return SkinProposalIR(
        influences,str(payload["surface_binding_hash"]),str(payload["skeleton_binding_hash"]),
        str(payload.get("model_provenance") or ""),
        schema_version=str(payload.get("schema_version") or "RealSaS.SkinProposalIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )


def preregister_geppetto_fit_stage(ctx:dict)->dict:
    surface=rigging_surface_from_dict(
        stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1")
    )
    qualification=rigging_surface_qualification_from_dict(
        stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceQualificationIR.v1")
    )
    prereg,out=build_fit_preregistration(
        ctx,lane="GEPPETTO",section_key="geppetto_fit",
        upstream_bindings={
            "rigging_surface":surface.geometry_lineage_hash,
            "rigging_surface_qualification":qualification.qualification_hash,
        },
        stage_id="26_GEPPETTO_FIT_PREREGISTERED",
    )
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "preregistration_hash":prereg.preregistration_hash,"architecture_id":prereg.architecture_id,
    }}


def execute_geppetto_fit_stage(ctx:dict)->dict:
    prereg=model_fit_preregistration_from_dict(
        stage_output_payload(ctx,"26_GEPPETTO_FIT_PREREGISTERED","RealSaS.ModelFitPreregistrationIR.v1")
    )
    execution,out=build_fit_execution(
        ctx,lane="GEPPETTO",section_key="geppetto_fit",prereg=prereg,
        stage_id="27_GEPPETTO_FIT",proposal_required=True,
    )
    payload=_execution_json(execution,expected_schema="RealSaS.SkeletonProposalIR.v1")
    proposal=_skeleton_proposal(payload)
    expected=dict(execution.upstream_bindings)["rigging_surface"]
    if proposal.surface_binding_hash!=expected:
        raise QualificationError("GEPPETTO_PROPOSAL_SURFACE_BINDING_DRIFT")
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "execution_hash":execution.execution_hash,"proposal_sha256":execution.proposal_sha256,
        "checkpoint_sha256":execution.checkpoint_sha256,"canonical_skeleton_minted":False,
    }}


def qualify_skeleton_stage(ctx:dict)->dict:
    surface=rigging_surface_from_dict(
        stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1")
    )
    execution=model_fit_execution_from_dict(
        stage_output_payload(ctx,"27_GEPPETTO_FIT","RealSaS.ModelFitExecutionIR.v1")
    )
    proposal=_skeleton_proposal(_execution_json(execution,expected_schema="RealSaS.SkeletonProposalIR.v1"))
    if proposal.surface_binding_hash!=surface.geometry_lineage_hash:
        raise QualificationError("SKELETON_QUALIFICATION_SURFACE_DRIFT")
    policy=dict(execution.qualification_policy or {})
    unknown=set(policy)-{"run_ilp_shadow"}
    if unknown:
        raise QualificationError(f"GEPPETTO_QUALIFICATION_POLICY_UNSUPPORTED:{sorted(unknown)}")
    skeleton=qualify_skeleton(surface,proposal,run_ilp_shadow=bool(policy.get("run_ilp_shadow",False)))
    root=ctx["run_root"]/"artifacts"/"28_SKELETON_QUALIFIED"
    return {"status":"PASS","outputs":[write_ir(root/"qualified_skeleton.json",skeleton,authority_class="QUALIFIED_SKELETON")],
        "diagnostics":{"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"joint_count":len(skeleton.joints),
                       "optimizer":skeleton.qualification_report.get("solver"),"optimality_proven":skeleton.qualification_report.get("optimality_proven")}}


def seal_geppetto_checkpoint_stage(ctx:dict)->dict:
    execution=model_fit_execution_from_dict(
        stage_output_payload(ctx,"27_GEPPETTO_FIT","RealSaS.ModelFitExecutionIR.v1")
    )
    skeleton=qualified_skeleton_from_dict(
        stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    value,out=seal_model_checkpoint(
        ctx,execution=execution,qualified_output_binding_hash=skeleton.skeleton_lineage_hash,
        stage_id="29_GEPPETTO_CHECKPOINT_SEALED",
    )
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "checkpoint_seal_hash":value.checkpoint_seal_hash,
        "qualified_skeleton_binding_hash":skeleton.skeleton_lineage_hash,
    }}


def preregister_arachne_fit_stage(ctx:dict)->dict:
    surface=rigging_surface_from_dict(
        stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1")
    )
    skeleton=qualified_skeleton_from_dict(
        stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    geppetto=model_checkpoint_seal_from_dict(
        stage_output_payload(ctx,"29_GEPPETTO_CHECKPOINT_SEALED","RealSaS.ModelCheckpointSealIR.v1")
    )
    prereg,out=build_fit_preregistration(
        ctx,lane="ARACHNE",section_key="arachne_fit",
        upstream_bindings={
            "rigging_surface":surface.geometry_lineage_hash,
            "qualified_skeleton":skeleton.skeleton_lineage_hash,
            "geppetto_checkpoint_seal":geppetto.checkpoint_seal_hash,
        },
        stage_id="30_ARACHNE_FIT_PREREGISTERED",
    )
    policy=dict(prereg.qualification_policy or {})
    required={"max_simplex_repair_l1","max_total_correction_l1","negative_tolerance"}
    if not required.issubset(policy):
        raise QualificationError("ARACHNE_PREREG_QUALIFICATION_POLICY_INCOMPLETE")
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "preregistration_hash":prereg.preregistration_hash,"architecture_id":prereg.architecture_id,
    }}


def execute_arachne_fit_stage(ctx:dict)->dict:
    prereg=model_fit_preregistration_from_dict(
        stage_output_payload(ctx,"30_ARACHNE_FIT_PREREGISTERED","RealSaS.ModelFitPreregistrationIR.v1")
    )
    execution,out=build_fit_execution(
        ctx,lane="ARACHNE",section_key="arachne_fit",prereg=prereg,
        stage_id="31_ARACHNE_FIT",proposal_required=True,
    )
    proposal=_skin_proposal(_execution_json(execution,expected_schema="RealSaS.SkinProposalIR.v1"))
    upstream=dict(execution.upstream_bindings)
    if proposal.surface_binding_hash!=upstream["rigging_surface"] or proposal.skeleton_binding_hash!=upstream["qualified_skeleton"]:
        raise QualificationError("ARACHNE_PROPOSAL_BINDING_DRIFT")
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "execution_hash":execution.execution_hash,"proposal_sha256":execution.proposal_sha256,
        "checkpoint_sha256":execution.checkpoint_sha256,"qualified_skin_minted":False,
    }}


def qualify_skin_stage(ctx:dict)->dict:
    surface=rigging_surface_from_dict(
        stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1")
    )
    skeleton=qualified_skeleton_from_dict(
        stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    execution=model_fit_execution_from_dict(
        stage_output_payload(ctx,"31_ARACHNE_FIT","RealSaS.ModelFitExecutionIR.v1")
    )
    proposal=_skin_proposal(_execution_json(execution,expected_schema="RealSaS.SkinProposalIR.v1"))
    policy=dict(execution.qualification_policy or {})
    allowed={"max_simplex_repair_l1","max_total_correction_l1","negative_tolerance","max_influences"}
    unknown=set(policy)-allowed
    if unknown:
        raise QualificationError(f"ARACHNE_QUALIFICATION_POLICY_UNSUPPORTED:{sorted(unknown)}")
    required={"max_simplex_repair_l1","max_total_correction_l1","negative_tolerance"}
    if not required.issubset(policy):
        raise QualificationError("ARACHNE_QUALIFICATION_POLICY_INCOMPLETE")
    max_influences=policy.get("max_influences")
    skin=qualify_skin(
        surface,skeleton,proposal,
        max_simplex_repair_l1=float(policy["max_simplex_repair_l1"]),
        max_total_correction_l1=float(policy["max_total_correction_l1"]),
        negative_tolerance=float(policy["negative_tolerance"]),
        max_influences=None if max_influences is None else int(max_influences),
    )
    root=ctx["run_root"]/"artifacts"/"32_SKIN_QUALIFIED"
    return {"status":"PASS","outputs":[write_ir(root/"qualified_skin.json",skin,authority_class="QUALIFIED_SKIN")],
        "diagnostics":{"skin_lineage_hash":skin.skin_lineage_hash,"row_count":len(skin.rows),
                       "total_correction_l1":skin.qualification_report.get("total_correction_l1")}}


def seal_arachne_checkpoint_stage(ctx:dict)->dict:
    execution=model_fit_execution_from_dict(
        stage_output_payload(ctx,"31_ARACHNE_FIT","RealSaS.ModelFitExecutionIR.v1")
    )
    skin=qualified_skin_from_dict(
        stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1")
    )
    value,out=seal_model_checkpoint(
        ctx,execution=execution,qualified_output_binding_hash=skin.skin_lineage_hash,
        stage_id="33_ARACHNE_CHECKPOINT_SEALED",
    )
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "checkpoint_seal_hash":value.checkpoint_seal_hash,
        "qualified_skin_binding_hash":skin.skin_lineage_hash,
    }}
