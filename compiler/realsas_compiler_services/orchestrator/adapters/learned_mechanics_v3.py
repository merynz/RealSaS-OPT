from __future__ import annotations

"""Carrier-native AXIS/MIRA adapters for stages 26-33.

Legacy V2 adapters remain available for historical replay. This module makes the
current research line explicit:
- AXIS hard parent decisions are validated/materialized, never reselected.
- MIRA predicts W_M directly on the exact Stage19 mechanical carrier.
- GSA is evidence only; no surface skin authority is minted in the normal path.
"""

import json

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import (
    carrier_skin_proposal_from_dict,
    qualified_carrier_skin_from_dict,
    qualify_carrier_skin_v1,
)
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    mechanical_carrier_evidence_from_dict,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    model_checkpoint_seal_from_dict,
    model_fit_execution_from_dict,
    model_fit_preregistration_from_dict,
)
from compiler.realsas_compiler_core.required_tree_v1 import (
    qualify_required_skeleton_tree_v1,
)
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    resolved_path,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_services.orchestrator.adapters.learned_mechanics_legacy_v2 import (
    _skeleton_proposal,
    execute_geppetto_fit_stage,
    preregister_geppetto_fit_stage,
    seal_geppetto_checkpoint_stage,
)
from compiler.realsas_compiler_services.orchestrator.adapters.model_execution_common_v2 import (
    build_fit_execution,
    build_fit_preregistration,
    seal_model_checkpoint,
)


def _execution_json_any(execution, *, allowed_schemas: set[str]) -> dict:
    if execution.proposal_path is None or execution.proposal_sha256 is None:
        raise QualificationError("MODEL_PROPOSAL_EXECUTION_REF_MISSING")
    path = resolved_path(execution.proposal_path)
    if not path.is_file() or sha256_file(path) != execution.proposal_sha256:
        raise QualificationError("MODEL_PROPOSAL_BYTES_DRIFT")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = str(payload.get("schema") or payload.get("schema_version") or "")
    if actual not in allowed_schemas:
        raise QualificationError(
            f"MODEL_PROPOSAL_SCHEMA_DRIFT:{actual}!={sorted(allowed_schemas)}"
        )
    return payload


def _load_carrier(ctx):
    return mechanical_carrier_evidence_from_dict(
        stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.MechanicalCarrierEvidenceIR.v1",
        )
    )


def qualify_skeleton_stage(ctx: dict) -> dict:
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    execution = model_fit_execution_from_dict(
        stage_output_payload(
            ctx, "27_GEPPETTO_FIT", "RealSaS.ModelFitExecutionIR.v1"
        )
    )
    carrier = _load_carrier(ctx)
    upstream = dict(execution.upstream_bindings)
    if (
        upstream.get("mechanical_carrier_evidence") != carrier.carrier_evidence_hash
        or upstream.get("mechanical_carrier_topology") != carrier.topology_hash
    ):
        raise QualificationError("SKELETON_QUALIFICATION_CARRIER_BINDING_DRIFT")

    payload = _execution_json_any(
        execution, allowed_schemas={"RealSaS.SkeletonProposalIR.v1"}
    )
    proposal = _skeleton_proposal(payload)
    if proposal.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("SKELETON_QUALIFICATION_SURFACE_DRIFT")

    authority_bindings = {
        "mechanical_carrier_evidence": carrier.carrier_evidence_hash,
        "mechanical_carrier_topology": carrier.topology_hash,
        "mechanical_carrier_geometry": carrier.geometry_hash,
        "candidate_mesh": carrier.candidate_mesh_binding_hash,
        "static_mesh_qualification": carrier.static_mesh_qualification_binding_hash,
    }
    meta = dict(proposal.metadata or {})
    if meta.get("hard_causal_parent_authority") is True:
        skeleton = qualify_required_skeleton_tree_v1(
            surface,
            proposal,
            authority_bindings=authority_bindings,
        )
    else:
        policy = dict(execution.qualification_policy or {})
        unknown = set(policy) - {"run_ilp_shadow"}
        if unknown:
            raise QualificationError(
                f"AXIS_QUALIFICATION_POLICY_UNSUPPORTED:{sorted(unknown)}"
            )
        skeleton = qualify_skeleton(
            surface,
            proposal,
            run_ilp_shadow=bool(policy.get("run_ilp_shadow", False)),
            authority_bindings=authority_bindings,
        )

    root = ctx["run_root"] / "artifacts" / "28_SKELETON_QUALIFIED"
    q = dict(skeleton.qualification_report or {})
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "qualified_skeleton.json",
                skeleton,
                authority_class="QUALIFIED_SKELETON",
            )
        ],
        "diagnostics": {
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "joint_count": len(skeleton.joints),
            "optimizer": q.get("solver"),
            "optimality_proven": q.get("optimality_proven"),
            "parent_reselection_performed": q.get("parent_reselection_performed"),
            "mechanical_carrier_evidence_hash": carrier.carrier_evidence_hash,
            "mechanical_carrier_topology_hash": carrier.topology_hash,
        },
    }


def preregister_arachne_fit_stage(ctx: dict) -> dict:
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    axis = model_checkpoint_seal_from_dict(
        stage_output_payload(
            ctx,
            "29_GEPPETTO_CHECKPOINT_SEALED",
            "RealSaS.ModelCheckpointSealIR.v1",
        )
    )
    carrier = _load_carrier(ctx)
    prereg, out = build_fit_preregistration(
        ctx,
        lane="ARACHNE",
        section_key="arachne_fit",
        upstream_bindings={
            "rigging_surface_evidence": surface.geometry_lineage_hash,
            "qualified_skeleton": skeleton.skeleton_lineage_hash,
            "axis_checkpoint_seal": axis.checkpoint_seal_hash,
            "mechanical_carrier_evidence": carrier.carrier_evidence_hash,
            "mechanical_carrier_topology": carrier.topology_hash,
            "mechanical_carrier_geometry": carrier.geometry_hash,
        },
        stage_id="30_ARACHNE_FIT_PREREGISTERED",
    )
    policy = dict(prereg.qualification_policy or {})
    required = {
        "max_simplex_repair_l1",
        "max_total_correction_l1",
        "negative_tolerance",
    }
    if not required.issubset(policy):
        raise QualificationError("MIRA_PREREG_QUALIFICATION_POLICY_INCOMPLETE")
    return {
        "status": "PASS",
        "outputs": [out],
        "diagnostics": {
            "preregistration_hash": prereg.preregistration_hash,
            "architecture_id": prereg.architecture_id,
            "output_domain": "MECHANICAL_CARRIER_M",
            "mechanical_carrier_evidence_hash": carrier.carrier_evidence_hash,
            "mechanical_carrier_topology_hash": carrier.topology_hash,
            "mechanical_carrier_geometry_hash": carrier.geometry_hash,
        },
    }


def execute_arachne_fit_stage(ctx: dict) -> dict:
    prereg = model_fit_preregistration_from_dict(
        stage_output_payload(
            ctx,
            "30_ARACHNE_FIT_PREREGISTERED",
            "RealSaS.ModelFitPreregistrationIR.v1",
        )
    )
    execution, out = build_fit_execution(
        ctx,
        lane="ARACHNE",
        section_key="arachne_fit",
        prereg=prereg,
        stage_id="31_ARACHNE_FIT",
        proposal_required=True,
    )
    proposal = carrier_skin_proposal_from_dict(
        _execution_json_any(
            execution, allowed_schemas={"RealSaS.CarrierSkinProposalIR.v1"}
        )
    )
    upstream = dict(execution.upstream_bindings)
    expected = {
        "mechanical_carrier_evidence": proposal.carrier_evidence_hash,
        "mechanical_carrier_topology": proposal.carrier_topology_hash,
        "mechanical_carrier_geometry": proposal.carrier_geometry_hash,
        "qualified_skeleton": proposal.skeleton_binding_hash,
    }
    for key, value in expected.items():
        if upstream.get(key) != value:
            raise QualificationError(f"MIRA_PROPOSAL_BINDING_DRIFT:{key}")
    return {
        "status": "PASS",
        "outputs": [out],
        "diagnostics": {
            "execution_hash": execution.execution_hash,
            "proposal_sha256": execution.proposal_sha256,
            "checkpoint_sha256": execution.checkpoint_sha256,
            "qualified_skin_minted": False,
            "output_domain": "MECHANICAL_CARRIER_M",
            "semantic_skin_transfer_performed": False,
        },
    }


def qualify_skin_stage(ctx: dict) -> dict:
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    execution = model_fit_execution_from_dict(
        stage_output_payload(
            ctx, "31_ARACHNE_FIT", "RealSaS.ModelFitExecutionIR.v1"
        )
    )
    carrier = _load_carrier(ctx)
    upstream = dict(execution.upstream_bindings)
    for key, expected in (
        ("mechanical_carrier_evidence", carrier.carrier_evidence_hash),
        ("mechanical_carrier_topology", carrier.topology_hash),
        ("mechanical_carrier_geometry", carrier.geometry_hash),
    ):
        if upstream.get(key) != expected:
            raise QualificationError(f"MIRA_QUALIFICATION_BINDING_DRIFT:{key}")

    proposal = carrier_skin_proposal_from_dict(
        _execution_json_any(
            execution, allowed_schemas={"RealSaS.CarrierSkinProposalIR.v1"}
        )
    )
    policy = dict(execution.qualification_policy or {})
    allowed = {
        "max_simplex_repair_l1",
        "max_total_correction_l1",
        "negative_tolerance",
        "max_influences",
    }
    unknown = set(policy) - allowed
    if unknown:
        raise QualificationError(
            f"MIRA_QUALIFICATION_POLICY_UNSUPPORTED:{sorted(unknown)}"
        )
    required = {
        "max_simplex_repair_l1",
        "max_total_correction_l1",
        "negative_tolerance",
    }
    if not required.issubset(policy):
        raise QualificationError("MIRA_QUALIFICATION_POLICY_INCOMPLETE")

    skin = qualify_carrier_skin_v1(
        carrier,
        skeleton,
        proposal,
        max_simplex_repair_l1=float(policy["max_simplex_repair_l1"]),
        max_total_correction_l1=float(policy["max_total_correction_l1"]),
        negative_tolerance=float(policy["negative_tolerance"]),
        max_influences=(
            None
            if policy.get("max_influences") is None
            else int(policy["max_influences"])
        ),
    )
    root = ctx["run_root"] / "artifacts" / "32_SKIN_QUALIFIED"
    q = dict(skin.qualification_report or {})
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "qualified_carrier_skin.json",
                skin,
                authority_class="QUALIFIED_CARRIER_NATIVE_SKIN",
            )
        ],
        "diagnostics": {
            "skin_lineage_hash": skin.skin_lineage_hash,
            "row_count": len(skin.rows),
            "output_domain": "MECHANICAL_CARRIER_M",
            "prediction_carrier_coverage": q.get("prediction_carrier_coverage"),
            "total_correction_l1": q.get("total_correction_l1"),
            "semantic_skin_transfer_performed": False,
            "gsa_skin_field_authority_minted": False,
            "mechanical_carrier_evidence_hash": carrier.carrier_evidence_hash,
            "mechanical_carrier_topology_hash": carrier.topology_hash,
            "mechanical_carrier_geometry_hash": carrier.geometry_hash,
        },
    }


def seal_arachne_checkpoint_stage(ctx: dict) -> dict:
    execution = model_fit_execution_from_dict(
        stage_output_payload(
            ctx, "31_ARACHNE_FIT", "RealSaS.ModelFitExecutionIR.v1"
        )
    )
    skin = qualified_carrier_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedCarrierSkinIR.v1",
        )
    )
    value, out = seal_model_checkpoint(
        ctx,
        execution=execution,
        qualified_output_binding_hash=skin.skin_lineage_hash,
        stage_id="33_ARACHNE_CHECKPOINT_SEALED",
    )
    return {
        "status": "PASS",
        "outputs": [out],
        "diagnostics": {
            "checkpoint_seal_hash": value.checkpoint_seal_hash,
            "qualified_skin_binding_hash": skin.skin_lineage_hash,
            "output_domain": "MECHANICAL_CARRIER_M",
        },
    }


__all__ = [
    "preregister_geppetto_fit_stage",
    "execute_geppetto_fit_stage",
    "qualify_skeleton_stage",
    "seal_geppetto_checkpoint_stage",
    "preregister_arachne_fit_stage",
    "execute_arachne_fit_stage",
    "qualify_skin_stage",
    "seal_arachne_checkpoint_stage",
]
