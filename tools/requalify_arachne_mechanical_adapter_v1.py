"""Requalify a fitted Arachne mechanical adapter field in exact compiler courts.

Audit only.  The adapted surface field is submitted as SkinProposalIR and must
pass the existing skin qualifier before it is allowed into G3B/G3/motion.
No product authority is minted by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import (
    SkinInfluenceProposal,
    SkinProposalIR,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_direct_lbs_mesh_weight_projection_v1 import exact_motion_court
from tools.demo.render_knight_motion_preview_v1 import _ctx


def read(path: Path):
    return json.loads(Path(path).read_text())


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def adapted_skin_proposal(
    *,
    surface,
    skeleton,
    adapted_npz: Path,
    fit_receipt: dict | None,
):
    with np.load(adapted_npz,allow_pickle=False) as z:
        weights=np.asarray(z["weights"],dtype=np.float64)
        surface_ids=tuple(map(str,z["surface_ids"].tolist()))
        joint_ids=tuple(map(str,z["joint_ids"].tolist()))
        row_joint_mask=(
            np.asarray(z["row_joint_mask"]).astype(bool)
            if "row_joint_mask" in z.files
            else np.ones_like(weights,dtype=bool)
        )
    expected_sids=tuple(str(x.surface_id) for x in surface.surface_nodes)
    expected_jids=tuple(
        str(x.canonical_joint_id)
        for x in sorted(skeleton.joints,key=lambda j:str(j.canonical_joint_id))
    )
    if surface_ids!=expected_sids:
        raise RuntimeError("ADAPTED_SKIN_SURFACE_AXIS_DRIFT")
    if joint_ids!=expected_jids:
        raise RuntimeError("ADAPTED_SKIN_JOINT_AXIS_DRIFT")
    if weights.shape!=(len(surface_ids),len(joint_ids)):
        raise RuntimeError("ADAPTED_SKIN_WEIGHT_SHAPE_DRIFT")
    if row_joint_mask.shape!=weights.shape:
        raise RuntimeError("ADAPTED_SKIN_SUPPORT_MASK_SHAPE_DRIFT")
    if not np.isfinite(weights).all() or np.any(weights<0.0):
        raise RuntimeError("ADAPTED_SKIN_WEIGHT_INVALID")
    if np.any(weights[~row_joint_mask]!=0.0):
        raise RuntimeError("ADAPTED_SKIN_MINTED_FORBIDDEN_SUPPORT")

    normalized=np.zeros_like(weights,dtype=np.float64)
    row_corrections=[]
    influences=[]
    for ri,sid in enumerate(surface_ids):
        row=weights[ri].copy()
        s=float(row.sum())
        if not np.isfinite(s) or s<=1e-15:
            raise RuntimeError(f"ADAPTED_SKIN_ZERO_ROW:{sid}")
        norm=row/s
        correction=float(np.abs(norm-row).sum())
        row_corrections.append(correction)
        normalized[ri]=norm
        for ji,jid in enumerate(joint_ids):
            w=float(norm[ji])
            if w>0.0:
                influences.append(SkinInfluenceProposal(sid,jid,w))

    max_corr=max(row_corrections,default=0.0)
    total_corr=float(sum(row_corrections))
    if max_corr>1e-6 or total_corr>1e-3:
        raise RuntimeError(
            f"ADAPTED_SKIN_SERIALIZATION_RENORMALIZATION_EXCEEDS_BUDGET:"
            f"{max_corr}:{total_corr}"
        )
    metadata={
        "adapter_artifact_sha256":sha256(adapted_npz),
        "adapter_serialization_renormalization_max_l1":max_corr,
        "adapter_serialization_renormalization_total_l1":total_corr,
        "evidence_role":"ARACHNE_MECHANICAL_RESIDUAL_ADAPTER_OUTPUT",
        "product_authority_claimed":False,
        "uncovered_row_semantics":"REQUIRES_DOWNSTREAM_MECHANICAL_COMPATIBILITY_PROOF",
    }
    if fit_receipt is not None:
        metadata.update({
            "fit_status":str(fit_receipt.get("status") or ""),
            "fit_bundle_sha256":str(fit_receipt.get("bundle_sha256") or ""),
            "fit_best_step":int(fit_receipt.get("best_step") or 0),
            "fit_semantic_budgets_passed":bool(
                fit_receipt.get("best_semantic_budgets_passed")
            ),
        })
        if str(fit_receipt.get("adapted_weights_sha256") or "")!=sha256(adapted_npz):
            raise RuntimeError("ADAPTED_SKIN_RECEIPT_HASH_DRIFT")
        if str(fit_receipt.get("status") or "")!="PASS_FIT_COMPLETED__AWAIT_COMPILER_REQUALIFICATION":
            raise RuntimeError("ADAPTED_SKIN_FIT_RECEIPT_NOT_PASS")

    proposal=SkinProposalIR(
        influences=tuple(influences),
        surface_binding_hash=surface.geometry_lineage_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        model_provenance="ARACHNE_V6_FROZEN_PLUS_MECHANICAL_RESIDUAL_ADAPTER_V1",
        metadata=metadata,
    )
    return proposal,metadata


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--skeleton-json",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--adapted-weights",type=Path,required=True)
    ap.add_argument("--fit-receipt",type=Path)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id)
    surface=rigging_surface_from_dict(read(a.surface_json))
    skeleton=qualified_skeleton_from_dict(read(a.skeleton_json))
    candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    if candidate.surface_binding_hash!=surface.geometry_lineage_hash:
        raise RuntimeError("ADAPTED_SKIN_CANDIDATE_SURFACE_DRIFT")

    fit_receipt=read(a.fit_receipt) if a.fit_receipt else None
    proposal,proposal_meta=adapted_skin_proposal(
        surface=surface,skeleton=skeleton,adapted_npz=a.adapted_weights,
        fit_receipt=fit_receipt,
    )
    skin=qualify_skin(
        surface,skeleton,proposal,
        max_simplex_repair_l1=1e-10,
        max_total_correction_l1=1e-8,
        negative_tolerance=0.0,
        max_influences=None,
    )
    if float(skin.qualification_report["total_correction_l1"])>1e-8:
        raise RuntimeError("ADAPTED_SKIN_COMPILER_QUALIFICATION_CORRECTION_EXCEEDED")

    camera_set=qualified_camera_set_from_dict(
        stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    _,envelope=derive_deformation_envelope_v1(
        skeleton=skeleton,camera_set=camera_set
    )
    policy=mesh_policy_from_dict(
        stage_output_payload(
            ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    g3b=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,
    )
    g3b_all=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,
        risk_l1_min=0.0,stress_all_faces=True,
    )
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,
    )
    rest,W,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin
    )
    motion=exact_motion_court(
        ctx,np.asarray(rest,np.float64),np.asarray(W,np.float64),np.asarray(faces,np.int64),
        tuple(str(j.canonical_joint_id) for j in skeleton.joints),
        skeleton,cameras,policy,
    )

    result={
        "schema":"RealSaS.ArachneMechanicalAdapterCompilerRequalification.v1",
        "status":"PASS" if (
            g3b["passed"] and g3b_all["passed"] and g3.passed
            and int(motion["failed_motion_frame_count"])==0
        ) else "FAIL",
        "product_authority_minted":False,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "skin_lineage_hash":skin.skin_lineage_hash,
        "proposal_metadata":proposal_meta,
        "skin_qualification":skin.qualification_report,
        "g3b":{
            "passed":bool(g3b["passed"]),
            "unsafe_face_count":int(g3b["unsafe_face_count"]),
        },
        "g3b_all_faces":{
            "passed":bool(g3b_all["passed"]),
            "unsafe_face_count":int(g3b_all["unsafe_face_count"]),
        },
        "g3":{
            "passed":bool(g3.passed),
            "maximum_condition_number":float(g3.maximum_condition_number),
            "minimum_area_ratio":float(g3.minimum_area_ratio),
            "maximum_area_ratio":float(g3.maximum_area_ratio),
            "maximum_edge_ratio":float(g3.maximum_edge_ratio),
            "failure_invariants":list(g3.failure_invariants),
        },
        "motion":{
            "failed_motion_frame_count":int(motion["failed_motion_frame_count"]),
            "maximum_motion_edge_ratio":float(motion["maximum_motion_edge_ratio"]),
            "maximum_motion_condition_number":float(motion["maximum_motion_condition_number"]),
        },
        "claim_boundary":[
            "Adapter output is re-submitted through the existing compiler skin qualifier.",
            "No adapter output can bypass G3B/G3/actual-motion qualification.",
            "PASS here is mechanical evidence for this exact candidate lineage only.",
            "No product authority is minted by this audit tool.",
        ],
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/"QUALIFIED_ADAPTED_SKIN.json").write_text(
        json.dumps(skin.to_dict(),indent=2,sort_keys=True)+"\n"
    )
    (a.out_dir/"REPORT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ARACHNE_ADAPTER_REQUALIFICATION="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
