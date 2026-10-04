from __future__ import annotations

"""C2/A2 frozen ATLAS carrier-first compatibility court.

No fitting. ATLAS keeps its current GSA perception input. The court derives the
exact Stage18/19 mechanical carrier, binds the frozen prediction to that carrier,
and measures whether the existing rig proposal remains geometrically/structurally
compatible before the joint ATLAS+MIRA Stage35 court.

This court intentionally does NOT claim that carrier conditioning is unnecessary.
That question belongs to A3.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import (
    GeppettoReferenceStrengthConfigV1,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    GeppettoReferenceStrengthNoLearnedSlotV1,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import (
    read,
    verified_checkpoint,
    write,
)


def _compare(predicted, archived, scale: float) -> dict:
    old = {j.source_proposal_id: j for j in archived.joints}
    new = {j.source_proposal_id: j for j in predicted.joints}
    report = {
        "same_generation_id_set": set(old) == set(new),
        "pred_joint_count": len(new),
        "archived_joint_count": len(old),
    }
    shared = sorted(set(old) & set(new))
    if shared:
        d = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(new[k].position, np.float64)
                    - np.asarray(old[k].position, np.float64)
                )
                / float(scale)
                for k in shared
            ],
            np.float64,
        )
        report.update(
            shared_joint_count=len(shared),
            normalized_position_rmse=float(np.sqrt(np.mean(d * d))),
            normalized_position_p95=float(np.quantile(d, 0.95)),
            normalized_position_max=float(d.max()),
        )
    else:
        report.update(
            shared_joint_count=0,
            normalized_position_rmse=None,
            normalized_position_p95=None,
            normalized_position_max=None,
        )

    old_source = {
        j.canonical_joint_id: j.source_proposal_id for j in archived.joints
    }
    new_source = {
        j.canonical_joint_id: j.source_proposal_id for j in predicted.joints
    }
    report["changed_parent_count_shared"] = int(
        sum(
            old_source.get(old[k].parent_canonical_id)
            != new_source.get(new[k].parent_canonical_id)
            for k in shared
        )
    )
    return report


def _carrier_relative_rig_metrics(carrier, skeleton, scale: float) -> dict:
    points = np.asarray(carrier.positions, np.float64)
    joints = np.asarray([j.position for j in skeleton.joints], np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError("ATLAS_C2_CARRIER_POINTS_INVALID")
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0:
        raise RuntimeError("ATLAS_C2_JOINT_POINTS_INVALID")

    distances = np.linalg.norm(
        joints[:, None, :] - points[None, :, :],
        axis=-1,
    )
    nearest = distances.min(axis=1) / float(scale)

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    margin = max(float(scale) * 0.05, 1e-8)
    inside = np.all((joints >= lo[None] - margin) & (joints <= hi[None] + margin), axis=1)

    return {
        "carrier_vertex_count": int(len(points)),
        "carrier_face_count": int(len(carrier.face_vertex_indices)),
        "joint_count": int(len(joints)),
        "joint_inside_carrier_bbox_margin_fraction": float(np.mean(inside)),
        "joint_to_nearest_carrier_vertex_mean_norm": float(nearest.mean()),
        "joint_to_nearest_carrier_vertex_p95_norm": float(np.quantile(nearest, 0.95)),
        "joint_to_nearest_carrier_vertex_max_norm": float(nearest.max()),
    }


def main(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ATLAS_C2_CUDA_DEVICE_NOT_AVAILABLE")

    ctx = _ctx(args.authority_root, args.run_id)
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    static_qualification = static_mesh_qualification_from_dict(
        stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )
    carrier = build_mechanical_carrier_evidence_v1(
        candidate,
        static_qualification=static_qualification,
        surface_addressing=addressing,
    )
    if carrier.source_surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("ATLAS_C2_CARRIER_SURFACE_LINEAGE_DRIFT")

    execution = read(
        args.fit_run / "artifacts/27_GEPPETTO_FIT/model_fit_execution.json"
    )
    source_path = Path(inspect.getfile(GeppettoReferenceStrengthNoLearnedSlotV1))
    if sha256_file(source_path) != execution["model_source_sha256"]:
        raise RuntimeError("ATLAS_C2_MODEL_SOURCE_DRIFT")
    checkpoint = verified_checkpoint(
        Path(execution["checkpoint_path"]),
        execution["checkpoint_sha256"],
    )
    config = GeppettoReferenceStrengthConfigV1(**checkpoint["config"])
    if config.config_hash != checkpoint["config_hash"]:
        raise RuntimeError("ATLAS_C2_CONFIG_DRIFT")

    tensor = tensorize_rigging_surface_v1(surface)
    model = GeppettoReferenceStrengthNoLearnedSlotV1(config).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    torch.manual_seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        proposal = model.propose(
            tensor,
            resource_step_limit=min(args.max_steps, tensor.node_count),
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
        )
    skeleton = qualify_skeleton(surface, proposal, run_ilp_shadow=False)

    archived = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )

    support_surface_ids = {
        str(sid)
        for vertex in candidate.vertices
        for sid, weight in vertex.support_binding.coefficients
        if float(weight) > 0.0
    }
    proposal_support_ids = {
        str(sid)
        for joint in proposal.joints
        for sid in joint.support_surface_ids
    }
    unknown_support = sorted(proposal_support_ids - support_surface_ids)
    if unknown_support:
        raise RuntimeError(
            "ATLAS_C2_PROPOSAL_SUPPORT_OUTSIDE_CARRIER_PROVENANCE:"
            + ",".join(unknown_support[:16])
        )

    report = {
        "schema": "RealSaS.ATLASCarrierFirstCompatibilityCourt.v1",
        "status": "MEASURED__ZERO_TRAIN__NO_PROMOTION_CLAIM",
        "run_id": args.run_id,
        "fit_run": str(args.fit_run),
        "device": args.device,
        "seed": int(args.seed),
        "checkpoint_sha256": execution["checkpoint_sha256"],
        "architecture_id": config.architecture_id,
        "gsa_surface_lineage_hash": surface.geometry_lineage_hash,
        "gsa_tensorization_hash": tensor.tensorization_hash,
        "carrier_candidate_hash": candidate.candidate_lineage_hash,
        "carrier_static_qualification_hash": static_qualification.qualification_hash,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "carrier_geometry_hash": carrier.geometry_hash,
        "carrier_neural_conditioning_used": False,
        "gsa_perception_memory_used": True,
        "carrier_bound_evaluation_domain": True,
        "teacher_predictor_input_used": False,
        "training_used": False,
        "product_authority_minted": False,
        "proposal_support_surface_count": int(len(proposal_support_ids)),
        "proposal_support_outside_carrier_provenance_count": 0,
        "comparison_to_archived_prediction": _compare(
            skeleton, archived, tensor.normalization_scale
        ),
        "carrier_relative_rig_metrics": _carrier_relative_rig_metrics(
            carrier, skeleton, tensor.normalization_scale
        ),
        "claim_boundary": [
            "This is the frozen GSA-input ATLAS baseline bound to an exact Stage19 carrier.",
            "It does not prove that ATLAS should remain carrier-blind.",
            "Carrier-awareness is authorized only by the A3 necessity court.",
            "Teacher/reference geometry is not an inference input.",
            "Hard joint mechanical quality is deferred to the C3 ATLAS+MIRA Stage35 court.",
        ],
        "next_gate": "C3_KNIGHT_CARRIER_FIRST_ONE_SHOT_MECHANICAL_COURT",
    }
    write(args.out_dir / "atlas_skeleton_proposal.json", proposal.to_dict())
    write(args.out_dir / "atlas_qualified_skeleton.json", skeleton.to_dict())
    write(args.out_dir / "mechanical_carrier_evidence.json", carrier.to_dict())
    write(args.out_dir / "REPORT.json", report)
    print("ATLAS_CARRIER_FIRST_C2_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fit-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-steps", type=int, default=128)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(
            args.out_dir / "ERROR.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
