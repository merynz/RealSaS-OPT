from __future__ import annotations

"""C3.3 actual-motion court for typed GSA-field -> carrier projection.

This court tests the architecture hypothesis suggested by C3.2:

1. MIRA predicts its semantic skin field in the GSA domain where it was trained.
2. The Compiler projects that field onto the already-existing Stage19 carrier
   through the carrier's admitted mechanical support coefficients.
3. The projected field is qualified as carrier-native skin.
4. Hard G3/G3B and the exact 51-frame Knight idle/run/slash court re-prove it.

The projection is explicit, hash-bound, recomputed for this carrier and followed
by proof. It is not silent reuse and no teacher weights are consumed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mechanical_carrier_skin_v1 import (
    qualify_mechanical_carrier_skin_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from models.mira.carrier_query_v1 import build_mira_mechanical_carrier_query_v1
from tools.audit_direct_lbs_mesh_weight_projection_v1 import exact_motion_court
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.audit_knight_mira_support_mixture_c32_v1 import (
    _legacy_transfer_ceiling,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


def _faces(candidate,vertex_ids):
    row={str(v):i for i,v in enumerate(vertex_ids)}
    return np.asarray(
        [tuple(row[str(v)] for v in face) for face in candidate.faces],
        dtype=np.int64,
    )


def main(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("C33_CUDA_NOT_AVAILABLE")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

    ctx=_ctx(args.authority_root,args.run_id)
    surface=rigging_surface_from_dict(stage_output_payload(
        ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    addressing=surface_addressing_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.SurfaceAddressingIR.v1"))
    static=static_mesh_qualification_from_dict(stage_output_payload(
        ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED","RealSaS.StaticCanonicalMeshQualificationIR.v1"))
    carrier=build_mechanical_carrier_evidence_v1(
        candidate,static_qualification=static,surface_addressing=addressing)
    policy=mesh_policy_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    envelope=deformation_envelope_from_dict(stage_output_payload(
        ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    surface_tensor=tensorize_rigging_surface_v1(surface)

    conditioning,ci,memory,tokens,decoder,mira_result,telemetry=_run_mira_backbone(
        surface=surface,skeleton=skeleton,arm_dir=args.arm_dir,
        device=args.device,query_chunk=args.backbone_query_chunk)
    legacy=_decode_legacy(
        conditioning=conditioning,ci=ci,memory=memory,tokens=tokens,
        decoder=decoder,device=args.device,chunk=args.readout_chunk)

    # Query object is used only for exact carrier vertex ordering, joint axis and
    # admitted mechanical support coefficients. Its face-cross normals do not
    # enter the projection.
    query=build_mira_mechanical_carrier_query_v1(
        candidate=candidate,carrier_evidence=carrier,
        surface_tensor=surface_tensor,conditioning=conditioning,allow_unoriented_carrier_normals_for_diagnostic=True)
    projected=_legacy_transfer_ceiling(query=query,legacy_weights=legacy)

    skin=qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        skeleton=skeleton,
        vertex_ids=query.carrier_vertex_ids,
        joint_ids=query.joint_ids,
        weights=projected,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=0.1,
    )

    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy)
    g3b=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,
        stress_all_faces=True)

    rest=np.asarray(carrier.positions,np.float64)
    faces=_faces(candidate,query.carrier_vertex_ids)
    motion=exact_motion_court(
        ctx,rest,projected,faces,query.joint_ids,
        skeleton,cameras,policy)

    g=_summarize_g3(g3,g3b)
    motion_summary={
        "frame_count":int(motion["frame_count"]),
        "failed_motion_frame_count":int(motion["failed_motion_frame_count"]),
        "maximum_motion_edge_ratio":float(motion["maximum_motion_edge_ratio"]),
        "maximum_motion_condition_number":float(
            motion["maximum_motion_condition_number"]),
    }
    passed=bool(
        g["g3_passed"]
        and g["g3b_passed"]
        and motion_summary["failed_motion_frame_count"]==0
    )
    report={
        "schema":"RealSaS.MIRACarrierProjectionActualMotionCourt.v1",
        "status":(
            "PASS_CARRIER_BOUND_COMPILER_PROJECTION_MECHANICS"
            if passed else
            "FAIL_CARRIER_BOUND_COMPILER_PROJECTION_MECHANICS"
        ),
        "run_id":args.run_id,
        "carrier_evidence_hash":carrier.carrier_evidence_hash,
        "carrier_topology_hash":carrier.topology_hash,
        "mira_checkpoint_sha256":mira_result["model_sha256"],
        "projection":{
            "source_domain":"FROZEN_MIRA_GSA_SEMANTIC_FIELD",
            "target_domain":"EXACT_STAGE19_MECHANICAL_CARRIER",
            "support":"EXACT_MECHANICAL_SUPPORT_COEFFICIENTS",
            "recomputed_for_exact_carrier":True,
            "teacher_weights_used":False,
            "silent_reuse":False,
            "direct_carrier_query_used":False,
        },
        "hard_mechanics":g,
        "actual_motion":motion_summary,
        "backbone_query_chunk_telemetry":telemetry,
        "product_authority_minted":False,
        "training_used":False,
        "claim_boundary":[
            "The same frozen MIRA checkpoint produces the GSA semantic field.",
            "Compiler projection is deterministic and carrier-bound.",
            "No teacher skin weights or direct carrier-query predictions are used.",
            "PASS requires both G3/G3B and exact 51-frame idle/run/slash motion closure.",
            "This audit does not itself promote the projection to product authority.",
        ],
        "decision":(
            "PROMOTE_TYPED_CARRIER_PROJECTION_CONTRACT__C4_NEURAL_ADAPTATION_NOT_NEEDED"
            if passed else
            "PROJECTION_NOT_SUFFICIENT__CONTINUE_C4_ATTRIBUTION"
        ),
    }
    write(args.out_dir/"PROJECTED_CARRIER_SKIN.json",skin.to_dict())
    write(args.out_dir/"G3.json",g3.to_dict())
    write(args.out_dir/"G3B.json",g3b)
    write(args.out_dir/"MOTION.json",motion)
    np.savez_compressed(
        args.out_dir/"PROJECTED_CARRIER_WEIGHTS.npz",
        weights=projected,
        vertex_ids=np.asarray(query.carrier_vertex_ids),
        joint_ids=np.asarray(query.joint_ids),
    )
    write(args.out_dir/"REPORT.json",report)
    print("MIRA_C33_RESULT="+json.dumps(report,sort_keys=True),flush=True)
    if not passed:
        raise SystemExit(2)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--arm-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--backbone-query-chunk",type=int,default=256)
    ap.add_argument("--readout-chunk",type=int,default=128)
    args=ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        write(args.out_dir/"ERROR.json",{
            "type":type(exc).__name__,"message":str(exc)
        })
        raise
