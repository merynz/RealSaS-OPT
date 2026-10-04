from __future__ import annotations

"""Knight C3 carrier-first one-shot mechanical court.

Frozen current ATLAS + frozen current MIRA. No fitting.

A/B:
LEGACY_SURFACE_TRANSFER:
  GSA backbone/readout -> QualifiedSkinIR on GSA -> legacy support transfer in G3.

DIRECT_CARRIER_QUERY:
  same GSA backbone/field tokens -> support-bound semantic-memory transport ->
  existing V6 readout queried at exact Stage19 carrier vertices ->
  QualifiedMechanicalCarrierSkinIR -> G3/G3B with no surface weight transfer.

The rig, checkpoint, carrier, skeleton, probe envelope and policy are shared.
"""

import argparse
import inspect
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
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
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.types import (
    SkinInfluenceProposal,
    SkinProposalIR,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1V4
from models.arachne.v6.readout_v6 import ArachneV6RawReadout
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import (
    GeppettoReferenceStrengthConfigV1,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    GeppettoReferenceStrengthNoLearnedSlotV1,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from models.mira.carrier_query_v1 import (
    build_mira_mechanical_carrier_query_v1,
    transport_surface_memory_to_carrier_v1,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.query_chunked_sdpa_v1 import query_chunked_cuda_bf16_sdpa
from tools.inference.refined_surface_rig_skin_v1 import (
    class_ast,
    read,
    verified_checkpoint,
    write,
)


def _run_atlas(*, surface, fit_run: Path, device: str, seed: int):
    execution = read(fit_run / "artifacts/27_GEPPETTO_FIT/model_fit_execution.json")
    source = Path(inspect.getfile(GeppettoReferenceStrengthNoLearnedSlotV1))
    if sha256_file(source) != execution["model_source_sha256"]:
        raise RuntimeError("C3_ATLAS_SOURCE_DRIFT")
    data = verified_checkpoint(Path(execution["checkpoint_path"]), execution["checkpoint_sha256"])
    cfg = GeppettoReferenceStrengthConfigV1(**data["config"])
    if cfg.config_hash != data["config_hash"]:
        raise RuntimeError("C3_ATLAS_CONFIG_DRIFT")
    model = GeppettoReferenceStrengthNoLearnedSlotV1(cfg).to(device).eval()
    model.load_state_dict(data["model"], strict=True)
    tensor = tensorize_rigging_surface_v1(surface)
    with torch.inference_mode():
        proposal = model.propose(
            tensor,
            resource_step_limit=min(128, tensor.node_count),
            generator=torch.Generator(device=device).manual_seed(seed),
        )
    skeleton = qualify_skeleton(surface, proposal, run_ilp_shadow=False)
    return proposal, skeleton, tensor, execution["checkpoint_sha256"]


def _run_mira_backbone(*, surface, skeleton, arm_dir: Path, device: str, query_chunk: int):
    result = read(arm_dir / "ARACHNE_KNIGHT_V6_RESULT.json")
    checkpoint = arm_dir / "ARACHNE_KNIGHT_V6_MODEL_FINAL_FP32.pt"
    data = verified_checkpoint(checkpoint, result["model_sha256"])
    conditioning = ArachneRichConditioningAdapterV3(require_scene_first=True)([surface], [skeleton])

    model = ArachneA1V4().to(device).eval()
    model.load_state_dict(data["backbone"], strict=True)
    ci = {
        key: torch.as_tensor(
            getattr(conditioning, "view_yaw_code" if key == "view_yaw_fourier" else key),
            device=device,
        )
        for key in inspect.signature(model.forward).parameters
    }
    attention_context = (
        query_chunked_cuda_bf16_sdpa(query_chunk)
        if device == "cuda" and query_chunk
        else nullcontext(None)
    )
    with attention_context as telemetry, torch.inference_mode(), torch.autocast(
        device, dtype=torch.bfloat16, enabled=(device == "cuda")
    ):
        raw = model(**ci)
    memory = raw.surface_memory[0].float()
    tokens = raw.field_tokens[0].float()
    decoder = ArachneV6RawReadout(
        memory.shape[-1], tokens.shape[-1], int(ci["pair_geometry"].shape[-1])
    ).to(device).eval()
    decoder.load_state_dict(data["decoder"], strict=True)
    return conditioning, ci, memory, tokens, decoder, result, telemetry


def _decode_legacy(*, conditioning, ci, memory, tokens, decoder, device: str, chunk: int):
    geom = torch.as_tensor(conditioning.geometry7, device=device)[0].float()
    pair = ci["pair_geometry"][0].float()
    legal = (
        ci["pair_mask"].bool()
        & ci["surface_mask"][:, :, None].bool()
        & ci["joint_mask"][:, None, :].bool()
    )[0]
    with torch.inference_mode(), torch.autocast(
        device, dtype=torch.bfloat16, enabled=(device == "cuda")
    ):
        _, pred = decoder.decode_all(memory, geom, pair, tokens, legal, chunk=chunk)
    weights = pred.float().cpu().numpy().astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _decode_carrier(
    *,
    candidate,
    carrier,
    surface_tensor,
    conditioning,
    memory,
    tokens,
    decoder,
    device: str,
    chunk: int,
):
    query = build_mira_mechanical_carrier_query_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        surface_tensor=surface_tensor,
        conditioning=conditioning,
    )
    qmemory = transport_surface_memory_to_carrier_v1(memory, query)
    geom = torch.as_tensor(query.geometry7, device=device)
    pair = torch.as_tensor(query.pair_geometry, device=device)
    legal = torch.as_tensor(query.legal_pair, device=device)
    with torch.inference_mode(), torch.autocast(
        device, dtype=torch.bfloat16, enabled=(device == "cuda")
    ):
        _, pred = decoder.decode_all(
            qmemory,
            geom,
            pair,
            tokens,
            legal,
            chunk=chunk,
        )
    weights = pred.float().cpu().numpy().astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    return query, weights


def _legacy_skin(*, surface, skeleton, conditioning, weights, model_hash: str, policy: dict):
    sids = conditioning.surface_ids[0]
    jids = conditioning.joint_ids[0]
    proposal = SkinProposalIR(
        tuple(
            SkinInfluenceProposal(sid, jid, float(weights[i, j]))
            for i, sid in enumerate(sids)
            for j, jid in enumerate(jids)
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        model_provenance=model_hash,
        metadata={
            "fresh_model_inference": True,
            "teacher_input_used": False,
            "carrier_native": False,
        },
    )
    return qualify_skin(
        surface,
        skeleton,
        proposal,
        max_simplex_repair_l1=float(policy["max_simplex_repair_l1"]),
        max_total_correction_l1=float(policy["max_total_correction_l1"]),
        negative_tolerance=float(policy["negative_tolerance"]),
        max_influences=None if policy.get("max_influences") is None else int(policy["max_influences"]),
    )


def _summarize_g3(g3, g3b):
    return {
        "g3_passed": bool(g3.passed),
        "g3_failures": list(g3.failure_invariants),
        "g3_min_area_ratio": float(g3.minimum_area_ratio),
        "g3_max_area_ratio": float(g3.maximum_area_ratio),
        "g3_max_condition": float(g3.maximum_condition_number),
        "g3_min_edge_ratio": float(g3.minimum_edge_ratio),
        "g3_max_edge_ratio": float(g3.maximum_edge_ratio),
        "g3b_passed": bool(g3b["passed"]),
        "g3b_risky_face_count": int(g3b["risky_face_count"]),
        "g3b_unsafe_face_count": int(g3b["unsafe_face_count"]),
        "g3b_report_hash": str(g3b["report_hash"]),
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C3_CUDA_NOT_AVAILABLE")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ctx = _ctx(args.authority_root, args.run_id)
    surface = rigging_surface_from_dict(stage_output_payload(
        ctx, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1"))
    candidate = canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.CanonicalMeshCandidateIR.v1"))
    addressing = surface_addressing_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.SurfaceAddressingIR.v1"))
    static = static_mesh_qualification_from_dict(stage_output_payload(
        ctx, "19_STATIC_CANONICAL_MESH_QUALIFIED", "RealSaS.StaticCanonicalMeshQualificationIR.v1"))
    carrier = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addressing)
    policy = mesh_policy_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"))
    envelope = deformation_envelope_from_dict(stage_output_payload(
        ctx, "34_DEFORMATION_CAPABILITY_ENVELOPE", "RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    cameras = qualified_camera_set_from_dict(stage_output_payload(
        ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"))

    proposal, skeleton, surface_tensor, atlas_hash = _run_atlas(
        surface=surface, fit_run=args.fit_run, device=args.device, seed=args.seed)

    prereg = read(args.fit_run / "artifacts/30_ARACHNE_FIT_PREREGISTERED/model_fit_preregistration.json")
    conditioning, ci, memory, tokens, decoder, mira_result, telemetry = _run_mira_backbone(
        surface=surface,
        skeleton=skeleton,
        arm_dir=args.arm_dir,
        device=args.device,
        query_chunk=args.backbone_query_chunk,
    )

    legacy_weights = _decode_legacy(
        conditioning=conditioning, ci=ci, memory=memory, tokens=tokens,
        decoder=decoder, device=args.device, chunk=args.readout_chunk)
    legacy_skin = _legacy_skin(
        surface=surface, skeleton=skeleton, conditioning=conditioning,
        weights=legacy_weights, model_hash=mira_result["model_sha256"],
        policy=dict(prereg["qualification_policy"]))

    query, carrier_weights = _decode_carrier(
        candidate=candidate, carrier=carrier, surface_tensor=surface_tensor,
        conditioning=conditioning, memory=memory, tokens=tokens, decoder=decoder,
        device=args.device, chunk=args.readout_chunk)
    carrier_skin = qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        skeleton=skeleton,
        vertex_ids=query.carrier_vertex_ids,
        joint_ids=query.joint_ids,
        weights=carrier_weights,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=0.1,
    )

    legacy_g3 = run_g3_local_frame_micro_stress_v2(
        candidate, surface=surface, skeleton=skeleton, skin=legacy_skin,
        envelope=envelope, cameras=cameras.cameras, policy=policy)
    legacy_g3b = run_skin_topology_compatibility_v1(
        candidate, surface=surface, skeleton=skeleton, skin=legacy_skin,
        envelope=envelope, cameras=cameras.cameras, policy=policy,
        stress_all_faces=True)

    carrier_g3 = run_g3_local_frame_micro_stress_v2(
        candidate, surface=surface, skeleton=skeleton, skin=carrier_skin,
        envelope=envelope, cameras=cameras.cameras, policy=policy)
    carrier_g3b = run_skin_topology_compatibility_v1(
        candidate, surface=surface, skeleton=skeleton, skin=carrier_skin,
        envelope=envelope, cameras=cameras.cameras, policy=policy,
        stress_all_faces=True)

    # Identity carrier vertices permit an exact same-query comparison against
    # the GSA-domain prediction; non-identity vertices are intentionally new queries.
    source_row = {sid: i for i, sid in enumerate(conditioning.surface_ids[0])}
    carrier_row = {vid: i for i, vid in enumerate(query.carrier_vertex_ids)}
    deltas = []
    identity_count = 0
    for vertex in candidate.vertices:
        coeffs = tuple(vertex.support_binding.coefficients)
        if str(vertex.support_binding.mode) != "IDENTITY_SURFACE_NODE" or len(coeffs) != 1:
            continue
        sid, coeff = coeffs[0]
        if abs(float(coeff) - 1.0) > 1e-12 or str(sid) not in source_row:
            continue
        identity_count += 1
        deltas.append(
            float(np.abs(
                carrier_weights[carrier_row[str(vertex.candidate_vertex_id)]]
                - legacy_weights[source_row[str(sid)]]
            ).sum())
        )

    report = {
        "schema": "RealSaS.KnightCarrierFirstOneShotMechanicalCourt.v1",
        "status": "MEASURED__NO_TRAIN__NO_PROMOTION_CLAIM",
        "workflow_contract": "C3_KNIGHT_CARRIER_FIRST_ONE_SHOT",
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "carrier_vertex_count": len(query.carrier_vertex_ids),
        "carrier_face_count": len(candidate.faces),
        "atlas_checkpoint_sha256": atlas_hash,
        "atlas_joint_count": len(skeleton.joints),
        "mira_checkpoint_sha256": mira_result["model_sha256"],
        "mira_query_hash": query.query_hash,
        "backbone_query_chunk_telemetry": telemetry,
        "legacy_surface_transfer": _summarize_g3(legacy_g3, legacy_g3b),
        "direct_carrier_query": _summarize_g3(carrier_g3, carrier_g3b),
        "identity_vertex_query_parity": {
            "count": int(identity_count),
            "row_l1_mean": float(np.mean(deltas)) if deltas else None,
            "row_l1_p95": float(np.quantile(deltas, 0.95)) if deltas else None,
            "row_l1_max": float(np.max(deltas)) if deltas else None,
        },
        "training_used": False,
        "teacher_predictor_input_used": False,
        "surface_skin_transfer_used_in_direct_arm": False,
        "hard_stage35_metric_family_used": True,
        "product_authority_minted": False,
        "decision": (
            "DIRECT_CARRIER_QUERY_PASS"
            if carrier_g3.passed and carrier_g3b["passed"]
            else "DIRECT_CARRIER_QUERY_RESIDUAL_REMAINS"
        ),
    }

    write(args.out_dir / "atlas_skeleton_proposal.json", proposal.to_dict())
    write(args.out_dir / "qualified_skeleton.json", skeleton.to_dict())
    write(args.out_dir / "carrier_skin.json", carrier_skin.to_dict())
    write(args.out_dir / "legacy_g3.json", legacy_g3.to_dict())
    write(args.out_dir / "legacy_g3b.json", legacy_g3b)
    write(args.out_dir / "carrier_g3.json", carrier_g3.to_dict())
    write(args.out_dir / "carrier_g3b.json", carrier_g3b)
    write(args.out_dir / "REPORT.json", report)
    np.savez_compressed(
        args.out_dir / "carrier_weights.npz",
        weights=carrier_weights,
        vertex_ids=np.asarray(query.carrier_vertex_ids),
        joint_ids=np.asarray(query.joint_ids),
    )
    print("KNIGHT_C3_CARRIER_FIRST_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fit-run", type=Path, required=True)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--backbone-query-chunk", type=int, default=256)
    ap.add_argument("--readout-chunk", type=int, default=128)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / "ERROR.json", {
            "type": type(exc).__name__,
            "message": str(exc),
        })
        raise
