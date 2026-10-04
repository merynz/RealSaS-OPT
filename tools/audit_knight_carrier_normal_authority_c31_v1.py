from __future__ import annotations

"""C3.1 carrier-normal authority causal court.

Tests whether the failed zero-train direct-carrier MIRA arm was caused by
unoriented Stage19 face-derived normals rather than MIRA capacity.

Arms:
A) current Stage19 carrier normals (canonical-sorted face cross products);
B) exact same carrier XYZ/connectivity/query support, but normals transported
   from the signed GSA normal field through exact geometry SurfaceSupportBinding.

No training, no topology mutation, no product authority.
"""

import argparse
import json
import math
from dataclasses import replace
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
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
    mechanical_carrier_evidence_hash,
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
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_carrier,
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import read, write


def _geometry_support(vertex):
    rows = tuple((str(sid), float(w)) for sid, w in vertex.support_binding.coefficients)
    if not rows:
        raise RuntimeError("C31_GEOMETRY_SUPPORT_EMPTY")
    total = sum(w for _, w in rows)
    if any((not math.isfinite(w) or w < 0.0) for _, w in rows) or abs(total - 1.0) > 1e-8:
        raise RuntimeError("C31_GEOMETRY_SUPPORT_INVALID")
    return rows


def _transport_gsa_normals(candidate, surface_tensor, ordered_vertex_ids):
    source_index = {str(sid): i for i, sid in enumerate(surface_tensor.surface_ids)}
    by_id = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    normals = np.asarray(surface_tensor.normals, np.float64)
    valid = np.asarray(surface_tensor.normal_valid, bool)

    out = np.zeros((len(ordered_vertex_ids), 3), np.float64)
    out_valid = np.zeros(len(ordered_vertex_ids), bool)
    valid_mass = np.zeros(len(ordered_vertex_ids), np.float64)
    support_count = np.zeros(len(ordered_vertex_ids), np.int64)

    for qi, vid in enumerate(ordered_vertex_ids):
        vertex = by_id[str(vid)]
        accum = np.zeros(3, np.float64)
        mass = 0.0
        count = 0
        for sid, coeff in _geometry_support(vertex):
            if sid not in source_index:
                raise RuntimeError("C31_SUPPORT_OUTSIDE_GSA")
            si = source_index[sid]
            count += 1
            if valid[si]:
                accum += coeff * normals[si]
                mass += coeff
        support_count[qi] = count
        valid_mass[qi] = mass
        length = float(np.linalg.norm(accum))
        if mass > 1e-8 and math.isfinite(length) and length > 1e-12:
            out[qi] = accum / length
            out_valid[qi] = True

    return out.astype(np.float32), out_valid, valid_mass, support_count


def _diagnostic_evidence(base, normals, normal_valid):
    geometry_hash = content_sha256({
        "schema": "RealSaS.MechanicalCarrierGeometryBasis.v1",
        "ordered_vertex_ids": base.ordered_vertex_ids,
        "positions": base.positions,
        "normals": tuple(tuple(map(float, row)) for row in normals),
        "normal_valid": tuple(bool(x) for x in normal_valid),
    })
    provisional = replace(
        base,
        normals=tuple(tuple(map(float, row)) for row in normals),
        normal_valid=tuple(bool(x) for x in normal_valid),
        geometry_hash=geometry_hash,
        carrier_evidence_hash="",
        metadata={
            **dict(base.metadata or {}),
            "authority": "DIAGNOSTIC_GSA_NORMAL_TRANSPORT_COUNTERFACTUAL",
            "normals_owner": "GSA_SIGNED_NORMAL_FIELD_X_GEOMETRY_SUPPORT",
            "product_authority": False,
        },
    )
    return replace(
        provisional,
        carrier_evidence_hash=mechanical_carrier_evidence_hash(provisional),
    )


def _normal_alignment(a, a_valid, b, b_valid, mask=None):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    valid = np.asarray(a_valid, bool) & np.asarray(b_valid, bool)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    dots = np.sum(a[valid] * b[valid], axis=1)
    return {
        "count": int(len(dots)),
        "signed_cos_mean": float(dots.mean()) if len(dots) else None,
        "signed_cos_p05": float(np.quantile(dots, 0.05)) if len(dots) else None,
        "signed_cos_p50": float(np.quantile(dots, 0.50)) if len(dots) else None,
        "signed_cos_p95": float(np.quantile(dots, 0.95)) if len(dots) else None,
        "absolute_cos_mean": float(np.abs(dots).mean()) if len(dots) else None,
        "negative_dot_fraction": float(np.mean(dots < 0.0)) if len(dots) else None,
        "strongly_opposed_fraction": float(np.mean(dots < -0.5)) if len(dots) else None,
    }


def _identity_mask(candidate, ordered_vertex_ids):
    by_id = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    out = np.zeros(len(ordered_vertex_ids), bool)
    for i, vid in enumerate(ordered_vertex_ids):
        v = by_id[str(vid)]
        coeffs = tuple(v.support_binding.coefficients)
        out[i] = (
            str(v.support_binding.mode) == "IDENTITY_SURFACE_NODE"
            and len(coeffs) == 1
            and abs(float(coeffs[0][1]) - 1.0) <= 1e-12
        )
    return out


def _identity_weight_parity(
    *,
    candidate,
    ordered_vertex_ids,
    conditioning,
    legacy_weights,
    direct_weights,
):
    source_row = {str(sid): i for i, sid in enumerate(conditioning.surface_ids[0])}
    carrier_row = {str(vid): i for i, vid in enumerate(ordered_vertex_ids)}
    rows = []
    for v in candidate.vertices:
        coeffs = tuple(v.support_binding.coefficients)
        if (
            str(v.support_binding.mode) != "IDENTITY_SURFACE_NODE"
            or len(coeffs) != 1
            or abs(float(coeffs[0][1]) - 1.0) > 1e-12
        ):
            continue
        sid = str(coeffs[0][0])
        if sid not in source_row:
            continue
        rows.append(float(np.abs(
            direct_weights[carrier_row[str(v.candidate_vertex_id)]]
            - legacy_weights[source_row[sid]]
        ).sum()))
    a = np.asarray(rows, np.float64)
    return {
        "count": int(len(a)),
        "row_l1_mean": float(a.mean()) if len(a) else None,
        "row_l1_p95": float(np.quantile(a, 0.95)) if len(a) else None,
        "row_l1_max": float(a.max()) if len(a) else None,
    }


def _run_arm(
    *,
    candidate,
    evidence,
    surface_tensor,
    conditioning,
    memory,
    tokens,
    decoder,
    skeleton,
    surface,
    envelope,
    cameras,
    policy,
    device,
    readout_chunk,
):
    query, weights = _decode_carrier(
        candidate=candidate,
        carrier=evidence,
        surface_tensor=surface_tensor,
        conditioning=conditioning,
        memory=memory,
        tokens=tokens,
        decoder=decoder,
        device=device,
        chunk=readout_chunk,
    )
    skin = qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=evidence,
        skeleton=skeleton,
        vertex_ids=query.carrier_vertex_ids,
        joint_ids=query.joint_ids,
        weights=weights,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=0.1,
    )
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras.cameras,
        policy=policy,
    )
    g3b = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras.cameras,
        policy=policy,
        stress_all_faces=True,
    )
    return query, weights, skin, g3, g3b


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C31_CUDA_NOT_AVAILABLE")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ctx = _ctx(args.authority_root, args.run_id)
    surface = rigging_surface_from_dict(stage_output_payload(
        ctx, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1"))
    skeleton = qualified_skeleton_from_dict(stage_output_payload(
        ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"))
    candidate = canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.CanonicalMeshCandidateIR.v1"))
    addressing = surface_addressing_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.SurfaceAddressingIR.v1"))
    static = static_mesh_qualification_from_dict(stage_output_payload(
        ctx, "19_STATIC_CANONICAL_MESH_QUALIFIED", "RealSaS.StaticCanonicalMeshQualificationIR.v1"))
    current = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addressing)

    surface_tensor = tensorize_rigging_surface_v1(surface)
    transported_normals, transported_valid, valid_mass, support_count = _transport_gsa_normals(
        candidate, surface_tensor, current.ordered_vertex_ids)
    transported = _diagnostic_evidence(current, transported_normals, transported_valid)

    policy = mesh_policy_from_dict(stage_output_payload(
        ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"))
    envelope = deformation_envelope_from_dict(stage_output_payload(
        ctx, "34_DEFORMATION_CAPABILITY_ENVELOPE", "RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    cameras = qualified_camera_set_from_dict(stage_output_payload(
        ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"))

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

    q_a, w_a, skin_a, g3_a, g3b_a = _run_arm(
        candidate=candidate, evidence=current, surface_tensor=surface_tensor,
        conditioning=conditioning, memory=memory, tokens=tokens, decoder=decoder,
        skeleton=skeleton, surface=surface, envelope=envelope, cameras=cameras,
        policy=policy, device=args.device, readout_chunk=args.readout_chunk)
    q_b, w_b, skin_b, g3_b, g3b_b = _run_arm(
        candidate=candidate, evidence=transported, surface_tensor=surface_tensor,
        conditioning=conditioning, memory=memory, tokens=tokens, decoder=decoder,
        skeleton=skeleton, surface=surface, envelope=envelope, cameras=cameras,
        policy=policy, device=args.device, readout_chunk=args.readout_chunk)

    identity = _identity_mask(candidate, current.ordered_vertex_ids)
    current_n = np.asarray(current.normals, np.float64)
    current_v = np.asarray(current.normal_valid, bool)

    report = {
        "schema": "RealSaS.KnightCarrierNormalAuthorityCourt.v1",
        "status": "MEASURED__NO_TRAIN__NO_PROMOTION_CLAIM",
        "run_id": args.run_id,
        "carrier_topology_hash": current.topology_hash,
        "current_carrier_evidence_hash": current.carrier_evidence_hash,
        "gsa_transport_counterfactual_hash": transported.carrier_evidence_hash,
        "mira_checkpoint_sha256": mira_result["model_sha256"],
        "transport_valid_fraction": float(np.mean(transported_valid)),
        "transport_valid_mass_p05": float(np.quantile(valid_mass, 0.05)),
        "support_count_p95": float(np.quantile(support_count, 0.95)),
        "normal_alignment_all": _normal_alignment(
            current_n, current_v, transported_normals, transported_valid),
        "normal_alignment_identity": _normal_alignment(
            current_n, current_v, transported_normals, transported_valid, identity),
        "current_face_cross_normal_arm": {
            "mechanics": _summarize_g3(g3_a, g3b_a),
            "identity_weight_parity": _identity_weight_parity(
                candidate=candidate,
                ordered_vertex_ids=q_a.carrier_vertex_ids,
                conditioning=conditioning,
                legacy_weights=legacy_weights,
                direct_weights=w_a,
            ),
        },
        "gsa_oriented_normal_counterfactual_arm": {
            "mechanics": _summarize_g3(g3_b, g3b_b),
            "identity_weight_parity": _identity_weight_parity(
                candidate=candidate,
                ordered_vertex_ids=q_b.carrier_vertex_ids,
                conditioning=conditioning,
                legacy_weights=legacy_weights,
                direct_weights=w_b,
            ),
        },
        "backbone_query_chunk_telemetry": telemetry,
        "training_used": False,
        "topology_changed": False,
        "positions_changed": False,
        "only_query_normal_authority_changed": True,
        "product_authority_minted": False,
    }

    before = report["current_face_cross_normal_arm"]["mechanics"]["g3b_unsafe_face_count"]
    after = report["gsa_oriented_normal_counterfactual_arm"]["mechanics"]["g3b_unsafe_face_count"]
    parity_before = report["current_face_cross_normal_arm"]["identity_weight_parity"]["row_l1_p95"]
    parity_after = report["gsa_oriented_normal_counterfactual_arm"]["identity_weight_parity"]["row_l1_p95"]
    report["causal_verdict"] = (
        "NORMAL_AUTHORITY_CAUSAL_DRIVER_SUPPORTED"
        if after < before and parity_after < parity_before
        else "NORMAL_AUTHORITY_ALONE_NOT_SUFFICIENT"
    )

    write(args.out_dir / "current_carrier_skin.json", skin_a.to_dict())
    write(args.out_dir / "transported_normal_carrier_skin.json", skin_b.to_dict())
    write(args.out_dir / "current_g3.json", g3_a.to_dict())
    write(args.out_dir / "current_g3b.json", g3b_a)
    write(args.out_dir / "transported_g3.json", g3_b.to_dict())
    write(args.out_dir / "transported_g3b.json", g3b_b)
    write(args.out_dir / "REPORT.json", report)
    print("KNIGHT_C31_NORMAL_AUTHORITY_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
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
