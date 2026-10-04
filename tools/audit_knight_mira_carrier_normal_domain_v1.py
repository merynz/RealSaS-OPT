from __future__ import annotations

"""C3N: isolate MIRA carrier-query normal-domain shift before any fit.

No fitting and no product promotion.

The current MIRA V6 readout was trained in the GSA semantic domain. C3 showed
that replacing its query normals with Stage19 carrier-face normals changes even
identity-vertex weights materially. This court keeps exact carrier XYZ and exact
carrier vertex domain fixed while factorizing only the normal evidence supplied
through:

- geometry7
- pair_geometry normal-axis terms

Arms:
  CARRIER_ALL: current C3 exact carrier normals everywhere.
  GSA_NORMAL_ALL: support-transported GSA normals in both query feature paths.
  GSA_NORMAL_GEOM_ONLY: GSA normals only in geometry7.
  GSA_NORMAL_PAIR_ONLY: GSA normals only in pair_geometry.

Hard G3/G3B is measured for every arm. This is an owner-attribution court, not
an authorization to make GSA normals mechanical authority.
"""

import argparse
import json
import math
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
from models.arachne.v2.arachne_geometry_v2 import arachne_pair_geometry_v2
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from models.mira.carrier_query_v1 import (
    build_mira_mechanical_carrier_query_v1,
    transport_surface_memory_to_carrier_v1,
)
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import read, write


def _transport_gsa_normals(query, surface_tensor):
    src_n = np.asarray(surface_tensor.normals, np.float64)
    src_v = np.asarray(surface_tensor.normal_valid, bool)
    out = np.zeros((query.vertex_count, 3), np.float64)
    valid_mass = np.zeros(query.vertex_count, np.float64)
    total_mass = np.zeros(query.vertex_count, np.float64)

    for row, col, coeff in zip(
        query.support_row_index.tolist(),
        query.support_source_index.tolist(),
        query.support_coefficients.tolist(),
    ):
        w = float(coeff)
        total_mass[int(row)] += w
        if bool(src_v[int(col)]):
            out[int(row)] += w * src_n[int(col)]
            valid_mass[int(row)] += w

    valid = (
        np.abs(total_mass - 1.0) <= 1e-6
    ) & (valid_mass >= 1.0 - 1e-6)
    norms = np.linalg.norm(out, axis=1)
    valid &= np.isfinite(norms) & (norms > 1e-12)
    out[valid] /= norms[valid, None]
    out[~valid] = 0.0
    return out.astype(np.float32), valid


def _pair(query, conditioning, normals, normal_valid):
    j = len(query.joint_ids)
    positions = np.asarray(query.positions_normalized, np.float32)
    joint_positions = np.asarray(conditioning.joint_positions_normalized[:, :j], np.float32)
    parent = np.asarray(conditioning.parent_indices[:, :j], np.int64)
    qmask = np.ones((1, query.vertex_count), bool)
    jmask = np.ones((1, j), bool)
    pair, legal = arachne_pair_geometry_v2(
        positions[None],
        np.asarray(normals, np.float32)[None],
        np.asarray(normal_valid, bool)[None],
        joint_positions,
        parent,
        qmask,
        jmask,
    )
    return pair[0], legal[0]


def _geom(query, normals, normal_valid):
    return np.concatenate(
        [
            2.0 * np.asarray(query.positions_normalized, np.float32),
            np.asarray(normals, np.float32),
            np.asarray(normal_valid, bool)[:, None].astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _decode(*, decoder, memory, tokens, geom, pair, legal, device, chunk):
    with torch.inference_mode(), torch.autocast(
        device, dtype=torch.bfloat16, enabled=(device == "cuda")
    ):
        _, pred = decoder.decode_all(
            memory,
            torch.as_tensor(geom, device=device),
            torch.as_tensor(pair, device=device),
            tokens,
            torch.as_tensor(legal, device=device),
            chunk=chunk,
        )
    w = pred.float().cpu().numpy().astype(np.float64)
    w /= w.sum(axis=1, keepdims=True)
    return w


def _identity_parity(*, candidate, query, carrier_weights, legacy_weights, conditioning):
    source_row = {sid: i for i, sid in enumerate(conditioning.surface_ids[0])}
    carrier_row = {vid: i for i, vid in enumerate(query.carrier_vertex_ids)}
    rows = []
    position_delta = []
    carrier_pos = np.asarray(query.positions_normalized, np.float64)
    source_pos = np.asarray(conditioning.surface_positions_normalized[0], np.float64)
    for vertex in candidate.vertices:
        coeffs = tuple(vertex.support_binding.coefficients)
        if str(vertex.support_binding.mode) != "IDENTITY_SURFACE_NODE" or len(coeffs) != 1:
            continue
        sid, coeff = coeffs[0]
        sid = str(sid)
        if abs(float(coeff) - 1.0) > 1e-12 or sid not in source_row:
            continue
        ci = carrier_row[str(vertex.candidate_vertex_id)]
        si = source_row[sid]
        rows.append(float(np.abs(carrier_weights[ci] - legacy_weights[si]).sum()))
        position_delta.append(float(np.linalg.norm(carrier_pos[ci] - source_pos[si])))
    a = np.asarray(rows, np.float64)
    p = np.asarray(position_delta, np.float64)
    return {
        "count": int(len(a)),
        "row_l1_mean": float(a.mean()) if len(a) else None,
        "row_l1_p95": float(np.quantile(a, 0.95)) if len(a) else None,
        "row_l1_max": float(a.max()) if len(a) else None,
        "position_delta_mean_norm": float(p.mean()) if len(p) else None,
        "position_delta_max_norm": float(p.max()) if len(p) else None,
    }


def _normal_delta(*, candidate, query, semantic_normals, semantic_valid):
    carrier = np.asarray(query.normals, np.float64)
    cvalid = np.asarray(query.normal_valid, bool)
    sem = np.asarray(semantic_normals, np.float64)
    svalid = np.asarray(semantic_valid, bool)
    carrier_row = {vid: i for i, vid in enumerate(query.carrier_vertex_ids)}
    angles = []
    for vertex in candidate.vertices:
        coeffs = tuple(vertex.support_binding.coefficients)
        if str(vertex.support_binding.mode) != "IDENTITY_SURFACE_NODE" or len(coeffs) != 1:
            continue
        _, coeff = coeffs[0]
        if abs(float(coeff) - 1.0) > 1e-12:
            continue
        i = carrier_row[str(vertex.candidate_vertex_id)]
        if not (cvalid[i] and svalid[i]):
            continue
        dot = float(np.clip(np.dot(carrier[i], sem[i]), -1.0, 1.0))
        angles.append(math.degrees(math.acos(dot)))
    a = np.asarray(angles, np.float64)
    return {
        "identity_both_valid_count": int(len(a)),
        "angle_deg_mean": float(a.mean()) if len(a) else None,
        "angle_deg_p50": float(np.quantile(a, 0.50)) if len(a) else None,
        "angle_deg_p95": float(np.quantile(a, 0.95)) if len(a) else None,
        "angle_deg_max": float(a.max()) if len(a) else None,
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C3N_CUDA_NOT_AVAILABLE")

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
    skeleton = qualified_skeleton_from_dict(stage_output_payload(
        ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"))
    surface_tensor = tensorize_rigging_surface_v1(surface)

    conditioning, ci, gsa_memory, tokens, decoder, mira_result, telemetry = _run_mira_backbone(
        surface=surface,
        skeleton=skeleton,
        arm_dir=args.arm_dir,
        device=args.device,
        query_chunk=args.backbone_query_chunk,
    )
    legacy_weights = _decode_legacy(
        conditioning=conditioning,
        ci=ci,
        memory=gsa_memory,
        tokens=tokens,
        decoder=decoder,
        device=args.device,
        chunk=args.readout_chunk,
    )

    query = build_mira_mechanical_carrier_query_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        surface_tensor=surface_tensor,
        conditioning=conditioning,
    )
    qmemory = transport_surface_memory_to_carrier_v1(gsa_memory, query)
    semantic_normals, semantic_valid = _transport_gsa_normals(query, surface_tensor)
    semantic_pair, semantic_legal = _pair(
        query, conditioning, semantic_normals, semantic_valid)
    semantic_geom = _geom(query, semantic_normals, semantic_valid)

    arms = {
        "CARRIER_ALL": (
            np.asarray(query.geometry7, np.float32),
            np.asarray(query.pair_geometry, np.float32),
            np.asarray(query.legal_pair, bool),
        ),
        "GSA_NORMAL_ALL": (
            semantic_geom,
            semantic_pair,
            semantic_legal,
        ),
        "GSA_NORMAL_GEOM_ONLY": (
            semantic_geom,
            np.asarray(query.pair_geometry, np.float32),
            np.asarray(query.legal_pair, bool),
        ),
        "GSA_NORMAL_PAIR_ONLY": (
            np.asarray(query.geometry7, np.float32),
            semantic_pair,
            semantic_legal,
        ),
    }

    reports = {}
    for name, (geom, pair, legal) in arms.items():
        print("C3N_ARM_BEGIN=" + name, flush=True)
        weights = _decode(
            decoder=decoder,
            memory=qmemory,
            tokens=tokens,
            geom=geom,
            pair=pair,
            legal=legal,
            device=args.device,
            chunk=args.readout_chunk,
        )
        skin = qualify_mechanical_carrier_skin_v1(
            candidate=candidate,
            carrier_evidence=carrier,
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
        report = {
            **_summarize_g3(g3, g3b),
            "identity_vertex_query_parity": _identity_parity(
                candidate=candidate,
                query=query,
                carrier_weights=weights,
                legacy_weights=legacy_weights,
                conditioning=conditioning,
            ),
            "skin_lineage_hash": skin.skin_lineage_hash,
        }
        reports[name] = report
        write(args.out_dir / (name + "_carrier_skin.json"), skin.to_dict())
        write(args.out_dir / (name + "_g3.json"), g3.to_dict())
        write(args.out_dir / (name + "_g3b.json"), g3b)
        np.savez_compressed(
            args.out_dir / (name + "_weights.npz"),
            weights=weights,
            vertex_ids=np.asarray(query.carrier_vertex_ids),
            joint_ids=np.asarray(query.joint_ids),
        )
        print("C3N_ARM_RESULT=" + json.dumps({name: report}, sort_keys=True), flush=True)

    exact = reports["CARRIER_ALL"]["identity_vertex_query_parity"]["row_l1_mean"]
    semantic = reports["GSA_NORMAL_ALL"]["identity_vertex_query_parity"]["row_l1_mean"]
    normal_domain_owner = (
        semantic is not None
        and exact is not None
        and semantic < exact * 0.25
    )

    result = {
        "schema": "RealSaS.MIRACarrierNormalDomainCourt.v1",
        "status": "MEASURED__NO_TRAIN__NO_PROMOTION_CLAIM",
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "mira_checkpoint_sha256": mira_result["model_sha256"],
        "mira_query_hash": query.query_hash,
        "backbone_query_chunk_telemetry": telemetry,
        "carrier_vs_transported_gsa_normal": _normal_delta(
            candidate=candidate,
            query=query,
            semantic_normals=semantic_normals,
            semantic_valid=semantic_valid,
        ),
        "arms": reports,
        "normal_domain_shift_material": bool(normal_domain_owner),
        "training_used": False,
        "teacher_predictor_input_used": False,
        "product_authority_minted": False,
        "decision": (
            "NORMAL_DOMAIN_SHIFT_IS_PRIMARY_OWNER_CANDIDATE"
            if normal_domain_owner
            else "NORMAL_DOMAIN_SHIFT_NOT_SUFFICIENT_TO_EXPLAIN_C3"
        ),
        "claim_boundary": [
            "All arms use exact Stage19 carrier XYZ and identical carrier vertex domain.",
            "Transported GSA normals are diagnostic semantic evidence, not mechanical carrier authority.",
            "No fit is authorized solely by this court.",
            "If normal-domain shift is insufficient, C4 minimal MIRA fit remains the next candidate.",
        ],
    }
    write(args.out_dir / "REPORT.json", result)
    print("MIRA_C3N_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


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
