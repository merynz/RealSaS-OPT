from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_holeless_partitioned_dense_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    _pose_skin_matrices,
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_EDGE_RATIO,
    DEFAULT_RISK_L1_MIN,
    _skin_l1_per_face,
    _stress_angle,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentBoundaryConstraintIR,
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json,
    replay_compacted_dense_face_provenance,
)
from tools.audit_knight_global_skin_region_sweep_v1 import skin_matrix
from tools.audit_knight_teacher_free_weight_completion_court_v1 import motion_metrics
from tools.demo.render_knight_motion_preview_v1 import _ctx


def pair(a, b):
    a, b = str(a), str(b)
    return (a, b) if a < b else (b, a)


def quantiles(values):
    a = np.asarray(tuple(values), dtype=np.float64)
    if not len(a):
        return {"count": 0}
    return {
        "count": int(len(a)),
        "min": float(np.min(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def probe_edge_risk(surface, skeleton, cameras, envelope, sids, W):
    idx = {sid: i for i, sid in enumerate(sids)}
    P = np.asarray(
        [next(n.P for n in surface.surface_nodes if str(n.surface_id) == sid) for sid in sids],
        dtype=np.float64,
    )
    edge_pairs = []
    seen = set()
    for rel in surface.local_relations:
        p = pair(rel.a_surface_id, rel.b_surface_id)
        if p in seen:
            continue
        seen.add(p)
        edge_pairs.append(p)
    edge_pairs = tuple(sorted(edge_pairs))
    ea = np.asarray([idx[a] for a, _ in edge_pairs], dtype=np.int64)
    eb = np.asarray([idx[b] for _, b in edge_pairs], dtype=np.int64)
    rest_len = np.linalg.norm(P[eb] - P[ea], axis=1)
    if np.any(rest_len <= 1e-12):
        raise RuntimeError("PROBE_EDGE_REST_LENGTH_DEGENERATE")

    max_ratio = np.ones(len(edge_pairs), dtype=np.float64)
    min_ratio = np.ones(len(edge_pairs), dtype=np.float64)

    frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    stress_angle = float(_stress_angle(envelope))
    hom = np.concatenate([P, np.ones((len(P), 1), dtype=np.float64)], axis=1)

    probe_count = 0
    for jid in sorted(joint_ids):
        for axis_index in range(3):
            for sign in (-1.0, 1.0):
                skin_by_id = _pose_skin_matrices(
                    skeleton,
                    frames,
                    joint_id=jid,
                    local_axis_index=axis_index,
                    degrees=sign * stress_angle,
                )
                matrices = np.stack([skin_by_id[x] for x in joint_ids], axis=0)
                per = np.stack(
                    [(hom @ matrices[k].T)[:, :3] for k in range(len(joint_ids))],
                    axis=1,
                )
                posed = np.sum(per * W[:, :, None], axis=1)
                ratio = np.linalg.norm(posed[eb] - posed[ea], axis=1) / rest_len
                max_ratio = np.maximum(max_ratio, ratio)
                min_ratio = np.minimum(min_ratio, ratio)
                probe_count += 1

    return {
        "edge_pairs": edge_pairs,
        "rest_length": rest_len,
        "max_ratio": max_ratio,
        "min_ratio": min_ratio,
        "probe_count": probe_count,
        "stress_angle_deg": stress_angle,
    }


def build_probe_partition(surface, risk, *, max_edge_ratio):
    edge_pairs = risk["edge_pairs"]
    max_ratio = risk["max_ratio"]
    overrides = []
    unsafe_pairs = []
    for i, p in enumerate(edge_pairs):
        value = float(max_ratio[i])
        if value <= float(max_edge_ratio):
            continue
        unsafe_pairs.append(p)
        overrides.append(ComponentBoundaryConstraintIR(
            constraint_id="PROBECUT:" + content_sha256({
                "pair": p,
                "max_edge_ratio": value,
                "limit": float(max_edge_ratio),
            })[:20],
            a_surface_id=p[0],
            b_surface_id=p[1],
            decision="SEPARATE",
            evidence_refs=(f"CANONICAL_PROBE_MAX_EDGE_RATIO:{value:.17g}",),
            confidence=min(1.0, max(0.0, value / max(float(max_edge_ratio), 1e-12) - 1.0)),
            metadata={
                "evidence_class": "CANONICAL_PROBE_CONDITIONED_EDGE_COMPATIBILITY",
                "stress_angle_deg": float(risk["stress_angle_deg"]),
                "max_edge_ratio": value,
                "edge_ratio_limit": float(max_edge_ratio),
                "automatic": False,
                "audit_only": True,
            },
        ))
    part = build_structural_partition(surface, boundary_overrides=tuple(overrides))
    return part, tuple(unsafe_pairs)


def residual_class_summary(candidate, full, *, surface, skeleton, skin):
    rest, weights, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    weights = np.asarray(weights, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    skin_l1 = _skin_l1_per_face(weights, faces)
    by = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    rows = {"HAS_GENERATED_SEAM": [], "IDENTITY_ONLY": [], "OTHER_SUPPORT_MODE": []}

    for fi in map(int, full["unsafe_face_indices"]):
        vids = tuple(map(str, candidate.faces[fi]))
        modes = tuple(str(by[v].support_binding.mode) for v in vids)
        if any(m == "SEAM_GEOMETRY_INTERPOLATION" for m in modes):
            cls = "HAS_GENERATED_SEAM"
        elif all(m == "IDENTITY_SURFACE_NODE" for m in modes):
            cls = "IDENTITY_ONLY"
        else:
            cls = "OTHER_SUPPORT_MODE"
        rows[cls].append(float(skin_l1[fi]))

    return {
        cls: {
            "unsafe_face_count": len(vals),
            "unsafe_l1_le_0p5_count": sum(x <= DEFAULT_RISK_L1_MIN for x in vals),
            "unsafe_skin_l1": quantiles(vals),
        }
        for cls, vals in rows.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--skin-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rr = _ctx(a.authority_root, a.run_id)["run_root"]
    surface = rigging_surface_from_dict(
        load_json(rr / "artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json")
    )
    skeleton = qualified_skeleton_from_dict(
        load_json(rr / "artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json")
    )
    cameras = tuple(sorted(
        qualified_camera_set_from_dict(
            load_json(rr / "artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json")
        ).cameras,
        key=lambda x: int(x.view_index),
    ))
    envelope = deformation_envelope_from_dict(
        load_json(rr / "artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json")
    )
    policy = mesh_policy_from_dict(
        load_json(rr / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json")
    )
    skin = qualified_skin_from_dict(load_json(a.skin_json))
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("SKIN_SURFACE_DRIFT")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise RuntimeError("SKIN_SKELETON_DRIFT")

    explicit_faces, prov = replay_compacted_dense_face_provenance(rr, surface)
    sids, jids, W = skin_matrix(surface, skeleton, skin)

    risk = probe_edge_risk(surface, skeleton, cameras, envelope, sids, W)
    part, unsafe_source_edges = build_probe_partition(
        surface, risk, max_edge_ratio=DEFAULT_MAX_EDGE_RATIO
    )
    carrier = build_component_carrier_policy(
        partition=part,
        decisions=tuple(
            ComponentCarrierDecisionIR(
                c.component_id,
                "MESH",
                ("CANONICAL_PROBE_CONDITIONED_EDGE_COMPATIBILITY",),
                metadata={"automatic": False, "audit_only": True},
            )
            for c in part.components
        ),
        metadata={
            "audit_only": True,
            "owner": "CANONICAL_PROBE_CONDITIONED_EDGE_COMPATIBILITY",
        },
    )
    candidate = build_holeless_partitioned_dense_candidate(
        surface,
        part,
        carrier,
        producer_policy_hash=content_sha256({
            "audit": "CANONICAL_PROBE_CONDITIONED_EDGE_COMPATIBILITY",
            "edge_ratio_limit": float(DEFAULT_MAX_EDGE_RATIO),
            "skin_lineage_hash": skin.skin_lineage_hash,
        }),
        explicit_face_provenance=explicit_faces,
    )
    validate_canonical_mesh_candidate(
        candidate, surface=surface, partition=part, carrier_policy=carrier
    )

    full = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        stress_all_faces=True,
    )

    rest, weights, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    skin_l1 = _skin_l1_per_face(
        np.asarray(weights, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )
    risky_mask = skin_l1 > float(DEFAULT_RISK_L1_MIN)
    full_unsafe = np.asarray(tuple(map(int, full["unsafe_face_indices"])), dtype=np.int64)
    default_g3b_unsafe = int(np.count_nonzero(risky_mask[full_unsafe])) if len(full_unsafe) else 0

    source_report = load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    motion = motion_metrics(
        np.asarray(rest, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        jids,
        skeleton,
        cameras,
        rr,
        source_report,
    )
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )

    sizes = sorted((len(c.surface_ids) for c in part.components), reverse=True)
    max_ratio = np.asarray(risk["max_ratio"], dtype=np.float64)
    min_ratio = np.asarray(risk["min_ratio"], dtype=np.float64)
    report = {
        "schema": "RealSaS.KnightProbeConditionedMechanicalRegionCourt.v1",
        "status": "COMPLETE__AUDIT_ONLY__NO_PARENT_MUTATION",
        "operator": {
            "region_owner": "CANONICAL_PROBE_CONDITIONED_SOURCE_EDGE_COMPATIBILITY",
            "stress_angle_deg": float(risk["stress_angle_deg"]),
            "probe_count": int(risk["probe_count"]),
            "edge_ratio_limit": float(DEFAULT_MAX_EDGE_RATIO),
            "edge_rule": "SEPARATE_IF_MAX_CANONICAL_PROBE_EDGE_RATIO_EXCEEDS_EXISTING_G3B_EDGE_LIMIT",
            "new_numeric_threshold_introduced": False,
        },
        "skin_lineage_hash": skin.skin_lineage_hash,
        "face_provenance_replay": prov,
        "source_edge_audit": {
            "edge_count": len(risk["edge_pairs"]),
            "unsafe_source_edge_count": len(unsafe_source_edges),
            "max_ratio": quantiles(max_ratio),
            "min_ratio": quantiles(min_ratio),
        },
        "partition": {
            "component_count": len(part.components),
            "largest_component_size": sizes[0],
            "singleton_component_count": sum(x == 1 for x in sizes),
            "small_le4_component_count": sum(x <= 4 for x in sizes),
            "separate_boundary_count": sum(
                x.decision == "SEPARATE" for x in part.boundary_constraints
            ),
        },
        "candidate": {
            "vertex_count": len(candidate.vertices),
            "face_count": len(candidate.faces),
            "metadata": dict(candidate.metadata or {}),
        },
        "all_face_stress": {
            "passed": bool(full["passed"]),
            "unsafe_face_count": int(full["unsafe_face_count"]),
            "report_hash": full["report_hash"],
            "residual_class_summary": residual_class_summary(
                candidate, full, surface=surface, skeleton=skeleton, skin=skin
            ),
        },
        "derived_default_g3b": {
            "risk_l1_min": float(DEFAULT_RISK_L1_MIN),
            "risky_face_count": int(np.count_nonzero(risky_mask)),
            "unsafe_face_count": default_g3b_unsafe,
            "passed": default_g3b_unsafe == 0,
        },
        "g3_local_10deg": {
            "passed": bool(g3.passed),
            "failure_invariants": list(g3.failure_invariants),
            "minimum_area_ratio": float(g3.minimum_area_ratio),
            "maximum_area_ratio": float(g3.maximum_area_ratio),
            "maximum_condition_number": float(g3.maximum_condition_number),
            "minimum_edge_ratio": float(g3.minimum_edge_ratio),
            "maximum_edge_ratio": float(g3.maximum_edge_ratio),
            "report_hash": g3.report_hash,
        },
        "actual_motion": motion,
        "claim_boundary": [
            "Source adjacency is partitioned only by measured canonical stress edge stretch using the existing Stage35 edge-ratio limit.",
            "No QualifiedSkinIR row is mutated.",
            "Stage18 remains the existing holeless dense subdivision builder.",
            "All-face stress, G3 10-degree micro-stress, and actual Knight motion are downstream validation only.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PROBE_CONDITIONED_REGION_COURT=" + json.dumps({
        "unsafe_source_edges": report["source_edge_audit"]["unsafe_source_edge_count"],
        "components": report["partition"]["component_count"],
        "faces": report["candidate"]["face_count"],
        "all_face_unsafe": report["all_face_stress"]["unsafe_face_count"],
        "residual_classes": report["all_face_stress"]["residual_class_summary"],
        "g3b_unsafe": report["derived_default_g3b"]["unsafe_face_count"],
        "g3_pass": report["g3_local_10deg"]["passed"],
        "g3_failures": report["g3_local_10deg"]["failure_invariants"],
        "motion_gt10": report["actual_motion"]["max_edge_gt_10"],
        "motion_gt4": report["actual_motion"]["max_edge_gt_4"],
        "motion_worst": report["actual_motion"]["worst_edge_max"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
