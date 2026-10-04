from __future__ import annotations

"""Classify residual G3B unsafe faces without mutating geometry or weights.

Portable diagnostic-only court for the checkpointed V9 mechanics notebook.
It reruns the existing skin/topology compatibility proof on an exact candidate,
then attributes each residual unsafe face by threshold family, rest-static status,
mechanical component, support-binding pattern, and worst motion probe.

No repair is executed. No product authority is minted.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    validate_mechanical_partition,
)
from tools.audit_v9_teacher_projection_oracle_v1 import teacher_to_skin


SCHEMA = "RealSaS.V9G3BResidualClassification.v1"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _triangle_rest_quality(points):
    p0, p1, p2 = (np.asarray(x, dtype=np.float64) for x in points)
    edges = (
        float(np.linalg.norm(p1 - p0)),
        float(np.linalg.norm(p2 - p1)),
        float(np.linalg.norm(p0 - p2)),
    )
    if min(edges) <= 1e-15:
        return {"min_angle_deg": 0.0, "aspect": float("inf"), "area": 0.0}
    area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
    if area <= 1e-15:
        return {"min_angle_deg": 0.0, "aspect": float("inf"), "area": area}

    def angle(a, b, c):
        u = b - a
        v = c - a
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        x = float(np.dot(u, v)) / max(denom, 1e-15)
        return math.degrees(math.acos(max(-1.0, min(1.0, x))))

    angles = (
        angle(p0, p1, p2),
        angle(p1, p2, p0),
        angle(p2, p0, p1),
    )
    longest = max(edges)
    min_altitude = 2.0 * area / longest
    aspect = longest / max(min_altitude, 1e-15)
    return {
        "min_angle_deg": float(min(angles)),
        "aspect": float(aspect),
        "area": float(area),
    }


def _dynamic_failures(row, report):
    failures = []
    if float(row["max_edge_ratio"]) > float(report["max_edge_ratio_limit"]):
        failures.append("EDGE_RATIO")
    if float(row["max_area_ratio"]) > float(report["max_area_ratio_limit"]):
        failures.append("AREA_EXPANSION")
    if float(row["min_area_ratio"]) < float(report["min_area_ratio_limit"]):
        failures.append("AREA_COLLAPSE")
    if float(row["max_condition_number"]) > float(report["max_condition_limit"]):
        failures.append("CONDITION")
    return tuple(failures)


def _severity(row, report):
    return float(max(
        float(row["max_edge_ratio"]) / float(report["max_edge_ratio_limit"]),
        float(row["max_area_ratio"]) / float(report["max_area_ratio_limit"]),
        float(report["min_area_ratio_limit"]) / max(float(row["min_area_ratio"]), 1e-15),
        float(row["max_condition_number"]) / float(report["max_condition_limit"]),
    ))


def _classify(report, candidate, policy):
    vertices = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    rows = []
    for row in tuple(report.get("top_unsafe_faces") or ()):
        vids = tuple(map(str, row["vertex_ids"]))
        vv = [vertices[x] for x in vids]
        rest = _triangle_rest_quality([v.P for v in vv])
        static_reasons = []
        if rest["min_angle_deg"] < float(policy.g3_min_angle_deg):
            static_reasons.append("MIN_ANGLE")
        if rest["aspect"] > float(policy.g3_max_aspect_longest_over_min_altitude):
            static_reasons.append("ASPECT")
        dyn = _dynamic_failures(row, report)
        if not dyn:
            raise RuntimeError(
                "G3B_RESIDUAL_UNSAFE_FACE_WITHOUT_DYNAMIC_THRESHOLD:"
                + str(row["face_index"])
            )
        modes = tuple(sorted(str(v.support_binding.mode) for v in vv))
        components = tuple(sorted(set(str(v.component_id) for v in vv)))
        probe = str(row["worst_probe_id"])
        joint = probe.split(":LOCAL_", 1)[0] if ":LOCAL_" in probe else probe
        rows.append({
            "face_index": int(row["face_index"]),
            "vertex_ids": vids,
            "component_ids": components,
            "component_pure": len(components) == 1,
            "support_binding_modes": modes,
            "contains_seam_vertex": any(x != "IDENTITY_SURFACE_NODE" for x in modes),
            "skin_l1_max": float(row["skin_l1_max"]),
            "dynamic_failures": dyn,
            "dynamic_failure_signature": "+".join(dyn),
            "worst_probe_id": probe,
            "worst_joint_id": joint,
            "max_edge_ratio": float(row["max_edge_ratio"]),
            "max_area_ratio": float(row["max_area_ratio"]),
            "min_area_ratio": float(row["min_area_ratio"]),
            "max_condition_number": float(row["max_condition_number"]),
            "dynamic_severity": _severity(row, report),
            "rest_min_angle_deg": rest["min_angle_deg"],
            "rest_aspect": rest["aspect"],
            "rest_area": rest["area"],
            "rest_static_failures": tuple(static_reasons),
            "rest_static_bad": bool(static_reasons),
        })
    if len(rows) != int(report["unsafe_face_count"]):
        raise RuntimeError(
            f"G3B_RESIDUAL_REPORT_TRUNCATED:{len(rows)}:{report['unsafe_face_count']}"
        )
    return rows


def _counter(rows, key):
    return dict(sorted(Counter(row[key] for row in rows).items(), key=lambda kv: (-kv[1], kv[0])))


def _summary(rows, report, all_face_report):
    sev = np.asarray([float(x["dynamic_severity"]) for x in rows], dtype=np.float64)
    prod = set(map(int, report["unsafe_face_indices"]))
    all_ids = set(map(int, all_face_report["unsafe_face_indices"]))
    return {
        "unsafe_face_count": len(rows),
        "all_face_unsafe_count": int(all_face_report["unsafe_face_count"]),
        "all_face_only_unsafe_count": len(all_ids - prod),
        "rest_static_bad_count": int(sum(bool(x["rest_static_bad"]) for x in rows)),
        "dynamic_only_count": int(sum(not bool(x["rest_static_bad"]) for x in rows)),
        "seam_face_count": int(sum(bool(x["contains_seam_vertex"]) for x in rows)),
        "identity_only_face_count": int(sum(not bool(x["contains_seam_vertex"]) for x in rows)),
        "component_pure_count": int(sum(bool(x["component_pure"]) for x in rows)),
        "dynamic_failure_signature_counts": _counter(rows, "dynamic_failure_signature"),
        "worst_joint_counts": _counter(rows, "worst_joint_id"),
        "component_counts": _counter(
            [{**x, "_component": "|".join(x["component_ids"])} for x in rows],
            "_component",
        ),
        "severity_p50": float(np.quantile(sev, 0.50)) if len(sev) else 0.0,
        "severity_p95": float(np.quantile(sev, 0.95)) if len(sev) else 0.0,
        "severity_max": float(sev.max(initial=0.0)),
        "stress_angle_deg": float(report["stress_angle_deg"]),
        "probe_count": int(report["probe_count"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--partition-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--fresh-skeleton-json", type=Path, required=True)
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--camera-set-json", type=Path, required=True)
    ap.add_argument("--mesh-policy-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    surface = rigging_surface_from_dict(read(args.surface_json))
    partition = mechanical_partition_from_dict(read(args.partition_json))
    validate_mechanical_partition(partition, surface)
    candidate = canonical_mesh_candidate_from_dict(read(args.candidate_json))
    skeleton = qualified_skeleton_from_dict(read(args.fresh_skeleton_json))
    camera_set = qualified_camera_set_from_dict(read(args.camera_set_json))
    policy = mesh_policy_from_dict(read(args.mesh_policy_json))
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    _, envelope = derive_deformation_envelope_v1(
        skeleton=skeleton, camera_set=camera_set
    )
    teacher_skin, _, valid, _, _, _ = teacher_to_skin(
        surface, skeleton, args.teacher_bank
    )

    production = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        risk_l1_min=0.5,
        stress_all_faces=False,
    )
    all_faces = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        risk_l1_min=0.0,
        stress_all_faces=True,
    )
    rows = _classify(production, candidate, policy)
    summary = _summary(rows, production, all_faces)

    report = {
        "schema": SCHEMA,
        "status": "PASS__DIAGNOSTIC_ONLY",
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "partition_lineage_hash": partition.partition_lineage_hash,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "teacher_clean_row_count": int(np.count_nonzero(valid)),
        "teacher_row_count": int(len(valid)),
        "summary": summary,
        "faces": rows,
        "claim_boundary": {
            "geometry_mutated": False,
            "skin_mutated": False,
            "partition_mutated": False,
            "product_authority_minted": False,
            "generalization_claimed": False,
        },
    }

    (args.out_dir / "PRODUCTION_G3B_REPORT.json").write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "ALL_FACE_G3B_REPORT.json").write_text(
        json.dumps(all_faces, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "G3B_RESIDUAL_CLASSIFICATION.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "G3B_RESIDUAL_FACES.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "face_index", "dynamic_failure_signature", "dynamic_severity",
            "rest_static_bad", "rest_min_angle_deg", "rest_aspect",
            "contains_seam_vertex", "skin_l1_max", "worst_joint_id",
            "worst_probe_id", "component_ids", "vertex_ids",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{k: row[k] for k in fields if k not in {"component_ids", "vertex_ids"}},
                "component_ids": "|".join(row["component_ids"]),
                "vertex_ids": "|".join(row["vertex_ids"]),
            })

    print("KNIGHT_V9_G3B_RESIDUAL_CLASSIFICATION=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
