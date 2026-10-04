from __future__ import annotations

"""Audit-only seam-support locality counterfactual for Knight V9.

Geometry, topology, partition, QualifiedSkinIR and skeleton are frozen.
Only generated seam vertices' mechanical skin-support coefficients are changed
in memory for counterfactual proof:
  baseline harmonic / top1 / top2 / top4 renormalized.

No candidate is serialized as product authority and no repair is applied.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
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


SCHEMA = "RealSaS.KnightV9SeamSupportLocalityCounterfactual.v1"
EXPECTED_BASELINE_PRODUCTION_UNSAFE = 48
EXPECTED_BASELINE_ALL_FACE_UNSAFE = 58


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_jsonable(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _topk_support(candidate, k: int | None):
    if k is None:
        return candidate, {
            "mode": "CURRENT_COMPONENT_HARMONIC_DIRICHLET_V1",
            "changed_vertex_count": 0,
            "support_entries_before": 0,
            "support_entries_after": 0,
            "mean_max_weight_before": None,
            "mean_max_weight_after": None,
        }

    changed = []
    before_entries = 0
    after_entries = 0
    max_before = []
    max_after = []
    out_vertices = []

    for vertex in candidate.vertices:
        if str(vertex.support_binding.mode) != "SEAM_GEOMETRY_INTERPOLATION":
            out_vertices.append(vertex)
            continue
        md = dict(vertex.support_binding.metadata or {})
        raw = tuple(md.get("skin_support_coefficients") or ())
        if not raw:
            raise RuntimeError(
                "SEAM_SUPPORT_COUNTERFACTUAL_SUPPORT_MISSING:"
                + str(vertex.candidate_vertex_id)
            )
        rows = [(str(sid), float(w)) for sid, w in raw if float(w) > 0.0]
        if not rows:
            raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_SUPPORT_EMPTY")
        rows.sort(key=lambda x: (-x[1], x[0]))
        total = sum(w for _, w in rows)
        if abs(total - 1.0) > 1e-8:
            raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_SUPPORT_NOT_SIMPLEX")
        before_entries += len(rows)
        max_before.append(rows[0][1])

        kept = rows[: min(int(k), len(rows))]
        denom = sum(w for _, w in kept)
        if denom <= 0.0:
            raise RuntimeError("SEAM_SUPPORT_COUNTERFACTUAL_TOPK_COLLAPSE")
        kept = tuple((sid, float(w / denom)) for sid, w in kept)
        after_entries += len(kept)
        max_after.append(max(w for _, w in kept))

        new_md = {
            **md,
            "skin_support_coefficients": kept,
            "audit_counterfactual_only": True,
            "audit_counterfactual_policy": f"HARMONIC_TOP{k}_RENORMALIZED",
        }
        out_vertices.append(
            replace(
                vertex,
                support_binding=replace(vertex.support_binding, metadata=new_md),
            )
        )
        changed.append(str(vertex.candidate_vertex_id))

    provisional = replace(candidate, vertices=tuple(out_vertices))
    return provisional, {
        "mode": f"HARMONIC_TOP{k}_RENORMALIZED",
        "changed_vertex_count": len(changed),
        "support_entries_before": int(before_entries),
        "support_entries_after": int(after_entries),
        "mean_max_weight_before": float(np.mean(max_before)) if max_before else 0.0,
        "mean_max_weight_after": float(np.mean(max_after)) if max_after else 0.0,
        "changed_vertex_set_sha256": _sha256_jsonable(sorted(changed)),
    }


def _failure_signature(row, report):
    out = []
    if float(row["max_edge_ratio"]) > float(report["max_edge_ratio_limit"]):
        out.append("EDGE_RATIO")
    if float(row["max_area_ratio"]) > float(report["max_area_ratio_limit"]):
        out.append("AREA_EXPANSION")
    if float(row["min_area_ratio"]) < float(report["min_area_ratio_limit"]):
        out.append("AREA_COLLAPSE")
    if float(row["max_condition_number"]) > float(report["max_condition_limit"]):
        out.append("CONDITION")
    return "+".join(out) if out else "UNKNOWN"


def _summary(report):
    rows = tuple(report.get("top_unsafe_faces") or ())
    if len(rows) != int(report["unsafe_face_count"]):
        raise RuntimeError(
            f"SEAM_SUPPORT_COUNTERFACTUAL_REPORT_TRUNCATED:{len(rows)}:"
            f"{report['unsafe_face_count']}"
        )
    sig = Counter(_failure_signature(row, report) for row in rows)
    cond = sum(
        float(row["max_condition_number"]) > float(report["max_condition_limit"])
        for row in rows
    )
    return {
        "unsafe_face_count": int(report["unsafe_face_count"]),
        "unsafe_face_indices": tuple(map(int, report["unsafe_face_indices"])),
        "failure_signature_counts": dict(sorted(sig.items())),
        "condition_unsafe_count": int(cond),
        "max_condition_number": max(
            (float(row["max_condition_number"]) for row in rows),
            default=1.0,
        ),
        "min_area_ratio": min(
            (float(row["min_area_ratio"]) for row in rows),
            default=1.0,
        ),
        "max_area_ratio": max(
            (float(row["max_area_ratio"]) for row in rows),
            default=1.0,
        ),
        "max_edge_ratio": max(
            (float(row["max_edge_ratio"]) for row in rows),
            default=1.0,
        ),
        "report_hash": str(report["report_hash"]),
    }


def _run(candidate, *, surface, skeleton, skin, envelope, cameras, policy):
    prod = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
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
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        risk_l1_min=0.0,
        stress_all_faces=True,
    )
    return _summary(prod), _summary(all_faces)


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

    surface = rigging_surface_from_dict(_read(args.surface_json))
    partition = mechanical_partition_from_dict(_read(args.partition_json))
    validate_mechanical_partition(partition, surface)
    candidate = canonical_mesh_candidate_from_dict(_read(args.candidate_json))
    skeleton = qualified_skeleton_from_dict(_read(args.fresh_skeleton_json))
    camera_set = qualified_camera_set_from_dict(_read(args.camera_set_json))
    policy = mesh_policy_from_dict(_read(args.mesh_policy_json))
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    _, envelope = derive_deformation_envelope_v1(
        skeleton=skeleton, camera_set=camera_set
    )
    teacher_skin, _, valid, _, _, _ = teacher_to_skin(
        surface, skeleton, args.teacher_bank
    )

    arms = []
    baseline_prod = None
    baseline_all = None
    for k in (None, 1, 2, 4):
        arm_candidate, mutation = _topk_support(candidate, k)
        prod, all_faces = _run(
            arm_candidate,
            surface=surface,
            skeleton=skeleton,
            skin=teacher_skin,
            envelope=envelope,
            cameras=cameras,
            policy=policy,
        )
        if k is None:
            if prod["unsafe_face_count"] != EXPECTED_BASELINE_PRODUCTION_UNSAFE:
                raise RuntimeError(
                    "SEAM_SUPPORT_COUNTERFACTUAL_BASELINE_PRODUCTION_DRIFT:"
                    + str(prod["unsafe_face_count"])
                )
            if all_faces["unsafe_face_count"] != EXPECTED_BASELINE_ALL_FACE_UNSAFE:
                raise RuntimeError(
                    "SEAM_SUPPORT_COUNTERFACTUAL_BASELINE_ALL_FACE_DRIFT:"
                    + str(all_faces["unsafe_face_count"])
                )
            baseline_prod = prod
            baseline_all = all_faces

        assert baseline_prod is not None and baseline_all is not None
        p0 = set(baseline_prod["unsafe_face_indices"])
        p1 = set(prod["unsafe_face_indices"])
        a0 = set(baseline_all["unsafe_face_indices"])
        a1 = set(all_faces["unsafe_face_indices"])
        arms.append({
            "arm": mutation["mode"],
            "support_mutation": mutation,
            "production": prod,
            "all_face": all_faces,
            "paired_delta": {
                "production_unsafe_delta":
                    int(prod["unsafe_face_count"] - baseline_prod["unsafe_face_count"]),
                "all_face_unsafe_delta":
                    int(all_faces["unsafe_face_count"] - baseline_all["unsafe_face_count"]),
                "production_resolved_count": len(p0 - p1),
                "production_introduced_count": len(p1 - p0),
                "all_face_resolved_count": len(a0 - a1),
                "all_face_introduced_count": len(a1 - a0),
                "condition_unsafe_delta":
                    int(prod["condition_unsafe_count"] - baseline_prod["condition_unsafe_count"]),
            },
        })
        print(
            "SEAM_SUPPORT_ARM "
            f"arm={mutation['mode']} "
            f"prod={prod['unsafe_face_count']} "
            f"all={all_faces['unsafe_face_count']} "
            f"condition={prod['condition_unsafe_count']} "
            f"prod_delta={arms[-1]['paired_delta']['production_unsafe_delta']} "
            f"introduced={arms[-1]['paired_delta']['production_introduced_count']}",
            flush=True,
        )

    nonbaseline = arms[1:]
    best = min(
        nonbaseline,
        key=lambda row: (
            row["production"]["unsafe_face_count"],
            row["all_face"]["unsafe_face_count"],
            row["paired_delta"]["production_introduced_count"],
            row["production"]["condition_unsafe_count"],
        ),
    )
    locality_supported = (
        best["production"]["unsafe_face_count"]
        < baseline_prod["unsafe_face_count"]
        and best["all_face"]["unsafe_face_count"]
        <= baseline_all["unsafe_face_count"]
        and best["paired_delta"]["production_introduced_count"] == 0
        and best["paired_delta"]["all_face_introduced_count"] == 0
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS__AUDIT_ONLY",
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "partition_lineage_hash": partition.partition_lineage_hash,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "teacher_clean_row_count": int(np.count_nonzero(valid)),
        "teacher_row_count": int(len(valid)),
        "arms": arms,
        "best_arm": best["arm"],
        "locality_hypothesis_supported": bool(locality_supported),
        "interpretation": (
            "BOUNDED_SUPPORT_LOCALITY_CAUSALLY_IMPROVES_G3B"
            if locality_supported else
            "TOPK_LOCALITY_NOT_SUFFICIENT_OR_MOVES_FAILURE"
        ),
        "claim_boundary": {
            "geometry_mutated": False,
            "topology_mutated": False,
            "partition_mutated": False,
            "qualified_skin_mutated": False,
            "candidate_support_mutated_in_memory_only": True,
            "product_authority_minted": False,
            "generalization_claimed": False,
        },
    }
    out = args.out_dir / "SEAM_SUPPORT_LOCALITY_COUNTERFACTUAL.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "KNIGHT_V9_SEAM_SUPPORT_LOCALITY_COUNTERFACTUAL="
        + json.dumps({
            "status": result["status"],
            "best_arm": result["best_arm"],
            "locality_hypothesis_supported": result["locality_hypothesis_supported"],
            "arms": [
                {
                    "arm": x["arm"],
                    "production": x["production"]["unsafe_face_count"],
                    "all_face": x["all_face"]["unsafe_face_count"],
                    "introduced": x["paired_delta"]["production_introduced_count"],
                }
                for x in arms
            ],
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
