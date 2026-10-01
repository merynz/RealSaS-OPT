from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
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
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    _skin_l1_per_face,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json,
    replay_compacted_dense_face_provenance,
)
from tools.audit_knight_global_skin_region_sweep_v1 import skin_matrix, threshold_partition
from tools.demo.render_knight_motion_preview_v1 import _ctx


def _face_class(candidate, face_index: int, *, by) -> str:
    modes = [str(by[str(vid)].support_binding.mode) for vid in candidate.faces[int(face_index)]]
    if any(m == "SEAM_GEOMETRY_INTERPOLATION" for m in modes):
        return "HAS_GENERATED_SEAM"
    if all(m == "IDENTITY_SURFACE_NODE" for m in modes):
        return "IDENTITY_ONLY"
    return "OTHER_SUPPORT_MODE"


def _static_face_rows(candidate, *, surface, skeleton, skin):
    rest, weights, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    rest = np.asarray(rest, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    skin_l1 = _skin_l1_per_face(weights, faces)
    by = {str(v.candidate_vertex_id): v for v in candidate.vertices}

    rows = []
    for fi, idx in enumerate(faces):
        tri = tuple(map(int, idx.tolist()))
        p = rest[np.asarray(tri, dtype=np.int64)]
        w = weights[np.asarray(tri, dtype=np.int64)]
        edge_rows = []
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            length = float(np.linalg.norm(p[ib] - p[ia]))
            l1 = float(np.abs(w[ib] - w[ia]).sum())
            edge_rows.append({
                "edge_local": [ia, ib],
                "rest_length": length,
                "skin_l1": l1,
                "skin_l1_per_rest_length": (
                    float(l1 / length) if length > 1e-15 else float("inf")
                ),
            })
        vids = tuple(map(str, candidate.faces[fi]))
        modes = tuple(str(by[v].support_binding.mode) for v in vids)
        seam_kinds = tuple(
            str(dict(by[v].metadata or {}).get("seam_kind") or "")
            for v in vids
        )
        rows.append({
            "face_index": int(fi),
            "face_class": _face_class(candidate, fi, by=by),
            "vertex_ids": vids,
            "support_modes": modes,
            "seam_kinds": seam_kinds,
            "skin_l1_max": float(skin_l1[fi]),
            "min_rest_edge_length": min(x["rest_length"] for x in edge_rows),
            "max_skin_l1_per_rest_length": max(
                x["skin_l1_per_rest_length"] for x in edge_rows
            ),
            "edges": edge_rows,
        })
    return rows


def _quantiles(values):
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


def _classify_candidate(
    name,
    candidate,
    *,
    surface,
    skeleton,
    skin,
    envelope,
    cameras,
    policy,
):
    static = _static_face_rows(candidate, surface=surface, skeleton=skeleton, skin=skin)
    by_index = {int(x["face_index"]): x for x in static}
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
    unsafe = tuple(map(int, full["unsafe_face_indices"]))
    unsafe_set = set(unsafe)

    classes = sorted({x["face_class"] for x in static})
    class_summary = {}
    for cls in classes:
        all_rows = [x for x in static if x["face_class"] == cls]
        bad_rows = [x for x in all_rows if int(x["face_index"]) in unsafe_set]
        class_summary[cls] = {
            "all_face_count": len(all_rows),
            "unsafe_face_count": len(bad_rows),
            "unsafe_fraction": (
                float(len(bad_rows) / len(all_rows)) if all_rows else 0.0
            ),
            "unsafe_skin_l1": _quantiles(x["skin_l1_max"] for x in bad_rows),
            "unsafe_skin_l1_per_rest_length": _quantiles(
                x["max_skin_l1_per_rest_length"] for x in bad_rows
            ),
            "unsafe_min_rest_edge_length": _quantiles(
                x["min_rest_edge_length"] for x in bad_rows
            ),
            "unsafe_l1_le_0p5_count": sum(
                float(x["skin_l1_max"]) <= 0.5 for x in bad_rows
            ),
        }

    top_dynamic = []
    for row in full["top_unsafe_faces"]:
        fi = int(row["face_index"])
        top_dynamic.append({
            **dict(row),
            "static": by_index[fi],
        })

    unsafe_static = [by_index[i] for i in unsafe]
    return {
        "name": name,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "candidate_face_count": len(candidate.faces),
        "candidate_vertex_count": len(candidate.vertices),
        "candidate_metadata": dict(candidate.metadata or {}),
        "stress_report_hash": full["report_hash"],
        "stress_angle_deg": full["stress_angle_deg"],
        "unsafe_face_count": len(unsafe),
        "unsafe_l1_le_0p5_count": sum(
            float(x["skin_l1_max"]) <= 0.5 for x in unsafe_static
        ),
        "unsafe_skin_l1": _quantiles(x["skin_l1_max"] for x in unsafe_static),
        "unsafe_skin_l1_per_rest_length": _quantiles(
            x["max_skin_l1_per_rest_length"] for x in unsafe_static
        ),
        "unsafe_min_rest_edge_length": _quantiles(
            x["min_rest_edge_length"] for x in unsafe_static
        ),
        "class_summary": class_summary,
        "top_dynamic_unsafe_faces": top_dynamic,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--skin-json", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
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
    baseline = canonical_mesh_candidate_from_dict(
        load_json(rr / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json")
    )
    skin = qualified_skin_from_dict(load_json(a.skin_json))

    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("SKIN_SURFACE_DRIFT")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise RuntimeError("SKIN_SKELETON_DRIFT")

    explicit_faces, prov = replay_compacted_dense_face_provenance(rr, surface)
    sids, _, W = skin_matrix(surface, skeleton, skin)
    groups, overrides, _ = threshold_partition(surface, sids, W, float(a.threshold))
    partition = build_structural_partition(surface, boundary_overrides=overrides)
    if len(partition.components) != len(groups):
        raise RuntimeError("REGION_COUNT_DRIFT")
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(
                c.component_id,
                "MESH",
                ("G3_OFFENDER_CLASSIFICATION_GLOBAL_SKIN_REGION",),
                metadata={"automatic": False, "audit_only": True},
            )
            for c in partition.components
        ),
        metadata={"audit_only": True, "threshold": float(a.threshold)},
    )
    global_candidate = build_holeless_partitioned_dense_candidate(
        surface,
        partition,
        carrier,
        producer_policy_hash=content_sha256({
            "audit": "G3_OFFENDER_CLASSIFICATION_GLOBAL_SKIN_REGION",
            "threshold": float(a.threshold),
            "arm": str(a.skin_json),
        }),
        explicit_face_provenance=explicit_faces,
    )
    validate_canonical_mesh_candidate(
        global_candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
    )

    baseline_report = _classify_candidate(
        "BASELINE_STAGE18_WITH_CORRECTED_SKIN",
        baseline,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    global_report = _classify_candidate(
        f"GLOBAL_REGION_THRESHOLD_{a.threshold:g}",
        global_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )

    report = {
        "schema": "RealSaS.KnightG3OffenderClassificationAudit.v1",
        "status": "COMPLETE__AUDIT_ONLY__NO_PARENT_MUTATION",
        "threshold": float(a.threshold),
        "skin_lineage_hash": skin.skin_lineage_hash,
        "face_provenance_replay": prov,
        "baseline": baseline_report,
        "global_region_child": global_report,
        "claim_boundary": [
            "This audit classifies all-face topology-skin stress offenders by candidate support mode and static skin-gradient geometry.",
            "It does not mutate QualifiedSkinIR, Stage17, Stage18, or product authority.",
            "HAS_GENERATED_SEAM means at least one face vertex uses SEAM_GEOMETRY_INTERPOLATION support.",
            "skin_l1_per_rest_length is diagnostic only and is not a production threshold or authority.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("G3_OFFENDER_CLASSIFICATION=" + json.dumps({
        "threshold": report["threshold"],
        "baseline_unsafe": baseline_report["unsafe_face_count"],
        "baseline_classes": baseline_report["class_summary"],
        "global_unsafe": global_report["unsafe_face_count"],
        "global_classes": global_report["class_summary"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
