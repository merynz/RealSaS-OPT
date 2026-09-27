from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)


def _context(authority_root: Path, run_id: str) -> dict:
    run_root = (authority_root / "runs" / run_id).resolve()
    return {
        "repo_root": Path(".").resolve(),
        "authority_root": authority_root.resolve(),
        "run_root": run_root,
        "run_id": run_id,
        "run_manifest_path": run_root / "run_manifest.json",
        "run_manifest": json.loads(
            (run_root / "run_manifest.json").read_text()
        ),
        "stage": {"id": "MOTION_TARGET_SKIN_SUPPORT_DIAGNOSTIC"},
        "ledger": json.loads((run_root / "ACTIVE_RUN_V2.json").read_text()),
    }


def diagnose(*, authority_root: Path, run_id: str) -> dict:
    ctx = _context(authority_root, run_id)
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )

    point_by_surface = {
        str(node.surface_id): np.asarray(node.P, dtype=np.float64)
        for node in surface.surface_nodes
    }
    joint_ids = tuple(
        str(joint.canonical_joint_id) for joint in skeleton.joints
    )
    accum = {
        jid: {
            "weight_sum": 0.0,
            "weighted_position_sum": np.zeros(3, dtype=np.float64),
            "weighted_outer_sum": np.zeros((3, 3), dtype=np.float64),
            "weighted_surface_count": 0,
            "maximum_surface_weight": 0.0,
        }
        for jid in joint_ids
    }
    unknown_surface_ids = []
    unknown_joint_ids = []
    for row in skin.rows:
        sid = str(row.surface_id)
        point = point_by_surface.get(sid)
        if point is None:
            unknown_surface_ids.append(sid)
            continue
        for jid, raw_weight in row.influences:
            jid = str(jid)
            if jid not in accum:
                unknown_joint_ids.append(jid)
                continue
            weight = float(raw_weight)
            if weight <= 0.0:
                continue
            target = accum[jid]
            target["weight_sum"] += weight
            target["weighted_position_sum"] += weight * point
            target["weighted_outer_sum"] += weight * np.outer(point, point)
            target["weighted_surface_count"] += 1
            target["maximum_surface_weight"] = max(
                float(target["maximum_surface_weight"]),
                weight,
            )

    if unknown_surface_ids:
        raise RuntimeError(
            "MOTION_TARGET_SUPPORT_UNKNOWN_SURFACE:"
            + ",".join(sorted(set(unknown_surface_ids))[:16])
        )
    if unknown_joint_ids:
        raise RuntimeError(
            "MOTION_TARGET_SUPPORT_UNKNOWN_JOINT:"
            + ",".join(sorted(set(unknown_joint_ids))[:16])
        )

    rows = []
    for joint in skeleton.joints:
        jid = str(joint.canonical_joint_id)
        row = accum[jid]
        total = float(row["weight_sum"])
        if total > 0.0:
            centroid = row["weighted_position_sum"] / total
            second = row["weighted_outer_sum"] / total
            covariance = second - np.outer(centroid, centroid)
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues = np.maximum(
                np.linalg.eigvalsh(covariance), 0.0
            )
        else:
            centroid = None
            covariance = None
            eigenvalues = None
        rows.append(
            {
                "canonical_joint_id": jid,
                "parent_canonical_id": (
                    None
                    if joint.parent_canonical_id is None
                    else str(joint.parent_canonical_id)
                ),
                "joint_rest_position": [
                    float(x) for x in joint.position
                ],
                "skin_weight_sum": total,
                "weighted_surface_count": int(
                    row["weighted_surface_count"]
                ),
                "maximum_surface_weight": float(
                    row["maximum_surface_weight"]
                ),
                "mesh_influence_centroid": (
                    None
                    if centroid is None
                    else [float(x) for x in centroid]
                ),
                "mesh_influence_covariance": (
                    None
                    if covariance is None
                    else [
                        [float(x) for x in r]
                        for r in covariance
                    ]
                ),
                "mesh_influence_spread_eigenvalues": (
                    None
                    if eigenvalues is None
                    else [float(x) for x in eigenvalues]
                ),
                "has_skin_support": total > 0.0,
            }
        )

    supported = [row for row in rows if row["has_skin_support"]]
    if not supported:
        raise RuntimeError("MOTION_TARGET_SUPPORT_EMPTY")
    return {
        "schema": "RealSaS.MotionTargetSkinSupportDiagnostic.v1",
        "status": "MEASURED",
        "run_id": run_id,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "joint_count": len(rows),
        "supported_joint_count": len(supported),
        "unsupported_joint_count": len(rows) - len(supported),
        "descriptor_frame": "REALSAS_OBJECT_FRAME",
        "descriptor_source": (
            "STAGE15_CANONICAL_SURFACE_P_X_STAGE32_QUALIFIED_SKIN_WEIGHT"
        ),
        "joint_supports": rows,
        "product_authority_claimed": False,
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = diagnose(
        authority_root=Path(args.authority_root).expanduser().resolve(),
        run_id=args.run_id,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "MOTION_TARGET_SKIN_SUPPORT_DIAGNOSTIC",
        json.dumps(
            {
                "joint_count": report["joint_count"],
                "supported_joint_count": report["supported_joint_count"],
                "unsupported_joint_count": report["unsupported_joint_count"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
