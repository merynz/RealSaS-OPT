"""Diagnostic counterfactual: archived rig kinematics/IDs on V9 support anchors.

This does not qualify or mint a product skeleton.  It preserves the archived
Stage28 canonical joint IDs, tree and rest positions while replacing only each
joint's support_surface_ids with the support set inferred/qualified on the V9
surface.  The result lets Arachne see a V9-valid support-anchor field without
conflating that experiment with fresh Geppetto kinematic drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualifiedJoint, QualifiedSkeletonIR
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _source_index(skeleton):
    rows = {}
    canonical_to_source = {}
    for joint in skeleton.joints:
        source = str(joint.source_proposal_id)
        if not source or source in rows:
            raise RuntimeError("COUNTERFACTUAL_SOURCE_PROPOSAL_ID_INVALID")
        rows[source] = joint
        canonical_to_source[str(joint.canonical_joint_id)] = source
    if len(canonical_to_source) != len(rows):
        raise RuntimeError("COUNTERFACTUAL_CANONICAL_ID_DUPLICATE")
    return rows, canonical_to_source


def _parent_source(joint, canonical_to_source):
    if joint.parent_canonical_id is None:
        return None
    parent = canonical_to_source.get(str(joint.parent_canonical_id))
    if parent is None:
        raise RuntimeError("COUNTERFACTUAL_PARENT_CANONICAL_ID_UNKNOWN")
    return parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fresh-skeleton-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    ctx = _ctx(args.authority_root, args.run_id)
    archived = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    fresh = qualified_skeleton_from_dict(read(args.fresh_skeleton_json))

    archived_by_source, archived_source_by_canonical = _source_index(archived)
    fresh_by_source, fresh_source_by_canonical = _source_index(fresh)
    archived_sources = set(archived_by_source)
    fresh_sources = set(fresh_by_source)
    if archived_sources != fresh_sources:
        raise RuntimeError(
            "COUNTERFACTUAL_SOURCE_SET_DRIFT:"
            f"archived_only={sorted(archived_sources-fresh_sources)}:"
            f"fresh_only={sorted(fresh_sources-archived_sources)}"
        )

    archived_root_source = archived_source_by_canonical.get(str(archived.root_id))
    fresh_root_source = fresh_source_by_canonical.get(str(fresh.root_id))
    if archived_root_source is None or fresh_root_source is None:
        raise RuntimeError("COUNTERFACTUAL_ROOT_SOURCE_UNKNOWN")

    old_positions = np.asarray(
        [archived_by_source[s].position for s in sorted(archived_sources)],
        dtype=np.float64,
    )
    span = float(np.max(old_positions.max(axis=0) - old_positions.min(axis=0)))
    if not np.isfinite(span) or span <= 1e-12:
        raise RuntimeError("COUNTERFACTUAL_ARCHIVED_RIG_SCALE_INVALID")

    joints = []
    rows = []
    deltas = []
    parent_changes = 0
    canonical_changes = 0
    for source in sorted(archived_sources):
        old = archived_by_source[source]
        new = fresh_by_source[source]
        old_parent_source = _parent_source(old, archived_source_by_canonical)
        fresh_parent_source = _parent_source(new, fresh_source_by_canonical)
        changed_parent = old_parent_source != fresh_parent_source
        parent_changes += int(changed_parent)
        canonical_changes += int(str(old.canonical_joint_id) != str(new.canonical_joint_id))

        delta = float(
            np.linalg.norm(
                np.asarray(new.position, dtype=np.float64)
                - np.asarray(old.position, dtype=np.float64)
            )
        )
        deltas.append(delta)
        joints.append(
            QualifiedJoint(
                canonical_joint_id=str(old.canonical_joint_id),
                position=tuple(map(float, old.position)),
                parent_canonical_id=(
                    None if old.parent_canonical_id is None
                    else str(old.parent_canonical_id)
                ),
                support_surface_ids=tuple(map(str, new.support_surface_ids)),
                source_proposal_id=source,
            )
        )
        rows.append(
            {
                "source_proposal_id": source,
                "archived_canonical_joint_id": str(old.canonical_joint_id),
                "fresh_v9_canonical_joint_id": str(new.canonical_joint_id),
                "archived_parent_source": old_parent_source,
                "fresh_v9_parent_source": fresh_parent_source,
                "parent_changed": bool(changed_parent),
                "archived_position": list(map(float, old.position)),
                "fresh_v9_position": list(map(float, new.position)),
                "position_delta": delta,
                "position_delta_over_archived_span": delta / span,
                "archived_support_count": len(old.support_surface_ids),
                "fresh_v9_support_count": len(new.support_surface_ids),
            }
        )

    qreport = {
        "status": "DIAGNOSTIC_COUNTERFACTUAL_NOT_QUALIFICATION",
        "product_authority_minted": False,
        "purpose": "ISOLATE_V9_SUBSTRATE_AND_SKIN_FROM_FRESH_GEPPETTO_KINEMATICS",
        "kinematics_source": "ARCHIVED_STAGE28",
        "canonical_joint_identity_source": "ARCHIVED_STAGE28",
        "support_anchor_source": "FRESH_V9_QUALIFIED_SKELETON",
        "archived_skeleton_lineage_hash": archived.skeleton_lineage_hash,
        "fresh_v9_skeleton_lineage_hash": fresh.skeleton_lineage_hash,
    }
    provisional = QualifiedSkeletonIR(
        joints=tuple(joints),
        root_id=str(archived.root_id),
        qualification_report=qreport,
        skeleton_lineage_hash="",
    )
    payload = provisional.to_dict()
    payload.pop("skeleton_lineage_hash", None)
    hybrid = QualifiedSkeletonIR(
        joints=tuple(joints),
        root_id=str(archived.root_id),
        qualification_report=qreport,
        skeleton_lineage_hash=content_sha256(payload),
    )

    d = np.asarray(deltas, dtype=np.float64)
    delta_report = {
        "schema": "RealSaS.KnightV9ArchivedKinematicsCounterfactual.v1",
        "status": "DIAGNOSTIC_ONLY",
        "product_authority_minted": False,
        "joint_count": len(joints),
        "source_proposal_set_equal": True,
        "archived_root_source": archived_root_source,
        "fresh_v9_root_source": fresh_root_source,
        "root_source_changed": archived_root_source != fresh_root_source,
        "parent_source_change_count": int(parent_changes),
        "canonical_joint_id_change_count": int(canonical_changes),
        "archived_body_span": span,
        "position_delta_mean": float(np.mean(d)),
        "position_delta_p95": float(np.quantile(d, 0.95)),
        "position_delta_max": float(np.max(d)),
        "position_delta_mean_over_span": float(np.mean(d) / span),
        "position_delta_p95_over_span": float(np.quantile(d, 0.95) / span),
        "position_delta_max_over_span": float(np.max(d) / span),
        "archived_skeleton_lineage_hash": archived.skeleton_lineage_hash,
        "fresh_v9_skeleton_lineage_hash": fresh.skeleton_lineage_hash,
        "hybrid_skeleton_lineage_hash": hybrid.skeleton_lineage_hash,
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "fresh_qualified_skeleton.json", hybrid.to_dict())
    write(args.out_dir / "RIG_DELTA_REPORT.json", delta_report)
    write(
        args.out_dir / "rig_REPORT.json",
        {
            "lane": "rig_counterfactual",
            "status": "DIAGNOSTIC_ARCHIVED_KINEMATICS_IDS_ON_V9_SUPPORTS",
            "product_authority_minted": False,
            "model_training_used": False,
            "teacher_inference_inputs_used": False,
            "joint_count": len(joints),
            "skeleton_lineage_hash": hybrid.skeleton_lineage_hash,
            "archived_skeleton_lineage_hash": archived.skeleton_lineage_hash,
            "fresh_v9_skeleton_lineage_hash": fresh.skeleton_lineage_hash,
        },
    )
    print(
        "V9_ARCHIVED_KINEMATICS_COUNTERFACTUAL="
        + json.dumps(
            {
                k: delta_report[k]
                for k in (
                    "joint_count",
                    "root_source_changed",
                    "parent_source_change_count",
                    "canonical_joint_id_change_count",
                    "position_delta_mean_over_span",
                    "position_delta_p95_over_span",
                    "position_delta_max_over_span",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
