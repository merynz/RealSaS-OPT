#!/usr/bin/env python3
from __future__ import annotations

"""Compile frozen semantic masks into Knight M relation diagnostics.

This is a research diagnostic only.  It reuses the sealed single-M Knight
baseline and recorded M poses; no IRIS geometry inference, refit, retarget,
reskinning or geometry mutation is executed.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.semantic_relation_evidence_v1 import (
    detect_pose_order_conflicts_v1,
    infer_behavior_evidence_v1,
    infer_grip_candidates_v1,
    infer_source_overlap_relations_v1,
    lift_mask_evidence_to_m_v1,
)
from models.iris.semantic_evidence_v1 import load_mask_evidence_v1
from tools.demo.render_knight_m_caa_v1 import load_inputs


def _relation(row):
    return {
        "relation_kind": row.relation_kind,
        "a_owner_id": row.a_owner_id,
        "b_owner_id": row.b_owner_id,
        "confidence": row.confidence,
        "support_count": row.support_count,
        "view_index": row.view_index,
        "a_face_index": row.a_face_index,
        "b_face_index": row.b_face_index,
        "normalized_distance": row.normalized_distance,
        "metadata": row.metadata or {},
    }


def _behavior(row):
    return {
        "owner_id": row.owner_id,
        "semantic_label": row.semantic_label,
        "behavior_class": row.behavior_class,
        "confidence": row.confidence,
        "metadata": row.metadata or {},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json",
    )
    p.add_argument("--semantic-evidence", type=Path, required=True)
    p.add_argument("--semantic-sha256", default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--face-confidence", type=float, default=0.45)
    p.add_argument("--grip-distance", type=float, default=0.08)
    p.add_argument("--diagnostic-resolution", type=int, default=256)
    args = p.parse_args()

    manifest, arrays, _, cameras, _ = load_inputs(args.baseline)
    rest_before = arrays["rest"].copy()
    poses_before = arrays["poses"].copy()
    evidence = load_mask_evidence_v1(
        args.semantic_evidence, expected_sha256=args.semantic_sha256
    )
    mesh = lift_mask_evidence_to_m_v1(
        rest=arrays["rest"],
        faces=arrays["faces"],
        cameras=cameras,
        evidence=evidence,
        minimum_face_confidence=args.face_confidence,
    )
    behavior = infer_behavior_evidence_v1(mesh)
    grip = infer_grip_candidates_v1(
        rest=arrays["rest"],
        faces=arrays["faces"],
        mesh=mesh,
        maximum_normalized_distance=args.grip_distance,
    )
    overlap = infer_source_overlap_relations_v1(
        rest=arrays["rest"],
        faces=arrays["faces"],
        cameras=cameras,
        evidence=evidence,
        mesh=mesh,
        resolution=args.diagnostic_resolution,
    )
    conflicts = []
    for frame_index, posed in enumerate(arrays["poses"]):
        frame_rows = detect_pose_order_conflicts_v1(
            posed=posed,
            faces=arrays["faces"],
            cameras=cameras,
            mesh=mesh,
            source_relations=overlap,
            resolution=args.diagnostic_resolution,
        )
        conflicts.extend({"frame_index": frame_index, **_relation(row)} for row in frame_rows)

    owner_rows = []
    for oi, owner_id in enumerate(mesh.owner_ids):
        fis = np.asarray(mesh.owner_face_indices[oi], dtype=int)
        owner_rows.append(
            {
                "owner_id": owner_id,
                "label": mesh.label_vocab[int(mesh.owner_label_index[oi])],
                "face_count": int(len(fis)),
                "mean_confidence": float(np.mean(mesh.face_confidence[fis])) if len(fis) else 0.0,
                "support_views": [
                    int(v)
                    for v in range(8)
                    if bool(np.any(mesh.face_support_views[fis, v]))
                ],
            }
        )

    report = {
        "schema": "RealSaS.KnightFrozenSemanticRelationDiagnostic.v1",
        "baseline_schema": manifest.get("schema"),
        "semantic_evidence_hash": evidence.evidence_hash(),
        "semantic_provider_id": evidence.provider_id,
        "canonical_geometry": "M_UNCHANGED",
        "geometry_model_loaded": False,
        "inference_geometry_executed": False,
        "refit_executed": False,
        "reskin_executed": False,
        "retarget_executed": False,
        "rest_geometry_byte_equal_after": bool(np.array_equal(arrays["rest"], rest_before)),
        "recorded_pose_byte_equal_after": bool(np.array_equal(arrays["poses"], poses_before)),
        "face_count": int(len(arrays["faces"])),
        "qualified_semantic_face_count": int(np.count_nonzero(mesh.face_label_index >= 0)),
        "unknown_semantic_face_count": int(np.count_nonzero(mesh.face_label_index < 0)),
        "owners": owner_rows,
        "behavior_evidence": [_behavior(x) for x in behavior],
        "grip_candidates": [_relation(x) for x in grip],
        "source_overlap_relations": [_relation(x) for x in overlap],
        "pose_order_conflicts": conflicts,
        "claims": {
            "semantic_masks_are_truth": False,
            "grip_is_qualified": False,
            "order_is_qualified": False,
            "secondary_motion_is_authorized": False,
            "runtime_behavior_changed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "owners": len(owner_rows),
                "qualified_faces": report["qualified_semantic_face_count"],
                "grip_candidates": len(grip),
                "source_overlap_relations": len(overlap),
                "pose_order_conflicts": len(conflicts),
                "geometry_unchanged": report["rest_geometry_byte_equal_after"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
