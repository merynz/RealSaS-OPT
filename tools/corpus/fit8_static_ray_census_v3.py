from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.corpus.fit8_static_ray_census_v1 import StaticRayCensusPolicyV1
from tools.corpus.fit8_static_ray_census_v2 import build_static_ray_census_v2

SCHEMA_VERSION = "RealSaS.FIT8StaticRayCensus.v3"
PARENT_SCHEMA_VERSION = "RealSaS.FIT8StaticRayCensus.v2"


def augment_static_ray_order_graph_v3(
    summary_v2: dict,
    arrays: dict[str, np.ndarray],
) -> dict:
    if summary_v2.get("schema") != PARENT_SCHEMA_VERSION:
        raise ValueError("FIT8_STATIC_RAY_V3_PARENT_SCHEMA_INVALID")
    pair_evidence = dict(summary_v2.get("pair_evidence") or {})
    if sorted(pair_evidence) != [str(i) for i in range(8)]:
        raise ValueError("FIT8_STATIC_RAY_V3_PAIR_EVIDENCE_VIEW_SET_INVALID")

    for view_index in range(8):
        key = f"v{view_index}_ordered_component_pair_counts"
        if key not in arrays:
            raise ValueError("FIT8_STATIC_RAY_V3_ORDERED_PAIR_ARRAY_MISSING")
        rows = np.asarray(arrays[key], dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1:] != (3,):
            raise ValueError("FIT8_STATIC_RAY_V3_ORDERED_PAIR_ARRAY_SHAPE_INVALID")
        if len(rows) and (
            np.any(rows[:, 0] < 0)
            or np.any(rows[:, 1] < 0)
            or np.any(rows[:, 0] == rows[:, 1])
            or np.any(rows[:, 2] <= 0)
        ):
            raise ValueError("FIT8_STATIC_RAY_V3_ORDERED_PAIR_ARRAY_VALUE_INVALID")
        canonical = sorted((int(a), int(b), int(count)) for a, b, count in rows.tolist())
        if len({(a, b) for a, b, _ in canonical}) != len(canonical):
            raise ValueError("FIT8_STATIC_RAY_V3_ORDERED_PAIR_DUPLICATE")
        row = dict(pair_evidence[str(view_index)])
        row["ordered_pair_count"] = len(canonical)
        row["ordered_pairs"] = [list(item) for item in canonical]
        pair_evidence[str(view_index)] = row

    summary = dict(summary_v2)
    summary["schema"] = SCHEMA_VERSION
    summary["parent_static_ray_schema"] = PARENT_SCHEMA_VERSION
    summary["pair_evidence"] = pair_evidence
    claims = dict(summary.get("claims") or {})
    claims["complete_ordered_component_relation_graph_exposed"] = True
    summary["claims"] = claims
    return summary


def build_static_ray_census_v3(
    *,
    subject_key: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_material_id: np.ndarray | None = None,
    face_owner_id: np.ndarray | None = None,
    face_component_id: np.ndarray | None = None,
    policy: StaticRayCensusPolicyV1,
    owner_authority_state: str = "UNSPECIFIED",
) -> tuple[dict, dict[str, np.ndarray]]:
    summary_v2, arrays = build_static_ray_census_v2(
        subject_key=subject_key,
        vertices=vertices,
        faces=faces,
        face_material_id=face_material_id,
        face_owner_id=face_owner_id,
        face_component_id=face_component_id,
        policy=policy,
        owner_authority_state=owner_authority_state,
    )
    return augment_static_ray_order_graph_v3(summary_v2, arrays), arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-npz", required=True, type=Path)
    parser.add_argument("--subject-key", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frame-scale", type=float, default=1.12)
    parser.add_argument(
        "--owner-authority-state",
        choices=("DIAGNOSTIC", "QUALIFIED", "UNSPECIFIED"),
        default="UNSPECIFIED",
    )
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.carrier_npz, allow_pickle=False) as archive:
        vertices = archive["vertices"]
        faces = archive["faces"]
        material = archive["face_material_id"] if "face_material_id" in archive.files else None
        owner = archive["face_owner_id"] if "face_owner_id" in archive.files else None
        component = archive["face_component_id"] if "face_component_id" in archive.files else None
    summary, arrays = build_static_ray_census_v3(
        subject_key=args.subject_key,
        vertices=vertices,
        faces=faces,
        face_material_id=material,
        face_owner_id=owner,
        face_component_id=component,
        policy=StaticRayCensusPolicyV1(
            resolution=args.resolution,
            frame_scale=args.frame_scale,
            provenance="CLI_STATIC_RAY_CENSUS_V3",
        ),
        owner_authority_state=args.owner_authority_state,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


__all__ = [
    "PARENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "augment_static_ray_order_graph_v3",
    "build_static_ray_census_v3",
]

if __name__ == "__main__":
    main()
