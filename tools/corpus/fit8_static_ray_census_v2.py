from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np

from tools.corpus.fit8_static_ray_census_v1 import (
    StaticRayCensusPolicyV1,
    build_static_ray_census_v1,
)

SCHEMA_VERSION = "RealSaS.FIT8StaticRayCensus.v2"
PARENT_SCHEMA_VERSION = "RealSaS.FIT8StaticRayCensus.v1"
OWNER_AUTHORITY_STATES = {"UNSPECIFIED", "DIAGNOSTIC", "QUALIFIED"}


def build_static_ray_census_v2(
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
    """Authority-safe wrapper around the tested v1 static visibility producer.

    V1 inferred owner authority from the presence of any nonnegative owner label.
    That is unsafe for FIT8 because some carrier producers intentionally emit
    diagnostic dominant-skin-joint labels. V2 preserves the exact v1 geometry,
    camera, raster and all-hit implementation while requiring the authority state
    to be explicit and never promoting label presence into qualification.
    """
    if owner_authority_state not in OWNER_AUTHORITY_STATES:
        raise ValueError("FIT8_STATIC_RAY_V2_OWNER_AUTHORITY_STATE_INVALID")

    summary, arrays = build_static_ray_census_v1(
        subject_key=subject_key,
        vertices=vertices,
        faces=faces,
        face_material_id=face_material_id,
        face_owner_id=face_owner_id,
        face_component_id=face_component_id,
        policy=policy,
    )
    owner = np.asarray(arrays["face_owner_id"], dtype=np.int32)
    labels_present = bool(np.any(owner >= 0))
    if owner_authority_state in {"DIAGNOSTIC", "QUALIFIED"} and not labels_present:
        raise ValueError("FIT8_STATIC_RAY_V2_OWNER_AUTHORITY_WITHOUT_LABELS")

    claims = dict(summary.get("claims") or {})
    claims.update(
        {
            "mechanical_owner_labels_present": labels_present,
            "mechanical_owner_authority_state": owner_authority_state,
            "mechanical_owner_authority": bool(
                labels_present and owner_authority_state == "QUALIFIED"
            ),
            "mechanical_owner_diagnostic_only": bool(
                labels_present and owner_authority_state == "DIAGNOSTIC"
            ),
            "owner_authority_inferred_from_label_presence": False,
        }
    )
    summary = dict(summary)
    summary["schema"] = SCHEMA_VERSION
    summary["parent_static_ray_schema"] = PARENT_SCHEMA_VERSION
    summary["claims"] = claims
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-npz", required=True, type=Path)
    parser.add_argument("--subject-key", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frame-scale", type=float, default=1.12)
    parser.add_argument(
        "--owner-authority-state",
        choices=sorted(OWNER_AUTHORITY_STATES),
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
    summary, arrays = build_static_ray_census_v2(
        subject_key=args.subject_key,
        vertices=vertices,
        faces=faces,
        face_material_id=material,
        face_owner_id=owner,
        face_component_id=component,
        policy=StaticRayCensusPolicyV1(
            resolution=args.resolution,
            frame_scale=args.frame_scale,
            provenance="CLI_STATIC_RAY_CENSUS_V2",
        ),
        owner_authority_state=args.owner_authority_state,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


__all__ = [
    "OWNER_AUTHORITY_STATES",
    "PARENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_static_ray_census_v2",
]

if __name__ == "__main__":
    main()
