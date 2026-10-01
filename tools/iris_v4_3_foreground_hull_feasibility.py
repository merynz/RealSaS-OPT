from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v3.dense_source_sampling_v3 import admitted_pixel_mask_for_camera
from models.iris.v4.source_constraint_v4 import (
    build_source_constraint_view_state_v4,
    build_ray_points_v4,
)
from models.iris.v4.source_constraint_v4_1 import sample_hard_positive_refresh_candidates_v41
from models.iris.v4.source_exterior_v4 import source_foreground_distance_fields_v4
from models.iris.v4.source_hull_lattice_v4_3 import (
    SourceHullLatticePolicyV43,
    source_hull_exterior_margin_v43,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_evidence(root: Path) -> tuple[np.ndarray, list[dict], np.ndarray, float]:
    norm = json.loads(
        (root / "08_NORMALIZATION_DOMAIN_QUALIFIED" / "normalization_domain.json").read_text()
    )
    camera_set = json.loads(
        (root / "05_CAMERA_CONTRACT_SOLVED" / "qualified_camera_set.json").read_text()
    )
    center = np.asarray(norm["center_xyz"], dtype=np.float64)
    half = float(norm["half_extent"])
    if not np.isfinite(center).all() or not np.isfinite(half) or half <= 0.0:
        raise ValueError("invalid normalization domain")

    cameras = []
    for camera in sorted(camera_set["cameras"], key=lambda row: int(row["view_index"])):
        if int(camera["resolution"]) != 1024:
            raise ValueError("V4.3 fixed-audit feasibility requires 1024 source cameras")
        cameras.append(
            {
                "view_index": int(camera["view_index"]),
                "origin": (np.asarray(camera["origin"], dtype=np.float64) - center) / half,
                "right": np.asarray(camera["right"], dtype=np.float64),
                "up": np.asarray(camera["screen_up"], dtype=np.float64),
                "forward": np.asarray(camera["forward"], dtype=np.float64),
                "half_extent": float(camera["half_extent"]) / half,
            }
        )
    if [row["view_index"] for row in cameras] != list(range(8)):
        raise ValueError("exact eight ordered views are required")

    masks = []
    for view in range(8):
        raw = np.fromfile(root / "observations" / f"V{view}.mask.bin", dtype=np.uint8)
        if raw.size != 1024 * 1024 or np.any((raw != 0) & (raw != 1)):
            raise ValueError(f"invalid source mask V{view}")
        masks.append(raw.reshape(1024, 1024).astype(bool))
    return np.stack(masks), cameras, center, half


def run_audit(
    *,
    evidence_zip: Path,
    fit_seed: int = 26091909,
    audit_training_step: int = 777777,
    rays_per_view: int = 256,
    query_chunk: int = 250000,
) -> dict:
    coverage = DenseSourceCoveragePolicyV3()
    coverage.validate()
    hull_policy = SourceHullLatticePolicyV43()
    hull_policy.validate()
    if int(rays_per_view) <= 0 or int(query_chunk) <= 0:
        raise ValueError("rays_per_view/query_chunk must be positive")

    with tempfile.TemporaryDirectory(prefix="realsas-v43-feas-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(evidence_zip, "r") as zf:
            zf.extractall(root)
        masks, cameras, _center, _half = _load_evidence(root)

        states = []
        for view, camera in enumerate(cameras):
            admitted = admitted_pixel_mask_for_camera(
                width=1024,
                height=1024,
                camera_origin_normalized=camera["origin"],
                camera_right=camera["right"],
                camera_screen_up=camera["up"],
                camera_forward=camera["forward"],
                camera_half_extent_normalized=camera["half_extent"],
            )
            states.append(
                build_source_constraint_view_state_v4(
                    masks[view], admitted, near_boundary_max_px=32.0
                )
            )

        distance_fields = torch.from_numpy(source_foreground_distance_fields_v4(masks))
        origins = torch.from_numpy(np.stack([row["origin"] for row in cameras]).astype(np.float32))
        right = torch.from_numpy(np.stack([row["right"] for row in cameras]).astype(np.float32))
        up = torch.from_numpy(np.stack([row["up"] for row in cameras]).astype(np.float32))
        forward = torch.from_numpy(np.stack([row["forward"] for row in cameras]).astype(np.float32))
        half_extent = torch.from_numpy(
            np.asarray([row["half_extent"] for row in cameras], dtype=np.float32)
        )

        rows = []
        total_feasible = 0
        total_infeasible = 0
        minimum_free = None
        for view, (state, camera) in enumerate(zip(states, cameras)):
            indices = sample_hard_positive_refresh_candidates_v41(
                state,
                fit_seed=int(fit_seed) + 404,
                refresh_index=0,
                view_index=view,
                count=int(rays_per_view),
            )
            points = build_ray_points_v4(
                indices,
                width=1024,
                height=1024,
                camera_origin_normalized=camera["origin"],
                camera_right=camera["right"],
                camera_screen_up=camera["up"],
                camera_forward=camera["forward"],
                camera_half_extent_normalized=camera["half_extent"],
                fit_seed=int(fit_seed),
                training_step=int(audit_training_step),
                view_index=view,
                coverage_policy=coverage,
            )
            ray_count, depth_count = map(int, points.shape[:2])
            certified_flat = []
            for start in range(0, ray_count * depth_count, int(query_chunk)):
                stop = min(ray_count * depth_count, start + int(query_chunk))
                row = source_hull_exterior_margin_v43(
                    points.reshape(-1, 3)[start:stop],
                    distance_fields_px=distance_fields,
                    camera_origins_normalized=origins,
                    camera_right=right,
                    camera_screen_up=up,
                    camera_forward=forward,
                    camera_half_extent_normalized=half_extent,
                    pixel_quantization_guard_px=float(hull_policy.pixel_quantization_guard_px),
                )
                certified_flat.append(row["certified"].cpu())
            certified = torch.cat(certified_flat).numpy().reshape(ray_count, depth_count)
            free_count = np.count_nonzero(~certified, axis=1)
            feasible = free_count > 0
            boundary = state.boundary.reshape(-1)[indices]

            view_feasible = int(np.count_nonzero(feasible))
            view_infeasible = int(ray_count - view_feasible)
            total_feasible += view_feasible
            total_infeasible += view_infeasible
            view_min = int(np.min(free_count))
            minimum_free = view_min if minimum_free is None else min(minimum_free, view_min)
            rows.append(
                {
                    "view": view,
                    "rays": ray_count,
                    "depth_samples": depth_count,
                    "source_hull_feasible_count": view_feasible,
                    "source_hull_infeasible_count": view_infeasible,
                    "min_free_samples_per_ray": view_min,
                    "median_free_samples_per_ray": float(np.median(free_count)),
                    "max_free_samples_per_ray": int(np.max(free_count)),
                    "boundary_rays": int(np.count_nonzero(boundary)),
                    "interior_rays": int(np.count_nonzero(~boundary)),
                }
            )

    total = total_feasible + total_infeasible
    return {
        "schema": "RealSaS.IRIS.V43ForegroundHullConstraintFeasibilityReproduction.v1",
        "status": (
            "SOURCE_ONLY_FIXED_AUDIT_FOREGROUND_TARGET_FEASIBLE__NOT_CAPACITY_PROOF"
            if total_infeasible == 0
            else "SOURCE_ONLY_FIXED_AUDIT_FOREGROUND_TARGET_CONFLICT_FOUND"
        ),
        "evidence_zip_sha256": _sha256(evidence_zip),
        "fit_seed": int(fit_seed),
        "audit_training_step": int(audit_training_step),
        "rays_per_view": int(rays_per_view),
        "total_audit_rays": int(total),
        "feasible_ray_count": int(total_feasible),
        "infeasible_ray_count": int(total_infeasible),
        "feasible_fraction": float(total_feasible / max(total, 1)),
        "minimum_uncertified_samples_on_any_ray": int(minimum_free or 0),
        "per_view": rows,
        "teacher_signed_sdf_used": False,
        "truth_mesh_signed_sdf_used": False,
        "claim_boundary": (
            "This tests source-mask/source-hull consistency on the exact fixed V4.3 audit "
            "foreground rays. It is not a TP64 capacity proof or product authority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-seed", type=int, default=26091909)
    parser.add_argument("--audit-training-step", type=int, default=777777)
    parser.add_argument("--rays-per-view", type=int, default=256)
    parser.add_argument("--query-chunk", type=int, default=250000)
    args = parser.parse_args()
    result = run_audit(
        evidence_zip=args.evidence_zip,
        fit_seed=args.fit_seed,
        audit_training_step=args.audit_training_step,
        rays_per_view=args.rays_per_view,
        query_chunk=args.query_chunk,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
