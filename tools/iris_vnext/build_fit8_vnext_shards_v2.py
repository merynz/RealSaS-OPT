from __future__ import annotations

"""Active IRIS vNext Fit8 cohort builder.

V2 reuses the proven V1 source-ray/mechanical bank machinery. Topology
supervision has an additional training-only authority court: mechanically
equivalent authored seam duplicates are safely welded, remaining TESSA
quantization identity collisions receive a bounded deterministic identity lift,
and serialization still requires zero residual coordinate collisions. Raw
authored faces/owners/materials remain unchanged for appearance, relations and
mechanical-demand truth.

The eight source views of one subject are independent deterministic courts. They
are materialized concurrently to keep the sealed preparation practical without
changing samples, seeds, ordering, or any scientific target semantics.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.topology_teacher_v1 import canonicalize_vnext_topology_teacher_v1
from models.tessa.v1.training_data_v1 import (
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
)
from tools.iris_vnext.build_fit8_vnext_shards_v1 import (
    SUBJECT_SOURCE_IDS,
    _build_view_banks,
    _mechanical_bank,
    _normalize_source_world,
    _pad_topology,
    _sample_surface,
    _seed,
    _sha,
    _tensor,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-shard-dir", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--stage01-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--surface-points", type=int, default=4096)
    p.add_argument("--mechanical-points", type=int, default=8192)
    p.add_argument("--appearance-points-per-view", type=int, default=4096)
    p.add_argument("--relation-pairs-per-view", type=int, default=2048)
    p.add_argument("--topology-target-tokens", type=int, default=512)
    p.add_argument("--topology-trajectory-tolerance", type=float, default=1.0e-7)
    p.add_argument("--ray-chunk", type=int, default=64)
    p.add_argument("--view-workers", type=int, default=8)
    return p.parse_args()


def _topology_windows_v2(rest_iris, posed_iris, faces, target_tokens: int, trajectory_tolerance: float):
    cfg = IRISVNextConfigV1().tessa_decoder_config()
    teacher = canonicalize_vnext_topology_teacher_v1(
        rest_iris,
        posed_iris,
        faces,
        coordinate_bins=cfg.coordinate_bins,
        trajectory_tolerance=trajectory_tolerance,
    )
    collision = teacher.metadata["quantized_collision_census"]
    if int(collision["collision_group_count"]) != 0:
        raise ValueError(
            "VNEXT_TESSA_UNRESOLVED_POSITION_COLLISION:"
            + json.dumps(collision, sort_keys=True, separators=(",", ":"))
        )
    sequence = build_teacher_asset_sequence_v1(
        teacher.vertices_iris * 0.5, teacher.faces, cfg=cfg
    )
    windows = build_truncated_teacher_windows_v1(
        sequence, cfg=cfg, target_tokens=target_tokens
    )
    return sequence, windows, cfg, teacher.metadata


def _source_provenance_v2() -> dict:
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repo / "models/iris/vnext/contracts_v1.py",
        repo / "models/iris/vnext/topology_teacher_v1.py",
        repo / "models/tessa/v1/training_data_v1.py",
        repo / "tools/iris_vnext/build_fit8_vnext_shards_v1.py",
    )
    return {
        str(path.relative_to(repo)): _sha(path)
        for path in paths
    }


def _build_subject_view_banks(
    *,
    source_id: str,
    cameras: list[dict],
    stage01: Path,
    rest_source: np.ndarray,
    faces: np.ndarray,
    face_owner: np.ndarray,
    face_material: np.ndarray,
    owner_rows,
    center: np.ndarray,
    half: float,
    appearance_count: int,
    relation_count: int,
    ray_chunk: int,
    view_workers: int,
):
    """Build V0..V7 independently while preserving deterministic output order."""
    base = stage01 / "observations" / source_id

    def one(view_index: int):
        camera = cameras[view_index]
        return _build_view_banks(
            subject_id=source_id,
            view_index=view_index,
            camera=camera,
            png_path=base / f"V{view_index}.png",
            mask_path=base / f"V{view_index}.mask.bin",
            vertices_source=rest_source,
            faces=faces,
            face_owner=face_owner,
            face_material=face_material,
            owner_rows=owner_rows,
            center=center,
            half=half,
            appearance_count=appearance_count,
            relation_count=relation_count,
            ray_chunk=ray_chunk,
        )

    workers = min(int(view_workers), len(cameras))
    if workers == 1:
        return [one(view_index) for view_index in range(len(cameras))]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vnext-view") as pool:
        # executor.map preserves input order, so the serialized bank remains V0..V7.
        return list(pool.map(one, range(len(cameras))))


def main():
    args = _args()
    for count in (
        args.surface_points,
        args.mechanical_points,
        args.appearance_points_per_view,
        args.relation_pairs_per_view,
        args.topology_target_tokens,
        args.ray_chunk,
        args.view_workers,
    ):
        if int(count) <= 0:
            raise ValueError("VNEXT_SHARD_POLICY_NONPOSITIVE")
    if int(args.view_workers) > 8:
        raise ValueError("VNEXT_SHARD_VIEW_WORKERS_EXCEED_VIEW_COUNT")
    if not (0.0 <= args.topology_trajectory_tolerance <= 1.0e-3):
        raise ValueError("VNEXT_SHARD_TOPOLOGY_TRAJECTORY_TOLERANCE_INVALID")

    legacy_dir = args.legacy_shard_dir.resolve()
    teacher_dir = args.teacher_dir.resolve()
    stage01 = args.stage01_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bootstrap_path = stage01 / "BOOTSTRAP_MANIFEST.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {row["subject_id"]: row for row in bootstrap["subjects"]}
    subject_rows = []
    topology_cfg = None

    for display in SUBJECT_IDS:
        source_id = SUBJECT_SOURCE_IDS[display]
        legacy_path = legacy_dir / f"{source_id}.pt"
        teacher_path = teacher_dir / f"{source_id}.teacher.npz"
        teacher_receipt_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
        if not legacy_path.is_file() or not teacher_path.is_file() or not teacher_receipt_path.is_file():
            raise FileNotFoundError(f"VNEXT_SHARD_INPUT_MISSING:{display}")

        legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
        if str(legacy.get("subject_id")) != source_id:
            raise ValueError("VNEXT_SHARD_LEGACY_SUBJECT_DRIFT:" + display)
        teacher_receipt = json.loads(teacher_receipt_path.read_text(encoding="utf-8"))
        boot = bootstrap_rows[source_id]
        if teacher_receipt["source"]["sha256"] != boot["raw_source"]["sha256"]:
            raise ValueError("VNEXT_SHARD_TEACHER_SOURCE_HASH_DRIFT:" + display)

        with np.load(teacher_path, allow_pickle=False) as data:
            rest_source = np.asarray(data["rest_vertices_source_world"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int64)
            face_owner = np.asarray(data["face_owner"], dtype=np.int32)
            face_material = np.asarray(data["face_material"], dtype=np.int32)
            posed_source = np.asarray(data["posed_vertices_source_world"], dtype=np.float64)

        center = np.asarray(legacy["center_xyz"], dtype=np.float64).reshape(3)
        half = float(torch.as_tensor(legacy["normalization_half_extent"]).item())
        if not np.isfinite(half) or half <= 0:
            raise ValueError("VNEXT_SHARD_LEGACY_NORMALIZATION_INVALID")
        rest_iris = _normalize_source_world(rest_source, center, half)
        posed_iris = _normalize_source_world(posed_source, center, half)
        if float(np.abs(rest_iris).max()) > 1.00001:
            raise ValueError("VNEXT_SHARD_REST_OUTSIDE_IRIS_DOMAIN:" + display)

        rng = np.random.default_rng(_seed(display))
        surface_points, _, _ = _sample_surface(rest_iris, faces, args.surface_points, rng)
        mechanical = _mechanical_bank(
            rest_iris, posed_iris, faces, args.mechanical_points, rng
        )
        sequence, windows, cfg, topology_canonicalization = _topology_windows_v2(
            rest_iris,
            posed_iris,
            faces,
            args.topology_target_tokens,
            args.topology_trajectory_tolerance,
        )
        if topology_cfg is None:
            topology_cfg = cfg
        elif topology_cfg != cfg:
            raise AssertionError("VNEXT_SHARD_TOPOLOGY_CONFIG_DRIFT")

        camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
        camera_bundle = json.loads(camera_bundle_path.read_text(encoding="utf-8"))
        cameras = list(camera_bundle["cameras"])
        if len(cameras) != 8:
            raise ValueError("VNEXT_SHARD_CAMERA_COUNT_DRIFT")
        view_banks = _build_subject_view_banks(
            source_id=source_id,
            cameras=cameras,
            stage01=stage01,
            rest_source=rest_source,
            faces=faces,
            face_owner=face_owner,
            face_material=face_material,
            owner_rows=teacher_receipt["owners"],
            center=center,
            half=half,
            appearance_count=args.appearance_points_per_view,
            relation_count=args.relation_pairs_per_view,
            ray_chunk=args.ray_chunk,
            view_workers=args.view_workers,
        )

        row = {
            "subject_id": display,
            "source_subject_id": source_id,
            "teacher_inputs_in_inference_forward": False,
            "image_tokens": legacy["image_tokens"].squeeze(0).contiguous(),
            "token_xy": legacy["token_xy"].squeeze(0).contiguous(),
            "camera_features": legacy["camera_features"].squeeze(0).contiguous(),
            "geometry_points": legacy["replay_points"].contiguous(),
            "geometry_targets": legacy["replay_targets"].contiguous(),
            "teacher_surface_points_iris": _tensor(surface_points, torch.float32),
            **{
                key: _tensor(value, torch.float32)
                for key, value in mechanical.items()
                if key != "mechanical_truth_metadata"
            },
            "appearance_points": _tensor(
                np.stack([v["appearance_points"] for v in view_banks]), torch.float32
            ),
            "appearance_rgba": _tensor(
                np.stack([v["appearance_rgba"] for v in view_banks]), torch.float32
            ),
            "appearance_alpha_boundary": _tensor(
                np.stack([v["appearance_alpha_boundary"] for v in view_banks]), torch.float32
            ),
            "appearance_boundary": _tensor(
                np.stack([v["appearance_boundary"] for v in view_banks]), torch.float32
            ),
            "_windows": windows,
            "_window_count": len(windows),
            "metadata": {
                "schema": "RealSaS.IRISVNextFit8ShardMetadata.v2",
                "legacy_shard_sha256": _sha(legacy_path),
                "teacher_export_sha256": _sha(teacher_path),
                "teacher_export_receipt_sha256": _sha(teacher_receipt_path),
                "stage01_bootstrap_sha256": _sha(bootstrap_path),
                "source_asset_sha256": boot["raw_source"]["sha256"],
                "mechanical_truth": mechanical["mechanical_truth_metadata"],
                "topology_teacher": {
                    "face_count": sequence.face_count,
                    "vertex_count": sequence.vertex_count,
                    "component_count": sequence.component_count,
                    "restart_count": sequence.restart_count,
                    "token_count": len(sequence.token_ids),
                    "window_count": len(windows),
                    "coordinate_bins": int(cfg.coordinate_bins),
                    "canonicalization": topology_canonicalization,
                },
                "anonymous_drawable_ids_exposed_to_inference": False,
                "teacher_rig_skin_motion_exposed_to_inference": False,
            },
        }
        relation_keys = [key for key in view_banks[0] if key.startswith("relation_")]
        for key in relation_keys:
            row[key] = _tensor(
                np.stack([v[key] for v in view_banks]), torch.float32
            )
        subject_rows.append(row)

    if topology_cfg is None:
        raise RuntimeError("VNEXT_SHARD_TOPOLOGY_CONFIG_MISSING")
    max_windows, max_length = _pad_topology(subject_rows, topology_cfg)

    manifest_rows = []
    for row in subject_rows:
        display = row["subject_id"]
        metadata = row.pop("metadata")
        source_subject_id = row.pop("source_subject_id")
        path = output / f"{display.lower().replace('-', '').replace(' ', '')}.vnext.pt"
        torch.save(row, path)
        manifest_rows.append(
            {
                "subject_id": display,
                "source_subject_id": source_subject_id,
                "shard_file": path.name,
                "shard_sha256": _sha(path),
                "shard_bytes": path.stat().st_size,
                "metadata": metadata,
            }
        )

    manifest = {
        "schema": "RealSaS.IRISVNextFit8Manifest.v4",
        "builder": "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
        "source_provenance": _source_provenance_v2(),
        "subjects": manifest_rows,
        "subject_order": list(SUBJECT_IDS),
        "teacher_inputs_in_inference_forward": False,
        "legacy_optimizer_state_reused": False,
        "legacy_5400_checkpoint_reused": False,
        "frozen_legacy_source_evidence_reused": True,
        "topology_window_shape": [max_windows, max_length],
        "topology_pad_token": int(topology_cfg.PAD),
        "topology_coordinate_bins": int(topology_cfg.coordinate_bins),
        "topology_teacher_collision_policy": "MOTION_EQUIVALENT_EXACT_SEAM_WELD_THEN_BOUNDED_TRAINING_ONLY_QUANTIZED_IDENTITY_LIFT_THEN_ZERO_RESIDUAL_COLLISIONS_REQUIRED",
        "policy": {
            "surface_points": int(args.surface_points),
            "mechanical_points": int(args.mechanical_points),
            "appearance_points_per_view": int(args.appearance_points_per_view),
            "relation_pairs_per_view": int(args.relation_pairs_per_view),
            "topology_target_tokens": int(args.topology_target_tokens),
            "topology_trajectory_tolerance": float(args.topology_trajectory_tolerance),
            "ray_chunk": int(args.ray_chunk),
            "view_workers": int(args.view_workers),
        },
    }
    manifest["manifest_value_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "COHORT_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        "VNEXT_FIT8_SHARDS_V2_PASS",
        "subjects=8",
        f"topology_windows={max_windows}",
        f"topology_length={max_length}",
        f"coordinate_bins={topology_cfg.coordinate_bins}",
        f"view_workers={args.view_workers}",
        f"manifest={manifest['manifest_value_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
