from __future__ import annotations

"""Build the exact IRIS vNext Fit8 cohort shards.

Inputs are deliberately separated:
- legacy IRIS shards: frozen DINO/camera/field-source evidence only;
- authored Blender teacher exports: rest topology + native deformation sequence;
- Stage01 observations: source-wins RGBA/masks/cameras.

No teacher rig, skin, animation, anonymous drawable ID, or source mesh identity is
placed on the inference forward path.  They are consumed only to derive targets.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.teacher_truth_v1 import build_mechanical_demand_truth_v1
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.training_data_v1 import (
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
)

SUBJECT_SOURCE_IDS = {
    "Snake": "fit8_snake_v1",
    "Bat": "fit8_bat_v1",
    "T-Rex": "fit8_trex_v1",
    "Horse": "fit8_horse_v1",
    "Spider": "fit8_spider_v1",
    "GreenBlob": "fit8_greenblob_v1",
    "Shark": "fit8_shark_v1",
    "George": "fit8_george_v1",
}


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
    p.add_argument("--ray-chunk", type=int, default=64)
    return p.parse_args()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**63 - 1)


def _normalize_source_world(value: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    out = (np.asarray(value, dtype=np.float64) - center) / float(half)
    if not np.isfinite(out).all():
        raise ValueError("VNEXT_SHARD_NORMALIZATION_NONFINITE")
    return out


def _triangle_geometry(vertices: np.ndarray, faces: np.ndarray):
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area <= 1e-12):
        raise ValueError("VNEXT_SHARD_DEGENERATE_TEACHER_FACE")
    normals = cross / double_area[:, None]
    centroids = tri.mean(axis=1)
    return tri, 0.5 * double_area, normals, centroids


def _sample_surface(vertices, faces, count: int, rng):
    tri, area, _, _ = _triangle_geometry(vertices, faces)
    probability = area / area.sum()
    face_index = rng.choice(len(faces), size=int(count), replace=True, p=probability)
    r1 = rng.random(int(count))
    r2 = rng.random(int(count))
    s = np.sqrt(r1)
    bary = np.stack((1.0 - s, s * (1.0 - r2), s * r2), axis=1)
    points = np.einsum("ni,nij->nj", bary, tri[face_index])
    return points.astype(np.float32), face_index.astype(np.int64), bary


def _camera_rays(camera: dict, pixels_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resolution = int(camera["resolution"])
    half = float(camera["half_extent"])
    x = pixels_xy[:, 0].astype(np.float64)
    y = pixels_xy[:, 1].astype(np.float64)
    sx = ((x + 0.5) / resolution * 2.0 - 1.0) * half
    sy = (1.0 - (y + 0.5) / resolution * 2.0) * half
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    origins = origin[None] + sx[:, None] * right[None] + sy[:, None] * up[None]
    directions = np.broadcast_to(forward, origins.shape).copy()
    return origins, directions


def _ray_triangle_first_two(origins, directions, vertices, faces, *, chunk=64):
    tri = vertices[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    ray_count = len(origins)
    first_face = np.full(ray_count, -1, dtype=np.int64)
    second_face = np.full(ray_count, -1, dtype=np.int64)
    first_point = np.zeros((ray_count, 3), dtype=np.float64)
    second_point = np.zeros((ray_count, 3), dtype=np.float64)
    first_depth = np.full(ray_count, np.inf, dtype=np.float64)
    second_depth = np.full(ray_count, np.inf, dtype=np.float64)
    for start in range(0, ray_count, int(chunk)):
        end = min(ray_count, start + int(chunk))
        o = origins[start:end]
        d = directions[start:end]
        h = np.cross(d[:, None, :], e2[None, :, :])
        a = np.einsum("rfi,fi->rf", h, e1)
        valid = np.abs(a) > 1e-10
        inv = np.zeros_like(a)
        inv[valid] = 1.0 / a[valid]
        s = o[:, None, :] - v0[None, :, :]
        u = inv * np.einsum("rfi,rfi->rf", s, h)
        q = np.cross(s, e1[None, :, :])
        vv = inv * np.einsum("ri,rfi->rf", d, q)
        t = inv * np.einsum("fi,rfi->rf", e2, q)
        valid &= (u >= -1e-9) & (vv >= -1e-9) & ((u + vv) <= 1.0 + 1e-9) & (t > 1e-8)
        t = np.where(valid, t, np.inf)
        for local in range(end - start):
            finite = np.flatnonzero(np.isfinite(t[local]))
            if not len(finite):
                continue
            order = finite[np.argsort(t[local, finite], kind="stable")]
            f0 = int(order[0])
            idx = start + local
            first_face[idx] = f0
            first_depth[idx] = float(t[local, f0])
            first_point[idx] = o[local] + first_depth[idx] * d[local]
            if len(order) > 1:
                f1 = int(order[1])
                second_face[idx] = f1
                second_depth[idx] = float(t[local, f1])
                second_point[idx] = o[local] + second_depth[idx] * d[local]
    return {
        "first_face": first_face,
        "second_face": second_face,
        "first_point": first_point,
        "second_point": second_point,
        "first_depth": first_depth,
        "second_depth": second_depth,
    }


_BOUNDARY_CACHE_RGBA = None
_BOUNDARY_CACHE_ALPHA = None
_BOUNDARY_CACHE_APPEARANCE = None


def _boundary_maps_v2(rgba: np.ndarray):
    h, w, _ = rgba.shape
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    pa = np.pad(alpha, ((1, 1), (1, 1)), mode="edge")
    pr = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    amin = np.full((h, w), np.inf, dtype=np.float32)
    amax = np.full((h, w), -np.inf, dtype=np.float32)
    dmax = np.zeros((h, w), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            aa = pa[dy : dy + h, dx : dx + w]
            rr = pr[dy : dy + h, dx : dx + w]
            amin = np.minimum(amin, aa)
            amax = np.maximum(amax, aa)
            dist = np.linalg.norm(rr - rgb, axis=-1)
            dmax = np.maximum(dmax, dist.astype(np.float32, copy=False))
    return (amax - amin > 0.20).astype(np.float32), (dmax > 0.20).astype(np.float32)


def _boundary_flags(rgba: np.ndarray, xy: np.ndarray):
    global _BOUNDARY_CACHE_RGBA, _BOUNDARY_CACHE_ALPHA, _BOUNDARY_CACHE_APPEARANCE
    if _BOUNDARY_CACHE_RGBA is not rgba:
        alpha_map, appearance_map = _boundary_maps_v2(rgba)
        _BOUNDARY_CACHE_RGBA = rgba
        _BOUNDARY_CACHE_ALPHA = alpha_map
        _BOUNDARY_CACHE_APPEARANCE = appearance_map
    q = np.asarray(xy, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("VNEXT_BOUNDARY_XY_SHAPE_INVALID")
    if len(q) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    x, y = q[:, 0], q[:, 1]
    h, w, _ = rgba.shape
    if np.any(x < 0) or np.any(x >= w) or np.any(y < 0) or np.any(y >= h):
        raise IndexError("VNEXT_BOUNDARY_XY_OUT_OF_RANGE")
    return _BOUNDARY_CACHE_ALPHA[y, x].copy(), _BOUNDARY_CACHE_APPEARANCE[y, x].copy()

def _adjacency(faces: np.ndarray) -> tuple[tuple[int, ...], ...]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(tuple(sorted((int(u), int(v)))), []).append(fi)
    out = [set() for _ in range(len(faces))]
    for rows in edge_faces.values():
        for a in rows:
            out[a].update(b for b in rows if b != a)
    return tuple(tuple(sorted(row)) for row in out)


def _owner_attachment_pairs(owner_rows: Iterable[dict]) -> set[tuple[int, int]]:
    rows = tuple(owner_rows)
    name_to_index = {str(row["source_import_name"]): int(row["anonymous_owner_index"]) for row in rows}
    pairs = set()
    for row in rows:
        child = int(row["anonymous_owner_index"])
        parent_name = row.get("parent_object_name")
        if parent_name in name_to_index:
            parent = name_to_index[parent_name]
            if parent != child:
                pairs.add(tuple(sorted((child, parent))))
    return pairs


def _build_view_banks(
    *,
    subject_id: str,
    view_index: int,
    camera: dict,
    png_path: Path,
    mask_path: Path,
    vertices_source: np.ndarray,
    faces: np.ndarray,
    face_owner: np.ndarray,
    face_material: np.ndarray,
    owner_rows: Iterable[dict],
    center: np.ndarray,
    half: float,
    appearance_count: int,
    relation_count: int,
    ray_chunk: int,
):
    rgba = np.asarray(Image.open(png_path).convert("RGBA"), dtype=np.uint8)
    resolution = int(camera["resolution"])
    if rgba.shape != (resolution, resolution, 4):
        raise ValueError("VNEXT_SHARD_SOURCE_RGBA_DIMENSION_DRIFT")
    mask = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8).reshape(resolution, resolution)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("VNEXT_SHARD_SOURCE_MASK_EMPTY")
    rng = np.random.default_rng(_seed(f"{subject_id}:view:{view_index}"))
    ray_count = max(int(appearance_count), int(relation_count)) * 3
    choice = rng.choice(len(xs), size=ray_count, replace=len(xs) < ray_count)
    pixels = np.stack((xs[choice], ys[choice]), axis=1).astype(np.int64)
    origins, directions = _camera_rays(camera, pixels)
    hits = _ray_triangle_first_two(
        origins, directions, vertices_source, faces, chunk=ray_chunk
    )
    valid = np.flatnonzero(hits["first_face"] >= 0)
    if len(valid) < appearance_count:
        raise ValueError(
            f"VNEXT_SHARD_VISIBLE_RAY_COVERAGE_INSUFFICIENT:view={view_index}:"
            f"{len(valid)}<{appearance_count}"
        )
    app_pick = valid[:appearance_count]
    app_xy = pixels[app_pick]
    app_points = _normalize_source_world(hits["first_point"][app_pick], center, half)
    app_rgba = rgba[app_xy[:, 1], app_xy[:, 0]].astype(np.float32) / 255.0
    app_alpha_boundary, app_boundary = _boundary_flags(rgba, app_xy)

    adjacency = _adjacency(faces)
    _, _, _, centroids = _triangle_geometry(vertices_source, faces)
    attachment_owner_pairs = _owner_attachment_pairs(owner_rows)
    visible_faces = set(map(int, hits["first_face"][valid].tolist()))

    relation_rows = []
    for ray_index in valid:
        a = int(hits["first_face"][ray_index])
        b = int(hits["second_face"][ray_index])
        if b < 0 or a == b:
            continue
        point_a = hits["first_point"][ray_index]
        point_b = hits["second_point"][ray_index]
        owner_a, owner_b = int(face_owner[a]), int(face_owner[b])
        same = float(owner_a == owner_b)
        material_boundary = float(
            owner_a != owner_b or int(face_material[a]) != int(face_material[b])
        )
        pair_owner = tuple(sorted((owner_a, owner_b)))
        alpha_b, appearance_b = _boundary_flags(rgba, pixels[ray_index : ray_index + 1])
        relation_rows.append(
            (
                point_a,
                point_b,
                same,
                material_boundary,
                float(pair_owner in attachment_owner_pairs),
                float(alpha_b[0]),
                max(float(appearance_b[0]), material_boundary),
                1.0,
                1.0,
                1.0,
                0.0,
            )
        )

    # Fallback/non-occlusion rows preserve static drawable/cut evidence even for
    # thin subjects that have few two-layer rays.
    for a in sorted(visible_faces):
        if len(relation_rows) >= relation_count:
            break
        candidates = adjacency[a]
        if not candidates:
            continue
        b = int(candidates[0])
        owner_a, owner_b = int(face_owner[a]), int(face_owner[b])
        same = float(owner_a == owner_b)
        boundary = float(owner_a != owner_b or int(face_material[a]) != int(face_material[b]))
        relation_rows.append(
            (
                centroids[a],
                centroids[b],
                same,
                boundary,
                float(tuple(sorted((owner_a, owner_b))) in attachment_owner_pairs),
                0.0,
                boundary,
                0.5,
                0.0,
                1.0,
                float(b in visible_faces),
            )
        )
    if not relation_rows:
        raise ValueError("VNEXT_SHARD_RELATION_TEACHER_EMPTY")
    relation_index = rng.choice(
        len(relation_rows), size=relation_count, replace=len(relation_rows) < relation_count
    )
    selected = [relation_rows[int(i)] for i in relation_index]
    point_a = _normalize_source_world(np.stack([row[0] for row in selected]), center, half)
    point_b = _normalize_source_world(np.stack([row[1] for row in selected]), center, half)
    labels = np.asarray([row[2:] for row in selected], dtype=np.float32)
    return {
        "appearance_points": app_points.astype(np.float32),
        "appearance_rgba": app_rgba.astype(np.float32),
        "appearance_alpha_boundary": app_alpha_boundary,
        "appearance_boundary": app_boundary,
        "relation_point_a": point_a.astype(np.float32),
        "relation_point_b": point_b.astype(np.float32),
        "relation_same_drawable_affinity": labels[:, 0],
        "relation_cut_probability": labels[:, 1],
        "relation_attachment_affinity": labels[:, 2],
        "relation_alpha_boundary_probability": labels[:, 3],
        "relation_appearance_boundary_probability": labels[:, 4],
        "relation_front_probability": labels[:, 5],
        "relation_occlusion_probability": labels[:, 6],
        "relation_visibility_probability_a": labels[:, 7],
        "relation_visibility_probability_b": labels[:, 8],
    }


def _mechanical_bank(rest_iris, posed_iris, faces, count, rng):
    truth = build_mechanical_demand_truth_v1(rest_iris, faces, posed_iris)
    _, area, normals, centroids = _triangle_geometry(rest_iris, faces)
    probability = area / area.sum()
    face_index = rng.choice(len(faces), size=int(count), replace=True, p=probability)
    def face_value(name, dtype=np.float32):
        return np.asarray(getattr(truth, name), dtype=dtype)[face_index]
    return {
        "mechanical_points": centroids[face_index].astype(np.float32),
        "mechanical_normals": normals[face_index].astype(np.float32),
        "mechanical_stretch_major": face_value("stretch_major"),
        "mechanical_stretch_minor": face_value("stretch_minor"),
        "mechanical_tangent_direction_2theta": face_value("tangent_direction_2theta"),
        "mechanical_bend_demand": face_value("bend_demand"),
        "mechanical_rigidity_probability": face_value("rigidity_probability"),
        "mechanical_truth_metadata": truth.metadata,
    }


def _topology_windows(rest_iris, faces, target_tokens: int):
    cfg = TESSAConfigV1(d_model=512, n_heads=8, surface_layers=0, decoder_layers=8, surface_latent_count=256)
    sequence = build_teacher_asset_sequence_v1(rest_iris * 0.5, faces, cfg=cfg)
    windows = build_truncated_teacher_windows_v1(sequence, cfg=cfg, target_tokens=target_tokens)
    return sequence, windows, cfg


def _pad_topology(subject_rows: list[dict], cfg: TESSAConfigV1):
    max_windows = max(len(row["_windows"]) for row in subject_rows)
    max_length = max(len(window.input_ids) for row in subject_rows for window in row["_windows"])
    for row in subject_rows:
        inputs = torch.full((max_windows, max_length), cfg.PAD, dtype=torch.long)
        labels = torch.full((max_windows, max_length), cfg.PAD, dtype=torch.long)
        offsets = torch.zeros(max_windows, dtype=torch.long)
        for wi, window in enumerate(row.pop("_windows")):
            n = len(window.input_ids)
            inputs[wi, :n] = torch.tensor(window.input_ids, dtype=torch.long)
            labels[wi, :n] = torch.tensor(window.labels, dtype=torch.long)
            offsets[wi] = int(window.sequence_position_offset)
        row["topology_input_ids"] = inputs
        row["topology_labels"] = labels
        row["topology_window_offsets"] = offsets
        row["topology_window_count"] = torch.tensor(int(row.pop("_window_count")), dtype=torch.long)
    return max_windows, max_length


def _tensor(value, dtype=None):
    return torch.as_tensor(np.asarray(value), dtype=dtype).contiguous()


def main():
    args = _args()
    for count in (
        args.surface_points,
        args.mechanical_points,
        args.appearance_points_per_view,
        args.relation_pairs_per_view,
        args.topology_target_tokens,
        args.ray_chunk,
    ):
        if int(count) <= 0:
            raise ValueError("VNEXT_SHARD_POLICY_NONPOSITIVE")
    legacy_dir = args.legacy_shard_dir.resolve()
    teacher_dir = args.teacher_dir.resolve()
    stage01 = args.stage01_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bootstrap = json.loads((stage01 / "BOOTSTRAP_MANIFEST.json").read_text())
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
        teacher_receipt = json.loads(teacher_receipt_path.read_text())
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
        sequence, windows, cfg = _topology_windows(
            rest_iris, faces, args.topology_target_tokens
        )
        topology_cfg = topology_cfg or cfg

        camera_bundle_path = stage01 / boot["source_scene_evidence"]["camera_bundle"]["path"]
        camera_bundle = json.loads(camera_bundle_path.read_text())
        if len(camera_bundle["cameras"]) != 8:
            raise ValueError("VNEXT_SHARD_CAMERA_COUNT_DRIFT")
        view_banks = []
        for view_index, camera in enumerate(camera_bundle["cameras"]):
            base = stage01 / "observations" / source_id
            view_banks.append(
                _build_view_banks(
                    subject_id=source_id,
                    view_index=view_index,
                    camera=camera,
                    png_path=base / f"V{view_index}.png",
                    mask_path=base / f"V{view_index}.mask.bin",
                    vertices_source=rest_source,
                    faces=faces,
                    face_owner=face_owner,
                    face_material=face_material,
                    owner_rows=teacher_receipt["owners"],
                    center=center,
                    half=half,
                    appearance_count=args.appearance_points_per_view,
                    relation_count=args.relation_pairs_per_view,
                    ray_chunk=args.ray_chunk,
                )
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
            "appearance_points": _tensor(np.stack([v["appearance_points"] for v in view_banks]), torch.float32),
            "appearance_rgba": _tensor(np.stack([v["appearance_rgba"] for v in view_banks]), torch.float32),
            "appearance_alpha_boundary": _tensor(np.stack([v["appearance_alpha_boundary"] for v in view_banks]), torch.float32),
            "appearance_boundary": _tensor(np.stack([v["appearance_boundary"] for v in view_banks]), torch.float32),
            "_windows": windows,
            "_window_count": len(windows),
            "metadata": {
                "schema": "RealSaS.IRISVNextFit8ShardMetadata.v1",
                "legacy_shard_sha256": _sha(legacy_path),
                "teacher_export_sha256": _sha(teacher_path),
                "teacher_export_receipt_sha256": _sha(teacher_receipt_path),
                "stage01_bootstrap_sha256": _sha(stage01 / "BOOTSTRAP_MANIFEST.json"),
                "source_asset_sha256": boot["raw_source"]["sha256"],
                "mechanical_truth": mechanical["mechanical_truth_metadata"],
                "topology_teacher": {
                    "face_count": sequence.face_count,
                    "vertex_count": sequence.vertex_count,
                    "component_count": sequence.component_count,
                    "restart_count": sequence.restart_count,
                    "token_count": len(sequence.token_ids),
                    "window_count": len(windows),
                },
                "anonymous_drawable_ids_exposed_to_inference": False,
                "teacher_rig_skin_motion_exposed_to_inference": False,
            },
        }
        relation_keys = [key for key in view_banks[0] if key.startswith("relation_")]
        for key in relation_keys:
            row[key] = _tensor(np.stack([v[key] for v in view_banks]), torch.float32)
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
        "schema": "RealSaS.IRISVNextFit8Manifest.v3",
        "subjects": manifest_rows,
        "subject_order": list(SUBJECT_IDS),
        "teacher_inputs_in_inference_forward": False,
        "legacy_optimizer_state_reused": False,
        "legacy_5400_checkpoint_reused": False,
        "frozen_legacy_source_evidence_reused": True,
        "topology_window_shape": [max_windows, max_length],
        "topology_pad_token": int(topology_cfg.PAD),
        "policy": {
            "surface_points": int(args.surface_points),
            "mechanical_points": int(args.mechanical_points),
            "appearance_points_per_view": int(args.appearance_points_per_view),
            "relation_pairs_per_view": int(args.relation_pairs_per_view),
            "topology_target_tokens": int(args.topology_target_tokens),
            "ray_chunk": int(args.ray_chunk),
        },
    }
    manifest["manifest_value_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "COHORT_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        "VNEXT_FIT8_SHARDS_PASS",
        "subjects=8",
        f"topology_windows={max_windows}",
        f"topology_length={max_length}",
        f"manifest={manifest['manifest_value_sha256']}",
    )


if __name__ == "__main__":
    main()
