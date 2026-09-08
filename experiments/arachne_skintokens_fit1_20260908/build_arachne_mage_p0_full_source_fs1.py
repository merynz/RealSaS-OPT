#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)

SCHEMA = "RealSaS.ArachneMageSkinTargetP0FullSource.v1"
PROJECTION_CONTRACT = "RealSaS.ArachneMageP0FullSourceHybridProjection.v1"

EXPECTED_NORMALIZED_SHA = "528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f"
EXPECTED_ZERO_SHA = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
EXPECTED_BODY_TEACHER_SHA = "ac2c37571b77fa2900cc042e36f74df111432b94fb21acfb54694d5b6f962e67"
EXPECTED_BRIDGE_SHA = "df8805a261326c3f58fc10d8f3f7a6354bb7241a0ba1fc3e272d89a2bc2a06e7"
EXPECTED_QUALIFIED_SKELETON_SHA = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"

EXPECTED_SURFACE_LINEAGE = "67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"

SOURCE_RUN_ID = "20260904T220929Z"
IRIS_CHECKPOINT_SHA = "766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2"
CAMERA_SHA = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)

CANDIDATE_COUNT = 8
DISTANCE_LOW_CONFIDENCE = 0.05
LOCAL_DISTANCE_DELTA = 0.005
LOCAL_SKIN_L1 = 0.2
VOTE_SKIN_L1 = 0.15
MULTIVIEW_VOTES = 2
SKIN_SUPPORT_EPS = 1e-8
RAY_BARY_TOL = 1e-9

MODE_VOCAB = {
    "MULTIVIEW_DISAMBIGUATED": 0,
    "MULTIVIEW_VALIDATED_NEAREST": 1,
    "NEAREST3D_LOW_CONF_FALLBACK": 2,
    "NEAREST3D_STABLE": 3,
    "SINGLE_VIEW_LOCAL_TIEBREAK": 4,
    "SINGLE_VIEW_VALIDATED_NEAREST": 5,
}
CONFIDENCE_VOCAB = {"HIGH": 0, "MEDIUM": 1, "MEDIUM_LOW": 2, "LOW": 3}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _npy_bytes(a: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.lib.format.write_array(bio, np.asarray(a), allow_pickle=False)
    return bio.getvalue()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Byte-deterministic NPZ: sorted members, fixed ZIP timestamp/metadata."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            zf.writestr(info, _npy_bytes(np.asarray(arrays[key])), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return out.getvalue()


def project(points: np.ndarray, camera: dict):
    origin = np.asarray(camera["origin"], np.float64)
    right = np.asarray(camera["right"], np.float64)
    up = np.asarray(camera["screen_up"], np.float64)
    forward = np.asarray(camera["forward"], np.float64)
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    forward /= np.linalg.norm(forward)
    half = float(camera["half_extent"])
    resolution = int(camera["resolution"])
    d = np.asarray(points, np.float64) - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    depth = d @ forward
    raster = np.stack(
        [
            (gx + 1.0) * 0.5 * resolution - 0.5,
            (gy + 1.0) * 0.5 * resolution - 0.5,
        ],
        axis=-1,
    )
    return raster, depth, resolution


def closest_points_point_to_triangles(P, A, B, C):
    """Vectorized Ericson point-to-triangle closest point with barycentric weights."""
    AB = B - A
    AC = C - A
    AP = P - A
    d1 = np.sum(AB * AP, axis=-1)
    d2 = np.sum(AC * AP, axis=-1)
    m, n = d1.shape
    Q = np.empty((m, n, 3), dtype=np.float64)
    bary = np.empty((m, n, 3), dtype=np.float64)
    assigned = np.zeros((m, n), bool)

    mask = (d1 <= 0) & (d2 <= 0)
    Q[mask] = np.broadcast_to(A, (m, n, 3))[mask]
    bary[mask] = (1, 0, 0)
    assigned |= mask

    BP = P - B
    d3 = np.sum(AB * BP, axis=-1)
    d4 = np.sum(AC * BP, axis=-1)
    mask = (d3 >= 0) & (d4 <= d3) & ~assigned
    Q[mask] = np.broadcast_to(B, (m, n, 3))[mask]
    bary[mask] = (0, 1, 0)
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~assigned
    v = np.zeros_like(d1)
    denom = d1 - d3
    v[mask] = d1[mask] / denom[mask]
    q = A + v[..., None] * AB
    Q[mask] = q[mask]
    bary[mask] = np.stack([1 - v, v, np.zeros_like(v)], -1)[mask]
    assigned |= mask

    CP = P - C
    d5 = np.sum(AB * CP, axis=-1)
    d6 = np.sum(AC * CP, axis=-1)
    mask = (d6 >= 0) & (d5 <= d6) & ~assigned
    Q[mask] = np.broadcast_to(C, (m, n, 3))[mask]
    bary[mask] = (0, 0, 1)
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~assigned
    w = np.zeros_like(d1)
    denom = d2 - d6
    w[mask] = d2[mask] / denom[mask]
    q = A + w[..., None] * AC
    Q[mask] = q[mask]
    bary[mask] = np.stack([1 - w, np.zeros_like(w), w], -1)[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~assigned
    w2 = np.zeros_like(d1)
    denom = (d4 - d3) + (d5 - d6)
    w2[mask] = (d4[mask] - d3[mask]) / denom[mask]
    q = B + w2[..., None] * (C - B)
    Q[mask] = q[mask]
    bary[mask] = np.stack([np.zeros_like(w2), 1 - w2, w2], -1)[mask]
    assigned |= mask

    mask = ~assigned
    denom = va + vb + vc
    vface = np.zeros_like(d1)
    wface = np.zeros_like(d1)
    vface[mask] = vb[mask] / denom[mask]
    wface[mask] = vc[mask] / denom[mask]
    q = A + AB * vface[..., None] + AC * wface[..., None]
    Q[mask] = q[mask]
    bary[mask] = np.stack([1 - vface - wface, vface, wface], -1)[mask]

    d2out = np.sum((P - Q) ** 2, axis=-1)
    return Q, d2out, bary


def topk_closest(points, vertices, faces, face_ids, k=CANDIDATE_COUNT, pchunk=32):
    """Exact top-k; stable argsort means ties resolve by ascending supplied face order/id."""
    tri = np.asarray(vertices, np.float64)[np.asarray(faces, np.int64)]
    A, B, C = tri[:, 0][None], tri[:, 1][None], tri[:, 2][None]
    all_idx, all_dist, all_bary = [], [], []
    for start in range(0, len(points), pchunk):
        P = np.asarray(points[start : start + pchunk], np.float64)[:, None, :]
        _, d2, bary = closest_points_point_to_triangles(P, A, B, C)
        order = np.argsort(d2, axis=1, kind="stable")[:, :k]
        dd = np.take_along_axis(d2, order, axis=1)
        bb = np.take_along_axis(bary, order[..., None], axis=1)
        all_idx.append(np.asarray(face_ids, np.int64)[order])
        all_dist.append(np.sqrt(dd))
        all_bary.append(bb)
    return np.vstack(all_idx), np.vstack(all_dist), np.vstack(all_bary)


def interpolate_skin(face_ids, bary, faces, skin22):
    tri_skin = np.asarray(skin22, np.float64)[np.asarray(faces, np.int64)[face_ids]]
    w = np.einsum("nkj,nkjc->nkc", np.asarray(bary, np.float64), tri_skin)
    mass = w.sum(axis=2)
    if np.any(mass <= SKIN_SUPPORT_EPS):
        raise RuntimeError("ZERO_SELECTED_SKIN_MASS_IN_ELIGIBLE_CANDIDATE")
    return w / mass[..., None]


def load_cameras(camera_dir: Path):
    cameras = []
    for i, expected in enumerate(CAMERA_SHA):
        p = camera_dir / f"V{i}.camera.json"
        if sha256_file(p) != expected:
            raise RuntimeError(f"CAMERA_SHA_DRIFT_V{i}")
        cameras.append(json.loads(p.read_text(encoding="utf-8")))
    return tuple(cameras)


def reconstruct_shipping_surface(zero_path: Path, cameras):
    if sha256_file(zero_path) != EXPECTED_ZERO_SHA:
        raise RuntimeError("ZERO_SURFACE_SHA_DRIFT")
    with np.load(zero_path, allow_pickle=False) as z:
        world = np.asarray(z["vertices"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
        hints = np.asarray(z["normals"], np.float64)
    center = np.asarray(cameras[0]["center"], np.float64)
    half = float(cameras[0]["half_extent"])
    surface = rigging_surface_from_scene_first_zero_mesh_v1(
        (world - center[None, :]) / half,
        faces,
        hints,
        cameras,
        normalization_center=center,
        normalization_half_extent=half,
        authority_label="IRIS_SCENE_FIRST_SIGNED_V3_PROMOTED_MAGE_FIT",
        source_run_id=SOURCE_RUN_ID,
        source_checkpoint_sha256=IRIS_CHECKPOINT_SHA,
        source_zero_surface_sha256=EXPECTED_ZERO_SHA,
        target_nodes=1024,
        normal_k=64,
        visibility_depth_tolerance_norm=0.02,
        metadata={
            "camera_contract": "CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG",
            "teacher_truth_used": False,
            "geppetto_reference_strength_fit1": True,
        },
    )
    if surface.geometry_lineage_hash != EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("SURFACE_LINEAGE_DRIFT")
    if len(surface.surface_nodes) != 950 or len(surface.local_relations) != 2813:
        raise RuntimeError("SURFACE_SHAPE_DRIFT")
    return surface


def load_bridge(path: Path):
    if sha256_file(path) != EXPECTED_BRIDGE_SHA:
        raise RuntimeError("JOINT_IDENTITY_BRIDGE_SHA_DRIFT")
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["bridge"]
    if len(rows) != 22:
        raise RuntimeError("BRIDGE_COUNT_DRIFT")
    mapping = {str(x["canonical_joint_id"]): int(x["teacher_source_control_index"]) for x in rows}
    joint_ids = tuple(sorted(mapping))
    source_cols = np.asarray([mapping[j] for j in joint_ids], np.int64)
    if len(set(source_cols.tolist())) != 22:
        raise RuntimeError("BRIDGE_SOURCE_CONTROL_DUPLICATE")
    return d, joint_ids, source_cols


def load_skeleton(path: Path, joint_ids):
    if sha256_file(path) != EXPECTED_QUALIFIED_SKELETON_SHA:
        raise RuntimeError("QUALIFIED_SKELETON_FILE_SHA_DRIFT")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("skeleton_lineage_hash") != EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("SKELETON_LINEAGE_DRIFT")
    got = tuple(sorted(str(x["canonical_joint_id"]) for x in d["joints"]))
    if got != tuple(joint_ids):
        raise RuntimeError("SKELETON_JOINT_ID_SET_DRIFT")
    return d


def source_skin_authority(normalized_path: Path, source_cols):
    if sha256_file(normalized_path) != EXPECTED_NORMALIZED_SHA:
        raise RuntimeError("NORMALIZED_SOURCE_SHA_DRIFT")
    with np.load(normalized_path, allow_pickle=False) as z:
        vertices = np.asarray(z["vertices_source"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
        skin41 = np.asarray(z["skin"], np.float64)
    if vertices.shape != (5321, 3) or faces.shape != (5763, 3) or skin41.shape != (5321, 41):
        raise RuntimeError("FULL_SOURCE_SHAPE_DRIFT")

    all_cols = np.arange(41, dtype=np.int64)
    excluded_cols = np.asarray(sorted(set(all_cols.tolist()) - set(source_cols.tolist())), np.int64)
    excluded_mass = float(np.abs(skin41[:, excluded_cols]).sum())
    if excluded_mass > 1e-12:
        raise RuntimeError(f"NON_BRIDGE_SOURCE_SKIN_MASS:{excluded_mass}")

    skin22 = skin41[:, source_cols]
    selected_mass = skin22.sum(axis=1)
    positive_vertex = selected_mass > SKIN_SUPPORT_EPS
    zero_vertex_count = int((~positive_vertex).sum())
    if zero_vertex_count != 42:
        raise RuntimeError(f"ZERO_SKIN_VERTEX_COUNT_DRIFT:{zero_vertex_count}")

    eligible = np.all(positive_vertex[faces], axis=1)
    eligible_count = int(eligible.sum())
    excluded_face_count = int((~eligible).sum())
    if eligible_count != 5683 or excluded_face_count != 80:
        raise RuntimeError(f"SKIN_SUPPORTED_FACE_COUNT_DRIFT:{eligible_count}:{excluded_face_count}")

    if np.max(np.abs(selected_mass[positive_vertex] - 1.0)) > 2e-6:
        raise RuntimeError("SOURCE_SELECTED_SKIN_SIMPLEX_DRIFT")
    skin22n = np.zeros_like(skin22)
    skin22n[positive_vertex] = skin22[positive_vertex] / selected_mass[positive_vertex, None]

    face_ids = np.flatnonzero(eligible).astype(np.int64)
    return {
        "vertices": vertices,
        "faces": faces,
        "skin22": skin22n,
        "eligible_face_ids": face_ids,
        "eligible_faces": faces[face_ids],
        "zero_vertex_count": zero_vertex_count,
        "excluded_face_count": excluded_face_count,
        "excluded_source_control_mass_l1": excluded_mass,
    }


def historical_body_foundation_regression(body_path: Path, points, source):
    if sha256_file(body_path) != EXPECTED_BODY_TEACHER_SHA:
        raise RuntimeError("HISTORICAL_BODY_TEACHER_SHA_DRIFT")
    with np.load(body_path, allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], np.float64)
        faces = np.asarray(z["faces"], np.int64)
    if vertices.shape != (3348, 3) or faces.shape != (4029, 3):
        raise RuntimeError("HISTORICAL_BODY_SHAPE_DRIFT")

    tree = cKDTree(source["vertices"])
    d, idx = tree.query(vertices, k=1)
    if float(np.max(d)) > 6.2e-8:
        raise RuntimeError("BODY_TO_FULL_SOURCE_VERTEX_MAP_DRIFT")
    body_skin22 = source["skin22"][np.asarray(idx, np.int64)]
    face_ids = np.arange(len(faces), dtype=np.int64)
    cidx, cdist, cbary = topk_closest(points, vertices, faces, face_ids)
    cw = interpolate_skin(cidx, cbary, faces, body_skin22)

    far = cdist[:, 0] > DISTANCE_LOW_CONFIDENCE
    ambiguity = np.zeros(len(points), bool)
    for i in range(len(points)):
        for k in range(1, CANDIDATE_COUNT):
            if (
                cdist[i, k] <= cdist[i, 0] + LOCAL_DISTANCE_DELTA
                and float(np.abs(cw[i, k] - cw[i, 0]).sum()) > LOCAL_SKIN_L1
            ):
                ambiguity[i] = True
                break
    flagged = far | ambiguity
    stats = {
        "nearest_mean": float(cdist[:, 0].mean()),
        "nearest_p50": float(np.quantile(cdist[:, 0], 0.50)),
        "nearest_p90": float(np.quantile(cdist[:, 0], 0.90)),
        "nearest_p95": float(np.quantile(cdist[:, 0], 0.95)),
        "nearest_p99": float(np.quantile(cdist[:, 0], 0.99)),
        "nearest_max": float(cdist[:, 0].max()),
        "distance_low_confidence_rows": int(far.sum()),
        "local_semantic_ambiguity_rows": int(ambiguity.sum()),
        "flagged_union_rows": int(flagged.sum()),
        "stable_rows": int((~flagged).sum()),
        "body_to_full_source_vertex_error_max": float(np.max(d)),
    }
    expected = {
        "nearest_mean": 0.009367747137483609,
        "nearest_p95": 0.03069111655845138,
        "nearest_p99": 0.06860541512023109,
        "nearest_max": 0.09476343516885569,
    }
    for key, val in expected.items():
        if abs(stats[key] - val) > 2e-12:
            raise RuntimeError(f"HISTORICAL_BODY_DISTANCE_REGRESSION_DRIFT:{key}:{stats[key]}:{val}")
    if stats["distance_low_confidence_rows"] != 16:
        raise RuntimeError("HISTORICAL_FAR_COUNT_DRIFT")
    if stats["local_semantic_ambiguity_rows"] != 24:
        raise RuntimeError("HISTORICAL_AMBIGUITY_COUNT_DRIFT")
    if stats["stable_rows"] != 915:
        raise RuntimeError("HISTORICAL_STABLE_COUNT_DRIFT")
    return stats


def prepare_teacher_projections(vertices, faces, eligible_face_ids, cameras):
    out = []
    ef = faces[eligible_face_ids]
    for camera in cameras:
        raster, depth, _ = project(vertices, camera)
        out.append((raster, depth, ef, eligible_face_ids))
    return tuple(out)


def frontmost_teacher_sample(raster_xy, view, projected, faces, skin22):
    """Continuous orthographic z-buffer: exact screen ray, no pixel quantization."""
    raster, depth, ef, eligible_face_ids = projected[view]
    tri2 = raster[ef]
    p = np.asarray(raster_xy, np.float64)
    a, b, c = tri2[:, 0], tri2[:, 1], tri2[:, 2]
    v0, v1, v2 = b - a, c - a, p[None, :] - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    good = np.abs(den) > 1e-12
    u = np.full(len(den), np.nan, np.float64)
    v = np.full(len(den), np.nan, np.float64)
    u[good] = (v2[good, 0] * v1[good, 1] - v1[good, 0] * v2[good, 1]) / den[good]
    v[good] = (v0[good, 0] * v2[good, 1] - v2[good, 0] * v0[good, 1]) / den[good]
    w = 1.0 - u - v
    inside = good & (u >= -RAY_BARY_TOL) & (v >= -RAY_BARY_TOL) & (w >= -RAY_BARY_TOL)
    local = np.flatnonzero(inside)
    if len(local) == 0:
        return None
    bary = np.stack([w[local], u[local], v[local]], axis=1)
    z = (bary * depth[ef[local]]).sum(axis=1)
    pos = z > 0.0
    if not np.any(pos):
        return None
    local = local[pos]
    bary = bary[pos]
    z = z[pos]
    global_ids = eligible_face_ids[local]
    order = np.lexsort((global_ids, z))
    j = int(order[0])
    face_id = int(global_ids[j])
    bb = bary[j]
    ww = bb @ skin22[faces[face_id]]
    mass = float(ww.sum())
    if mass <= SKIN_SUPPORT_EPS:
        raise RuntimeError("FRONTMOST_ZERO_SKIN_FACE")
    ww = ww / mass
    return face_id, float(z[j]), ww, bb


def hybrid_projection(surface, source, cameras):
    nodes = tuple(surface.surface_nodes)
    points = np.asarray([n.P for n in nodes], np.float64)
    face_ids = source["eligible_face_ids"]
    cidx, cdist, cbary = topk_closest(
        points, source["vertices"], source["eligible_faces"], face_ids
    )
    cw = interpolate_skin(cidx, cbary, source["faces"], source["skin22"])

    far = cdist[:, 0] > DISTANCE_LOW_CONFIDENCE
    ambiguity = np.zeros(len(nodes), bool)
    for i in range(len(nodes)):
        for k in range(1, CANDIDATE_COUNT):
            if (
                cdist[i, k] <= cdist[i, 0] + LOCAL_DISTANCE_DELTA
                and float(np.abs(cw[i, k] - cw[i, 0]).sum()) > LOCAL_SKIN_L1
            ):
                ambiguity[i] = True
                break
    flagged = far | ambiguity

    final = cw[:, 0].copy()
    chosen_rank = np.zeros(len(nodes), np.uint8)
    chosen_face = cidx[:, 0].copy()
    modes = np.asarray(["NEAREST3D_STABLE"] * len(nodes), dtype=object)
    confidence = np.asarray(["HIGH"] * len(nodes), dtype=object)
    support_count = np.asarray([len(n.support_views) for n in nodes], np.uint8)
    ray_sample_count = np.zeros(len(nodes), np.uint8)
    winning_vote_count = np.zeros(len(nodes), np.uint8)

    projected = prepare_teacher_projections(
        source["vertices"], source["faces"], source["eligible_face_ids"], cameras
    )

    for i in np.flatnonzero(flagged):
        node = nodes[int(i)]
        if not node.support_views:
            modes[i] = "NEAREST3D_LOW_CONF_FALLBACK"
            confidence[i] = "LOW"
            continue

        bind = {int(v): np.asarray(xy, np.float64) for v, xy in node.raster_bindings}
        counts = np.zeros(CANDIDATE_COUNT, np.int64)
        samples = 0
        for view in node.support_views:
            view = int(view)
            if view not in bind:
                raise RuntimeError("SUPPORT_VIEW_WITHOUT_RASTER_BINDING")
            sample = frontmost_teacher_sample(
                bind[view], view, projected, source["faces"], source["skin22"]
            )
            if sample is None:
                continue
            samples += 1
            sample_w = sample[2]
            l1 = np.abs(cw[i] - sample_w[None, :]).sum(axis=1)
            counts += (l1 <= VOTE_SKIN_L1).astype(np.int64)

        ray_sample_count[i] = np.uint8(samples)
        mx = int(counts.max(initial=0))
        winning_vote_count[i] = np.uint8(mx)

        if mx >= MULTIVIEW_VOTES:
            winners = np.flatnonzero(counts == mx)
            k = int(winners[0])
            chosen_rank[i] = np.uint8(k)
            chosen_face[i] = cidx[i, k]
            final[i] = cw[i, k]
            if k == 0:
                modes[i] = "MULTIVIEW_VALIDATED_NEAREST"
                confidence[i] = "HIGH"
            else:
                modes[i] = "MULTIVIEW_DISAMBIGUATED"
                confidence[i] = "MEDIUM"
            continue

        if mx == 1:
            winners = np.flatnonzero(
                (counts == 1) & (cdist[i] <= cdist[i, 0] + LOCAL_DISTANCE_DELTA)
            )
            if len(winners):
                k = int(winners[0])
                chosen_rank[i] = np.uint8(k)
                chosen_face[i] = cidx[i, k]
                final[i] = cw[i, k]
                if k == 0:
                    modes[i] = "SINGLE_VIEW_VALIDATED_NEAREST"
                    confidence[i] = "MEDIUM_LOW"
                else:
                    modes[i] = "SINGLE_VIEW_LOCAL_TIEBREAK"
                    confidence[i] = "MEDIUM_LOW"
                continue

        modes[i] = "NEAREST3D_LOW_CONF_FALLBACK"
        confidence[i] = "LOW"

    if not np.isfinite(final).all():
        raise RuntimeError("NONFINITE_FINAL_WEIGHT")
    if np.any(final < -1e-12):
        raise RuntimeError("NEGATIVE_FINAL_WEIGHT")
    simplex = np.abs(final.sum(axis=1) - 1.0)
    if float(simplex.max()) > 1e-9:
        raise RuntimeError(f"FINAL_SIMPLEX_DRIFT:{simplex.max()}")

    return {
        "weights": final,
        "mode": modes,
        "confidence": confidence,
        "nearest_distance": cdist[:, 0],
        "far": far,
        "ambiguity": ambiguity,
        "flagged": flagged,
        "chosen_rank": chosen_rank,
        "chosen_face": chosen_face,
        "support_count": support_count,
        "ray_sample_count": ray_sample_count,
        "winning_vote_count": winning_vote_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", type=Path, required=True)
    ap.add_argument("--zero-surface", type=Path, required=True)
    ap.add_argument("--camera-dir", type=Path, required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--qualified-skeleton", type=Path, required=True)
    ap.add_argument("--historical-body-teacher", type=Path, required=True)
    ap.add_argument("--output-npz", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    args = ap.parse_args()

    cameras = load_cameras(args.camera_dir)
    surface = reconstruct_shipping_surface(args.zero_surface, cameras)
    bridge, joint_ids, source_cols = load_bridge(args.bridge)
    skeleton = load_skeleton(args.qualified_skeleton, joint_ids)
    source = source_skin_authority(args.normalized, source_cols)

    points = np.asarray([n.P for n in surface.surface_nodes], np.float64)
    historical = historical_body_foundation_regression(
        args.historical_body_teacher, points, source
    )
    result = hybrid_projection(surface, source, cameras)

    order = np.argsort(
        np.asarray([n.surface_id for n in surface.surface_nodes], dtype=str),
        kind="stable",
    )
    surface_ids = np.asarray(
        [surface.surface_nodes[int(i)].surface_id for i in order],
        dtype=f"<U{max(len(n.surface_id) for n in surface.surface_nodes)}",
    )
    joint_ids_arr = np.asarray(joint_ids, dtype=f"<U{max(map(len, joint_ids))}")
    weights = np.asarray(result["weights"][order], np.float32)
    mode_code = np.asarray([MODE_VOCAB[x] for x in result["mode"][order]], np.uint8)
    confidence_code = np.asarray(
        [CONFIDENCE_VOCAB[x] for x in result["confidence"][order]], np.uint8
    )

    arrays = {
        "surface_ids": surface_ids,
        "joint_ids": joint_ids_arr,
        "weights": weights,
        "mode_code": mode_code,
        "confidence_code": confidence_code,
        "nearest_distance_world": np.asarray(result["nearest_distance"][order], np.float32),
        "distance_low_confidence": np.asarray(result["far"][order], np.uint8),
        "local_semantic_ambiguity": np.asarray(result["ambiguity"][order], np.uint8),
        "selected_candidate_rank": np.asarray(result["chosen_rank"][order], np.uint8),
        "selected_teacher_face_id": np.asarray(result["chosen_face"][order], np.int32),
        "support_view_count": np.asarray(result["support_count"][order], np.uint8),
        "teacher_ray_sample_count": np.asarray(result["ray_sample_count"][order], np.uint8),
        "winning_vote_count": np.asarray(result["winning_vote_count"][order], np.uint8),
    }
    npz_data = deterministic_npz_bytes(arrays)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.write_bytes(npz_data)

    weight_content_sha = sha256_bytes(weights.astype("<f4", copy=False).tobytes(order="C"))
    binding_sha = sha256_bytes(
        json_bytes(
            {
                "schema": "RealSaS.ArachneMageSkinTargetBinding.v1",
                "surface_geometry_lineage_hash": surface.geometry_lineage_hash,
                "skeleton_lineage_hash": skeleton["skeleton_lineage_hash"],
                "surface_ids": surface_ids.tolist(),
                "joint_ids": joint_ids_arr.tolist(),
                "teacher_weights_content_sha256": weight_content_sha,
            }
        )
    )

    mode_counts = dict(sorted(Counter(result["mode"].tolist()).items()))
    conf_counts = dict(sorted(Counter(result["confidence"].tolist()).items()))
    manifest = {
        "schema": SCHEMA,
        "status": "P0_FULL_SOURCE_FS1_PROJECTED__A0_STILL_BLOCKED_UNTIL_CACHE_RUNNER_REGRESSIONS",
        "scientific_scope": "TEACHER_ONLY_TARGET_PROJECTION__NO_LEARNED_OPTIMIZER",
        "inputs": {
            "normalized_source_sha256": EXPECTED_NORMALIZED_SHA,
            "zero_surface_sha256": EXPECTED_ZERO_SHA,
            "camera_sha256": list(CAMERA_SHA),
            "joint_identity_bridge_sha256": EXPECTED_BRIDGE_SHA,
            "qualified_skeleton_file_sha256": EXPECTED_QUALIFIED_SKELETON_SHA,
            "historical_body_teacher_sha256": EXPECTED_BODY_TEACHER_SHA,
        },
        "binding": {
            "surface_geometry_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton["skeleton_lineage_hash"],
            "surface_node_count": 950,
            "joint_count": 22,
            "weights_shape": list(weights.shape),
            "weights_dtype": str(weights.dtype),
            "teacher_weights_content_sha256": weight_content_sha,
            "target_binding_sha256": binding_sha,
            "target_npz_sha256": sha256_bytes(npz_data),
        },
        "source_deformation_surface": {
            "full_normalized_vertices": 5321,
            "full_normalized_faces": 5763,
            "zero_selected_skin_vertices": source["zero_vertex_count"],
            "excluded_zero_skin_faces": source["excluded_face_count"],
            "eligible_skin_supported_faces": int(len(source["eligible_face_ids"])),
            "excluded_non_bridge_source_control_mass_l1": source[
                "excluded_source_control_mass_l1"
            ],
            "eligibility_rule": "ALL_3_FACE_VERTICES_SELECTED_22_CONTROL_MASS_GT_1E-8",
        },
        "projection_policy": {
            "contract": PROJECTION_CONTRACT,
            "base": "EXACT_CLOSEST_POINT_ON_SKIN_SUPPORTED_FULL_SOURCE_FACES__BARYCENTRIC_22_CONTROL_SKIN",
            "candidate_count": CANDIDATE_COUNT,
            "distance_low_confidence_threshold_world": DISTANCE_LOW_CONFIDENCE,
            "local_ambiguity_distance_delta_world": LOCAL_DISTANCE_DELTA,
            "local_ambiguity_skin_l1_threshold": LOCAL_SKIN_L1,
            "teacher_view_sample": "CONTINUOUS_ORTHOGRAPHIC_FRONTMOST_RAY_ON_EXACT_PRODUCT_CAMERA__NO_PIXEL_QUANTIZATION",
            "teacher_view_candidate_vote_l1_threshold": VOTE_SKIN_L1,
            "multiview_override_requires_votes": MULTIVIEW_VOTES,
            "single_view_override_requires_candidate_within_nearest_plus_world": LOCAL_DISTANCE_DELTA,
            "support_views": "EXACT_SHIPPING_GSA_SUPPORT_VIEWS_AND_RASTER_BINDINGS",
            "tie_breaks": "NEAREST_CANDIDATE_RANK_THEN_ASCENDING_GLOBAL_FACE_ID",
            "teacher_source_geometry_is_product_inference_dependency": False,
        },
        "historical_body_foundation_regression": historical,
        "result_counts": {
            "observed_rows": int(sum(bool(n.support_views) for n in surface.surface_nodes)),
            "completed_rows": int(sum(not bool(n.support_views) for n in surface.surface_nodes)),
            "distance_low_confidence_rows": int(result["far"].sum()),
            "local_semantic_ambiguity_rows": int(result["ambiguity"].sum()),
            "flagged_union_rows": int(result["flagged"].sum()),
            "nearest_candidate_changed_rows": int((result["chosen_rank"] != 0).sum()),
            "mode": mode_counts,
            "confidence": conf_counts,
        },
        "validation": {
            "finite": bool(np.isfinite(weights).all()),
            "negative_weight_count": int((weights < 0).sum()),
            "simplex_max_abs_error_float32": float(
                np.max(np.abs(weights.astype(np.float64).sum(axis=1) - 1.0))
            ),
            "surface_ids_canonical_sorted": bool(
                surface_ids.tolist() == sorted(surface_ids.tolist())
            ),
            "joint_ids_canonical_sorted": bool(
                joint_ids_arr.tolist() == sorted(joint_ids_arr.tolist())
            ),
        },
        "authorization": {
            "a0_optimizer_authorized": False,
            "a1_optimizer_authorized": False,
            "next_gate": "REBUILD_CONDITIONING_CACHE__REBIND_A0_PREREG_RUNNER__RERUN_CPU_RESUME_TERMINAL_IDEMPOTENCE",
        },
    }
    args.output_manifest.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
