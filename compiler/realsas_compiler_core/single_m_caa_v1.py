from __future__ import annotations

"""Canonical single-M source-wins appearance transport.

The approved Knight baseline uses the recorded M poses directly. Completion
is compiled beforehand by CAA; playback only samples sealed source images and
M-face/barycentric completion. No independent visual topology or rebind exists.
The baseline's sRGB composition is explicit; linear-light shipping parity,
semantic overlap ownership, grip and depth-tie closure remain separate gates.
"""
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image

from .appearance_bake_v2 import _sample_index_map, _nearest_triangle_lattice
from .appearance_compile_v2 import triangular_barycentric_samples
from .appearance_color_v2 import (
    straight_srgb_rgba_u8_to_premultiplied_linear,
    premultiplied_linear_to_straight_srgb_u8,
)
from .camera_geometry_v2 import qualify_camera_v3, project_points_xyz_v3
from .types import QualificationError

MATERIAL_CONTRACT = "CANONICAL_M_SOURCE_WINS_PLUS_SEALED_CAA_V1"
POLICY = {"depth_tie_epsilon": 1e-9, "maximum_fragment_layers": 32}


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def prepare_caa_field(arrays, rest, faces):
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    offsets = np.asarray(arrays["face_sample_offsets"], dtype=np.int64)
    resolution = np.asarray(arrays["face_tile_resolutions"], dtype=np.int32)
    if (
        resolution.shape != (len(faces),)
        or np.any(resolution < 4)
        or offsets.shape != (len(faces) + 1,)
        or offsets[0] != 0
        or not np.array_equal(np.diff(offsets), resolution * (resolution + 1) // 2)
        or not np.array_equal(
            arrays["sample_face_index"],
            np.repeat(np.arange(len(faces)), np.diff(offsets)),
        )
    ):
        raise QualificationError("M_CAA_FACE_LATTICE_ADDRESS_INVALID")
    patterns = {
        int(n): triangular_barycentric_samples(int(n)) for n in np.unique(resolution)
    }
    if np.all(resolution == 4):
        bary = np.asarray(arrays.get("barycentric", patterns[4]))
        if not np.array_equal(bary, patterns[4]):
            raise QualificationError("M_CAA_BARYCENTRIC_PATTERN_DRIFT")
        expected = np.einsum("sc,fcd->fsd", bary, rest[faces]).reshape(-1, 3)
    else:
        expected = np.empty((int(offsets[-1]), 3), dtype=np.float64)
        for fi, n in enumerate(resolution):
            expected[offsets[fi] : offsets[fi + 1]] = patterns[int(n)] @ rest[faces[fi]]
    residual = float(np.max(np.abs(expected - arrays["sample_positions"])))
    if not np.isfinite(residual) or residual > 1e-10:
        raise QualificationError("M_CAA_SURFACE_ADDRESS_DRIFT")
    luts = {}
    for n in patterns:
        sample_map = _sample_index_map(n)
        luts[n] = np.array(
            [
                [
                    sample_map[_nearest_triangle_lattice(i, j, tile_resolution=n)]
                    for i in range(n)
                ]
                for j in range(n)
            ]
        )
    rgba = np.asarray(arrays["rgba"])
    provenance = np.asarray(arrays["provenance"])
    donors = np.asarray(arrays["source_view"])
    v = np.arange(8)[:, None]
    if (
        rgba.shape != (8, len(expected), 4)
        or provenance.shape != rgba.shape[:2]
        or donors.shape != provenance.shape
    ):
        raise QualificationError("M_CAA_FIELD_SHAPE_INVALID")
    valid = (
        ((provenance == 0) & (donors == v))
        | ((provenance == 1) & (donors >= 0) & (donors < 8) & (donors != v))
        | ((provenance == 2) & (donors == -2))
        | ((provenance == 3) & (donors == -4))
        | ((provenance == 4) & (donors == -3))
    )
    if not valid.all() or np.any(rgba[provenance == 3]):
        raise QualificationError("M_CAA_PROVENANCE_OR_ABSTENTION_DRIFT")
    return {
        "offsets": offsets,
        "resolutions": resolution,
        "pm": straight_srgb_rgba_u8_to_premultiplied_linear(rgba),
        "provenance": provenance,
        "source_view": donors,
        "sample_lut": luts.get(4),
        "sample_luts": luts,
        "position_residual": residual,
    }


def compile_rest_visibility(rest, faces, cameras):
    return [
        rasterize(project_points_xyz_v3(rest, camera), faces, camera.resolution)[:2]
        for camera in cameras
    ]


def render_single_m_caa(
    *, rest, posed, faces, source, maps, cameras, caa, view, resolution=256,
    face_draw_rank=None, output_camera=None,
):
    if (
        np.asarray(posed).shape != np.asarray(rest).shape
        or not np.isfinite(posed).all()
    ):
        raise QualificationError("M_CAA_POSED_DOMAIN_DRIFT")
    projected = project_points_xyz_v3(posed, cameras[view] if output_camera is None else output_camera)
    sampler, counts = material_sampler(
        rest, faces, source, maps, cameras, caa, view, True
    )
    result = render_m_depth(
        positions=projected[:, :2] - 0.5,
        depths=projected[:, 2],
        faces=faces,
        uv=np.zeros((len(rest), 2)),
        texture=source[view],
        resolution=resolution,
        sample_material=sampler,
        face_draw_rank=face_draw_rank,
        position_resolution=None if output_camera is None else (output_camera.resolution, output_camera.resolution),
    )
    result.material_fragments = counts
    return result


def _proof_fields(result, geometry, ambiguity, overflow, layer_owners, contributions):
    result.geometry_visible = geometry
    result.final_alpha = result.straight_rgba_u8[:, :, 3] > 0
    result.exact_depth_ambiguity = ambiguity
    result.layer_overflow = overflow
    result.layer_owner_face_index = layer_owners.transpose(1, 2, 0)
    result.contributing_layer_mask = contributions.transpose(1, 2, 0)
    result.coverage_sample_owner_face_index = result.owner_face_index[:, :, None]
    result.coverage_sample_count = 1
    return result


def write_material_bundle(
    path, *, rest, faces, source, cameras, caa_arrays, input_identities
):
    """Offline transport only; retains the exact compiled CAA samples."""
    prepare_caa_field(caa_arrays, rest, faces)
    maps = compile_rest_visibility(rest, faces, cameras)
    payload = {"rest": np.asarray(rest), "faces": np.asarray(faces)}
    for key in [
        "face_sample_offsets",
        "face_tile_resolutions",
        "sample_face_index",
        "sample_positions",
        "rgba",
        "provenance",
        "source_view",
    ]:
        payload["caa_" + key] = np.asarray(caa_arrays[key])
    if caa_arrays.get("barycentric") is not None:
        payload["caa_barycentric"] = np.asarray(caa_arrays["barycentric"])
    for v in range(8):
        payload[f"source_{v}"] = np.asarray(source[v], dtype=np.uint8)
        payload[f"owner_{v}"] = maps[v][0]
        payload[f"depth_{v}"] = maps[v][1]
    metadata = {
        "contract": MATERIAL_CONTRACT,
        "rest_hash": array_hash(rest),
        "faces_hash": array_hash(faces),
        "input_identities": dict(input_identities),
        "cameras": [vars(x) for x in cameras],
        "composition": "APPROVED_BASELINE_STRAIGHT_SRGB_FRONT_TO_BACK",
        "caa_filtering": "LINEAR_PREMULTIPLIED",
        "runtime_generation": False,
        "product_qualification_claimed": False,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "contract": MATERIAL_CONTRACT,
    }


def load_material_bundle(path, *, expected_sha256, rest=None, faces=None):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise QualificationError("M_CAA_MATERIAL_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        if meta.get("contract") != MATERIAL_CONTRACT:
            raise QualificationError("M_CAA_MATERIAL_CONTRACT_INVALID")
        r = data["rest"]
        f = data["faces"]
        if array_hash(r) != meta["rest_hash"] or array_hash(f) != meta["faces_hash"]:
            raise QualificationError("M_CAA_MATERIAL_DOMAIN_HASH_DRIFT")
        if rest is not None and not np.array_equal(rest, r):
            raise QualificationError("M_CAA_RUNTIME_REST_DRIFT")
        if faces is not None and not np.array_equal(faces, f):
            raise QualificationError("M_CAA_RUNTIME_TOPOLOGY_DRIFT")
        arrays = {
            k.removeprefix("caa_"): data[k] for k in data.files if k.startswith("caa_")
        }
        cameras = [
            qualify_camera_v3(row, view_id=row["view_id"], view_index=row["view_index"])
            for row in meta["cameras"]
        ]
        # Use sealed coefficients exactly after structural qualification.
        cameras = [type(cameras[v])(**row) for v, row in enumerate(meta["cameras"])]
        return {
            "rest": r,
            "faces": f,
            "source": [data[f"source_{v}"] for v in range(8)],
            "maps": [(data[f"owner_{v}"], data[f"depth_{v}"]) for v in range(8)],
            "cameras": cameras,
            "caa": prepare_caa_field(arrays, r, f),
            "metadata": meta,
        }


def native_material_payload(path, *, expected_sha256, rest, faces):
    """Exact sealed source/CAA transport for the native M consumer, no bake."""
    material = load_material_bundle(
        path, expected_sha256=expected_sha256, rest=rest, faces=faces
    )
    with np.load(path, allow_pickle=False) as data:
        caa = {
            k.removeprefix("caa_"): data[k] for k in data.files if k.startswith("caa_")
        }
    h, w, _ = material["source"][0].shape
    if any(image.shape != (h, w, 4) for image in material["source"]):
        raise QualificationError("M_CAA_SOURCE_DIMENSIONS_MUST_MATCH")
    blobs = [
        struct.pack("<5I", len(rest), len(faces), len(caa["sample_positions"]), w, h),
        np.asarray(rest, dtype="<f8").tobytes(),
        np.asarray(faces, dtype="<u4").tobytes(),
    ]
    for camera in material["cameras"]:
        blobs.append(
            struct.pack(
                "<13dI",
                *camera.origin,
                *camera.right,
                *camera.screen_up,
                *camera.forward,
                camera.half_extent,
                camera.resolution,
            )
        )
    for array, dtype in [
        (np.stack(material["source"]), "u1"),
        (np.stack([x[0] for x in material["maps"]]), "<i4"),
        (np.stack([x[1] for x in material["maps"]]), "<f8"),
        (caa["face_sample_offsets"], "<u8"),
        (caa["face_tile_resolutions"], "<u4"),
        (caa["rgba"], "u1"),
        (caa["provenance"], "u1"),
        (caa["source_view"], "<i2"),
    ]:
        blobs.append(np.asarray(array, dtype=dtype).tobytes())
    luts = material["caa"]["sample_luts"]
    blobs.append(struct.pack("<I", len(luts)))
    for n, lut in sorted(luts.items()):
        blobs.extend([struct.pack("<I", n), np.asarray(lut, dtype="<u4").tobytes()])
    raw = b"".join(blobs)
    return struct.pack("<Q", len(raw)) + zlib.compress(raw, 6)


def rasterize(projected, faces, resolution):
    """Opaque M first hit; alpha never changes geometric ownership."""
    owner = np.full((resolution, resolution), -1, dtype=np.int32)
    depth = np.full((resolution, resolution), np.inf)
    bary = np.zeros((resolution, resolution, 3), dtype=np.float64)

    def orient(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    def top_left(a, b):
        dy, dx = b[1] - a[1], b[0] - a[0]
        return dy < 0 or (abs(dy) <= 1e-12 and dx > 0)

    for fi, ids in enumerate(faces):
        tri = projected[ids]
        a, b, c = tri[:, :2]
        area = orient(a, b, c[0], c[1])
        if abs(area) <= 1e-12:
            continue
        lo = np.maximum(0, np.floor(tri[:, :2].min(axis=0) - 0.5).astype(int))
        hi = np.minimum(
            resolution - 1, np.ceil(tri[:, :2].max(axis=0) - 0.5).astype(int)
        )
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        x, y = xx + 0.5, yy + 0.5
        edges = [orient(b, c, x, y), orient(c, a, x, y), orient(a, b, x, y)]
        sign = 1 if area > 0 else -1
        tl = (
            [top_left(b, c), top_left(c, a), top_left(a, b)]
            if area > 0
            else [top_left(c, b), top_left(a, c), top_left(b, a)]
        )
        inside = np.ones(x.shape, dtype=bool)
        for e, t in zip(edges, tl):
            q = sign * e
            inside &= (q > 1e-12) | ((np.abs(q) <= 1e-12) & t)
        weights = np.stack([e / area for e in edges], axis=-1)
        z = weights @ tri[:, 2]
        dst = depth[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        win = inside & (z < dst - 1e-12)
        dst[win] = z[win]
        owner[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1][win] = fi
        bary[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1][win] = weights[win]
    return owner, depth, bary


def sample_caa(caa, face_index, barycentric, view):
    """The existing 4-texel face atlas and linear PM bilinear filtering.

    Identical lattice scatter for RGBA/provenance/source-view. No RGB predictor
    is replaced or rerun. Code 3 remains an explicit unsupported abstention.
    """
    f = np.asarray(face_index)
    b = np.asarray(barycentric)
    if not np.all(caa["resolutions"][f] == 4):
        rgba = np.empty((len(f), 4), dtype=np.uint8)
        codes = np.empty(len(f), dtype=np.uint8)
        for n in np.unique(caa["resolutions"][f]):
            mask = caa["resolutions"][f] == n
            rgba[mask], codes[mask] = _sample_lattice(
                caa, f[mask], b[mask], view, int(n)
            )
        return rgba, codes
    return _sample_lattice(caa, f, b, view, 4)


def _sample_lattice(caa, f, b, view, resolution):
    if resolution != 4:
        scale = resolution - 1
        x = np.clip(b[:, 1] * scale, 0, scale)
        y = np.clip(b[:, 2] * scale, 0, scale)
    else:
        # Preserve the exact arithmetic of the approved four-texel baseline.
        x = np.clip(b[:, 1] * 3, 0, 3)
        y = np.clip(b[:, 2] * 3, 0, 3)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, resolution - 1)
    y1 = np.minimum(y0 + 1, resolution - 1)
    fx = x - x0
    fy = y - y0
    sample_lut = caa["sample_luts"][resolution]
    index = (
        np.stack(
            [
                sample_lut[y0, x0],
                sample_lut[y0, x1],
                sample_lut[y1, x0],
                sample_lut[y1, x1],
            ],
            axis=1,
        )
        + caa["offsets"][f, None]
    )
    weights = np.stack(
        [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1
    )
    pm = np.sum(caa["pm"][view, index] * weights[:, :, None], axis=1)
    codes = caa["provenance"][view, index]
    rank = np.array([0, 1, 2, 4, 3])[codes]
    active = weights > 1e-12
    worst = np.max(np.where(active, rank, -1), axis=1)
    code = np.array([0, 1, 2, 4, 3], dtype=np.uint8)[worst]
    pm[code == 3] = 0
    rgba = premultiplied_linear_to_straight_srgb_u8(pm[:, None, :])[:, 0, :]
    return rgba, code


def source_bilinear(points, view, source, rest_maps, cameras):
    projected = project_points_xyz_v3(points, cameras[view])
    h, w, _ = source[view].shape
    # Camera coordinates are pixel boundaries; integer texture coordinates are
    # pixel centers. The V reference's +0.5 placement uses the same convention.
    x = np.clip(projected[:, 0] - 0.5, 0, w - 1)
    y = np.clip(projected[:, 1] - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    xy = np.floor(projected[:, :2]).astype(int)
    in_bounds = np.all((xy >= 0) & (xy < cameras[view].resolution), axis=1)
    xy = np.clip(xy, 0, cameras[view].resolution - 1)
    ix, iy = xy.T
    owner, depth = rest_maps[view]
    eps = 2 * cameras[view].half_extent / cameras[view].resolution
    visible = (
        in_bounds
        & (owner[iy, ix] >= 0)
        & (np.abs(projected[:, 2] - depth[iy, ix]) <= eps)
    )
    tex = source[view]
    row = (
        (
            tex[y0, x0].astype(float) * (1 - fx[:, None])
            + tex[y0, x1].astype(float) * fx[:, None]
        )
        * (1 - fy[:, None])
        + (
            tex[y1, x0].astype(float) * (1 - fx[:, None])
            + tex[y1, x1].astype(float) * fx[:, None]
        )
        * fy[:, None]
    ) / 255
    known = visible & (row[:, 3] > 1e-12)
    row[~known] = 0
    return row, known


def material_sampler(rest, faces, source, maps, cameras, caa, view, use_caa):
    count = {
        "source_fragments": 0,
        "caa_fragments": 0,
        "unsupported_fragments": 0,
        "provenance": {str(i): 0 for i in range(5)},
    }

    def sample(fi, bary):
        fi = np.asarray(fi, dtype=int)
        b = np.clip(bary, 0, 1)
        b /= b.sum(axis=1, keepdims=True)
        points = np.einsum("nc,ncd->nd", b, rest[faces[fi]])
        row, known = source_bilinear(points, view, source, maps, cameras)
        provenance = np.zeros(len(fi), dtype=np.uint8)
        count["source_fragments"] += int(known.sum())
        count["provenance"]["0"] += int(known.sum())
        missing = ~known
        if use_caa and missing.any():
            ids = fi[missing]
            rgba, codes = sample_caa(caa, ids, b[missing], view)
            row[missing] = rgba.astype(float) / 255
            provenance[missing] = codes
            count["caa_fragments"] += int(np.count_nonzero(codes != 3))
            count["unsupported_fragments"] += int(np.count_nonzero(codes == 3))
            for c in range(5):
                count["provenance"][str(c)] += int(np.count_nonzero(codes == c))
        else:
            count["unsupported_fragments"] += int(missing.sum())
            provenance[missing] = 3
        sample.provenance_codes = provenance
        return row

    return sample, count


def render_m_depth(
    *, positions, depths, faces, uv, texture, resolution, sample_material,
    face_draw_rank=None, position_resolution=None,
):
    p, z, f, uv = (
        np.asarray(positions, dtype=float),
        np.asarray(depths, dtype=float),
        np.asarray(faces, dtype=np.int64),
        np.asarray(uv, dtype=float),
    )
    tex = np.asarray(texture, dtype=np.uint8)
    ranks=None if face_draw_rank is None else np.asarray(face_draw_rank,dtype=np.int32)
    if ranks is not None and (ranks.shape!=(len(f),) or np.any(ranks<0)):
        raise QualificationError("PRESENTATION_FACE_DRAW_RANK_INVALID")
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or z.shape != (len(p),)
        or uv.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(z).all()
        or np.any(z <= 0)
        or not np.isfinite(uv).all()
        or f.ndim != 2
        or f.shape[1] != 3
        or np.any(f < 0)
        or np.any(f >= len(p))
        or tex.ndim != 3
        or tex.shape[2] != 4
        or resolution <= 0
    ):
        raise QualificationError("VISUAL_CANONICAL_DEPTH_INPUT_INVALID")
    height, width = tex.shape[:2] if position_resolution is None else position_resolution[::-1]
    if width <= 0 or height <= 0:
        raise QualificationError("VISUAL_CANONICAL_POSITION_RESOLUTION_INVALID")
    p = (p + 0.5) * np.array([resolution / width, resolution / height])
    pixels, values, colors, owners = [], [], [], []

    def orient(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    def top_left(a, b):
        dy, dx = b[1] - a[1], b[0] - a[0]
        return dy < 0 or (abs(dy) <= 1e-12 and dx > 0)

    for fi, face in enumerate(f):
        a, b, c = p[face]
        area = orient(a, b, c[0], c[1])
        if abs(area) <= 1e-12:
            continue
        lo = np.maximum(0, np.floor(np.min(p[face], axis=0) - 0.5).astype(int))
        hi = np.minimum(
            resolution - 1, np.ceil(np.max(p[face], axis=0) - 0.5).astype(int)
        )
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        x, y = xx.ravel() + 0.5, yy.ravel() + 0.5
        w0, w1 = orient(b, c, x, y), orient(c, a, x, y)
        w2 = orient(a, b, x, y)
        sign = 1 if area > 0 else -1
        tl = (
            (top_left(b, c), top_left(c, a), top_left(a, b))
            if area > 0
            else (top_left(c, b), top_left(a, c), top_left(b, a))
        )
        inside = np.ones(len(x), dtype=bool)
        for q, edge in zip((w0, w1, w2), tl):
            q = sign * q
            inside &= (q > 1e-12) | ((np.abs(q) <= 1e-12) & edge)
        if not np.any(inside):
            continue
        w0, w1 = w0[inside] / area, w1[inside] / area
        w2 = 1 - w0 - w1
        row = np.column_stack((w0, w1, w2, np.ones_like(w0)))
        live = np.ones(len(w0), dtype=bool)
        pixels.append((yy.ravel()[inside] * resolution + xx.ravel()[inside])[live])
        values.append((w0 * z[face[0]] + w1 * z[face[1]] + w2 * z[face[2]])[live])
        colors.append(row[live])
        owners.append(np.full(np.count_nonzero(live), fi, dtype=np.int32))
    n = resolution * resolution
    accum = np.zeros((n, 4), dtype=float)
    owner = np.full(n, -1, dtype=np.int32)
    output_provenance = np.full(n, 255, dtype=np.uint8)
    geometry = np.zeros(n, dtype=bool)
    ambiguity = np.zeros(n, dtype=bool)
    overflow_mask = np.zeros(n, dtype=bool)
    layer_owners = np.full((1, n), -1, dtype=np.int32)
    contributions = np.zeros((1, n), dtype=bool)
    ties = overflow = overlap = 0
    if pixels and sum(len(x) for x in pixels):
        pixel, depth, color, ancestry = (
            np.concatenate(x) for x in (pixels, values, colors, owners)
        )
        geometry[pixel] = True
        color = sample_material(ancestry, color[:, :3])
        fragment_provenance = np.asarray(
            getattr(
                sample_material,
                "provenance_codes",
                np.zeros(len(color), dtype=np.uint8),
            )
        )
        all_order = (np.lexsort((depth, pixel)) if ranks is None else
                     np.lexsort((depth,ranks[ancestry],pixel)))
        ordered_pixel = pixel[all_order]
        first = np.r_[True, ordered_pixel[1:] != ordered_pixel[:-1]]
        output_provenance[ordered_pixel[first]] = fragment_provenance[all_order[first]]
        live = color[:, 3] > 1e-12
        fragment_provenance = fragment_provenance[live]
        pixel, depth, color, ancestry = (
            a[live] for a in (pixel, depth, color, ancestry)
        )
        if len(pixel) == 0:
            result = SimpleNamespace(
                straight_rgba_u8=np.zeros((resolution, resolution, 4), np.uint8),
                provenance_code=output_provenance.reshape(resolution, resolution),
                owner_face_index=np.full((resolution, resolution), -1, np.int32),
                unresolved_depth_tie_count=0,
                fragment_overflow_count=0,
                overlap_pixel_count=0,
            )
            return _proof_fields(
                result,
                geometry.reshape(resolution, resolution),
                ambiguity.reshape(resolution, resolution),
                overflow_mask.reshape(resolution, resolution),
                layer_owners.reshape(1, resolution, resolution),
                contributions.reshape(1, resolution, resolution),
            )
        raw_order = np.lexsort((depth, pixel))
        raw_pixel,raw_depth=pixel[raw_order],depth[raw_order]
        raw_tied=(raw_pixel[1:]==raw_pixel[:-1]) & (np.abs(raw_depth[1:]-raw_depth[:-1])<=POLICY["depth_tie_epsilon"])
        order = raw_order if ranks is None else np.lexsort((depth,ranks[ancestry],pixel))
        fragment_provenance = fragment_provenance[order]
        pixel, depth, color, ancestry = (
            x[order] for x in (pixel, depth, color, ancestry)
        )
        same = pixel[1:] == pixel[:-1]
        ties = int(np.count_nonzero(raw_tied))
        ambiguity[raw_pixel[1:][raw_tied]] = True
        starts = np.r_[True, ~same]
        rank = np.arange(len(pixel)) - np.maximum.accumulate(
            np.where(starts, np.arange(len(pixel)), 0)
        )
        overflow = int(np.count_nonzero(rank >= POLICY["maximum_fragment_layers"]))
        overflow_mask[pixel[rank >= POLICY["maximum_fragment_layers"]]] = True
        retained = min(int(np.max(rank)) + 1, POLICY["maximum_fragment_layers"])
        layer_owners = np.full((retained, n), -1, dtype=np.int32)
        contributions = np.zeros((retained, n), dtype=bool)
        overlap = int(len(np.unique(pixel[rank > 0])))
        # Front to back. Ownership is the nearest nontransparent canonical
        # fragment, independent of triangle enumeration and source alpha.
        for layer in range(int(np.max(rank)) + 1):
            take = rank == layer
            dst, src = pixel[take], color[take]
            alpha = src[:, 3]
            transmission = 1 - accum[dst, 3]
            if layer < retained:
                layer_owners[layer, dst] = ancestry[take]
                contributions[layer, dst] = (alpha * transmission) > 1e-12
            accum[dst, :3] += src[:, :3] * alpha[:, None] * transmission[:, None]
            accum[dst, 3] += alpha * transmission
            if layer == 0:
                owner[dst] = ancestry[take]
                output_provenance[dst] = fragment_provenance[take]
    visible = accum[:, 3] > 1e-12
    accum[visible, :3] /= accum[visible, 3, None]
    rgba = (
        np.floor(np.clip(accum, 0, 1) * 255 + 0.5)
        .astype(np.uint8)
        .reshape(resolution, resolution, 4)
    )
    provenance = output_provenance.reshape(resolution, resolution)
    result = SimpleNamespace(
        straight_rgba_u8=rgba,
        provenance_code=provenance,
        owner_face_index=owner.reshape(resolution, resolution),
        unresolved_depth_tie_count=ties,
        fragment_overflow_count=overflow,
        overlap_pixel_count=overlap,
    )
    return _proof_fields(
        result,
        geometry.reshape(resolution, resolution),
        ambiguity.reshape(resolution, resolution),
        overflow_mask.reshape(resolution, resolution),
        layer_owners.reshape(-1, resolution, resolution),
        contributions.reshape(-1, resolution, resolution),
    )
