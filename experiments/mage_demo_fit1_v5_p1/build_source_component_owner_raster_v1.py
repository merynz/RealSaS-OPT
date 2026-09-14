from __future__ import annotations

"""Build exact source-component owner rasters for the bounded Mage demo.

The authority is derived only from the sealed full-subject FIT target, the exact typed
source-component truth witness, and the exact eight orthographic source cameras.  It
does not use P1/P1Q geometry, V5 weights, filenames, nearest-owner completion, or a
learned component classifier.  Source triangles are first proven to stay entirely
inside one typed component range, then a deterministic orthographic z-buffer assigns
one source owner to every rendered source pixel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import polygon

SCHEMA = "RealSaS.MageExactSourceComponentOwnerRasterAuthority.v1"
EXPECTED_FIT_TARGET_SHA256 = "dec2e816ea2bac5bfca44b1669484b0cde3c87d8d4f5052890cee07469c6825a"
EXPECTED_SOURCE_TRUTH_SHA256 = "a23565b0904e9683d11083984a87405f4e0a1069984ca11b690ad431f35f5e86"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _project(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    forward /= np.linalg.norm(forward)
    delta = np.asarray(points, dtype=np.float64) - origin[None, :]
    half = float(camera["half_extent"])
    resolution = int(camera["resolution"])
    if resolution <= 0 or half <= 0.0:
        raise RuntimeError("SOURCE_OWNER_CAMERA_INVALID")
    gx = (delta @ right) / half
    gy = -(delta @ up) / half
    depth = delta @ forward
    raster = np.stack(
        ((gx + 1.0) * 0.5 * resolution - 0.5, (gy + 1.0) * 0.5 * resolution - 0.5),
        axis=1,
    )
    return raster, depth


def _raster_owner(vertices, faces, face_owner, camera):
    resolution = int(camera["resolution"])
    xy, depth = _project(vertices, camera)
    z = np.full((resolution, resolution), np.inf, dtype=np.float64)
    owner = np.full((resolution, resolution), -1, dtype=np.int16)

    for triangle_index, triangle in enumerate(np.asarray(faces, dtype=np.int64)):
        points = xy[triangle]
        tri_depth = depth[triangle]
        if np.any(tri_depth <= 0.0):
            continue
        rr, cc = polygon(points[:, 1], points[:, 0], shape=(resolution, resolution))
        if not len(rr):
            continue
        x = cc.astype(np.float64)
        y = rr.astype(np.float64)
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1.0e-15:
            continue
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
        g = 1.0 - a - b
        raster_depth = a * tri_depth[0] + b * tri_depth[1] + g * tri_depth[2]
        old_depth = z[rr, cc]
        take = raster_depth < old_depth
        if np.any(take):
            z[rr[take], cc[take]] = raster_depth[take]
            owner[rr[take], cc[take]] = int(face_owner[triangle_index])
    return owner, z


def run(args) -> dict:
    fit_target = Path(args.fit_target)
    source_truth_path = Path(args.source_truth)
    if _sha(fit_target) != EXPECTED_FIT_TARGET_SHA256:
        raise RuntimeError("SOURCE_OWNER_FIT_TARGET_SHA_DRIFT")
    if _sha(source_truth_path) != EXPECTED_SOURCE_TRUTH_SHA256:
        raise RuntimeError("SOURCE_OWNER_TRUTH_SHA_DRIFT")

    cameras = tuple(Path(path) for path in args.cameras)
    observations = tuple(Path(path) for path in args.observations)
    if len(cameras) != 8 or len(observations) != 8:
        raise RuntimeError("SOURCE_OWNER_REQUIRES_8_VIEWS")

    with np.load(fit_target, allow_pickle=False) as payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)

    source_truth = json.loads(source_truth_path.read_text(encoding="utf-8"))
    components = [
        component
        for component in source_truth["components"]
        if int(component["vertex_range"][0]) < len(vertices)
    ]
    component_names = [str(component["component_family"]) for component in components]
    vertex_owner = np.full(len(vertices), -1, dtype=np.int16)
    for component_index, component in enumerate(components):
        start, end = map(int, component["vertex_range"])
        vertex_owner[start:min(end, len(vertices))] = component_index
    if np.any(vertex_owner < 0):
        raise RuntimeError("SOURCE_OWNER_UNOWNED_FIT_VERTEX")

    owner_per_corner = vertex_owner[faces]
    same_owner = (
        (owner_per_corner[:, 0] == owner_per_corner[:, 1])
        & (owner_per_corner[:, 1] == owner_per_corner[:, 2])
    )
    if not bool(np.all(same_owner)):
        raise RuntimeError("SOURCE_OWNER_CROSS_COMPONENT_FACE")
    face_owner = owner_per_corner[:, 0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    artifacts = []
    for view, (camera_path, observation_path) in enumerate(zip(cameras, observations)):
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        owner, depth = _raster_owner(vertices, faces, face_owner, camera)
        output_path = output_dir / f"V{view}_SOURCE_COMPONENT_OWNER_RASTER.npz"
        np.savez_compressed(
            output_path,
            owner=owner,
            depth=depth,
            component_names=np.asarray(component_names),
        )
        with Image.open(observation_path) as image:
            alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3] >= 8
        predicted = owner >= 0
        intersection = int(np.count_nonzero(predicted & alpha))
        union = int(np.count_nonzero(predicted | alpha))
        predicted_count = int(np.count_nonzero(predicted))
        truth_count = int(np.count_nonzero(alpha))
        row = {
            "view": view,
            "camera_sha256": _sha(camera_path),
            "observation_sha256": _sha(observation_path),
            "owner_raster_sha256": _sha(output_path),
            "source_render_pixel_count": predicted_count,
            "observation_alpha_pixel_count": truth_count,
            "recall": float(intersection / max(truth_count, 1)),
            "precision": float(intersection / max(predicted_count, 1)),
            "iou": float(intersection / max(union, 1)),
            "component_pixel_counts": {
                component_names[index]: int(np.count_nonzero(owner == index))
                for index in range(len(component_names))
            },
        }
        if row["recall"] < 0.99 or row["precision"] < 0.99:
            raise RuntimeError(f"SOURCE_OWNER_RASTER_GATE_FAIL_V{view}")
        rows.append(row)
        artifacts.append({"path": output_path.name, "sha256": row["owner_raster_sha256"]})
        print("SOURCE_OWNER_VIEW=" + json.dumps(row, sort_keys=True), flush=True)

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__EXACT_SOURCE_FIT_TARGET_COMPONENT_OWNER_RASTER_V0_V7",
        "fit_target_sha256": _sha(fit_target),
        "source_truth_sha256": _sha(source_truth_path),
        "component_names": component_names,
        "source_face_count": int(len(faces)),
        "cross_component_face_count": 0,
        "views": rows,
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    manifest_path = output_dir / "SOURCE_COMPONENT_OWNER_RASTER_AUTHORITY.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_OWNER_MANIFEST_SHA256=" + _sha(manifest_path), flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-target", required=True)
    parser.add_argument("--source-truth", required=True)
    parser.add_argument("--cameras", nargs=8, required=True)
    parser.add_argument("--observations", nargs=8, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
