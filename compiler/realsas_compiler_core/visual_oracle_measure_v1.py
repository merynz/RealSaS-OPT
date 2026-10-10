"""Independent oracle: source cached abs poses + inverse-affine sprite sampler.

Does not call the hierarchy, corner builder, RSS writer or native rasterizer.
Reads source ZIP again. Expected owners/order come from the raw SCML, not a
consumer receipt. The native face-ID output is measured against this reference.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import math
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image


def cached_source(section):
    raw = Path(section["archive_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != section["archive_sha256"]:
        raise ValueError("ORACLE_REFERENCE_ARCHIVE_DRIFT")
    images, frames, owners = {}, [], set()
    with zipfile.ZipFile(BytesIO(raw)) as z:
        scml = z.read(section["scml_member"])
        if hashlib.sha256(scml).hexdigest() != section["scml_sha256"]:
            raise ValueError("ORACLE_REFERENCE_SCML_DRIFT")
        root = ET.fromstring(scml)
        folder_files = {(d.get("id"), f.get("id")): f for d in root.findall("folder") for f in d.findall("file")}
        entity = next(e for e in root.findall("entity") if e.get("id") == str(section["entity_id"]))
        base = PurePosixPath(section["scml_member"]).parent
        for clip in section["clips"]:
            animation = next(a for a in entity.findall("animation") if a.get("name") == clip)
            timelines = {t.get("id"): t for t in animation.findall("timeline")}
            for key in animation.find("mainline").findall("key"):
                objects = []
                for ref in key.findall("object_ref"):
                    if any("abs_" + k not in ref.attrib for k in ("x", "y", "angle", "scale_x", "scale_y", "a")):
                        raise ValueError("ORACLE_INDEPENDENT_CACHED_POSE_MISSING")
                    # The authored cached file identity is also independent of
                    # the producer's timeline/file lookup.
                    if ref.get("folder") is None or ref.get("file") is None:
                        raise ValueError("ORACLE_CACHED_ASSET_ID_MISSING")
                    f = folder_files[(ref.get("folder"), ref.get("file"))]
                    member = str(base / f.get("name"))
                    if member not in images:
                        image = Image.open(BytesIO(z.read(member))).convert("RGBA")
                        images[member] = np.asarray(image, dtype=float) / 255
                    timeline = timelines[ref.get("timeline")]
                    owner = f"entity:{entity.get('id')}/object:{timeline.get('obj')}"
                    owners.add(owner)
                    objects.append({"owner": owner, "name": timeline.get("name"), "member": member,
                        "pose": [float(ref.get("abs_" + k)) for k in ("x", "y", "angle", "scale_x", "scale_y", "a")],
                        "pivot": [float(ref.get("abs_" + k, f.get(k, d))) for k, d in (("pivot_x", "0"), ("pivot_y", "1"))],
                        "order": int(ref.get("z_index"))})
                frames.append({"clip": clip, "key_id": key.get("id"), "time_ms": int(key.get("time", 0)),
                               "objects": sorted(objects, key=lambda o: o["order"])})
    return images, frames, sorted(owners)


def independent_reference(frame, images, owners, resolution, offset):
    """Inverse affine on pixel centers, bilinear straight RGBA, source-over.

    Canonical addresses are (authored owner, source image texel). Per-owner
    visibility is measured from the final authored composite, not parent links.
    """
    pm = np.zeros((resolution, resolution, 4))
    owner_map = np.full((resolution, resolution), -1, dtype=np.int32)
    robust = np.zeros((resolution, resolution), dtype=bool)
    address = np.full((resolution, resolution), -1, dtype=np.int64)
    for obj in frame["objects"]:
        image = images[obj["member"]]; h, w = image.shape[:2]
        x, y, degrees, sx, sy, opacity = obj["pose"]
        if opacity != 1 or sx == 0 or sy == 0:
            raise ValueError("ORACLE_REFERENCE_OPACITY_OR_SCALE_UNSUPPORTED")
        a = math.radians(degrees); c, s = math.cos(a), math.sin(a)
        # Matrix maps sprite top-left edge coordinates directly into y-down
        # screen coordinates. It does not use the producer's corner function.
        matrix = np.array([[c*sx, s*sy], [-s*sx, c*sy]])
        pivot_edge = np.array([obj["pivot"][0]*w, (1-obj["pivot"][1])*h])
        origin = np.array([x, -y]) + offset - matrix @ pivot_edge
        quad = np.array([[0, 0], [w, 0], [0, h], [w, h]]) @ matrix.T + origin
        low = np.maximum(np.floor(quad.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(quad.max(axis=0)).astype(int), resolution)
        xx, yy = np.meshgrid(np.arange(low[0], high[0]) + .5, np.arange(low[1], high[1]) + .5)
        coords = (np.stack((xx, yy), -1) - origin) @ np.linalg.inv(matrix).T
        u, v = coords[..., 0] - .5, coords[..., 1] - .5
        domain = (coords[..., 0] >= 0) & (coords[..., 0] < w) & (coords[..., 1] >= 0) & (coords[..., 1] < h)
        iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u-iu, v-iv
        samples = np.zeros((*u.shape, 4))
        for dx, dy, weight in ((0, 0, (1-fu)*(1-fv)), (1, 0, fu*(1-fv)),
                               (0, 1, (1-fu)*fv), (1, 1, fu*fv)):
            ix, iy = iu+dx, iv+dy
            valid = domain & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            samples[valid] += image[iy[valid], ix[valid]] * weight[valid, None]
        area = np.s_[low[1]:high[1], low[0]:high[0]]
        alpha = samples[..., 3]
        pm[area][:, :, :3] = samples[..., :3]*alpha[..., None] + pm[area][:, :, :3]*(1-alpha[..., None])
        pm[area][:, :, 3] = alpha + pm[area][:, :, 3]*(1-alpha)
        visible = alpha > 1e-12
        owner_map[area][visible] = owners.index(obj["owner"])
        # Integer canonical texel addresses, with sufficient foreground support.
        tx = np.clip(np.floor(coords[..., 0]).astype(int), 0, w-1)
        ty = np.clip(np.floor(coords[..., 1]).astype(int), 0, h-1)
        address[area][visible] = ty[visible]*w + tx[visible]
        robust[area][visible] = (alpha[visible] >= .99) & (u[visible] >= 1) & (u[visible] < w-2) & (v[visible] >= 1) & (v[visible] < h-2)
    rgba = pm.copy()
    rgba[..., :3] = np.divide(pm[..., :3], pm[..., 3, None], out=np.zeros_like(pm[..., :3]), where=pm[..., 3, None] > 1e-12)
    rgba = np.clip(np.floor(rgba*255 + .5), 0, 255).astype(np.uint8)
    return rgba, owner_map, robust, address


def metrics(expected, actual, expected_owner, actual_owner, robust, budgets):
    e, a = expected.astype(float)/255, actual.astype(float)/255
    epm, apm = e.copy(), a.copy()
    epm[..., :3] *= e[..., 3, None]; apm[..., :3] *= a[..., 3, None]
    union = (e[..., 3] > .01) | (a[..., 3] > .01)
    intersection = (e[..., 3] > .01) & (a[..., 3] > .01)
    error = np.abs(epm-apm)[union]
    iou = float(intersection.sum()/max(1, union.sum()))
    mean = float(error.mean()) if error.size else 0
    p99 = float(np.quantile(error, .99)) if error.size else 0
    mismatch = int((robust & (expected_owner != actual_owner)).sum())
    checks = {"alpha_iou": iou >= budgets["minimum_alpha_iou"],
              "pm_mean": mean <= budgets["maximum_pm_mean"],
              "pm_p99": p99 <= budgets["maximum_pm_p99"],
              "ownership": mismatch <= budgets["maximum_robust_owner_mismatch_pixels"]}
    return {"alpha_iou": iou, "pm_mean": mean, "pm_p99": p99,
            "robust_owner_tested_pixels": int(robust.sum()),
            "robust_owner_mismatch_pixels": mismatch,
            "checks": checks, "passed": all(checks.values())}


def canonical_visible(frame, owners, owner_map, address, robust):
    result = {}
    for obj in frame["objects"]:
        mask = robust & (owner_map == owners.index(obj["owner"]))
        result[(obj["owner"], obj["member"])] = set(map(int, address[mask]))
    return result


def initially_occluded(frame, images, owners, owner_map, robust, offset):
    """Authored opaque texel centers covered by another robust authored owner.

    Unseen due to sampling, offscreen texels and transparent source support do
    not become hidden truth. This is sprite material, not completed 3D anatomy.
    """
    result = {}
    resolution = owner_map.shape[0]
    for obj in frame["objects"]:
        image = images[obj["member"]]; h, w = image.shape[:2]
        yy, xx = np.nonzero(image[..., 3] >= .99)
        x, y, angle, sx, sy, _ = obj["pose"]
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        matrix = np.array([[c*sx, s*sy], [-s*sx, c*sy]])
        pivot = [obj["pivot"][0]*w, (1-obj["pivot"][1])*h]
        screen = (np.stack((xx+.5, yy+.5), -1)-pivot) @ matrix.T + [x, -y] + offset
        cells = np.floor(screen).astype(int)
        valid = ((cells >= 0) & (cells < resolution)).all(axis=1)
        ix, iy = cells[valid, 0], cells[valid, 1]
        occluded = robust[iy, ix] & (owner_map[iy, ix] >= 0) & (owner_map[iy, ix] != owners.index(obj["owner"]))
        result[(obj["owner"], obj["member"])] = set(map(int, (yy[valid]*w + xx[valid])[occluded]))
    return result
