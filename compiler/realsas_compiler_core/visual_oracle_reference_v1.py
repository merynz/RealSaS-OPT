"""Authored SCML -> explicit layered research evidence, not product authority.

Local hierarchy/spin interpolation is the producer. Cached source abs_* values
are deliberately NOT its pose input; the independent measurement reads those.
Only declared clips/mainline timestamps, linear/instant curves and opacity=1
are supported. Unsupported source features fail instead of approximating them.
"""
from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.visual_oracle_io_v1 import sha, file_ref


def checked_archive(section: dict):
    path = Path(section["archive_path"])
    if sha(path.read_bytes()) != section["archive_sha256"]:
        raise ValueError("ORACLE_ARCHIVE_DRIFT")
    z = zipfile.ZipFile(path)
    member = section["scml_member"]
    raw = z.read(member)
    if sha(raw) != section["scml_sha256"]:
        z.close()
        raise ValueError("ORACLE_SCML_DRIFT")
    return z, ET.fromstring(raw), PurePosixPath(member).parent


def spatial(element):
    return np.array([float(element.get(k, d)) for k, d in
                     (("x", "0"), ("y", "0"), ("angle", "0"),
                      ("scale_x", "1"), ("scale_y", "1"), ("a", "1"))])


def compose(local, parent):
    """Spriter component transform: translation, reflection-aware angle, scales."""
    x, y, angle, sx, sy, alpha = local
    px, py, pa, psx, psy, palpha = parent
    c, s = math.cos(math.radians(pa)), math.sin(math.radians(pa))
    return np.array([px + x * psx * c - y * psy * s,
                     py + x * psx * s + y * psy * c,
                     (pa + (-angle if psx * psy < 0 else angle)) % 360,
                     sx * psx, sy * psy, alpha * palpha])


def timeline_at(timeline, key_id, time_ms, length, looping):
    keys = timeline.findall("key")
    index = next(i for i, k in enumerate(keys) if k.get("id") == key_id)
    a = keys[index]
    b = keys[(index + 1) % len(keys)]
    elem_a, elem_b = list(a)[0], list(b)[0]
    v, w = spatial(elem_a), spatial(elem_b)
    ta, tb = int(a.get("time", 0)), int(b.get("time", 0))
    curve = a.get("curve_type", "linear")
    if curve not in ("linear", "instant"):
        raise ValueError("ORACLE_UNSUPPORTED_CURVE:" + curve)
    if index == len(keys) - 1 and not looping:
        return elem_a, v
    if tb <= ta:
        tb += length
    t = 0.0 if len(keys) == 1 or curve == "instant" else (time_ms - ta) / (tb - ta)
    if not 0 <= t <= 1:
        raise ValueError("ORACLE_TIMELINE_REFERENCE_TIME_INVALID")
    spin = int(a.get("spin", 1))
    delta = w[2] - v[2]
    if spin == 0:
        delta = 0
    elif spin > 0 and delta < 0:
        delta += 360
    elif spin < 0 and delta > 0:
        delta -= 360
    result = v + (w - v) * t
    result[2] = (v[2] + delta * t) % 360
    # Pivot changes require their own interpolation contract, not silent defaults.
    if any(elem_a.get(k) != elem_b.get(k) for k in ("pivot_x", "pivot_y")) and t != 0:
        raise ValueError("ORACLE_INTERPOLATED_PIVOT_UNSUPPORTED")
    return elem_a, result


def corners(pose, width, height, pivot):
    x, y, angle, sx, sy, _ = pose
    px, py = pivot
    local = np.array([[0, height], [width, height], [width, 0], [0, 0]], dtype=float)
    local -= [px * width, py * height]
    local *= [sx, sy]
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    world = local @ np.array([[c, s], [-s, c]]) + [x, y]
    world[:, 1] *= -1  # SCML y-up -> output y-down.
    return world


def produce(section: dict, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    z, root, base = checked_archive(section)
    with z:
        entity = next(e for e in root.findall("entity") if e.get("id") == str(section["entity_id"]))
        files = {(folder.get("id"), f.get("id")): f
                 for folder in root.findall("folder") for f in folder.findall("file")}
        frames, assets, owners = [], {}, {}
        for clip_name in section["clips"]:
            animation = next(a for a in entity.findall("animation") if a.get("name") == clip_name)
            timelines = {t.get("id"): t for t in animation.findall("timeline")}
            for mainkey in animation.find("mainline").findall("key"):
                time_ms = int(mainkey.get("time", 0))
                bones, objects, seen_z = {}, [], set()
                def evaluate(ref):
                    timeline = timelines[ref.get("timeline")]
                    elem, local = timeline_at(timeline, ref.get("key"), time_ms,
                        int(animation.get("length")), animation.get("looping", "true") == "true")
                    parent = bones[ref.get("parent")] if ref.get("parent", "-1") != "-1" else np.array([0, 0, 0, 1, 1, 1])
                    return timeline, elem, compose(local, parent)
                for ref in mainkey.findall("bone_ref"):
                    _, _, bones[ref.get("id")] = evaluate(ref)
                for ref in mainkey.findall("object_ref"):
                    timeline, elem, pose = evaluate(ref)
                    if timeline.get("object_type", "sprite") != "sprite" or abs(pose[5] - 1) > 1e-12:
                        raise ValueError("ORACLE_UNSUPPORTED_OBJECT_TYPE_OR_OPACITY")
                    owner = f"entity:{entity.get('id')}/object:{timeline.get('obj')}"
                    if timeline.get("obj") is None:
                        raise ValueError("ORACLE_STABLE_OWNER_ID_MISSING")
                    name = timeline.get("name")
                    if owner in owners and owners[owner] != name:
                        raise ValueError("ORACLE_OWNER_NAME_DRIFT")
                    owners[owner] = name
                    f = files[(elem.get("folder"), elem.get("file"))]
                    member = str(base / f.get("name"))
                    if ".." in PurePosixPath(member).parts:
                        raise ValueError("ORACLE_ASSET_PATH_ESCAPE")
                    if member not in assets:
                        raw = z.read(member)
                        image = Image.open(BytesIO(raw)).convert("RGBA")
                        if image.size != (int(f.get("width")), int(f.get("height"))):
                            raise ValueError("ORACLE_IMAGE_DIMENSION_DRIFT")
                        path = destination / (sha(raw) + ".png")
                        path.write_bytes(raw)
                        assets[member] = {"member": member, "ref": file_ref(path, "image/png"),
                                          "width": image.width, "height": image.height}
                    asset = assets[member]
                    pivot = [float(elem.get(k, f.get(k, d))) for k, d in (("pivot_x", "0"), ("pivot_y", "1"))]
                    order = int(ref.get("z_index"))
                    if order in seen_z:
                        raise ValueError("ORACLE_AMBIGUOUS_ORDER")
                    seen_z.add(order)
                    objects.append({"owner": owner, "name": name, "member": member,
                        "order": order, "pose": pose.tolist(), "pivot": pivot,
                        "corners": corners(pose, asset["width"], asset["height"], pivot).tolist(),
                        "parent_ref": ref.get("parent", "-1"), "source_ref_id": ref.get("id")})
                if len({o["owner"] for o in objects}) != len(objects):
                    raise ValueError("ORACLE_DUPLICATE_OWNER_IN_FRAME")
                frames.append({"clip": clip_name, "time_ms": time_ms,
                    "key_id": mainkey.get("id"), "length_ms": int(animation.get("length")),
                    "objects": sorted(objects, key=lambda o: o["order"])})
    bounds = np.concatenate([np.array(o["corners"]) for f in frames for o in f["objects"]])
    resolution, margin = int(section["resolution"]), int(section["margin_pixels"])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    if np.max(high - low) > resolution - 2 * margin:
        raise ValueError("ORACLE_FIXED_VIEWPORT_CROPS_SOURCE")
    offset = ((resolution - (high - low)) / 2 - low).tolist()
    return {"schema": "RealSaS.AuthoredVisualOracleEvidence.v1", "source": section,
        "scope": "ONE_AUTHORED_2D_VIEW__SOURCE_MAINLINE_KEYS__NOT_KNIGHT_3D_BIND",
        "resolution": resolution, "offset": offset, "frames": frames,
        "owners": owners, "assets": assets, "required_contact_cardinality": None,
        "required_grip_cardinality": None, "full_visual_qualification": False}


