"""Explicit layered research package for the existing native RSS consumer."""
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.runtime_package_v2 import write_rss_v2
from compiler.realsas_compiler_core.visual_oracle_io_v1 import sha


def package_frame(evidence, frame, path: Path, variant="baseline"):
    """Bake authored order into existing RSVM face order; no fake product seal.

    One-view research RSS uses the real native player, bypassing only the
    product packager's eight-camera/mechanical admission, explicitly unqualified.
    """
    if variant not in {"baseline", "wrong_order", "missing_layer", "wrong_owner", "wrong_pose"}:
        raise ValueError("ORACLE_UNKNOWN_VARIANT")
    resolution = evidence["resolution"]
    atlas_size, gutter = 1024, 2
    atlas = Image.new("RGBA", (atlas_size, atlas_size))
    uv, positions, faces, face_owners = [], [], [], []
    objects = list(frame["objects"])
    if variant == "wrong_order":
        objects.reverse()
    x = y = row_height = 0
    owner_ids = sorted(evidence["owners"])
    for obj in objects:
        asset = evidence["assets"][obj["member"]]
        raw = Path(asset["ref"]["path"]).read_bytes()
        if sha(raw) != asset["ref"]["sha256"]:
            raise ValueError("ORACLE_ASSET_BYTES_DRIFT")
        image = Image.open(BytesIO(raw)).convert("RGBA")
        w, h = image.size
        if x + w + 2 * gutter > atlas_size:
            x = 0; y += row_height; row_height = 0
        if y + h + 2 * gutter > atlas_size:
            raise ValueError("ORACLE_ATLAS_CAPACITY")
        if not (variant == "missing_layer" and obj["name"] == "cape"):
            atlas.paste(image, (x + gutter, y + gutter))
        lo = np.array([x + gutter - .5, y + gutter - .5]) / (atlas_size - 1)
        hi = (np.array([x + gutter + w, y + gutter + h]) - .5) / (atlas_size - 1)
        uv.extend([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        vertices = (np.array(obj["corners"]) + evidence["offset"]) * atlas_size / resolution - .5
        if variant == "wrong_pose" and obj["name"] == "head":
            vertices[:, 0] += 12 * atlas_size / resolution
        start = len(positions)
        positions.extend(vertices)
        faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
        owner = obj["owner"]
        if variant == "wrong_owner":
            owner = owner_ids[(owner_ids.index(owner) + 1) % len(owner_ids)]
        face_owners.extend([owner, owner])
        x += w + 2 * gutter; row_height = max(row_height, h + 2 * gutter)
    texture = BytesIO(); atlas.save(texture, format="PNG")
    entries = OrderedDict()
    entries["visual_mesh_v0.bin"] = (b"RSVM1\0\0\0" + struct.pack("<IIII", atlas_size, atlas_size,
        len(positions), len(faces)) + np.array(uv, dtype="<f8").tobytes()
        + np.array(faces, dtype="<u4").tobytes())
    entries["visual_texture_v0.png"] = texture.getvalue()
    entries["clip_0_v0.positions.bin"] = (b"RSVP1\0\0\0" + struct.pack("<II", 1, len(positions))
        + np.array(positions, dtype="<f8").tobytes())
    manifest = {"schema": "RealSaS.RuntimePackage.v2",
        "presentation_geometry_mode": "SOURCE_OWNED_VISUAL_PRESENTATION_V1",
        "mechanical_mesh_render_authority": 0, "runtime_visual_mesh_rebuild": 0,
        "runtime_binding_solve": 0, "runtime_generation": 0, "donor_search_at_runtime": 0,
        "playback_sampling_contract": "SEALED_FRAME_INDEX_ONLY",
        "view_selection_contract": "SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized": 0, "host_interpolation_authorized": 0,
        "presentation_state_execution_authorized": 0, "clipping_authorized": 0,
        "tint_order_visibility_authorized": 0, "mip_generation_authorized": 0,
        "texture_sampling_contract": "SOURCE_RGBA8_BILINEAR_STRAIGHT_TO_PM_V1",
        "clip_count": 1, "view_count": 1, "view.0.id": "V0",
        "view.0.mesh_entry": "visual_mesh_v0.bin", "view.0.texture_entry": "visual_texture_v0.png",
        "view.0.resolution": resolution, "clip.0.id": "oracle", "clip.0.frame_count": 1,
        "clip.0.view.0.positions_entry": "clip_0_v0.positions.bin",
        "research_oracle_only": 1, "full_visual_acceptance_passed": 0}
    entries["manifest.txt"] = ("\n".join(f"{k}={v}" for k, v in manifest.items()) + "\n").encode()
    # Receipt is inside the hash-bound package, not an unbound sidecar.
    entries["authority/oracle_face_owners.json"] = json.dumps(face_owners).encode()
    write_rss_v2(path, entries)
    return face_owners
