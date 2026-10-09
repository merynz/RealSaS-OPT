"""V4 contact-aware safety court plus explicit V6 presentation debug witnesses.

The debug outputs make the two remaining hypotheses observable instead of
inferring them from the beauty render: visible semantic-slot ownership, visible
source-domain ownership, and the qualified contact anchors at RUN/V6 frame0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools import replay_relation_court_v4 as v4


court = v4.court
_original_render_run_v6 = court._render_run_v6


def _id_color(value: int) -> tuple[int, int, int]:
    # Deterministic high-contrast diagnostic palette without carrying semantics.
    x = (int(value) + 1) * 2654435761 & 0xFFFFFFFF
    return (64 + ((x >> 0) & 127), 64 + ((x >> 8) & 127), 64 + ((x >> 16) & 127))


def _owner_debug(owner_face_index: np.ndarray, face_labels: np.ndarray) -> Image.Image:
    owner = np.asarray(owner_face_index, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int32)
    rgb = np.zeros((*owner.shape, 3), dtype=np.uint8)
    rgb[:] = np.asarray([34, 38, 46], dtype=np.uint8)
    valid = owner >= 0
    if np.any(valid):
        visible_labels = labels[owner[valid]]
        for label in np.unique(visible_labels):
            mask = np.zeros(owner.shape, dtype=bool)
            mask[valid] = visible_labels == int(label)
            rgb[mask] = np.asarray(_id_color(int(label)), dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _debug_render_run_v6(*, projection, arrays, package: Path, player: Path, out: Path):
    gif, metrics = _original_render_run_v6(
        projection=projection, arrays=arrays, package=package, player=player, out=out
    )
    clip = next(row for row in projection.clips if row.clip_id == "demo_run_v1")
    view = next(row for row in projection.views if int(row.view_index) == 6)
    vi = 6
    faces = np.asarray(arrays[f"view_{vi}_faces"], dtype=np.int64)
    binding = court.domain_binding_from_arrays(arrays, vi)
    domain_id = np.asarray(binding["domain_id"], dtype=np.int32)
    face_domains = domain_id[faces[:, 0]]
    face_slots = np.asarray(arrays[f"view_{vi}_semantic_face_slot"], dtype=np.int32)

    slot_frames = []
    domain_frames = []
    refs = []
    for fi in range(int(clip.frame_count)):
        ref = court._source_owned_visual_reference_frame(
            projection, arrays, clip=clip, view=view, frame_index=fi
        )
        refs.append(ref)
        slot_frames.append(
            _owner_debug(ref.owner_face_index, face_slots).resize(
                (768, 768), Image.Resampling.NEAREST
            )
        )
        domain_frames.append(
            _owner_debug(ref.owner_face_index, face_domains).resize(
                (768, 768), Image.Resampling.NEAREST
            )
        )

    slot_gif = out / "Knight_RUN_V6_SEMANTIC_SLOT_DEBUG_768.gif"
    domain_gif = out / "Knight_RUN_V6_DOMAIN_DEBUG_768.gif"
    for path, frames in ((slot_gif, slot_frames), (domain_gif, domain_frames)):
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 / 24,
            loop=0,
        )

    # Frame0 contact overlay and machine-readable relation table. Coordinates
    # are emitted in the native 256px source-view space and drawn at 3x.
    frame0_path = out / "Knight_RUN_V6_FRAME0_CONTACT_DEBUG_768.png"
    beauty = Image.open(gif)
    beauty.seek(0)
    overlay = beauty.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    positions = np.asarray(
        arrays[f"{clip.array_prefix}_view_{vi}_positions"][0], dtype=np.float64
    )
    pairs = np.asarray(
        arrays[f"view_{vi}_qualified_contact_pairs"], dtype=np.int64
    ).reshape(-1, 2)
    codes = np.asarray(
        arrays[f"view_{vi}_qualified_contact_relation_codes"], dtype=np.int8
    )
    rows = []
    relation_rgb = {
        0: (40, 255, 80),
        1: (255, 40, 220),
        2: (40, 220, 255),
    }
    for index, ((a, b), code) in enumerate(zip(pairs.tolist(), codes.tolist())):
        pa = positions[int(a)]
        pb = positions[int(b)]
        color = relation_rgb.get(int(code), (255, 255, 255))
        p0 = (float(pa[0]) * 3.0, float(pa[1]) * 3.0)
        p1 = (float(pb[0]) * 3.0, float(pb[1]) * 3.0)
        draw.line((p0, p1), fill=color, width=2)
        r = 2
        for px, py in (p0, p1):
            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=1)
        rows.append(
            {
                "pair_index": int(index),
                "relation_code": int(code),
                "vertex_a": int(a),
                "vertex_b": int(b),
                "domain_a": int(domain_id[int(a)]),
                "domain_b": int(domain_id[int(b)]),
                "semantic_slot_a": int(arrays[f"view_{vi}_semantic_vertex_slot"][int(a)]),
                "semantic_slot_b": int(arrays[f"view_{vi}_semantic_vertex_slot"][int(b)]),
                "frame0_a_xy": [float(pa[0]), float(pa[1])],
                "frame0_b_xy": [float(pb[0]), float(pb[1])],
                "frame0_gap_px": float(np.linalg.norm(pa - pb)),
            }
        )
    overlay.save(frame0_path)
    relation_path = out / "Knight_RUN_V6_FRAME0_CONTACT_DEBUG.json"
    relation_path.write_text(
        json.dumps(
            {
                "schema": "RealSaS.FastPresentationDebugWitness.v1",
                "view_id": view.view_id,
                "clip_id": clip.clip_id,
                "frame_index": 0,
                "native_resolution": int(view.camera["resolution"]),
                "relations": rows,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    metrics = {
        **metrics,
        "semantic_slot_debug_gif": str(slot_gif),
        "domain_debug_gif": str(domain_gif),
        "frame0_contact_debug_png": str(frame0_path),
        "frame0_contact_debug_json": str(relation_path),
    }
    return gif, metrics


court._render_run_v6 = _debug_render_run_v6


if __name__ == "__main__":
    court.main()
