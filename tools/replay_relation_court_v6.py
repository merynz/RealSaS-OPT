"""V5 debug court plus two causal presentation diagnostics.

1. Measure each semantic slot in isolation at RUN/V6 frame0 and compare its
   drawable alpha support with final front-owner visibility.  This distinguishes
   "the limb is absent" from "the limb exists but composition suppresses it".
2. Enumerate exact coincident source-cut domain pairs before mechanical-support
   qualification and compare them with the qualified Stage37 contact graph.
   This exposes source seams that contact qualification filtered out.

Research-only; no product authority and no M/G/W mutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.visual_depth_v2 import render_visual_depth
from tools import replay_relation_court_v5 as v5


court = v5.court
_original_render_run_v6 = court._render_run_v6


def _boundary_vertices(faces: np.ndarray, domains: np.ndarray) -> dict[int, np.ndarray]:
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    out = {}
    for domain in np.unique(domains):
        rows = boundary[domains[boundary[:, 0]] == int(domain)]
        vertices = np.unique(rows) if len(rows) else np.flatnonzero(domains == int(domain))
        out[int(domain)] = np.asarray(vertices, dtype=np.int64)
    return out


def _exact_source_cut_table(*, rest, faces, domains, qualified_pairs, qualified_codes):
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    domains = np.asarray(domains, dtype=np.int32)
    qpairs = np.asarray(qualified_pairs, dtype=np.int64).reshape(-1, 2)
    qcodes = np.asarray(qualified_codes, dtype=np.int8)
    boundary = _boundary_vertices(faces, domains)

    qualified_domain_codes: dict[tuple[int, int], set[int]] = {}
    for (a, b), code in zip(qpairs.tolist(), qcodes.tolist()):
        key = tuple(sorted((int(domains[int(a)]), int(domains[int(b)]))))
        qualified_domain_codes.setdefault(key, set()).add(int(code))

    rows = []
    values = sorted(map(int, np.unique(domains)))
    for offset, da in enumerate(values):
        va = boundary[da]
        if not len(va):
            continue
        for db in values[offset + 1 :]:
            vb = boundary[db]
            if not len(vb):
                continue
            tree = cKDTree(rest[vb])
            distance, index = tree.query(rest[va], k=1)
            exact = np.flatnonzero(np.asarray(distance) <= 1.0e-9)
            if not len(exact):
                continue
            pairs = sorted(
                {
                    (int(va[int(i)]), int(vb[int(np.asarray(index)[int(i)])]))
                    for i in exact.tolist()
                }
            )
            key = (int(da), int(db))
            rows.append(
                {
                    "domain_a": int(da),
                    "domain_b": int(db),
                    "exact_source_cut_vertex_pair_count": int(len(pairs)),
                    "sample_vertex_pairs": [list(map(int, p)) for p in pairs[:8]],
                    "qualified": key in qualified_domain_codes,
                    "qualified_relation_codes": sorted(qualified_domain_codes.get(key, set())),
                }
            )
    return rows


def _isolated_slot_metrics(*, projection, arrays, clip, view, frame_index, ref):
    vi = int(view.view_index)
    faces = np.asarray(arrays[f"view_{vi}_faces"], dtype=np.int64)
    uv = np.asarray(arrays[f"view_{vi}_uv"], dtype=np.float64)
    slots = np.asarray(arrays[f"view_{vi}_semantic_face_slot"], dtype=np.int32)
    positions = np.asarray(
        arrays[f"{clip.array_prefix}_view_{vi}_positions"][frame_index], dtype=np.float64
    )
    canonical_key = f"{clip.array_prefix}_view_{vi}_canonical_depths"
    if canonical_key in arrays:
        depths = np.asarray(arrays[canonical_key][frame_index], dtype=np.float64)
    else:
        depths = np.asarray(arrays[f"{clip.array_prefix}_view_{vi}_depths"][frame_index], dtype=np.float64)
    texture = np.asarray(Image.open(Path(view.texture_path)).convert("RGBA"), dtype=np.uint8)
    resolution = int(view.camera["resolution"])
    final_owner = np.asarray(ref.owner_face_index, dtype=np.int64)

    rows = []
    isolated_masks = {}
    for slot in sorted(map(int, np.unique(slots))):
        selected_faces = np.flatnonzero(slots == slot)
        isolated = render_visual_depth(
            positions=positions,
            depths=depths,
            faces=faces[selected_faces],
            uv=uv,
            texture=texture,
            resolution=resolution,
        )
        mask = np.asarray(isolated.straight_rgba_u8[:, :, 3]) > 0
        isolated_masks[slot] = mask
        final_front = np.isin(final_owner, selected_faces)
        isolated_count = int(np.count_nonzero(mask))
        front_count = int(np.count_nonzero(final_front))
        rows.append(
            {
                "semantic_slot_id": int(slot),
                "face_count": int(len(selected_faces)),
                "isolated_alpha_pixel_count": isolated_count,
                "final_front_owner_pixel_count": front_count,
                "front_owner_to_isolated_ratio": (
                    float(front_count / isolated_count) if isolated_count else 1.0
                ),
            }
        )

    # Pairwise drawable overlap is independent of final ordering and identifies
    # exactly which slots compete for pixels in this frame.
    overlap_rows = []
    slot_ids = sorted(isolated_masks)
    for ai, a in enumerate(slot_ids):
        for b in slot_ids[ai + 1 :]:
            overlap = int(np.count_nonzero(isolated_masks[a] & isolated_masks[b]))
            if overlap:
                overlap_rows.append(
                    {
                        "semantic_slot_a": int(a),
                        "semantic_slot_b": int(b),
                        "drawable_overlap_pixel_count": overlap,
                    }
                )
    return rows, sorted(
        overlap_rows,
        key=lambda row: (-row["drawable_overlap_pixel_count"], row["semantic_slot_a"], row["semantic_slot_b"]),
    )


def _render_with_causal_debug(*, projection, arrays, package: Path, player: Path, out: Path):
    gif, metrics = _original_render_run_v6(
        projection=projection, arrays=arrays, package=package, player=player, out=out
    )
    clip = next(row for row in projection.clips if row.clip_id == "demo_run_v1")
    view = next(row for row in projection.views if int(row.view_index) == 6)
    vi = int(view.view_index)
    ref = court._source_owned_visual_reference_frame(
        projection, arrays, clip=clip, view=view, frame_index=0
    )
    slot_rows, overlap_rows = _isolated_slot_metrics(
        projection=projection,
        arrays=arrays,
        clip=clip,
        view=view,
        frame_index=0,
        ref=ref,
    )
    binding = court.domain_binding_from_arrays(arrays, vi)
    exact_rows = _exact_source_cut_table(
        rest=arrays[f"view_{vi}_rest_positions"],
        faces=arrays[f"view_{vi}_faces"],
        domains=binding["domain_id"],
        qualified_pairs=arrays[f"view_{vi}_qualified_contact_pairs"],
        qualified_codes=arrays[f"view_{vi}_qualified_contact_relation_codes"],
    )
    debug = {
        "schema": "RealSaS.PresentationCausalDebugWitness.v1",
        "clip_id": clip.clip_id,
        "view_id": view.view_id,
        "frame_index": 0,
        "semantic_slot_isolated_support": slot_rows,
        "semantic_slot_drawable_overlap": overlap_rows,
        "exact_source_cut_domain_pairs": exact_rows,
        "unqualified_exact_source_cut_domain_pair_count": int(
            sum(not row["qualified"] for row in exact_rows)
        ),
    }
    path = out / "Knight_RUN_V6_CAUSAL_DEBUG.json"
    path.write_text(json.dumps(debug, sort_keys=True, indent=2) + "\n")
    return gif, {
        **metrics,
        "causal_debug_json": str(path),
        "unqualified_exact_source_cut_domain_pair_count": debug[
            "unqualified_exact_source_cut_domain_pair_count"
        ],
    }


court._render_run_v6 = _render_with_causal_debug


if __name__ == "__main__":
    court.main()
