"""Material-seam coefficient coupling court over the sealed V6 witness.

V7 showed that preserving contacts only after motion safety is too late: two
source charts can already disagree before the repair because their harmonic
motion coefficients were solved independently.  This court keeps the same
qualified source-cut relations, makes MATERIAL_CONTINUITY vertices share one
presentation coefficient row, recomputes the connected-palette baseline, then
runs the existing contact-aware safety and semantic-order runtime unchanged.

This is a causal research intervention only.  Mechanical M/G/W, palettes and
motion witnesses are immutable.
"""

from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.visual_contact_v1 import MATERIAL_CONTINUITY
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import (
    ConnectedPresentationPalette,
)
from compiler.realsas_compiler_services.orchestrator.adapters import (
    presentation_research_v4_impl as connected_pose,
)
from tools import replay_relation_court_v7 as v7


court = v7.court
_original_relation_compile = court._relation_compile


def _material_equivalence_couple(blend, pairs, codes):
    value = np.asarray(blend, dtype=np.float64).copy()
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(codes, dtype=np.int8)
    material = pairs[codes == MATERIAL_CONTINUITY]
    if not len(material):
        return value, {
            "material_pair_count": 0,
            "material_equivalence_class_count": 0,
            "maximum_coefficient_l1_change": 0.0,
            "maximum_material_pair_l1_after": 0.0,
        }

    vertices = sorted(set(map(int, material.reshape(-1).tolist())))
    parent = {v: v for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for a, b in material.tolist():
        union(a, b)
    groups = {}
    for vertex in vertices:
        groups.setdefault(find(vertex), []).append(vertex)

    original = value.copy()
    for group in groups.values():
        shared = np.mean(original[np.asarray(group, dtype=np.int64)], axis=0)
        shared = np.maximum(shared, 0.0)
        total = float(shared.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("MATERIAL_SEAM_COEFFICIENT_SIMPLEX_INVALID")
        shared /= total
        value[np.asarray(group, dtype=np.int64)] = shared

    delta = np.abs(value - original).sum(axis=1)
    pair_l1 = np.abs(value[material[:, 0]] - value[material[:, 1]]).sum(axis=1)
    return value, {
        "material_pair_count": int(len(material)),
        "material_equivalence_class_count": int(len(groups)),
        "maximum_coefficient_l1_change": float(np.max(delta, initial=0.0)),
        "maximum_material_pair_l1_after": float(np.max(pair_l1, initial=0.0)),
    }


def _coupled_relation_compile(topology, projection, arrays):
    target_map = court._attachment_target_map(topology)
    coupling = {}

    # Stage37-like relation evidence is evaluated first from the sealed V6
    # coefficients.  Only material relations are used to couple coefficients.
    for view in projection.views:
        vi = int(view.view_index)
        binding = court.domain_binding_from_arrays(arrays, vi)
        rest = np.asarray(arrays[f"view_{vi}_rest_positions"], dtype=np.float64)
        faces = np.asarray(arrays[f"view_{vi}_faces"], dtype=np.int64)
        owners = np.asarray(
            arrays[f"view_{vi}_vertex_attachment_owner"], dtype=np.int32
        )
        blend = np.asarray(
            arrays[f"view_{vi}_motion_blend_coefficients"], dtype=np.float64
        )
        contacts = court.qualify_visual_contacts(
            view_index=vi,
            rest_positions=rest,
            visual_faces=faces,
            domain_id=binding["domain_id"],
            anchor_vertex=binding["anchor_vertex"],
            anchor_mechanical_vertices=binding["anchor_mechanical_vertices"],
            motion_blend_coefficients=blend,
            vertex_attachment_owner=owners,
            attachment_target_joint_by_owner=target_map,
        )
        coupled, diagnostic = _material_equivalence_couple(
            blend, contacts.pairs, contacts.relation_codes
        )
        arrays[f"view_{vi}_pre_contact_motion_blend_coefficients"] = blend.copy()
        arrays[f"view_{vi}_motion_blend_coefficients"] = coupled
        coupling[f"V{vi}"] = diagnostic

        # Re-evaluate the exact connected-palette baseline with only the
        # presentation coefficients changed.  V6 already sealed these palettes;
        # no rig/skin/motion inference occurs here.
        for clip in projection.clips:
            p = clip.array_prefix
            old = np.asarray(
                arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"],
                dtype=np.float64,
            ).copy()
            arrays[f"{p}_view_{vi}_pre_contact_motion_safety_baseline_positions"] = old
            frames = []
            rotations = np.asarray(
                arrays[f"{p}_view_{vi}_palette_rotations"], dtype=np.float64
            )
            translations = np.asarray(
                arrays[f"{p}_view_{vi}_palette_translations"], dtype=np.float64
            )
            depths = np.asarray(arrays[f"{p}_view_{vi}_depths"], dtype=np.float64)
            for fi in range(len(rotations)):
                palette = ConnectedPresentationPalette(
                    rotations[fi], translations[fi]
                )
                field = connected_pose._field(
                    np.c_[rest, depths[fi]],
                    palette,
                    rest,
                    coupled,
                    owners,
                    topology["attachments"],
                )
                frames.append(field[:, :2])
            arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"] = np.asarray(
                frames, dtype=np.float64
            )

    relation_contract, contact_summaries, order_summaries = _original_relation_compile(
        topology, projection, arrays
    )
    for view_id, row in coupling.items():
        contact_summaries[view_id]["material_coefficient_coupling"] = row
    relation_contract["material_coefficient_coupling"] = coupling
    return relation_contract, contact_summaries, order_summaries


court._relation_compile = _coupled_relation_compile


if __name__ == "__main__":
    court.main()
