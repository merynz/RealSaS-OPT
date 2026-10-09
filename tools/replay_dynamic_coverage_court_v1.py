"""Canonical dynamic-exposure court for source-owned visual presentation.

The existing Stage45 coverage predicate only asks whether a visual slot that
*already has source faces* produced pixels.  It cannot detect a body slot that
becomes mechanically visible during motion but has zero drawable faces in the
selected source view.  This court measures exactly that missing contract.

It reuses the sealed V6 M/G/W/motion witness, compiles the current V9
contact/order candidate in-memory, and compares canonical front-surface slot
visibility against source-owned visual slot support.  No appearance is invented
and no product authority is minted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from tools import replay_relation_court_v1 as base
from tools import replay_relation_court_v9 as v9


SCHEMA = "RealSaS.CanonicalDynamicExposureCourt.v1"
MIN_VISIBLE_PIXELS = 4


def _mechanical_mesh(witness):
    vertices = [
        SimpleNamespace(canonical_mesh_vertex_id=f"v{i}", P=tuple(map(float, xyz)))
        for i, xyz in enumerate(np.asarray(witness["vertices"], dtype=np.float64))
    ]
    faces = tuple(
        tuple(f"v{int(j)}" for j in face)
        for face in np.asarray(witness["faces"], dtype=np.int64)
    )
    return SimpleNamespace(vertices=vertices, faces=faces)


def _face_slots(weights, faces):
    weights = np.asarray(weights, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if weights.ndim != 2 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("DYNAMIC_EXPOSURE_INPUT_SHAPE_INVALID")
    return np.argmax(weights[faces].mean(axis=1), axis=1).astype(np.int32)


def _slot_histogram(values):
    if not len(values):
        return {}
    keys, counts = np.unique(np.asarray(values, dtype=np.int32), return_counts=True)
    return {int(k): int(v) for k, v in zip(keys.tolist(), counts.tolist())}


def run(*, evidence_root: Path, out: Path, view_index: int, clip_id: str):
    out.mkdir(parents=True, exist_ok=True)
    topology_path = base._candidate_stage(
        evidence_root, "37_QUALIFIED_PRESENTATION_STRUCTURE", "source_domains.json"
    )
    projection_path = base._candidate_stage(
        evidence_root, "42_RUNTIME_PROJECTION_AND_CAA_BINDING", "projection.json"
    )
    arrays_path = base._candidate_stage(
        evidence_root, "42_RUNTIME_PROJECTION_AND_CAA_BINDING", "projection_arrays.npz"
    )
    witness_path = base._candidate_stage(
        evidence_root, "37_QUALIFIED_PRESENTATION_STRUCTURE", "sealed_witness_replay.npz"
    )

    topology = json.loads(topology_path.read_text())
    projection = base.source_owned_visual_runtime_projection_from_dict(
        json.loads(projection_path.read_text())
    )
    arrays = base._load_npz(arrays_path)
    witness = base._load_npz(witness_path)

    # Compile current candidate relation state so visual semantic slots match
    # the exact contact/order candidate under test.  The returned contract is
    # diagnostic only; no package or native consumer is needed for this court.
    relation_contract, contact_summary, order_summary = v9.court._relation_compile(
        topology, projection, arrays
    )

    view = next(row for row in projection.views if int(row.view_index) == int(view_index))
    clip = next(row for row in projection.clips if row.clip_id == str(clip_id))
    p = clip.array_prefix
    vi = int(view.view_index)

    mechanical_weights = np.asarray(witness["canonical_motion_weights"], dtype=np.float64)
    mechanical_faces = np.asarray(witness["faces"], dtype=np.int64)
    mechanical_face_slot = _face_slots(mechanical_weights, mechanical_faces)
    mesh = _mechanical_mesh(witness)
    camera = qualify_camera_v3(
        dict(view.camera, resolution=int(view.source_width)),
        view_id=view.view_id,
        view_index=vi,
    )

    visual_face_slot = np.asarray(arrays[f"view_{vi}_semantic_face_slot"], dtype=np.int32)
    visual_source_face_counts = _slot_histogram(visual_face_slot)
    body_slot_count = int(mechanical_weights.shape[1])

    frame_rows = []
    missing_observations = []
    max_visible_by_slot = {slot: 0 for slot in range(body_slot_count)}
    for fi, posed_xyz in enumerate(np.asarray(witness[f"{p}_canonical_xyz"], dtype=np.float64)):
        visibility = rasterize_visible_owner(
            mesh,
            camera,
            width=int(view.source_width),
            height=int(view.source_height),
            positions=posed_xyz,
            max_layers=4,
        )
        owner = np.asarray(visibility.owner_face_index, dtype=np.int64)
        visible_faces = owner[owner >= 0]
        visible_slots = mechanical_face_slot[visible_faces] if len(visible_faces) else np.empty((0,), dtype=np.int32)
        mechanical_visible = _slot_histogram(visible_slots)
        frame_missing = []
        for slot, pixel_count in sorted(mechanical_visible.items()):
            max_visible_by_slot[slot] = max(max_visible_by_slot.get(slot, 0), int(pixel_count))
            source_faces = int(visual_source_face_counts.get(slot, 0))
            if int(pixel_count) >= MIN_VISIBLE_PIXELS and source_faces == 0:
                row = {
                    "frame_index": int(fi),
                    "semantic_slot_id": int(slot),
                    "canonical_visible_pixel_count": int(pixel_count),
                    "source_visual_face_count": 0,
                }
                frame_missing.append(row)
                missing_observations.append(row)
        frame_rows.append(
            {
                "frame_index": int(fi),
                "canonical_visible_pixels_by_slot": mechanical_visible,
                "missing_source_support": frame_missing,
            }
        )

    missing_slots = sorted({int(row["semantic_slot_id"]) for row in missing_observations})
    result = {
        "schema": SCHEMA,
        "status": "FAIL_DIAGNOSTIC" if missing_slots else "PASS_DIAGNOSTIC",
        "view_id": view.view_id,
        "view_index": vi,
        "clip_id": clip.clip_id,
        "body_slot_count": body_slot_count,
        "minimum_visible_pixels": MIN_VISIBLE_PIXELS,
        "visual_source_face_counts_by_slot": visual_source_face_counts,
        "canonical_max_visible_pixels_by_slot": {
            int(k): int(v) for k, v in max_visible_by_slot.items() if int(v) > 0
        },
        "missing_source_support_slot_ids": missing_slots,
        "missing_source_support_observation_count": int(len(missing_observations)),
        "frames": frame_rows,
        "contact_summary": contact_summary.get(f"V{vi}"),
        "order_summary": order_summary.get(f"V{vi}"),
        "presentation_relations_hash": relation_contract.get("presentation_relations_hash"),
        "mechanics_reopened": False,
        "appearance_completion_consumed": False,
        "product_authority": False,
        "interpretation": (
            "A missing slot means canonical posed geometry exposes a body motion owner "
            "for which this source-view visual mesh has no drawable faces. Draw order "
            "cannot repair that absence; admitted appearance/dynamic-exposure support is required."
        ),
    }
    path = out / "dynamic_exposure_court.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=6)
    parser.add_argument("--clip-id", default="demo_run_v1")
    args = parser.parse_args()
    result = run(
        evidence_root=args.evidence_root,
        out=args.out,
        view_index=args.view_index,
        clip_id=args.clip_id,
    )
    raise SystemExit(1 if result["status"] != "PASS_DIAGNOSTIC" else 0)


if __name__ == "__main__":
    main()
