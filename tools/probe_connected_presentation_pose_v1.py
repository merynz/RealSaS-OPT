#!/usr/bin/env python3
"""Compare palette relations on an exact exported witness; diagnostic only.

This does not invoke Stage45, execute a scientific Attempt, or qualify final
visual charts/appearance. The imported topology/projection/witness bindings are
verified before either palette is evaluated. No input bytes are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import slot_rigid_transform_2d
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import (
    ConnectedPresentationPalette, compile_connected_palette, prove_connected_palette, POLICY,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(baseline):
    projection = json.loads((baseline / "projection.json").read_text())
    topology = json.loads((baseline / "source_domains.json").read_text())
    witness_sha = digest(baseline / "sealed_witness_replay.npz")
    projection_sha = digest(baseline / "projection_arrays.npz")
    if (witness_sha != topology["witness"]["sha256"]
            or projection_sha != projection["projection_npz_sha256"]
            or projection["qualified_visual_presentation_binding_hash"] != topology["topology_hash"]
            or projection["mechanical_mesh_binding_hash"] != topology["mechanical_mesh_binding_hash"]
            or projection["dynamic_motion_binding_hash"] != topology["dynamic_motion_binding_hash"]):
        raise RuntimeError("CONNECTED_PALETTE_DIAGNOSTIC_BASELINE_BINDING_DRIFT")
    with np.load(baseline / "sealed_witness_replay.npz", allow_pickle=False) as data:
        witness = {key: data[key].copy() for key in data.files}
    rows = []
    for view in projection["views"]:
        vi = int(view["view_index"])
        camera = qualify_camera_v3(dict(view["camera"], resolution=view["source_width"]),
                                  view_id=f"V{vi}", view_index=vi)
        for clip in projection["clips"]:
            legacy_max = connected_max = root_max = 0.
            legacy_failed = connected_failed = 0
            matrices = witness[clip["array_prefix"] + "_skin_matrices_source"]
            if len(matrices) != clip["frame_count"]:
                raise RuntimeError("CONNECTED_PALETTE_DIAGNOSTIC_FRAME_MATRIX_DRIFT")
            for matrix in matrices:
                args = dict(axis_positions_source=witness["axis_positions_source"],
                    axis_parents=witness["axis_parents"], skin_matrices_source=matrix, camera=camera)
                transforms = [slot_rigid_transform_2d(slot_rest_xyz=point,
                    skin_matrix_source=transform, camera=camera)
                    for point, transform in zip(witness["axis_positions_source"], matrix)]
                legacy = ConnectedPresentationPalette(np.asarray([row[0] for row in transforms]),
                                                       np.asarray([row[1] for row in transforms]))
                old = prove_connected_palette(legacy, **args)
                new = prove_connected_palette(compile_connected_palette(**args), **args)
                legacy_failed += not old["connected_palette_relations_passed"]
                connected_failed += not new["connected_palette_relations_passed"]
                legacy_max = max(legacy_max, old["maximum_parent_child_relation_residual_px"])
                connected_max = max(connected_max, new["maximum_parent_child_relation_residual_px"])
                root_max = max(root_max, new["maximum_root_transport_residual_px"])
            rows.append(dict(clip_id=clip["clip_id"], view_id=f"V{vi}", frame_count=len(matrices),
                legacy_relation_failed_frames=legacy_failed, connected_relation_failed_frames=connected_failed,
                legacy_maximum_relation_residual_source_px=legacy_max,
                connected_maximum_relation_residual_source_px=connected_max,
                connected_maximum_root_transport_residual_source_px=root_max))
    return dict(schema="RealSaS.ConnectedPresentationPaletteDiagnostic.v1",
        qualification="UNQUALIFIED_LOCAL_DIAGNOSTIC_ONLY",
        baseline_projection_hash=projection["projection_hash"],
        baseline_projection_json_sha256=digest(baseline / "projection.json"),
        baseline_projection_npz_sha256=projection_sha, witness_npz_sha256=witness_sha,
        mechanical_mesh_binding_hash=topology["mechanical_mesh_binding_hash"],
        dynamic_motion_binding_hash=topology["dynamic_motion_binding_hash"],
        pose_policy_hash=content_sha256(POLICY),
        implementation_file_sha256={name: digest(ROOT / name) for name in (
            "compiler/realsas_compiler_core/visual_presentation_pose_v1.py",
            "compiler/realsas_compiler_core/visual_attachment_motion_v1.py",
            "tools/probe_connected_presentation_pose_v1.py")},
        numpy_version=np.__version__, source_coordinate_units="SOURCE_CAMERA_PIXELS",
        mechanics_modified=False, stage45_executed=False, native_render_executed=False,
        final_visual_chart_contacts_qualified=False, hidden_appearance_coverage_qualified=False, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2) + "\n")
    rows = value["rows"]
    print(json.dumps(dict(qualification=value["qualification"],
        frame_views=sum(row["frame_count"] for row in rows),
        legacy_failed_frame_views=sum(row["legacy_relation_failed_frames"] for row in rows),
        connected_failed_frame_views=sum(row["connected_relation_failed_frames"] for row in rows),
        legacy_maximum_relation_residual_source_px=max(row["legacy_maximum_relation_residual_source_px"] for row in rows),
        connected_maximum_relation_residual_source_px=max(row["connected_maximum_relation_residual_source_px"] for row in rows))))


if __name__ == "__main__":
    main()
