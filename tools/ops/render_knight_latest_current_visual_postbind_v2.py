from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.joint_frames_v2 import (
    derive_joint_frames_post_bind_v2,
    frame_set_hash_v2,
)
from compiler.realsas_compiler_core import motion_dynamic_proof_v2 as motion_proof
from compiler.realsas_compiler_core import visual_material_render_v1 as visual_render
from compiler.realsas_compiler_core.types import QualificationError
from tools.demo import render_knight_motion_preview_v1 as motion_preview
from tools.ops import render_knight_latest_current_visual_v1 as renderer

CAA_RUN = "KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010"
MAT_RUN = "KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010"
SKELETON_NAME = "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json"


def _load_stage_payload(ledger: dict, stage_id: str, schema: str) -> dict:
    row = next(x for x in ledger["stages"] if x["id"] == stage_id)
    out = next(x for x in row.get("outputs") or () if x.get("schema") == schema)
    return json.loads(Path(out["path"]).read_text())


def main() -> None:
    authority_root = Path(os.environ["AUTHORITY_ROOT"]).resolve()
    input_root = Path(os.environ["INPUT_ROOT"]).resolve()

    ledger = json.loads(
        (authority_root / "runs" / CAA_RUN / "ACTIVE_RUN_V2.json").read_text()
    )
    camera_set = qualified_camera_set_from_dict(
        _load_stage_payload(
            ledger,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda row: int(row.view_index)))
    skeleton = qualified_skeleton_from_dict(
        json.loads((input_root / SKELETON_NAME).read_text())
    )
    carrier_skin = qualified_carrier_skin_from_dict(
        json.loads(
            (
                authority_root
                / "runs"
                / MAT_RUN
                / "qualified_carrier_skin.json"
            ).read_text()
        )
    )

    frames, qualification = derive_joint_frames_post_bind_v2(
        skeleton,
        carrier_skin=carrier_skin,
        cameras=cameras,
    )
    qualification_hash = content_sha256(qualification)
    frame_hash = frame_set_hash_v2(
        frames,
        carrier_skin_lineage_hash=carrier_skin.skin_lineage_hash,
    )
    print(
        "POST_BIND_JOINT_FRAME_QUALIFICATION",
        json.dumps(
            {
                "status": qualification["status"],
                "schema": qualification["schema"],
                "coincident_edge_count": qualification["coincident_edge_count"],
                "events": qualification["events"],
                "qualification_content_hash": qualification_hash,
                "frame_set_hash": frame_hash,
                "carrier_skin_lineage_hash": carrier_skin.skin_lineage_hash,
                "subject_specific_code_used": qualification["subject_specific_code_used"],
                "invented_epsilon": qualification["invented_epsilon"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def _canonical_target_frames(_skeleton, *, cameras):
        if str(_skeleton.skeleton_lineage_hash) != str(skeleton.skeleton_lineage_hash):
            raise RuntimeError("POST_BIND_FRAME_SKELETON_DRIFT")
        return frames

    # Reuse the legacy demo motion-channel mapper, but force every target-frame
    # consumer to the exact compiler-qualified Stage35 post-bind frame set.
    motion_preview.derive_joint_frames_from_skeleton = _canonical_target_frames
    motion_proof.derive_joint_frames_from_skeleton = _canonical_target_frames

    # Capture the eight exact rest-space visual binding evaluations performed
    # during renderer setup. This also gives us the exact per-visual-vertex
    # mechanical affine domains consumed later by Stage42.
    original_eval = renderer.evaluate_region_visual_binding_v1
    eval_call_index = 0
    rest_eval_by_view: dict[int, dict] = {}

    def _capture_eval(*args, **kwargs):
        nonlocal eval_call_index
        result = original_eval(*args, **kwargs)
        if eval_call_index < 8:
            binding = args[0] if args else kwargs["binding"]
            rest_eval_by_view[eval_call_index] = {
                "positions": np.asarray(result, dtype=np.float64).copy(),
                "binding": binding,
                "mechanical": np.asarray(
                    kwargs["posed_mechanical_positions_xyz"], dtype=np.float64
                ).copy(),
                "camera": kwargs["camera"],
            }
        eval_call_index += 1
        return result

    renderer.evaluate_region_visual_binding_v1 = _capture_eval

    # Keep the sealed visual-depth policy unchanged. Diagnostic replays vary one
    # guard at a time only to classify the already-failing call; the production
    # call still fails under the canonical epsilon and four-fragment limit.
    original_render = renderer.render_visual_material
    call_index = 0
    clip_ids = ("demo_idle_v1", "demo_run_v1", "demo_slash_v1")

    def _classify_depth_call(kwargs):
        old_eps = visual_render.DEPTH_TIE_EPSILON
        old_layers = visual_render.MAXIMUM_FRAGMENT_LAYERS
        tie_present = False
        overflow_present = False
        minimum_layers = None
        try:
            visual_render.MAXIMUM_FRAGMENT_LAYERS = 1_000_000
            try:
                original_render(**kwargs)
            except QualificationError as diag:
                tie_present = str(diag) == "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW"

            visual_render.DEPTH_TIE_EPSILON = -1.0
            visual_render.MAXIMUM_FRAGMENT_LAYERS = old_layers
            try:
                original_render(**kwargs)
            except QualificationError as diag:
                overflow_present = str(diag) == "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW"

            if overflow_present:
                for layers in range(int(old_layers) + 1, 33):
                    visual_render.MAXIMUM_FRAGMENT_LAYERS = layers
                    try:
                        original_render(**kwargs)
                    except QualificationError as diag:
                        if str(diag) == "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW":
                            continue
                        raise
                    minimum_layers = layers
                    break
        finally:
            visual_render.DEPTH_TIE_EPSILON = old_eps
            visual_render.MAXIMUM_FRAGMENT_LAYERS = old_layers
        return tie_present, overflow_present, minimum_layers

    def _diagnostic_render(**kwargs):
        nonlocal call_index
        current = call_index
        call_index += 1
        try:
            return original_render(**kwargs)
        except QualificationError as exc:
            if str(exc) != "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW":
                raise

            clip_index = current // 80
            within_clip = current % 80
            frame_index = within_clip // 8
            view_index = within_clip % 8
            clip_id = (
                clip_ids[clip_index]
                if clip_index < len(clip_ids)
                else f"clip_{clip_index}"
            )

            posed_tie, posed_overflow, posed_min_layers = _classify_depth_call(kwargs)

            rest_tie = None
            rest_overflow = None
            rest_min_layers = None
            single_domain_tie = None
            single_domain_overflow = None
            single_domain_min_layers = None
            mixed_domain_tie = None
            mixed_domain_overflow = None
            mixed_domain_min_layers = None
            mixed_domain_face_count = None
            total_visual_face_count = None
            max_domains_per_visual_face = None

            rest = rest_eval_by_view.get(view_index)
            if rest is not None:
                projected = np.asarray(
                    renderer.project_points_xyz_v3(
                        rest["mechanical"], rest["camera"]
                    ),
                    dtype=np.float64,
                )
                binding = rest["binding"]
                face_indices = np.asarray(
                    binding["mechanical_face_indices"], dtype=np.int64
                )
                bary = np.asarray(
                    binding["mechanical_barycentric"], dtype=np.float64
                )
                rest_depths = np.sum(
                    projected[face_indices, 2] * bary,
                    axis=1,
                )
                rest_kwargs = dict(kwargs)
                rest_kwargs["positions"] = rest["positions"]
                rest_kwargs["depths"] = rest_depths
                rest_tie, rest_overflow, rest_min_layers = _classify_depth_call(
                    rest_kwargs
                )

                # F6 counterfactual: preserve positions/depths/material exactly,
                # changing only whether visual triangles whose three vertices are
                # bound to different mechanical affine faces participate. If the
                # overflow disappears for single-domain triangles, the dynamic
                # failure is owned by deformation-domain coherence rather than by
                # the four-layer policy itself.
                visual_faces = np.asarray(kwargs["faces"], dtype=np.int64)
                vertex_domains = np.sort(face_indices, axis=1)
                tri_domains = vertex_domains[visual_faces]
                domain_counts = np.asarray(
                    [
                        len({tuple(row.tolist()) for row in rows})
                        for rows in tri_domains
                    ],
                    dtype=np.int32,
                )
                single_mask = domain_counts == 1
                mixed_mask = ~single_mask
                total_visual_face_count = int(len(visual_faces))
                mixed_domain_face_count = int(np.count_nonzero(mixed_mask))
                max_domains_per_visual_face = int(domain_counts.max(initial=0))

                single_kwargs = dict(kwargs)
                single_kwargs["faces"] = visual_faces[single_mask]
                (
                    single_domain_tie,
                    single_domain_overflow,
                    single_domain_min_layers,
                ) = _classify_depth_call(single_kwargs)

                mixed_kwargs = dict(kwargs)
                mixed_kwargs["faces"] = visual_faces[mixed_mask]
                (
                    mixed_domain_tie,
                    mixed_domain_overflow,
                    mixed_domain_min_layers,
                ) = _classify_depth_call(mixed_kwargs)

            raise RuntimeError(
                "SEALED_VISUAL_DEPTH_FAIL::"
                f"clip={clip_id}::frame={frame_index}::view={view_index}::"
                f"posed_tie={posed_tie}::posed_overflow={posed_overflow}::"
                f"posed_minimum_fragment_layers={posed_min_layers}::"
                f"rest_tie={rest_tie}::rest_overflow={rest_overflow}::"
                f"rest_minimum_fragment_layers={rest_min_layers}::"
                f"total_visual_faces={total_visual_face_count}::"
                f"mixed_domain_faces={mixed_domain_face_count}::"
                f"max_domains_per_visual_face={max_domains_per_visual_face}::"
                f"single_domain_only_tie={single_domain_tie}::"
                f"single_domain_only_overflow={single_domain_overflow}::"
                f"single_domain_only_minimum_fragment_layers={single_domain_min_layers}::"
                f"mixed_domain_only_tie={mixed_domain_tie}::"
                f"mixed_domain_only_overflow={mixed_domain_overflow}::"
                f"mixed_domain_only_minimum_fragment_layers={mixed_domain_min_layers}"
            ) from exc

    renderer.render_visual_material = _diagnostic_render
    renderer.main()


if __name__ == "__main__":
    main()
