from __future__ import annotations

import json
import os
from pathlib import Path

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

    # Legacy demo retargeting helper is reused only for motion-channel mapping.
    # Its target-frame provider is replaced by the current compiler's post-bind
    # frame qualification, so coincident joints follow Stage35 authority rather
    # than the obsolete V1 bone-vector derivation.
    motion_preview.derive_joint_frames_from_skeleton = _canonical_target_frames
    renderer.main()


if __name__ == "__main__":
    main()
