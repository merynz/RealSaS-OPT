import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools.motion.materialize_quaternius_preset_v1 import (
    EXPECTED_CHANNELS,
    EXPECTED_CLIP_SCHEMA,
    EXPECTED_COORDINATE_FRAME,
    EXPECTED_EXTRACTOR_SCHEMA,
    EXPECTED_JOINT_FRAME,
    EXPECTED_ROOT_TRANSLATION_SEMANTICS,
    build_motion_fragment,
    patch_run_manifest,
    sha256,
    verify_archive_and_extract,
    verify_direct_fbx,
    verify_extraction_outputs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _clip_payload(clip_id: str, clip_kind: str, take: str, *, root_x: float = 0.0) -> dict:
    return {
        "schema": EXPECTED_CLIP_SCHEMA,
        "clip_id": clip_id,
        "clip_kind": clip_kind,
        "source_space": "SOURCE_RIG_TRACKS_V2",
        "coordinate_frame": EXPECTED_COORDINATE_FRAME,
        "joint_frame_semantics": EXPECTED_JOINT_FRAME,
        "duration_seconds": 1.0,
        "loop": clip_kind in {"IDLE", "RUN"},
        "channel_contract": list(EXPECTED_CHANNELS),
        "source_skeleton": [
            {
                "source_joint_id": "root",
                "parent_source_joint_id": None,
                "rest_position": [0.0, 0.0, 0.0],
            },
            {
                "source_joint_id": "child",
                "parent_source_joint_id": "root",
                "rest_position": [0.0, 0.0, 1.0],
            },
        ],
        "tracks": [
            {
                "source_joint_id": "root",
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "local_rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "local_translation_xyz": [root_x, 0.0, 0.0],
                        "local_scale_xyz": [1.0, 1.0, 1.0],
                    }
                ],
                "metadata": {},
            },
            {
                "source_joint_id": "child",
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "local_rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "local_translation_xyz": [0.0, 0.0, 0.0],
                        "local_scale_xyz": [1.0, 1.0, 1.0],
                    }
                ],
                "metadata": {},
            },
        ],
        "metadata": {
            "extractor_schema": EXPECTED_EXTRACTOR_SCHEMA,
            "source_fbx_sha256": "f" * 64,
            "license_evidence_sha256": "e" * 64,
            "source_take": take,
            "root_translation_semantics": EXPECTED_ROOT_TRANSLATION_SEMANTICS,
            "source_mesh_used_as_product_authority": False,
            "source_skin_used_as_product_authority": False,
            "source_material_used_as_product_authority": False,
            "source_texture_used_as_product_authority": False,
            "source_skeleton_used_as_final_target_authority": False,
            "motion_semantics_only": True,
        },
    }


def _spec() -> dict:
    return {
        "schema": "RealSaS.MotionPresetSourceManifest.v1",
        "status": "SEALED_SOURCE__EXTRACT_WITH_BLENDER_V2",
        "source": {
            "fbx_sha256": "f" * 64,
            "license_evidence_sha256": "e" * 64,
        },
        "clips": [
            {
                "clip_id": "demo_idle_v1",
                "clip_kind": "IDLE",
                "source_take": "HumanArmature|Idle",
                "output_filename": "idle.json",
            },
            {
                "clip_id": "demo_run_v1",
                "clip_kind": "RUN",
                "source_take": "HumanArmature|Run",
                "output_filename": "run.json",
            },
            {
                "clip_id": "demo_slash_v1",
                "clip_kind": "SLASH",
                "source_take": "HumanArmature|Run_swordAttack",
                "output_filename": "slash.json",
            },
        ],
    }


def _write_verified_fixture(tmp_path: Path, *, run_root_x: float = 0.0):
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, spec)
    rows = []
    for row in spec["clips"]:
        payload = _clip_payload(
            row["clip_id"],
            row["clip_kind"],
            row["source_take"],
            root_x=run_root_x if row["clip_kind"] == "RUN" else 0.0,
        )
        target = tmp_path / row["output_filename"]
        _write_json(target, payload)
        rows.append(
            {
                "clip_id": row["clip_id"],
                "clip_kind": row["clip_kind"],
                "source_take": row["source_take"],
                "path": str(target),
                "sha256": sha256(target),
                "payload_sha256": "0" * 64,
            }
        )
    receipt = {
        "schema": "RealSaS.MotionPresetExtractionReceipt.v1",
        "status": "PASS",
        "source_fbx_sha256": "f" * 64,
        "spec_sha256": sha256(spec_path),
        "extractor_schema": EXPECTED_EXTRACTOR_SCHEMA,
        "outputs": rows,
        "source_mesh_skin_appearance_product_authority": False,
    }
    _write_json(tmp_path / "EXTRACTION_RECEIPT.json", receipt)
    return spec_path, spec


def test_archive_member_and_hash_verification_is_exact(tmp_path):
    archive = tmp_path / "source.zip"
    fbx_bytes = b"fbx-bytes"
    license_bytes = b"license-bytes"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Knight/FBX/KnightCharacter.fbx", fbx_bytes)
        zf.writestr("Knight/License.docx", license_bytes)
    spec = {
        "source": {
            "archive_sha256": sha256(archive),
            "fbx_member": "Knight/FBX/KnightCharacter.fbx",
            "fbx_sha256": hashlib.sha256(fbx_bytes).hexdigest(),
            "license_member": "Knight/License.docx",
            "license_evidence_sha256": hashlib.sha256(license_bytes).hexdigest(),
        }
    }
    fbx, license_path = verify_archive_and_extract(
        archive=archive,
        spec=spec,
        temp_root=tmp_path / "temp",
    )
    assert fbx.read_bytes() == fbx_bytes
    assert license_path.read_bytes() == license_bytes


def test_direct_fbx_mode_requires_exact_sealed_hash(tmp_path):
    source_fbx = tmp_path / "KnightCharacter.fbx"
    source_fbx.write_bytes(b"exact-fbx")
    spec = {"source": {"fbx_sha256": hashlib.sha256(b"exact-fbx").hexdigest()}}
    assert verify_direct_fbx(source_fbx=source_fbx, spec=spec) == source_fbx.resolve()

    drifted = {"source": {"fbx_sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="MOTION_MATERIALIZER_FBX_SHA_DRIFT"):
        verify_direct_fbx(source_fbx=source_fbx, spec=drifted)


def test_verified_outputs_build_stage33_ready_fragment_with_authority_root(tmp_path):
    spec_path, spec = _write_verified_fixture(tmp_path)
    verified = verify_extraction_outputs(
        out_dir=tmp_path,
        spec_path=spec_path,
        spec=spec,
    )
    fragment = build_motion_fragment(
        spec=spec,
        verified_outputs=verified,
        authority_root=tmp_path,
    )
    assert [x["clip_id"] for x in fragment["motion"]["sources"]] == [
        "demo_idle_v1",
        "demo_run_v1",
        "demo_slash_v1",
    ]
    assert all(
        x["source_kind"] == "EXTERNAL_ARTIST_CLIP_V1"
        for x in fragment["motion"]["sources"]
    )
    assert all(
        x["file"]["path"].startswith("$REALSAS_AUTHORITY_ROOT/")
        for x in fragment["motion"]["sources"]
    )
    assert fragment["motion"]["compiler"]["root_trajectory_modes"] == {
        "demo_idle_v1": "IN_PLACE",
        "demo_run_v1": "IN_PLACE",
        "demo_slash_v1": "IN_PLACE",
    }


def test_materializer_fails_closed_if_expected_in_place_clip_has_root_motion(tmp_path):
    spec_path, spec = _write_verified_fixture(tmp_path, run_root_x=1e-3)
    with pytest.raises(RuntimeError, match="EXPECTED_IN_PLACE_ROOT_TRANSLATION"):
        verify_extraction_outputs(
            out_dir=tmp_path,
            spec_path=spec_path,
            spec=spec,
        )


def test_run_manifest_patch_requires_explicit_replace_for_existing_motion(tmp_path):
    spec_path, spec = _write_verified_fixture(tmp_path)
    verified = verify_extraction_outputs(
        out_dir=tmp_path,
        spec_path=spec_path,
        spec=spec,
    )
    fragment = build_motion_fragment(
        spec=spec,
        verified_outputs=verified,
        authority_root=tmp_path,
    )
    run_manifest = tmp_path / "run_manifest.json"
    _write_json(
        run_manifest,
        {
            "run_id": "SUBJECT2_KNIGHT_V1",
            "motion": {"sources": [{"clip_id": "legacy"}], "compiler": {}},
        },
    )

    with pytest.raises(RuntimeError, match="USE_REPLACE_MOTION"):
        patch_run_manifest(
            path=run_manifest,
            fragment=fragment,
            replace_motion=False,
        )

    before, after = patch_run_manifest(
        path=run_manifest,
        fragment=fragment,
        replace_motion=True,
    )
    assert before != after
    assert json.loads(run_manifest.read_text(encoding="utf-8"))["motion"] == fragment["motion"]
