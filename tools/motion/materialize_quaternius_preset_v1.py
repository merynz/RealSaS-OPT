from __future__ import annotations

"""Materialize the sealed Quaternius Knight motion preset without adding source FBX to git.

This host-side wrapper verifies the sealed archive/member identities, invokes the
Blender-only MotionSourceClip.v2 extractor, re-verifies every emitted clip, and
writes a Stage33-ready motion manifest fragment. Optionally it atomically patches
an existing run_manifest.json.

Examples:
  python tools/motion/materialize_quaternius_preset_v1.py \
    --archive "/path/to/Knight by @Quaternius-20260919T105040Z-1-001.zip" \
    --out-dir "$REALSAS_AUTHORITY_ROOT/runs/SUBJECT2_KNIGHT_V1/inputs/motion/quaternius_knight_v1"

  python tools/motion/materialize_quaternius_preset_v1.py \
    --source-fbx "/path/to/KnightCharacter.fbx" \
    --out-dir "/tmp/quaternius_knight_v1"
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "canonical" / "MOTION_PRESET_SOURCE_QUATERNIUS_KNIGHT_V1.json"
EXTRACTOR = REPO_ROOT / "tools" / "motion" / "blender_extract_motion_source_v2.py"
MATERIALIZER_SCHEMA = "RealSaS.MotionPresetMaterializationReceipt.v1"
EXPECTED_EXTRACTOR_SCHEMA = "RealSaS.BlenderMotionExtractor.v2"
EXPECTED_CLIP_SCHEMA = "RealSaS.MotionSourceClip.v2"
EXPECTED_CHANNELS = (
    "LOCAL_ROTATION_QUAT_XYZW",
    "LOCAL_TRANSLATION_XYZ",
    "LOCAL_SCALE_XYZ",
)
EXPECTED_COORDINATE_FRAME = "REALSAS_OBJECT_FRAME_V1"
EXPECTED_JOINT_FRAME = "REALSAS_DERIVED_JOINT_FRAME_V1"
EXPECTED_ROOT_TRANSLATION_SEMANTICS = (
    "LOCAL_DERIVED_JOINT_FRAME_NORMALIZED_BY_SOURCE_BODY_SCALE"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_canonical_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical_json_bytes(value))
    tmp.replace(path)


def write_pretty_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def _exact_member_bytes(zf: zipfile.ZipFile, member: str) -> bytes:
    matches = [name for name in zf.namelist() if name == member]
    _require(len(matches) == 1, f"MOTION_MATERIALIZER_MEMBER_CARDINALITY:{member}:{len(matches)}")
    return zf.read(matches[0])


def verify_archive_and_extract(
    *,
    archive: Path,
    spec: dict,
    temp_root: Path,
) -> tuple[Path, Path]:
    source = dict(spec.get("source") or {})
    expected_archive = str(source.get("archive_sha256") or "")
    _require(archive.is_file(), "MOTION_MATERIALIZER_ARCHIVE_MISSING")
    _require(len(expected_archive) == 64, "MOTION_MATERIALIZER_ARCHIVE_SHA_SPEC_INVALID")
    _require(sha256(archive) == expected_archive, "MOTION_MATERIALIZER_ARCHIVE_SHA_DRIFT")

    fbx_member = str(source.get("fbx_member") or "")
    license_member = str(source.get("license_member") or "")
    _require(fbx_member and license_member, "MOTION_MATERIALIZER_MEMBER_SPEC_MISSING")

    with zipfile.ZipFile(archive, "r") as zf:
        fbx_bytes = _exact_member_bytes(zf, fbx_member)
        license_bytes = _exact_member_bytes(zf, license_member)

    fbx_path = temp_root / "source" / Path(fbx_member).name
    license_path = temp_root / "source" / Path(license_member).name
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    fbx_path.write_bytes(fbx_bytes)
    license_path.write_bytes(license_bytes)

    _require(
        sha256(fbx_path) == str(source.get("fbx_sha256") or ""),
        "MOTION_MATERIALIZER_FBX_SHA_DRIFT",
    )
    _require(
        sha256(license_path) == str(source.get("license_evidence_sha256") or ""),
        "MOTION_MATERIALIZER_LICENSE_SHA_DRIFT",
    )
    return fbx_path, license_path


def verify_direct_fbx(*, source_fbx: Path, spec: dict) -> Path:
    source = dict(spec.get("source") or {})
    expected = str(source.get("fbx_sha256") or "")
    _require(source_fbx.is_file(), "MOTION_MATERIALIZER_DIRECT_FBX_MISSING")
    _require(len(expected) == 64, "MOTION_MATERIALIZER_FBX_SHA_SPEC_INVALID")
    _require(sha256(source_fbx) == expected, "MOTION_MATERIALIZER_FBX_SHA_DRIFT")
    return source_fbx.resolve()


def resolve_blender(raw: str | None) -> Path:
    candidate = (raw or os.environ.get("BLENDER_EXE") or "").strip()
    if candidate:
        direct = Path(candidate).expanduser()
        if direct.is_file():
            return direct.resolve()
        found = shutil.which(candidate)
        if found:
            return Path(found).resolve()
        raise RuntimeError(f"MOTION_MATERIALIZER_BLENDER_NOT_FOUND:{candidate}")

    found = shutil.which("blender")
    if found:
        return Path(found).resolve()

    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        root = program_files / "Blender Foundation"
        if root.is_dir():
            matches = sorted(root.glob("Blender */blender.exe"), reverse=True)
            if matches:
                return matches[0].resolve()

    for fixed in (
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
    ):
        if fixed.is_file():
            return fixed.resolve()

    raise RuntimeError("MOTION_MATERIALIZER_BLENDER_NOT_FOUND")


def _take_matches(actual: str, requested: str) -> bool:
    short = requested.split("|")[-1]
    return actual == requested or actual == short or actual.endswith("|" + short)


def _root_joint_id(payload: dict) -> str:
    rows = tuple(payload.get("source_skeleton") or ())
    roots = [
        str(row.get("source_joint_id") or "")
        for row in rows
        if row.get("parent_source_joint_id") in (None, "")
    ]
    _require(len(roots) == 1 and roots[0] != "", "MOTION_MATERIALIZER_ROOT_CARDINALITY")
    return roots[0]


def _assert_in_place_root(payload: dict, *, tolerance: float = 1e-7) -> None:
    root_id = _root_joint_id(payload)
    tracks = [
        row for row in tuple(payload.get("tracks") or ())
        if str(row.get("source_joint_id") or "") == root_id
    ]
    _require(len(tracks) == 1, "MOTION_MATERIALIZER_ROOT_TRACK_CARDINALITY")
    keys = tuple(tracks[0].get("keyframes") or ())
    _require(bool(keys), "MOTION_MATERIALIZER_ROOT_TRACK_EMPTY")
    for key in keys:
        xyz = tuple(float(x) for x in key.get("local_translation_xyz") or ())
        _require(len(xyz) == 3 and all(math.isfinite(x) for x in xyz), "MOTION_MATERIALIZER_ROOT_TRANSLATION_INVALID")
        norm = math.sqrt(sum(x * x for x in xyz))
        _require(norm <= tolerance, f"MOTION_MATERIALIZER_EXPECTED_IN_PLACE_ROOT_TRANSLATION:{norm}")


def verify_extraction_outputs(
    *,
    out_dir: Path,
    spec_path: Path,
    spec: dict,
) -> list[dict]:
    receipt_path = out_dir / "EXTRACTION_RECEIPT.json"
    _require(receipt_path.is_file(), "MOTION_MATERIALIZER_EXTRACTION_RECEIPT_MISSING")
    receipt = _load_json(receipt_path)
    _require(
        str(receipt.get("schema") or "") == "RealSaS.MotionPresetExtractionReceipt.v1",
        "MOTION_MATERIALIZER_EXTRACTION_RECEIPT_SCHEMA",
    )
    _require(receipt.get("status") == "PASS", "MOTION_MATERIALIZER_EXTRACTION_NOT_PASS")
    source = dict(spec.get("source") or {})
    _require(
        str(receipt.get("source_fbx_sha256") or "") == str(source.get("fbx_sha256") or ""),
        "MOTION_MATERIALIZER_RECEIPT_FBX_SHA_DRIFT",
    )
    _require(
        str(receipt.get("spec_sha256") or "") == sha256(spec_path),
        "MOTION_MATERIALIZER_RECEIPT_SPEC_SHA_DRIFT",
    )
    _require(
        str(receipt.get("extractor_schema") or "") == EXPECTED_EXTRACTOR_SCHEMA,
        "MOTION_MATERIALIZER_EXTRACTOR_SCHEMA_DRIFT",
    )
    _require(
        receipt.get("source_mesh_skin_appearance_product_authority") is False,
        "MOTION_MATERIALIZER_SOURCE_APPEARANCE_AUTHORITY_DRIFT",
    )

    receipt_rows = tuple(receipt.get("outputs") or ())
    by_id = {str(row.get("clip_id") or ""): dict(row) for row in receipt_rows}
    _require(
        len(by_id) == len(receipt_rows) == len(tuple(spec.get("clips") or ())),
        "MOTION_MATERIALIZER_OUTPUT_CARDINALITY",
    )

    verified: list[dict] = []
    for clip_spec in tuple(spec.get("clips") or ()):
        clip_id = str(clip_spec.get("clip_id") or "")
        row = by_id.get(clip_id)
        _require(row is not None, f"MOTION_MATERIALIZER_OUTPUT_MISSING:{clip_id}")
        path = (out_dir / str(clip_spec.get("output_filename") or "")).resolve()
        _require(path.is_file(), f"MOTION_MATERIALIZER_CLIP_FILE_MISSING:{clip_id}")
        digest = sha256(path)
        _require(digest == str(row.get("sha256") or ""), f"MOTION_MATERIALIZER_CLIP_SHA_DRIFT:{clip_id}")

        payload = _load_json(path)
        _require(str(payload.get("schema") or "") == EXPECTED_CLIP_SCHEMA, f"MOTION_MATERIALIZER_CLIP_SCHEMA:{clip_id}")
        _require(str(payload.get("clip_id") or "") == clip_id, f"MOTION_MATERIALIZER_CLIP_ID:{clip_id}")
        _require(
            str(payload.get("clip_kind") or "") == str(clip_spec.get("clip_kind") or ""),
            f"MOTION_MATERIALIZER_CLIP_KIND:{clip_id}",
        )
        _require(
            str(payload.get("source_space") or "") == "SOURCE_RIG_TRACKS_V2",
            f"MOTION_MATERIALIZER_SOURCE_SPACE:{clip_id}",
        )
        _require(
            str(payload.get("coordinate_frame") or "") == EXPECTED_COORDINATE_FRAME,
            f"MOTION_MATERIALIZER_COORDINATE_FRAME:{clip_id}",
        )
        _require(
            str(payload.get("joint_frame_semantics") or "") == EXPECTED_JOINT_FRAME,
            f"MOTION_MATERIALIZER_JOINT_FRAME:{clip_id}",
        )
        _require(
            tuple(map(str, payload.get("channel_contract") or ())) == EXPECTED_CHANNELS,
            f"MOTION_MATERIALIZER_CHANNEL_CONTRACT:{clip_id}",
        )
        _require(bool(payload.get("source_skeleton")), f"MOTION_MATERIALIZER_SOURCE_SKELETON_EMPTY:{clip_id}")
        _require(bool(payload.get("tracks")), f"MOTION_MATERIALIZER_TRACKS_EMPTY:{clip_id}")

        meta = dict(payload.get("metadata") or {})
        _require(
            str(meta.get("extractor_schema") or "") == EXPECTED_EXTRACTOR_SCHEMA,
            f"MOTION_MATERIALIZER_CLIP_EXTRACTOR_SCHEMA:{clip_id}",
        )
        _require(
            str(meta.get("source_fbx_sha256") or "") == str(source.get("fbx_sha256") or ""),
            f"MOTION_MATERIALIZER_CLIP_FBX_SHA:{clip_id}",
        )
        _require(
            str(meta.get("license_evidence_sha256") or "") == str(source.get("license_evidence_sha256") or ""),
            f"MOTION_MATERIALIZER_CLIP_LICENSE_SHA:{clip_id}",
        )
        _require(
            _take_matches(str(meta.get("source_take") or ""), str(clip_spec.get("source_take") or "")),
            f"MOTION_MATERIALIZER_CLIP_TAKE:{clip_id}",
        )
        _require(
            str(meta.get("root_translation_semantics") or "") == EXPECTED_ROOT_TRANSLATION_SEMANTICS,
            f"MOTION_MATERIALIZER_ROOT_TRANSLATION_SEMANTICS:{clip_id}",
        )
        for key in (
            "source_mesh_used_as_product_authority",
            "source_skin_used_as_product_authority",
            "source_material_used_as_product_authority",
            "source_texture_used_as_product_authority",
            "source_skeleton_used_as_final_target_authority",
        ):
            _require(meta.get(key) is False, f"MOTION_MATERIALIZER_FORBIDDEN_AUTHORITY:{clip_id}:{key}")
        _require(meta.get("motion_semantics_only") is True, f"MOTION_MATERIALIZER_MOTION_ONLY_DRIFT:{clip_id}")
        _assert_in_place_root(payload)

        verified.append(
            {
                "clip_id": clip_id,
                "clip_kind": str(clip_spec.get("clip_kind") or ""),
                "source_take": str(meta.get("source_take") or ""),
                "path": str(path),
                "sha256": digest,
            }
        )

    return verified


def _manifest_path(path: Path, authority_root: Path | None) -> str:
    resolved = path.resolve()
    if authority_root is not None:
        try:
            rel = resolved.relative_to(authority_root.resolve())
        except ValueError:
            pass
        else:
            return "$REALSAS_AUTHORITY_ROOT/" + rel.as_posix()
    return str(resolved)


def build_motion_fragment(
    *,
    spec: dict,
    verified_outputs: list[dict],
    authority_root: Path | None,
) -> dict:
    by_id = {row["clip_id"]: row for row in verified_outputs}
    sources = []
    root_modes = {}
    for clip_spec in tuple(spec.get("clips") or ()):
        clip_id = str(clip_spec["clip_id"])
        row = by_id[clip_id]
        sources.append(
            {
                "clip_id": clip_id,
                "clip_kind": str(clip_spec["clip_kind"]),
                "source_kind": "EXTERNAL_ARTIST_CLIP_V1",
                "file": {
                    "path": _manifest_path(Path(row["path"]), authority_root),
                    "sha256": row["sha256"],
                },
            }
        )
        root_modes[clip_id] = "IN_PLACE"
    return {
        "motion": {
            "sources": sources,
            "compiler": {
                "root_trajectory_modes": root_modes,
                "contacts": {},
            },
        }
    }


def patch_run_manifest(
    *,
    path: Path,
    fragment: dict,
    replace_motion: bool,
) -> tuple[str, str]:
    _require(path.is_file(), "MOTION_MATERIALIZER_RUN_MANIFEST_MISSING")
    before_sha = sha256(path)
    manifest = _load_json(path)
    new_motion = dict(fragment["motion"])
    old_motion = manifest.get("motion")
    if old_motion not in (None, {}, new_motion) and not replace_motion:
        raise RuntimeError("MOTION_MATERIALIZER_RUN_MANIFEST_MOTION_EXISTS__USE_REPLACE_MOTION")
    manifest["motion"] = new_motion
    write_pretty_json(path, manifest)
    return before_sha, sha256(path)


def run_blender(
    *,
    blender: Path,
    source_fbx: Path,
    spec_path: Path,
    out_dir: Path,
) -> None:
    command = [
        str(blender),
        "--background",
        "--python",
        str(EXTRACTOR),
        "--",
        "--source-fbx",
        str(source_fbx),
        "--spec",
        str(spec_path),
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"MOTION_MATERIALIZER_BLENDER_FAILED:{completed.returncode}")


def default_out_dir(run_id: str) -> Path:
    raw = os.environ.get("REALSAS_AUTHORITY_ROOT", "").strip()
    if not raw:
        raise RuntimeError("MOTION_MATERIALIZER_OUT_DIR_REQUIRED_WITHOUT_REALSAS_AUTHORITY_ROOT")
    return (
        Path(raw).expanduser().resolve()
        / "runs"
        / run_id
        / "inputs"
        / "motion"
        / "quaternius_knight_v1"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--archive", default="")
    source_group.add_argument("--source-fbx", default="")
    ap.add_argument("--blender", default="")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--run-id", default="SUBJECT2_KNIGHT_V1")
    ap.add_argument("--patch-run-manifest", default="")
    ap.add_argument("--replace-motion", action="store_true")
    args = ap.parse_args(argv)

    archive = Path(args.archive).expanduser().resolve() if str(args.archive).strip() else None
    direct_fbx = Path(args.source_fbx).expanduser().resolve() if str(args.source_fbx).strip() else None
    spec_path = Path(args.spec).expanduser().resolve()
    _require(spec_path.is_file(), "MOTION_MATERIALIZER_SPEC_MISSING")
    spec = _load_json(spec_path)
    _require(
        str(spec.get("schema") or "") == "RealSaS.MotionPresetSourceManifest.v1",
        "MOTION_MATERIALIZER_SPEC_SCHEMA",
    )
    _require(
        str(spec.get("status") or "") == "SEALED_SOURCE__EXTRACT_WITH_BLENDER_V2",
        "MOTION_MATERIALIZER_SPEC_STATUS",
    )

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else default_out_dir(str(args.run_id))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    blender = resolve_blender(args.blender)

    source_input_mode = "SEALED_ARCHIVE"
    archive_receipt = None
    source_fbx_receipt_path = ""
    with tempfile.TemporaryDirectory(prefix="realsas_motion_materialize_") as tmp:
        temp_root = Path(tmp)
        if archive is not None:
            source_fbx, _license = verify_archive_and_extract(
                archive=archive,
                spec=spec,
                temp_root=temp_root,
            )
            archive_receipt = {
                "path": str(archive),
                "sha256": sha256(archive),
            }
            source_fbx_receipt_path = f"{archive}!/{spec['source']['fbx_member']}"
        else:
            source_input_mode = "EXACT_FBX_SHA"
            source_fbx = verify_direct_fbx(source_fbx=direct_fbx, spec=spec)
            source_fbx_receipt_path = str(source_fbx)
        run_blender(
            blender=blender,
            source_fbx=source_fbx,
            spec_path=spec_path,
            out_dir=out_dir,
        )

    verified = verify_extraction_outputs(
        out_dir=out_dir,
        spec_path=spec_path,
        spec=spec,
    )

    raw_authority = os.environ.get("REALSAS_AUTHORITY_ROOT", "").strip()
    authority_root = Path(raw_authority).expanduser().resolve() if raw_authority else None
    fragment = build_motion_fragment(
        spec=spec,
        verified_outputs=verified,
        authority_root=authority_root,
    )
    fragment_path = out_dir / "MOTION_RUN_MANIFEST_FRAGMENT.json"
    write_pretty_json(fragment_path, fragment)

    run_manifest_result = None
    if str(args.patch_run_manifest).strip():
        run_manifest_path = Path(args.patch_run_manifest).expanduser().resolve()
        before_sha, after_sha = patch_run_manifest(
            path=run_manifest_path,
            fragment=fragment,
            replace_motion=bool(args.replace_motion),
        )
        run_manifest_result = {
            "path": str(run_manifest_path),
            "sha256_before": before_sha,
            "sha256_after": after_sha,
        }

    source = dict(spec["source"])
    receipt = {
        "schema": MATERIALIZER_SCHEMA,
        "status": "PASS",
        "source_input_mode": source_input_mode,
        "archive": archive_receipt,
        "source_fbx": {
            "path": source_fbx_receipt_path,
            "sha256": str(source["fbx_sha256"]),
        },
        "source_fbx_sha256": str(source["fbx_sha256"]),
        "license_evidence_sha256": str(source["license_evidence_sha256"]),
        "spec": {
            "path": str(spec_path),
            "sha256": sha256(spec_path),
        },
        "blender_executable": str(blender),
        "extractor": {
            "path": str(EXTRACTOR),
            "sha256": sha256(EXTRACTOR),
            "schema": EXPECTED_EXTRACTOR_SCHEMA,
        },
        "outputs": verified,
        "motion_manifest_fragment": {
            "path": str(fragment_path),
            "sha256": sha256(fragment_path),
        },
        "run_manifest_patch": run_manifest_result,
        "source_asset_committed_to_repository": False,
        "source_mesh_skin_material_texture_product_authority": False,
        "source_skeleton_final_target_authority": False,
    }
    materialization_path = out_dir / "MATERIALIZATION_RECEIPT.json"
    write_pretty_json(materialization_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
