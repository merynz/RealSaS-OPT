from __future__ import annotations

"""Run all eight authored-motion teacher exports through one exact-bound entrypoint.

This wrapper must run on a machine with Blender. It verifies the canonical vNext
input binding before launching Blender, and verifies every produced receipt after
export. Teacher artifacts are training-only and never enter inference inputs.
"""

import argparse
import json
from pathlib import Path
import subprocess

from tools.iris_vnext.input_binding_v1 import (
    build_input_receipt_v1,
    load_binding_v1,
    sha256_file,
)


TEACHER_SOURCE_PATHS = (
    "tools/iris_vnext/export_fit8_authored_teachers_v1.py",
    "tools/iris_vnext/blender_authored_teacher_export_v1.py",
    "tools/iris_vnext/blender_authored_teacher_export_v2.py",
    "tools/iris_vnext/blender_authored_teacher_export_v3.py",
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blender", required=True)
    p.add_argument("--binding", type=Path, default=Path("canonical/VNEXT_FIT8_INPUT_BINDING_V1.json"))
    p.add_argument("--stage01-bundle", type=Path, required=True)
    p.add_argument("--stage01-root", type=Path, required=True)
    p.add_argument("--legacy-shard-dir", type=Path, required=True)
    p.add_argument("--legacy-cohort-manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-frames-per-clip", type=int, default=16)
    return p.parse_args()


def _teacher_source_provenance() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    out = {}
    for relative in TEACHER_SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError("VNEXT_TEACHER_SOURCE_MISSING:" + relative)
        out[relative] = sha256_file(path)
    return out


def main():
    args = _args()
    if args.max_frames_per_clip <= 0:
        raise ValueError("VNEXT_TEACHER_MAX_FRAMES_INVALID")
    receipt = build_input_receipt_v1(
        binding_path=args.binding,
        stage01_bundle=args.stage01_bundle,
        stage01_root=args.stage01_root,
        legacy_shard_dir=args.legacy_shard_dir,
        legacy_cohort_manifest=args.legacy_cohort_manifest,
    )
    binding = load_binding_v1(args.binding)
    root = args.stage01_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher_source_provenance = _teacher_source_provenance()
    # V3 preserves V2's exact action-name/slot transport and freezes the rest
    # triangulation while verifying pose-time vertex indexing plus source polygon
    # connectivity/materials. Pose-dependent Blender tessellation is not topology.
    exporter = Path(__file__).with_name("blender_authored_teacher_export_v3.py").resolve()

    version = subprocess.run(
        [args.blender, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0].strip()
    exported = []
    for row in binding["subjects"]:
        legacy_id = str(row["legacy_subject_id"])
        target = output / f"{legacy_id}.teacher.npz"
        admission = root / str(row["observation_admission_path_in_stage01"])
        source = root / str(row["raw_source_path_in_stage01"])
        cmd = [
            args.blender,
            "-b",
            "--python",
            str(exporter),
            "--",
            "--source",
            str(source),
            "--expected-source-sha256",
            str(row["raw_source_sha256"]),
            "--subject-id",
            legacy_id,
            "--observation-admission",
            str(admission),
            "--output",
            str(target),
            "--max-frames-per-clip",
            str(args.max_frames_per_clip),
        ]
        subprocess.run(cmd, check=True)
        teacher_receipt_path = target.with_suffix(target.suffix + ".json")
        if not teacher_receipt_path.is_file():
            raise RuntimeError(f"VNEXT_TEACHER_RECEIPT_MISSING:{legacy_id}")
        teacher = json.loads(teacher_receipt_path.read_text(encoding="utf-8"))
        if teacher.get("subject_id") != legacy_id:
            raise ValueError(f"VNEXT_TEACHER_RECEIPT_SUBJECT_DRIFT:{legacy_id}")
        if teacher.get("source", {}).get("sha256") != row["raw_source_sha256"]:
            raise ValueError(f"VNEXT_TEACHER_RECEIPT_SOURCE_DRIFT:{legacy_id}")
        if teacher.get("observation_admission_sha256") != row["observation_admission_sha256"]:
            raise ValueError(f"VNEXT_TEACHER_RECEIPT_ADMISSION_DRIFT:{legacy_id}")
        if int(teacher.get("rest_vertex_count", -1)) != int(row["raw_vertex_count"]):
            raise ValueError(f"VNEXT_TEACHER_REST_VERTEX_CENSUS_DRIFT:{legacy_id}")
        if tuple(map(str, teacher.get("native_motion_inventory_exported") or ())) != tuple(
            map(str, row["native_motion_inventory"])
        ):
            raise ValueError(f"VNEXT_TEACHER_NATIVE_MOTION_INVENTORY_DRIFT:{legacy_id}")
        if teacher.get("teacher_rig_skin_motion_allowed_as_inference_input") is not False:
            raise ValueError(f"VNEXT_TEACHER_INFERENCE_LEAK:{legacy_id}")
        if teacher.get("canonical_M_G_W_minted") is not False:
            raise ValueError(f"VNEXT_TEACHER_CANONICAL_AUTHORITY_VIOLATION:{legacy_id}")
        if sha256_file(target) != teacher.get("output", {}).get("sha256"):
            raise ValueError(f"VNEXT_TEACHER_PAYLOAD_HASH_DRIFT:{legacy_id}")
        exported.append(
            {
                "subject_id": row["subject_id"],
                "legacy_subject_id": legacy_id,
                "teacher_file": target.name,
                "teacher_sha256": sha256_file(target),
                "receipt_file": teacher_receipt_path.name,
                "receipt_sha256": sha256_file(teacher_receipt_path),
                "sampled_frame_count": int(teacher["sampled_frame_count"]),
                "rest_vertex_count": int(teacher["rest_vertex_count"]),
                "native_motion_count": len(teacher["native_motion_inventory_exported"]),
                "teacher_transport_revision": int(teacher.get("transport_revision", 1)),
            }
        )

    final = {
        "schema": "RealSaS.IRISVNextFit8TeacherBatch.v1",
        "blender_version": version,
        "blender_teacher_exporter": exporter.name,
        "blender_teacher_exporter_sha256": sha256_file(exporter),
        "teacher_source_provenance_schema": "RealSaS.IRISVNextFit8TeacherSourceProvenance.v1",
        "teacher_source_provenance": teacher_source_provenance,
        "verified_input_receipt": receipt,
        "subjects": exported,
        "teacher_inputs_in_inference_forward": False,
        "canonical_M_G_W_minted": False,
    }
    path = output / "TEACHER_BATCH_RECEIPT.json"
    path.write_text(json.dumps(final, sort_keys=True, indent=2), encoding="utf-8")
    print(
        "VNEXT_FIT8_TEACHER_BATCH_PASS",
        "subjects=8",
        f"teacher_sources={len(teacher_source_provenance)}",
        f"receipt_sha256={sha256_file(path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
