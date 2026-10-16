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
    exporter = Path(__file__).with_name("blender_authored_teacher_export_v1.py").resolve()

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
            }
        )

    final = {
        "schema": "RealSaS.IRISVNextFit8TeacherBatch.v1",
        "blender_version": version,
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
        f"receipt_sha256={sha256_file(path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
