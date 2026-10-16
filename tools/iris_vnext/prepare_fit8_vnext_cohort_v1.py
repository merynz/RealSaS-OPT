from __future__ import annotations

"""Authorized preparation entrypoint for the fresh IRIS vNext Fit8 cohort.

The lower-level builders remain deterministic machinery. This wrapper is the
authority boundary: exact Stage01/legacy inputs and the authored teacher batch
must all be hash-bound before V2 shard materialization is allowed.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.iris_vnext.input_binding_v1 import (
    build_input_receipt_v1,
    load_binding_v1,
    sha256_file,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", type=Path, default=Path("canonical/VNEXT_FIT8_INPUT_BINDING_V1.json"))
    p.add_argument("--stage01-bundle", type=Path, required=True)
    p.add_argument("--stage01-root", type=Path, required=True)
    p.add_argument("--legacy-shard-dir", type=Path, required=True)
    p.add_argument("--legacy-cohort-manifest", type=Path)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--surface-points", type=int, default=4096)
    p.add_argument("--mechanical-points", type=int, default=8192)
    p.add_argument("--appearance-points-per-view", type=int, default=4096)
    p.add_argument("--relation-pairs-per-view", type=int, default=2048)
    p.add_argument("--topology-target-tokens", type=int, default=512)
    p.add_argument("--topology-trajectory-tolerance", type=float, default=1.0e-7)
    p.add_argument("--ray-chunk", type=int, default=64)
    return p.parse_args()


def _verify_teacher_batch(binding, teacher_dir: Path) -> dict:
    receipt_path = teacher_dir / "TEACHER_BATCH_RECEIPT.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("VNEXT_TEACHER_BATCH_RECEIPT_MISSING")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "RealSaS.IRISVNextFit8TeacherBatch.v1":
        raise ValueError("VNEXT_TEACHER_BATCH_SCHEMA_INVALID")
    if receipt.get("teacher_inputs_in_inference_forward") is not False:
        raise ValueError("VNEXT_TEACHER_BATCH_INFERENCE_LEAK")
    if receipt.get("canonical_M_G_W_minted") is not False:
        raise ValueError("VNEXT_TEACHER_BATCH_AUTHORITY_VIOLATION")
    rows = {str(row["legacy_subject_id"]): row for row in receipt.get("subjects") or ()}
    if len(rows) != 8:
        raise ValueError("VNEXT_TEACHER_BATCH_SUBJECT_COUNT_DRIFT")
    verified = []
    for source in binding["subjects"]:
        legacy_id = str(source["legacy_subject_id"])
        if legacy_id not in rows:
            raise ValueError(f"VNEXT_TEACHER_BATCH_SUBJECT_MISSING:{legacy_id}")
        row = rows[legacy_id]
        teacher = teacher_dir / str(row["teacher_file"])
        teacher_receipt = teacher_dir / str(row["receipt_file"])
        if sha256_file(teacher) != row["teacher_sha256"]:
            raise ValueError(f"VNEXT_TEACHER_BATCH_PAYLOAD_HASH_DRIFT:{legacy_id}")
        if sha256_file(teacher_receipt) != row["receipt_sha256"]:
            raise ValueError(f"VNEXT_TEACHER_BATCH_RECEIPT_HASH_DRIFT:{legacy_id}")
        value = json.loads(teacher_receipt.read_text(encoding="utf-8"))
        if value.get("source", {}).get("sha256") != source["raw_source_sha256"]:
            raise ValueError(f"VNEXT_TEACHER_BATCH_SOURCE_DRIFT:{legacy_id}")
        if value.get("observation_admission_sha256") != source["observation_admission_sha256"]:
            raise ValueError(f"VNEXT_TEACHER_BATCH_ADMISSION_DRIFT:{legacy_id}")
        verified.append(
            {
                "subject_id": source["subject_id"],
                "legacy_subject_id": legacy_id,
                "teacher_sha256": row["teacher_sha256"],
                "receipt_sha256": row["receipt_sha256"],
                "sampled_frame_count": int(row["sampled_frame_count"]),
            }
        )
    return {
        "batch_receipt_sha256": sha256_file(receipt_path),
        "blender_version": receipt.get("blender_version"),
        "subjects": verified,
    }


def main():
    args = _args()
    binding = load_binding_v1(args.binding)
    input_receipt = build_input_receipt_v1(
        binding_path=args.binding,
        stage01_bundle=args.stage01_bundle,
        stage01_root=args.stage01_root,
        legacy_shard_dir=args.legacy_shard_dir,
        legacy_cohort_manifest=args.legacy_cohort_manifest,
    )
    teacher_receipt = _verify_teacher_batch(binding, args.teacher_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    builder = Path(__file__).with_name("build_fit8_vnext_shards_v2.py").resolve()
    cmd = [
        sys.executable,
        str(builder),
        "--legacy-shard-dir",
        str(args.legacy_shard_dir.resolve()),
        "--teacher-dir",
        str(args.teacher_dir.resolve()),
        "--stage01-root",
        str(args.stage01_root.resolve()),
        "--output-dir",
        str(output),
        "--surface-points",
        str(args.surface_points),
        "--mechanical-points",
        str(args.mechanical_points),
        "--appearance-points-per-view",
        str(args.appearance_points_per_view),
        "--relation-pairs-per-view",
        str(args.relation_pairs_per_view),
        "--topology-target-tokens",
        str(args.topology_target_tokens),
        "--topology-trajectory-tolerance",
        str(args.topology_trajectory_tolerance),
        "--ray-chunk",
        str(args.ray_chunk),
    ]
    subprocess.run(cmd, check=True)
    manifest_path = output / "COHORT_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("VNEXT_COHORT_MANIFEST_MISSING_AFTER_BUILD")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("VNEXT_COHORT_MANIFEST_SCHEMA_DRIFT")
    if manifest.get("builder") != "tools/iris_vnext/build_fit8_vnext_shards_v2.py":
        raise ValueError("VNEXT_COHORT_BUILDER_AUTHORITY_DRIFT")
    if tuple(row["subject_id"] for row in manifest.get("subjects") or ()) != tuple(
        row["subject_id"] for row in binding["subjects"]
    ):
        raise ValueError("VNEXT_COHORT_MANIFEST_SUBJECT_DRIFT")
    if manifest.get("legacy_optimizer_state_reused") is not False:
        raise ValueError("VNEXT_COHORT_LEGACY_OPTIMIZER_REUSE_FORBIDDEN")
    if manifest.get("legacy_5400_checkpoint_reused") is not False:
        raise ValueError("VNEXT_COHORT_LEGACY_CHECKPOINT_REUSE_FORBIDDEN")
    final = {
        "schema": "RealSaS.IRISVNextFit8PreparationReceipt.v2",
        "verified_inputs": input_receipt,
        "verified_teacher_batch": teacher_receipt,
        "builder": "tools/iris_vnext/build_fit8_vnext_shards_v2.py",
        "builder_sha256": sha256_file(builder),
        "cohort_manifest_file": manifest_path.name,
        "cohort_manifest_sha256": sha256_file(manifest_path),
        "cohort_manifest_value_sha256": manifest["manifest_value_sha256"],
        "topology_coordinate_bins": int(manifest["topology_coordinate_bins"]),
        "topology_teacher_collision_policy": manifest["topology_teacher_collision_policy"],
        "subject_count": 8,
        "teacher_inputs_in_inference_forward": False,
        "legacy_optimizer_state_reused": False,
        "legacy_5400_checkpoint_reused": False,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
    }
    receipt_path = output / "PREPARATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(final, sort_keys=True, indent=2), encoding="utf-8")
    print(
        "VNEXT_FIT8_PREPARATION_V2_PASS",
        "subjects=8",
        f"coordinate_bins={manifest['topology_coordinate_bins']}",
        f"manifest={manifest['manifest_value_sha256']}",
        f"receipt_sha256={sha256_file(receipt_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
