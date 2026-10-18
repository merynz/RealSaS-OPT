from __future__ import annotations

"""Fresh scientific IRIS vNext Fit8 runner.

Do not invoke until PREPARATION_RECEIPT and A100 engineering preflight both PASS.
This runner never consumes the legacy step-5400 optimizer/checkpoint. Completing
training is not a Fit8 PASS; scientific evaluation remains a separate court.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from models.iris.vnext.cohort_v1 import VNextFit8PolicyV1
from models.iris.vnext.cohort_v2 import IRISVNextFit8TrainerV2
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from tools.iris_vnext.input_binding_v1 import sha256_file


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", type=Path, required=True)
    p.add_argument("--a100-preflight", type=Path, required=True)
    p.add_argument("--flexicubes-root", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def _load_gate(cohort_dir: Path, preflight_path: Path):
    manifest_path = cohort_dir / "COHORT_MANIFEST.json"
    preparation_path = cohort_dir / "PREPARATION_RECEIPT.json"
    if not manifest_path.is_file() or not preparation_path.is_file():
        raise FileNotFoundError("VNEXT_FIT8_PREPARATION_RECEIPT_REQUIRED")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preparation.get("schema") != "RealSaS.IRISVNextFit8PreparationReceipt.v2":
        raise ValueError("VNEXT_FIT8_PREPARATION_SCHEMA_INVALID")
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("VNEXT_FIT8_COHORT_MANIFEST_SCHEMA_INVALID")
    if manifest.get("builder") != "tools/iris_vnext/build_fit8_vnext_shards_v2.py":
        raise ValueError("VNEXT_FIT8_COHORT_BUILDER_AUTHORITY_DRIFT")
    if preparation.get("cohort_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("VNEXT_FIT8_PREPARATION_MANIFEST_HASH_DRIFT")
    if preflight.get("schema") != "RealSaS.IRISVNextA100EngineeringPreflight.v1":
        raise ValueError("VNEXT_FIT8_A100_PREFLIGHT_SCHEMA_INVALID")
    if preflight.get("status") != "PASS":
        raise ValueError("VNEXT_FIT8_A100_PREFLIGHT_NOT_PASS")
    if preflight.get("preflight_checkpoint_reusable") is not False:
        raise ValueError("VNEXT_FIT8_PREFLIGHT_CHECKPOINT_REUSE_FORBIDDEN")
    if preflight.get("scientific_fit_started") is not False:
        raise ValueError("VNEXT_FIT8_PREFLIGHT_ALREADY_CLAIMS_SCIENTIFIC_START")
    if preflight.get("cohort", {}).get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("VNEXT_FIT8_PREFLIGHT_COHORT_HASH_DRIFT")
    if preflight.get("cohort", {}).get("preparation_receipt_sha256") != sha256_file(preparation_path):
        raise ValueError("VNEXT_FIT8_PREFLIGHT_PREPARATION_HASH_DRIFT")
    if int(preflight.get("cohort", {}).get("topology_coordinate_bins", -1)) != int(
        manifest.get("topology_coordinate_bins", -2)
    ):
        raise ValueError("VNEXT_FIT8_PREFLIGHT_TOPOLOGY_GRAMMAR_DRIFT")
    if manifest.get("legacy_optimizer_state_reused") is not False:
        raise ValueError("VNEXT_FIT8_LEGACY_OPTIMIZER_REUSE_FORBIDDEN")
    if manifest.get("legacy_5400_checkpoint_reused") is not False:
        raise ValueError("VNEXT_FIT8_LEGACY_5400_REUSE_FORBIDDEN")
    bound = dict(manifest)
    bound["preparation_receipt_sha256"] = sha256_file(preparation_path)
    bound["a100_preflight_report_sha256"] = sha256_file(preflight_path)
    bound["input_binding_sha256"] = preparation["verified_inputs"]["binding_sha256"]
    return bound, manifest_path, preparation_path


def _append_jsonl(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("VNEXT_FIT8_CUDA_REQUIRED")
    device_name = torch.cuda.get_device_name(0)
    if "A100" not in device_name.upper():
        raise RuntimeError(f"VNEXT_FIT8_A100_REQUIRED:{device_name}")
    cohort_dir = args.cohort_dir.resolve()
    bound_manifest, manifest_path, preparation_path = _load_gate(
        cohort_dir, args.a100_preflight.resolve()
    )
    cfg = IRISVNextConfigV1()
    if int(bound_manifest["topology_coordinate_bins"]) != int(
        cfg.tessa_decoder_config().coordinate_bins
    ):
        raise ValueError("VNEXT_FIT8_MODEL_TOPOLOGY_GRAMMAR_BIN_DRIFT")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_receipt = {
        "schema": "RealSaS.IRISVNextFit8ScientificStart.v2",
        "cohort_manifest_sha256": sha256_file(manifest_path),
        "preparation_receipt_sha256": sha256_file(preparation_path),
        "a100_preflight_sha256": sha256_file(args.a100_preflight.resolve()),
        "topology_coordinate_bins": int(bound_manifest["topology_coordinate_bins"]),
        "legacy_5400_checkpoint_reused": False,
        "legacy_optimizer_state_reused": False,
        "device": device_name,
        "scientific_fit_started": True,
        "scientific_fit_pass_claimed": False,
    }
    start_path = args.checkpoint_dir / "SCIENTIFIC_START.json"
    if start_path.exists():
        previous = json.loads(start_path.read_text(encoding="utf-8"))
        if previous != start_receipt:
            raise ValueError("VNEXT_FIT8_SCIENTIFIC_START_BINDING_DRIFT")
    else:
        start_path.write_text(json.dumps(start_receipt, sort_keys=True, indent=2), encoding="utf-8")

    trainer = IRISVNextFit8TrainerV2(
        data_root=cohort_dir,
        manifest=bound_manifest,
        checkpoint_root=args.checkpoint_dir,
        device="cuda",
        model_config=cfg,
        policy=VNextFit8PolicyV1(),
        flexicubes_root=args.flexicubes_root.resolve(),
    )
    while True:
        try:
            row = trainer.one_step()
        except StopIteration:
            trainer.save()
            complete = {
                "schema": "RealSaS.IRISVNextFit8TrainingComplete.v2",
                "global_step": int(trainer.step),
                "binding_sha256": trainer.binding_hash,
                "training_complete": True,
                "scientific_fit_pass_claimed": False,
            }
            (args.checkpoint_dir / "TRAINING_COMPLETE.json").write_text(
                json.dumps(complete, sort_keys=True, indent=2), encoding="utf-8"
            )
            print(
                "VNEXT_FIT8_TRAINING_COMPLETE__SCIENTIFIC_PASS_NOT_CLAIMED",
                f"global_step={trainer.step}",
                f"binding={trainer.binding_hash}",
                flush=True,
            )
            break
        row["binding_sha256"] = trainer.binding_hash
        _append_jsonl(args.log.resolve(), row)
        print(
            "VNEXT_FIT8_STEP",
            row["global_step"],
            row["stage"],
            f"loss={row['loss']:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
