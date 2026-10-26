#!/usr/bin/env python3
"""Dependency-free audit of current V3 occlusion supervision balance.

The probe scans non-test Python sources for GraphicsPairTruthRowV3 producers,
then verifies the current adapter admits only positive occlusion support while
training_v3 applies BCE to every valid relation target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3OcclusionTruthBalanceProbe.v1"
PRODUCER_TOKEN = "GraphicsPairTruthRowV3("
EXPECTED_PRODUCER = "models/iris/vnext/relation_truth_v3_adapter_v1.py"


def _py_sources(repo_root: Path):
    for root_name in ("models", "compiler", "tools"):
        root = repo_root / root_name
        if not root.exists():
            continue
        yield from root.rglob("*.py")


def run(repo_root: Path) -> dict:
    producers = []
    for path in _py_sources(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if PRODUCER_TOKEN in text:
            producers.append(rel)
    producers = sorted(set(producers))
    if producers != [EXPECTED_PRODUCER]:
        raise RuntimeError("OCCLUSION_TRUTH_PRODUCER_SET_DRIFT:" + ",".join(producers))

    adapter = (repo_root / EXPECTED_PRODUCER).read_text(encoding="utf-8")
    training = (repo_root / "models/iris/vnext/training_v3.py").read_text(encoding="utf-8")

    required_adapter_fragments = (
        'required = ("face_a", "face_b", "front_probability", "front_valid", "occlusion_positive_support")',
        'if np.any(occlusion != 1.0):',
        'raise ValueError("RELATION_TRUTH_V3_OCCLUSION_NOT_POSITIVE_SUPPORT_ONLY")',
        'occlusion_probability=QualifiedProbabilityTargetV2(float(occlusion[index])',
    )
    missing = [fragment for fragment in required_adapter_fragments if fragment not in adapter]
    if missing:
        raise RuntimeError("OCCLUSION_TRUTH_ADAPTER_SEMANTICS_DRIFT:" + repr(missing))
    if 'raw = F.binary_cross_entropy(' not in training:
        raise RuntimeError("OCCLUSION_TRUTH_BCE_OBJECTIVE_DRIFT")
    if 'RELATION_VIEW_KEYS_V3' not in training or '"occlusion_probability"' not in training:
        raise RuntimeError("OCCLUSION_TRUTH_RELATION_KEY_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "CURRENT_V3_OCCLUSION_NEGATIVE_SUPERVISION_ABSENT",
        "producer_sources": producers,
        "facts": {
            "current_pair_truth_requires_field": "occlusion_positive_support",
            "current_adapter_rejects_any_valid_occlusion_value_other_than_one": True,
            "current_valid_occlusion_target_set": [1.0],
            "current_relation_objective": "VALIDITY_MASKED_BINARY_CROSS_ENTROPY",
            "negative_occlusion_truth_producer_found": False,
        },
        "mathematical_consequence": {
            "statement": "for the occlusion sub-objective supplied only by this producer, predicting probability 1 on every valid sample minimizes BCE; the current truth path cannot teach calibrated negative/non-occluding cases",
            "scope": "current V3 pair-truth producer path only; this does not claim future Truth Foundry supervision is impossible",
        },
        "required_before_causal_truth_training": [
            "define qualified negative occlusion support and explicit N/A/abstention semantics",
            "report positive/negative/abstain counts by morphology, view, overlap and alpha/compositing stratum before optimizer step 0",
            "evaluate calibration (Brier/ECE) in addition to ranking/classification metrics",
        ],
        "training_executed": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.repo_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
