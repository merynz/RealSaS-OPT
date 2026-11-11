#!/usr/bin/env python3
"""Build a full Corpus370 V5 semantic-target CANDIDATE readback.

An annotation file contains *reviewed*, source-face-addressed semantic teaching
choices. This command does not infer semantics from source names, qualify a
family, touch the source roster, generate a split, or authorize training.

Usage:
  python tools/corpus/build_corpus370_semantic_teacher_v1.py \\
      --roster corpus370_observation_packets_v5.json \\
      --annotations corpus370_semantic_annotations_v1.json \\
      --output semantic_candidates_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.iris.truth.semantic_prior_contract_v1 import build_corpus370_candidate_cohort


DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "canonical" / "REALSAS_SEMANTIC_PRIOR_POLICY_V1.json"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError("SEMANTIC_PRIOR_V1_INPUT_OBJECT_REQUIRED")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = _load(args.policy)
    roster = _load(args.roster)
    annotations = _load(args.annotations)
    result = build_corpus370_candidate_cohort(
        policy=policy,
        roster=roster,
        annotations=annotations,
        require_370=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    print(json.dumps({
        "schema": result["schema"],
        "counts": result["counts"],
        "target_matrix_status": result["target_matrix_status"],
        "training_authorized": result["semantic_training_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
