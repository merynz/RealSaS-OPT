#!/usr/bin/env python3
"""Stage48 FIT8 bundle V4: exact pair-observability admission.

V3 binds Platform subject UUIDs to exact logical FIT8 keys and constructs the
objective-exact target packets.  V4 makes the final student-facing decision for
the pair family explicit: an attenuation objective is authorized only when at
least two distinct FIT8 subjects contain valid jointly-contributing same-ray
rows, which guarantees every 7/1 fold retains pair evidence.  If the exact
cohort contains zero such rows, the family is an explicit ABSTAIN rather than a
fake negative target or a reason to block the independently observable geometry,
mechanics and appearance families.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256
from tools.truth_foundry.build_fit8_presentation_truth_bundle_v1 import write_json
from tools.truth_foundry.build_fit8_presentation_truth_bundle_v3 import build as build_v3


def _require(value, code):
    if not value:
        raise ValueError(code)


def build(observation, *, api_root, artifact_root, repo_root, output):
    bundle = build_v3(
        observation,
        api_root=api_root,
        artifact_root=artifact_root,
        repo_root=repo_root,
        output=output,
    )
    root = Path(output).resolve()
    cohort = list(bundle.get("cohort") or ())
    pair_subjects = sum(
        int((row.get("counts") or {}).get("v31_observable_pair_count", 0)) > 0
        for row in cohort
    )
    total_pairs = int((bundle.get("aggregate_counts") or {}).get("observable_pairs", 0))
    _require(len(cohort) == 8, "FIT8_PRESENTATION_V4_EIGHT_SUBJECTS_REQUIRED")

    projection = dict(bundle.get("objective_projection") or {})
    abstain = list(projection.get("explicit_abstain") or ())
    if pair_subjects >= 2 and total_pairs > 0:
        pair_authorized = True
        pair_observability = "MECHANISM_CONDITIONAL__CROSS_FOLD_PAIR_EVIDENCE_AUTHORIZED"
        projection["pair_probe"] = ["occlusion_attenuation_probe__JOINTLY_CONTRIBUTING_SAME_RAY_ONLY"]
    elif pair_subjects == 0 and total_pairs == 0:
        pair_authorized = False
        pair_observability = "UNAVAILABLE__ZERO_JOINTLY_CONTRIBUTING_SAME_RAY_ROWS__ABSTAIN"
        projection["pair_probe"] = []
        marker = "occlusion_attenuation__NO_OBSERVATION_CONDITIONED_PAIR_EVIDENCE"
        if marker not in abstain:
            abstain.append(marker)
    else:
        raise ValueError("FIT8_PRESENTATION_V4_CROSS_FOLD_PAIR_EVIDENCE_INSUFFICIENT")

    projection["explicit_abstain"] = abstain
    bundle["objective_projection"] = projection
    bundle["pair_student_supervision_authorized"] = pair_authorized
    bundle["pair_evidence_subject_count"] = int(pair_subjects)
    bundle["pair_evidence_valid_rows"] = int(total_pairs)
    bundle["pair_observability"] = pair_observability
    bundle["bundle_builder_revision"] = "V4_OBJECTIVE_EXACT_PAIR_OBSERVABILITY_ADMISSION"
    bundle["missing_pair_truth_becomes_negative"] = False

    manifest_path = root / "FIT8_PRESENTATION_TRUTH_BUNDLE_V1.json"
    write_json(manifest_path, bundle)
    print(
        json.dumps(
            {
                "bundle": str(manifest_path),
                "sha256": sha256(manifest_path),
                "builder_revision": bundle["bundle_builder_revision"],
                "pair_student_supervision_authorized": pair_authorized,
                "pair_evidence_subject_count": pair_subjects,
                "pair_evidence_valid_rows": total_pairs,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "artifact-root", "repo-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    build(
        json.loads(Path(args.observation).read_text()),
        api_root=args.api_root,
        artifact_root=args.artifact_root,
        repo_root=args.repo_root,
        output=args.output,
    )
