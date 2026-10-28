#!/usr/bin/env python3
"""Objective-exact FIT8 presentation truth bundle V2.

V1 performs the expensive Platform/CAS rebinding and target construction. This
wrapper keeps those bytes, then seals the student-facing manifest against the
actual target validity masks: privileged or overlap-factored appearance rows are
not counted as deterministic student supervision, and geometric front order
remains Compiler-only rather than being advertised as a learned pair probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.truth_foundry.build_fit8_presentation_truth_bundle_v1 import build as build_v1, write_json
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256


def _require(value, code):
    if not value:
        raise ValueError(code)


def build(observation, *, api_root, artifact_root, repo_root, output):
    bundle = build_v1(
        observation,
        api_root=api_root,
        artifact_root=artifact_root,
        repo_root=repo_root,
        output=output,
    )
    root = Path(output).resolve()
    aggregate = dict(bundle.get("aggregate_counts") or {})
    aggregate["appearance_rows_total"] = 0
    aggregate["observable_appearance_hits"] = 0
    aggregate["pair_rows_total"] = 0
    aggregate["observable_pairs"] = 0

    for row in bundle.get("cohort") or ():
        path = root / row["v31_target_file"]
        _require(path.is_file() and sha256(path) == row["v31_target_sha256"], "FIT8_PRESENTATION_V2_TARGET_DRIFT")
        with np.load(path, allow_pickle=False) as z:
            appearance_valid = np.asarray(z["appearance_valid"], dtype=np.bool_)
            pair_valid = np.asarray(z["pair_valid"], dtype=np.bool_)
            mechanical_valid = np.asarray(z["mechanical_valid"], dtype=np.bool_)
        observable_appearance = int(np.count_nonzero(appearance_valid))
        observable_pair = int(np.count_nonzero(pair_valid))
        _require(observable_appearance > 0, "FIT8_PRESENTATION_V2_NO_OBSERVABLE_APPEARANCE:" + row["subject_id"])
        counts = dict(row.get("counts") or {})
        counts["v31_appearance_row_count"] = int(len(appearance_valid))
        counts["v31_observable_appearance_hit_count"] = observable_appearance
        counts["v31_pair_count"] = int(len(pair_valid))
        counts["v31_observable_pair_count"] = observable_pair
        counts["v31_mechanical_face_count"] = int(np.count_nonzero(mechanical_valid))
        row["counts"] = counts
        aggregate["appearance_rows_total"] += int(len(appearance_valid))
        aggregate["observable_appearance_hits"] += observable_appearance
        aggregate["pair_rows_total"] += int(len(pair_valid))
        aggregate["observable_pairs"] += observable_pair

    _require(aggregate["observable_appearance_hits"] > 0, "FIT8_PRESENTATION_V2_EMPTY_OBSERVABLE_APPEARANCE")
    bundle["aggregate_counts"] = aggregate
    bundle["bundle_builder_revision"] = "V2_OBJECTIVE_EXACT_OBSERVABILITY_COUNTS"
    bundle["objective_projection"] = {
        "mechanical": ["nonrigid_response_conditional", "normalized_response_energy_conditional"],
        "appearance": ["source_equivalent_rgba__SINGLE_MEASURABLE_CONTRIBUTOR_RAYS_ONLY"],
        "pair_probe": ["occlusion_attenuation_probe__JOINTLY_CONTRIBUTING_SAME_RAY_ONLY"],
        "compiler_oracle_only": [
            "pose_conditioned_complete_ray_order",
            "geometric_front_order",
            "hit_removal_response",
            "full_source_over_stack",
        ],
        "explicit_abstain": [
            "semantic_part_class",
            "same_drawable_owner",
            "attachment_class",
            "grip_class",
            "physical_stiffness",
            "physical_impulse",
        ],
    }
    bundle["appearance_observability"] = (
        "DETERMINISTIC_TARGET_VALID_IFF_EXACTLY_ONE_HIT_EXCEEDS_SEALED_SOURCE_OVER_CONTRIBUTION_FLOOR"
    )
    bundle["pair_observability"] = (
        "MECHANISM_CONDITIONAL_TARGET_VALID_IFF_BOTH_SAME_RAY_ENDPOINTS_MEASURABLY_CONTRIBUTE"
    )
    bundle["hidden_or_overlap_factored_teacher_appearance_becomes_negative_or_valid"] = False
    manifest_path = root / "FIT8_PRESENTATION_TRUTH_BUNDLE_V1.json"
    write_json(manifest_path, bundle)
    print(
        json.dumps(
            {
                "bundle": str(manifest_path),
                "sha256": sha256(manifest_path),
                "observable_appearance_hits": aggregate["observable_appearance_hits"],
                "observable_pairs": aggregate["observable_pairs"],
                "builder_revision": bundle["bundle_builder_revision"],
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
