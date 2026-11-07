#!/usr/bin/env python3
"""Minimal, bounded SOURCE-HIT visibility oracle consumer for Corpus370.

Demonstrates a causal, per-pixel editable hidden-layer decision using sealed
source-render intervention evidence. This is TEACHER-ONLY; neither source face
addresses nor this oracle output are V3.2 student inputs. It does not identify
persistent drawables, cuts, mechanical owners, or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.corpus.court_corpus370_source_locus_visibility_v1 import qualify
from tools.corpus.generic_source_locus_reciprocal_palette_v2 import TOL, distance
from tools.corpus.generic_source_locus_counterfactual_v1 import _rgba

SCHEMA = "RealSaS.Corpus370SourceHitOracleConsumer.v1"


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def _render_over(foreground, background, carrier_mode):
    a = _rgba(foreground, carrier_mode)
    b = _rgba(background, carrier_mode)
    opacity = a[3]
    return [a[k] + (1.0 - opacity) * b[k] for k in range(3)] + [
        opacity + (1.0 - opacity) * b[3]
    ]


def consume(source_document: dict, visibility_document: dict | None = None) -> dict:
    """Re-qualify exact measured evidence; do not trust free-form labels."""
    visibility = qualify(source_document)
    if visibility_document is not None:
        require(isinstance(visibility_document, dict), "ORACLE_VISIBILITY_NOT_OBJECT")
        supplied = {k: v for k, v in visibility_document.items() if k != "parents"}
        require(supplied == visibility, "ORACLE_VISIBILITY_COURT_REPLAY_MISMATCH")
        require(visibility_document.get("training_authorized") is False,
                "ORACLE_FALSE_TRAINING_AUTHORITY")
    require(visibility["training_authorized"] is False
            and visibility["qualified_same_drawable_positives"] == 0
            and visibility["qualified_different_drawable_negatives"] == 0,
            "ORACLE_FORGED_DRAWABLE_OWNERSHIP")
    require(len(source_document["rows"]) == len(visibility["rows"]),
            "ORACLE_PIXEL_CENSUS_DRIFT")
    mode = source_document.get("carrier_mode")
    results = []
    valid_count = 0
    max_replay_error = 0.0
    for original, qualified in zip(source_document["rows"], visibility["rows"]):
        if qualified["status"] != "MEASURED_SOURCE_HIT_VISIBILITY_CANDIDATE_ONLY":
            results.append({
                "status": "ABSTAIN_SOURCE_RAY_OR_RENDER_NOT_QUALIFIED",
                "teacher_oracle_only": True, "training_authorized": False,
            })
            continue
        targets = qualified["targets"]
        require(len(targets) == 2, "ORACLE_SOURCE_HIT_COUNT")
        require(qualified["view_index"] == original["view_index"]
                and qualified["pixel_x"] == original["pixel_x"]
                and qualified["pixel_y"] == original["pixel_y"]
                and qualified["source_sha256"] == source_document["source_sha256"],
                "ORACLE_SOURCE_RAY_IDENTITY_DRIFT")
        measured_full = _rgba(original["full_soup_rgba"], mode)
        independent_over = _render_over(original["standalone_rgba"],
                                        original["removal_rgba"], mode)
        error = max(abs(v - m) for v, m in zip(independent_over, measured_full))
        require(error <= TOL, "ORACLE_RENDER_COMPOSITING_REPLAY_DRIFT")
        max_replay_error = max(max_replay_error, error)
        top, hidden = targets
        if (top["visibility_label"] == "POSITIVE_VISIBLE_CONTRIBUTOR"
            and hidden["visibility_label"] == "NEGATIVE_OPAQUE_OCCLUDED_CONTRIBUTOR"):
            controls = original["generic_counterfactual_v2"]["rgba"]
            reveal = distance(controls["remove_a"],
                              controls["recolor_b_given_remove_a"], mode)
            require(reveal > TOL,
                    "ORACLE_HIDDEN_EDIT_TARGET_NOT_CAUSALLY_REVEALED")
            require(original["standalone_rgba"][3] >= 1 - 1e-6,
                    "ORACLE_FRONT_NOT_OPAQUE")
            result = {
                "status": "PASS_BOUNDED_OCCLUDED_SOURCE_EDIT_REVEAL",
                "render_decision": "SOURCE_HIT_A_FRONT__RETAIN_HIDDEN_SOURCE_B",
                "source_sha256": qualified["source_sha256"],
                "camera_sha256": qualified["camera_sha256"],
                "view_index": qualified["view_index"],
                "pixel_x": qualified["pixel_x"], "pixel_y": qualified["pixel_y"],
                "front_source_locus_teacher_only": top["source_face_teacher_only"],
                "hidden_source_locus_teacher_only": hidden["source_face_teacher_only"],
                "hidden_recolor_effect_given_front_removal": reveal,
                "measured_full_rgba_replay_max_error": error,
                "first_hit_only_baseline_would_discard_hidden_source": True,
                "same_drawable": "UNKNOWN", "different_drawable": "UNKNOWN",
                "cut": "UNKNOWN", "attachment": "UNKNOWN",
                "mechanical_owner": "UNKNOWN", "graphics_owner": "UNKNOWN",
                "teacher_oracle_only": True, "student_ids_exposed": False,
                "training_authorized": False,
            }
            valid_count += 1
        else:
            result = {
                "status": "ABSTAIN_EDITABLE_HIDDEN_SOURCE_NOT_PROVEN",
                "source_sha256": qualified["source_sha256"],
                "view_index": qualified["view_index"],
                "pixel_x": qualified["pixel_x"], "pixel_y": qualified["pixel_y"],
                "measured_full_rgba_replay_max_error": error,
                "first_hit_only_baseline_would_discard_hidden_source": None,
                "same_drawable": "UNKNOWN", "different_drawable": "UNKNOWN",
                "cut": "UNKNOWN", "attachment": "UNKNOWN",
                "mechanical_owner": "UNKNOWN", "graphics_owner": "UNKNOWN",
                "teacher_oracle_only": True, "student_ids_exposed": False,
                "training_authorized": False,
            }
        results.append(result)
    return {
        "schema": SCHEMA,
        "status": "BOUNDED_VISIBILITY_ORACLE_DECISION_COURT__NO_DRAWABLE_OR_TRAINING_AUTHORITY",
        "subject_id": source_document["subject_id"],
        "source_sha256": source_document["source_sha256"],
        "evaluated_pixel_count": len(results),
        "occluded_edit_sources_recoverable": valid_count,
        "abstain_pixel_count": len(results) - valid_count,
        "max_full_rgba_replay_error": max_replay_error,
        "first_hit_only_baseline_potential_hidden_source_losses": valid_count,
        "qualified_same_drawable_positive_pairs": 0,
        "qualified_different_drawable_negative_pairs": 0,
        "rows": results, "teacher_oracle_only": True,
        "graphics_owner_truth_minted": False, "training_authorized": False,
    }


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--visibility", type=Path, required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--expected-visibility-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(sha256(a.source) == a.expected_source_sha256,
            "ORACLE_SOURCE_MANIFEST_SHA_DRIFT")
    require(sha256(a.visibility) == a.expected_visibility_sha256,
            "ORACLE_VISIBILITY_RECEIPT_SHA_DRIFT")
    result = consume(json.loads(a.source.read_text()),
                     json.loads(a.visibility.read_text()))
    result["source_manifest_sha256"] = a.expected_source_sha256
    result["visibility_receipt_sha256"] = a.expected_visibility_sha256
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
