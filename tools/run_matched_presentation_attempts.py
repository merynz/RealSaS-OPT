"""Matched V6 -> V7 presentation Attempts over the same sealed raw inputs.

Unlike the previous Stage42-only court, V7 intentionally changes Stage37 contact
qualification, Stage42 contact/composition, Stage43 runtime transport and Stage45
proof. M/G/W and the sealed motion witness remain byte-identical inputs.
"""
import argparse
from copy import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_scoped_presentation_attempt import run
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import POLICY as CONNECTED_POLICY
from compiler.realsas_compiler_core.visual_contact_v1 import POLICY as CONTACT_POLICY
from compiler.realsas_compiler_core.visual_contact_motion_v1 import POLICY as CONTACT_MOTION_POLICY
from compiler.realsas_compiler_core.visual_semantic_order_v1 import POLICY as SEMANTIC_ORDER_POLICY
from compiler.realsas_compiler_services.proof.causal_attribution import (
    ControlledInterventionEvidenceV1,
    proof_probe_fingerprint_v1,
    build_controlled_owner_attribution_v1,
)


def compare(out, baseline, candidate):
    before = json.loads((out / "baseline/presentation_proof.json").read_text())
    after = json.loads((out / "candidate/presentation_proof.json").read_text())
    same_inputs = baseline["subject_input_id"] == candidate["subject_input_id"]
    if not same_inputs:
        raise RuntimeError("MATCHED_PRESENTATION_SEALED_INPUT_DRIFT")

    protected = tuple(
        key
        for key in (
            "area_collapse_count",
            "condition_failure_count",
            "flipped_triangles",
            "edge_gt_4_count",
            "native_reference_mismatch_frame_views",
            "unresolved_depth_ties",
            "fragment_overflow",
        )
        if int(after.get(key, 0)) > int(before.get(key, 0))
    )
    probe = {
        "schema": "RealSaS.PresentationRelationClosureProbe.v2",
        "metric": "relational_presentation_passed",
        "units": "BOOLEAN",
        "required_predicates": [
            "qualified_contact_relations_passed",
            "qualified_dynamic_coverage_passed",
            "semantic_occlusion_passed",
            "setup_identity_passed",
            "frame0_relations_passed",
            "temporal_relations_passed",
        ],
    }
    signatures = [
        {"signature_id": "PRESENTATION_CONTACT_GAP", "failure_family": "QUALIFIED_CONTACT"},
        {"signature_id": "PRESENTATION_SEMANTIC_OCCLUSION", "failure_family": "DRAW_ORDER"},
        {"signature_id": "PRESENTATION_DYNAMIC_COVERAGE", "failure_family": "COVERAGE"},
    ]
    policies = [
        content_sha256(CONNECTED_POLICY),
        content_sha256(CONTACT_POLICY),
        content_sha256(CONTACT_MOTION_POLICY),
        content_sha256(SEMANTIC_ORDER_POLICY),
    ]
    evidence = ControlledInterventionEvidenceV1(
        intervention_id=candidate["attempt_id"],
        target_signature_id=signatures[0]["signature_id"],
        proof_domain="SCOPED_RESEARCH_PRESENTATION",
        owner_id="STAGE37_45_PRESENTATION_RELATIONS",
        baseline_source_product_state_hash=before["projection_hash"],
        baseline_measurement_report_hash=content_sha256(before),
        counterfactual_source_product_state_hash=after["projection_hash"],
        probe_fingerprint=proof_probe_fingerprint_v1(
            operator_policy_hashes=policies, probe_specification=probe
        ),
        changed_owner_ids=(
            "STAGE37_CONTACT_AND_SEMANTIC_SLOTS",
            "STAGE42_CONTACT_PRESERVING_COMPOSITION",
            "STAGE43_RUNTIME_RELATION_TRANSPORT",
            "STAGE45_RELATION_PROOF",
        ),
        bounded_change_passed=same_inputs,
        target_metric=probe["metric"],
        target_metric_before=float(bool(before.get(probe["metric"]))),
        target_metric_after=float(bool(after.get(probe["metric"]))),
        higher_is_better=True,
        material_improvement_margin=0.5,
        baseline_status="PASS" if before.get(probe["metric"]) else "FAIL",
        counterfactual_status="PASS" if after.get(probe["metric"]) else "FAIL",
        protected_invariant_regressions=protected,
        mutation_summary={
            "changed_stage_ids": [
                "37_QUALIFIED_PRESENTATION_STRUCTURE",
                "42_RUNTIME_PROJECTION_AND_CAA_BINDING",
                "43_RSS_MATERIALIZE_COMPACT",
                "45_DYNAMIC_VISUAL_INTEGRITY_PROOF",
            ],
            "mechanics_changed": False,
            "sealed_motion_witness_changed": False,
        },
        metadata={
            "product_authority": False,
            "state_hash_scope": "RESEARCH_PRESENTATION_RELATION_CLOSURE",
        },
    )
    findings = build_controlled_owner_attribution_v1(
        failure_signatures=signatures,
        proof_domain=evidence.proof_domain,
        baseline_source_product_state_hash=before["projection_hash"],
        baseline_measurement_report_hash=content_sha256(before),
        operator_policy_hashes=policies,
        probe_specification=probe,
        interventions=[evidence],
    )
    result = {
        "schema": "RealSaS.MatchedPresentationAttempts.v2",
        "baseline": baseline,
        "candidate": candidate,
        "exact_sealed_subject_input_reused": same_inputs,
        "probe": probe,
        "protected_invariant_regressions": protected,
        "owner_findings": findings,
        "visual_closure_passed": after["status"] == "PASS_DEMO_ONLY",
        "product_authority": False,
    }
    (out / "matched_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input-root", "ctl", "player", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--api", default="http://127.0.0.1:8080")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    baseline_args = copy(args)
    baseline_args.out = args.out / "baseline"
    baseline_args.variant = "v4"
    baseline_args.relational_gate = True
    baseline_args.allow_proof_fail = True
    baseline = run(baseline_args)

    candidate_args = copy(args)
    candidate_args.out = args.out / "candidate"
    candidate_args.variant = "v5"
    candidate_args.relational_gate = False
    candidate_args.allow_proof_fail = True
    candidate_args.subject_id = baseline["subject_id"]
    candidate_args.reuse_inputs = baseline
    candidate_args.baseline_release = baseline["engine_release_id"]
    candidate_args.parent_attempt = baseline["attempt_id"]
    candidate = run(candidate_args)

    result = compare(args.out, baseline, candidate)
    print(
        json.dumps({
            "matched_comparison": str(args.out / "matched_comparison.json"),
            "visual_closure_passed": result["visual_closure_passed"],
        }),
        flush=True,
    )
    if not result["visual_closure_passed"]:
        raise SystemExit("MATCHED_PRESENTATION_VISUAL_CLOSURE_PENDING__EVIDENCE_EXPORTED")


if __name__ == "__main__":
    main()
