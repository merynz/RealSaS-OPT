"""Matched Go child Attempts with the same S37 artifacts and relational probes."""
import argparse
from copy import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_scoped_presentation_attempt import run
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_presentation_pose_v1 import POLICY
from compiler.realsas_compiler_services.proof.causal_attribution import (
    ControlledInterventionEvidenceV1, proof_probe_fingerprint_v1, build_controlled_owner_attribution_v1,
)


def compare(out, baseline, candidate):
    before = json.loads((out / "baseline/presentation_proof.json").read_text())
    after = json.loads((out / "candidate/presentation_proof.json").read_text())
    ledgers = [json.loads((out / label / "execution_ledger.json").read_text()) for label in ("baseline", "candidate")]
    upstream = [next(s for s in ledger["stages"] if s["id"].startswith("37_")) for ledger in ledgers]
    exact = ([x["sha256"] for x in upstream[0]["outputs"]] == [x["sha256"] for x in upstream[1]["outputs"]]
             and bool(upstream[1].get("platform_artifact_id"))
             and baseline["subject_input_id"] == candidate["subject_input_id"])
    if not exact: raise RuntimeError("MATCHED_PRESENTATION_UPSTREAM_BYTES_OR_REUSE_DRIFT")
    probe = {"schema": "RealSaS.ConnectedPresentationProbe.v1", "frame_views": [
        [r["clip_id"], r["view_id"], r["frame_index"]] for r in before["relational_frames"]],
        "metric": "maximum_parent_child_relation_residual_source_px", "units": "SOURCE_CAMERA_PIXELS"}
    if probe["frame_views"] != [[r["clip_id"], r["view_id"], r["frame_index"]] for r in after["relational_frames"]]:
        raise RuntimeError("MATCHED_PRESENTATION_PROBE_MATRIX_DRIFT")
    protected = tuple(key for key in ("area_collapse_count", "condition_failure_count", "flipped_triangles",
        "edge_gt_4_count", "native_reference_mismatch_frame_views", "unresolved_depth_ties", "fragment_overflow")
        if after[key] > before[key])
    signatures = [{"signature_id": "PRESENTATION_PARENT_CHILD_DISCONNECTION", "failure_family": "JOINT_RELATION"},
                  {"signature_id": "PRESENTATION_VISUAL_GAPS", "failure_family": "CONTACT_COVERAGE_OR_OCCLUSION"}]
    policies = [content_sha256(POLICY)]
    evidence = ControlledInterventionEvidenceV1(
        intervention_id=candidate["attempt_id"], target_signature_id=signatures[0]["signature_id"],
        proof_domain="SCOPED_RESEARCH_PRESENTATION", owner_id="STAGE42_PRESENTATION_POSE",
        baseline_source_product_state_hash=before["projection_hash"], baseline_measurement_report_hash=content_sha256(before),
        counterfactual_source_product_state_hash=after["projection_hash"],
        probe_fingerprint=proof_probe_fingerprint_v1(operator_policy_hashes=policies, probe_specification=probe),
        changed_owner_ids=("STAGE42_PRESENTATION_POSE",), bounded_change_passed=exact,
        target_metric=probe["metric"], target_metric_before=before[probe["metric"]], target_metric_after=after[probe["metric"]],
        higher_is_better=False, material_improvement_margin=1.,
        baseline_status="PASS" if before["connected_palette_relations_passed"] else "FAIL",
        counterfactual_status="PASS" if after["connected_palette_relations_passed"] else "FAIL",
        protected_invariant_regressions=protected,
        mutation_summary={"changed_stage_id": "42_RUNTIME_PROJECTION_AND_CAA_BINDING", "mechanics_changed": False},
        metadata={"product_authority": False, "state_hash_scope": "RESEARCH_PROJECTION_ONLY"})
    findings = build_controlled_owner_attribution_v1(failure_signatures=signatures,
        proof_domain=evidence.proof_domain, baseline_source_product_state_hash=before["projection_hash"],
        baseline_measurement_report_hash=content_sha256(before), operator_policy_hashes=policies,
        probe_specification=probe, interventions=[evidence])
    result = {"schema": "RealSaS.MatchedPresentationAttempts.v1", "baseline": baseline, "candidate": candidate,
        "exact_stage37_reused": exact, "probe": probe, "protected_invariant_regressions": protected,
        "owner_findings": findings, "visual_closure_passed": after["status"] == "PASS_DEMO_ONLY",
        "product_authority": False}
    (out / "matched_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "input-root", "ctl", "player", "out"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--api", default="http://127.0.0.1:8080")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    baseline_args = copy(args)
    baseline_args.out = args.out / "baseline"
    baseline_args.variant, baseline_args.relational_gate, baseline_args.allow_proof_fail = "v3", True, True
    baseline = run(baseline_args)
    candidate_args = copy(baseline_args)
    candidate_args.out, candidate_args.variant = args.out / "candidate", "v4"
    candidate_args.subject_id = baseline["subject_id"]
    candidate_args.reuse_inputs = baseline
    candidate_args.baseline_release = baseline["engine_release_id"]
    candidate_args.parent_attempt = baseline["attempt_id"]
    candidate_args.require_stage37_reuse = True
    candidate = run(candidate_args)
    result = compare(args.out, baseline, candidate)
    print(json.dumps({"matched_comparison": str(args.out / "matched_comparison.json"),
                      "visual_closure_passed": result["visual_closure_passed"]}), flush=True)
    if not result["visual_closure_passed"]: raise SystemExit("MATCHED_PRESENTATION_VISUAL_CLOSURE_PENDING__EVIDENCE_EXPORTED")


if __name__ == "__main__": main()
