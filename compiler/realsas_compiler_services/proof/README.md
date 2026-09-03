# Proof Services

This directory contains **subordinate diagnostic/measurement services** used by `compiler/realsas_compiler_core/`.

Authority rules:

1. `realsas_compiler_core` owns proof plans, proof status and canonical product truth.
2. A proof service may derive measurements, failure signatures or controlled-intervention causal evidence from already-bound current-compiler inputs.
3. A failure signature is diagnostic evidence, **not owner attribution**.
4. Owner attribution requires a separate controlled counterfactual child attempt under the same proof probe/policy.
5. No service here may mutate canonical identity, silently repair a product, or convert FAIL/ABSTAIN to PASS.

## Current services

### `failure_signatures.py`

Semantic promotion/rebind of the useful diagnostic separation in historical v0.5 `realsas_deformation/failure_signatures.py`. It localizes failed invariants but does not import/reactivate the historical contract stack and never invents a causal owner.

### `motion_probe.py` + `motion_probe_geometry.py`

Current V4 semantic rebind of the valuable historical `realsas_deformation/motion_proof.py` rule: **authored/requested motion must be exercised and its deformation consequences measured**.

The probe consumes the exact current `CanonicalPuppetGraph.v3` mechanical, directional mesh/mesh-skin and puppet-local `MotionStateIR` state. It samples authored clip keys plus uniform times, evaluates qualified mesh deformation through the promoted LBS numerical service, and reports effective motion, edge relative change, triangle area compression/expansion and degeneration, non-finite deformation, and loop-seam/return consequences.

Its 4x4 matrices are internal LBS measurement carriers derived from current puppet-local `translation_xy / rotation_deg / scale_xy / depth_offset`; they are **not** full-3D reconstruction or shipping-motion authority.

### `causal_attribution.py`

Controlled-intervention causal attribution. This deliberately replaces the historical heuristic owner-ranking classifier with a stricter evidence rule.

An owner is attributed only when a counterfactual record:

- starts from the exact baseline product state and measurement report;
- uses the same operator-policy + probe-specification fingerprint;
- creates a distinct child product state;
- changes exactly one declared owner domain;
- passes an explicit bounded-change audit;
- materially improves the target metric or converts the target proof from non-PASS to PASS;
- introduces no protected-invariant regression.

If multiple owners materially improve the same failure, attribution **ABSTAINS** rather than pretending the cause is unique. If all interventions are invalid or none materially improve, attribution also abstains.

The service emits JSON-ready controlled evidence for the existing `DomainProofReportIR.owner_attribution` field, but current `proof_engine.py` does not yet accept arbitrary external intervention records. Binding waits for the core child-attempt/repair lineage gate so callers cannot inject fabricated causal evidence.

### `repair_loop.py`

Bounded repair-directive and mandatory re-proof contract. It does **not** mutate `CanonicalPuppetGraph.v3`; it defines what a separately executed repair child attempt must prove before any repair effect can receive credit.

A repair directive is executable only when:

- causal attribution is `attributed`, never merely diagnosed;
- the selected owner matches the repair operation owner;
- the operation explicitly targets the attributed signature;
- the operation carries a non-empty qualification hash and is qualified for automatic execution;
- allowed change paths and a bounded-change specification are explicit.

Each directive represents **one owner-local counterfactual operation per child attempt**. The application record must produce a distinct child state whose `parent_state_hash` is the baseline product state, change only the attributed owner and only permitted paths, and pass the bounded-change audit.

Repair acceptance then requires a non-empty child proof bundle, the exact same proof-probe fingerprint, material target improvement or target resolution, and zero protected-invariant regressions. Any probe change, cross-owner mutation, lineage mismatch or protected regression rejects repair credit.

Owner-specific executors (rig/weight/mesh/deformation) are intentionally separate and remain unpromoted until their operation families are individually source-diffed and qualified.
