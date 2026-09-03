# Proof Services

This directory contains **subordinate diagnostic/measurement services** used by `compiler/realsas_compiler_core/`.

Authority rules:

1. `realsas_compiler_core` owns proof plans, proof status and canonical product truth.
2. A proof service may derive measurements or failure signatures from already-bound current-compiler inputs.
3. A failure signature is diagnostic evidence, **not owner attribution**.
4. Owner attribution requires a separate controlled causal mutation/fault experiment.
5. No service here may mutate canonical identity, silently repair a product, or convert FAIL/ABSTAIN to PASS.

## Current services

### `failure_signatures.py`

Semantic promotion/rebind of the useful diagnostic separation in historical v0.5 `realsas_deformation/failure_signatures.py`. It localizes failed invariants but does not import/reactivate the historical contract stack and never invents a causal owner.

### `motion_probe.py` + `motion_probe_geometry.py`

Current V4 semantic rebind of the valuable historical `realsas_deformation/motion_proof.py` rule: **authored/requested motion must be exercised and its deformation consequences measured**.

The probe consumes the exact current `CanonicalPuppetGraph.v3` mechanical, directional mesh/mesh-skin and puppet-local `MotionStateIR` state. It samples authored clip keys plus uniform times, evaluates qualified mesh deformation through the promoted LBS numerical service, and reports:

- effective authored motion;
- edge relative change;
- triangle area compression/expansion and degeneration;
- non-finite deformation;
- loop seam / return-to-rest consequence;
- per-clip and per-direction/component measurements.

Its 4x4 matrices are internal LBS measurement carriers derived from current puppet-local `translation_xy / rotation_deg / scale_xy / depth_offset`; they are **not** a full-3D reconstruction or shipping-motion authority.

Current Compiler `proof_engine.py` binds the exact motion-probe policy into `ProofPlanIR` and owns the resulting MOTION PASS/FAIL status.
