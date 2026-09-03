# Proof Services

This directory contains **subordinate diagnostic/measurement services** used by `compiler/realsas_compiler_core/`.

Authority rules:

1. `realsas_compiler_core` owns proof plans, proof status and canonical product truth.
2. A proof service may bind or measure already-qualified current-Compiler evidence, but cannot manufacture missing coordinate authority.
3. A failure signature is diagnostic evidence, **not owner attribution**.
4. Owner attribution requires a controlled counterfactual child attempt under the same probe/policy.
5. No service here may mutate canonical identity, silently repair a product, or convert FAIL/ABSTAIN to PASS.

## Current services

### `failure_signatures.py`

Rebound historical failure-localization semantics. It localizes failed invariants and never invents a causal owner.

### `motion_bake.py`

Current qualification-owned directional frame binding. It accepts frames only from a separately qualified directional evaluator and binds them to the exact product-state hash, proof-plan hash, clip and evaluator identity. It does **not** derive a view pivot, execute LBS, or assume that mechanical `QualifiedJoint.position` and directional mesh `P.xy` share a coordinate frame.

Missing qualified frames make the MOTION domain **ABSTAIN**. The same bound frames must be used by export; solver replay during export is forbidden.

### `motion_frame_metrics.py`

Evaluator-independent measurement over qualification-owned 2D frame bakes. It measures requested mobility, symmetric edge stretch, signed triangle-area consequences/flips, loop seam and return-to-rest under restored bounded policy. Measurement failure does not imply causal ownership.

### Retracted direct authored-motion evaluator

`motion_probe.py` and `motion_probe_geometry.py` were promoted during restoration and then explicitly **RETRACTED** after audit found an unauthorized coordinate shortcut: mechanical joint `Vec3` and directional editable-mesh `P.xy` were treated as if they shared a qualified frame.

Blocker: `CURRENT_DIRECTIONAL_JOINT_VIEW_BINDING_MISSING`.

No production evaluator may manufacture directional frames from mechanical joint positions plus directional mesh coordinates until a typed Compiler-owned directional joint/view binding is qualified. See `canonical/AUTHORED_MOTION_PROOF_RETRACTION_V1_20260903.json`.

### `causal_attribution.py`

Controlled-intervention causal attribution. Owner credit requires same-probe, single-owner bounded counterfactual evidence, material target improvement and no protected-invariant regression. Ambiguous improvements abstain.

### `repair_loop.py`

Bounded repair-directive and mandatory same-probe re-proof contract. It does not mutate `CanonicalPuppetGraph.v3`. Owner-specific executors remain separate and unpromoted until individually source-diffed and qualified.

## Runtime/export boundary

`compiler/realsas_compiler_services/export/runtime_v2.py` is a valid **pure native package writer** over already-projected views and already-baked frames. Full current-V4 projection/native interlock remains blocked until the directional joint/view binding and qualified evaluator are closed.
