# Compiler Export Services

This package is subordinate to the canonical Compiler. Export is a **projection of an already-qualified product and proof**, never a second solver/evaluator path.

## Current path

```text
CanonicalPuppetGraph.v3
  + ProductProofEvaluationV1 / PASS ProductProofBundleIR
  + exact MOTION MeasurementReportIR
      -> proof-owned RuntimeDeployBake.v2
      -> current_v4_projection.py
      -> runtime_v2.py
      -> .rss / .rsr
      -> runtime/realsas_cpp/
```

## Files

- `runtime_deploy_bake.py` — compact codec for the exact sampled frames produced during motion proof. Export must not rerun deformation.
- `current_v4_projection.py` — proof-bound projection of current directional meshes/face-corner appearance into native runtime-v2 deployment topology. UV seams use runtime-only vertex duplication; canonical editable meshes are unchanged.
- `runtime_v2.py` — pure deterministic `.rsr`/`.rss` writer. It validates external texture bytes by SHA-256, CRC32 and dimensions and contains no model/solver/proof/repair execution.

## Fail-closed compatibility boundaries

Native runtime v2 currently requires one texture atlas per direction and vertex UVs. Current V4 appearance owns face-corner UVs, so projection duplicates runtime vertices only where a UV seam requires it. Current material UV uses bottom-left normalized V; native runtime v2 uses PNG top-left V, so projection explicitly applies `v_runtime = 1 - v_material`.

Dynamic order/visibility tracks are not guessed. Until their JSON key semantics are typed for deployment, proof-owned runtime bake abstains when either track family is present.

Raw texture bytes are external content-addressed payloads: the Compiler product binds their identity through appearance lineage, and materialization verifies the supplied PNG bytes before packaging.

## Authority

A runtime archive is deployable only when it is bound to the exact current product state, exact PASS proof bundle, exact MOTION measurement report and exact texture payload identities. No export operation may promote FAIL/ABSTAIN to PASS or reinterpret canonical rig/skin/motion truth.
