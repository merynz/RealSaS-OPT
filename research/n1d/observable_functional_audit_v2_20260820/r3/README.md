# N1D Observable Functional Audit V2 — Source Bundle

This bundle implements a fresh functional-geometry audit from the frozen real raster-only N1D Hybrid V11 inference front door.

## Phases

0. `src/prepare_runtime.py`
   - verifies the frozen checkpoint and canonical N1D source ZIP by SHA-256;
   - deterministically prepares the historical absolute runtime aliases required by the byte-frozen V11 dependencies;
   - builds the family/episode raster view from the padded corpus layout without changing raster bytes.

1. `src/observable_phase.py`
   - reads only paired e00 rasters plus frozen N1D source/checkpoint dependencies;
   - runs the frozen Hybrid V11 baseline;
   - rebuilds the route-matched feasible candidate set H from descriptors + calibrated triangulation;
   - verifies that raster visual-hull `V_B` and frozen normal-head `N_B` replay the baseline exactly;
   - freezes candidate geometry and baseline prediction as observable state.

2. `src/evaluation_phase.py`
   - opens evaluator sidecars only after the observable state is frozen;
   - maps carriers to evaluator truth, identifies active/reliable baseline-hard but H-recoverable witnesses;
   - teacher truth may select the nearest feasible H candidate for the counterfactual only;
   - calls `decorate_pb_observable()` to regenerate counterfactual `N_B` and `V_B` from rasters/model, never from sidecar normals/visibility;
   - computes frozen GFDR-V2 and non-tautological mechanical effect / truth-relative metrics.

3. `src/mechanics_metrics.py`
   - primary equivalence deliberately excludes raw coordinate-identity blocks `F_delta` and `R_rel_B`;
   - state-vs-state equivalence NRMS is symmetric, while prediction-vs-truth NRMS is truth-normalized;
   - uses response, co-response, differential, transfer, affinity, and support-aware articulation-locus consequences.

## Frozen dependencies copied byte-for-byte

- `realsas_n1d_hybrid_v11_frozen_runner.py`
- `v8_base_frozen.py`
- `v5_seed_geometry_frozen.py`
- `realsas_gfdr_v2.py`

The model implementation ZIP and checkpoint are external immutable inputs and are recorded by the experiment input manifest.

## Tests

```bash
python tests/test_static.py
python tests/test_decorator_replay.py --root <raster-root> --baseline <frozen-v11-baseline.npz> --family 9908 --episode e00 --route V8_BASE
```

Canonical execution is authorized only after the repository `RESEARCH_EXECUTION_PROTOCOL_V2` and three-store source persistence gate pass.
