# RealSaS — R6 Observable-Surface Coverage Stress — Preregistration

**Date:** 2026-09-04  
**Status:** `PREREGISTERED_BEFORE_STRESS_OUTPUTS__NO_ARCHITECTURE_CHANGE_AUTHORIZED`

## Question

Does the current shipping observation route retain enough information for downstream rigging when the observation-grounded substrate is materially incomplete, rather than the ~99.4–100% coverage regime of the earlier capsule-shell U0/U1 gate?

This is an information-sufficiency strengthening experiment. It does not redefine IRIS truth, visibility, or the GeometricSubstrateAssembler (GSA).

## Frozen route

The compared paths are:

`U0_REFERENCE_FULL_SURFACE -> downstream consumer`

and

`U1 observation evidence -> GeometricSubstrateAssembler.current -> DTB-ND1 where qualified -> RiggingSurfaceIR S -> downstream consumer`.

The stress mask is applied to `ObservationEvidenceIR` **before** GSA. No `RiggingSurfaceIR` node is deleted after compilation. Hidden-surface completion, direct source-mesh consumer input, source topology injection, and learned visibility/occlusion authority are forbidden.

## Witness and unchanged consumers

Witness: `branch_blend_4` from the frozen R6 synthetic panel.

Downstream consumers remain unchanged:

- shipping Geppetto V2 configuration and optimizer protocol used by `test_r6_oracle_substrate_geppetto_one_family_v1.py`;
- shipping SkinFieldCodec and Arachne V2 configuration/protocol used by `test_r6_oracle_substrate_arachne_one_family_v1.py`;
- real Compiler qualification remains in the route.

No model width, loss, seed, optimizer, acceptance band, Compiler threshold, DTB-ND1 rule, camera, shell, or skeleton target may be modified after stress outputs are observed.

## Frozen deprivation operators

Coordinates are normalized from the full U0 shell bounding box into approximately `[-1,1]` per axis. The following contiguous mechanical-region masks are fixed before outputs:

### `CORE_75`

Hide a full-surface sample group iff:

`abs(x_norm) < 0.35 AND -0.20 < y_norm < 0.55`.

Input-geometry-only preflight predicts roughly 75% full-surface retention on `branch_blend_4`.

### `CORE_60`

Hide a full-surface sample group iff:

`abs(x_norm) < 0.45 AND -0.35 < y_norm < 0.55`.

Input-geometry-only preflight predicts roughly 60% full-surface retention on `branch_blend_4`.

These masks are not claimed to be a photorealistic garment renderer. They are deliberately contiguous missing-surface stresses intended to answer whether the current GSA + consumers depend on near-complete surface coverage. No random/scattered dropout is used.

## Required telemetry

For U0 and each stressed U1 arm record at minimum:

- full U0 sample count;
- naturally observable pre-stress sample count;
- stress-hidden sample count;
- GSA output surface-node count;
- retained fraction relative to U0;
- support-view min/mean/max;
- DTB-ND1 normal count/fraction;
- local-relation count;
- hidden-surface completion flag;
- direct source-mesh consumer-input flag.

Geppetto records generated control count, matched MAE/p95, Compiler-qualified mechanical status, pass step and stable passes.

Arachne records Codec/Arachne row-L1 p95, verified deformation ratio, exact admitted row coverage, Compiler correction, pass step and stable passes.

## Interpretation frozen before outputs

- `U0 FAIL`: apparatus invalid; no information-sufficiency interpretation.
- `U0 PASS + CORE_75 FAIL`: current observable-substrate contract is materially coverage-sensitive; near-complete prior U1 results are insufficient.
- `U0 PASS + CORE_75 PASS + CORE_60 FAIL`: current route tolerates substantial deprivation but a coverage boundary exists between the two frozen stresses.
- `U0 PASS + CORE_75 PASS + CORE_60 PASS`: strong synthetic evidence that the current GSA + consumer route does not require near-complete surface coverage on this witness.

A Geppetto result and an Arachne result answer different questions. Arachne uses the frozen oracle mechanical skeleton as in the existing U1 Arachne gate and therefore does not rescue a Geppetto failure.

No outcome from this synthetic strengthening experiment alone establishes arbitrary real-character or RigAnything-equivalent performance. It determines whether **surface incompleteness itself** is already a demonstrated bottleneck under the current GSA/consumer architecture.

## Authority rule

This preregistration authorizes only the stress measurement. It authorizes no IRIS per-view visibility head, hidden-surface teacher, completion mechanism, new GSA heuristic, or optimizer/FIT change.
