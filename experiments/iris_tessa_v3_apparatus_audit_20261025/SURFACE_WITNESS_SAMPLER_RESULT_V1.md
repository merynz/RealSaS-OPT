# Surface Witness Sampler Court — Result V1

**Date:** 2026-10-26  
**Preregistration:** `SURFACE_WITNESS_SAMPLER_PREREG_V1.md`  
**Workflow:** `iris-surface-witness-sampler-v1-stage`  
**Run:** `40312976720`  
**Exact SHA:** `1b9fa8687d304aa247c829bae1ff512355644956`  
**Artifact:** `iris-surface-witness-sampler-v1-report`, digest `sha256:a61cc8b32ec7c30c4896e2632fa70ad92239eb4c8ab3e0f32246f6e81a7ebaa3`  
**Status:** `GEOMETRY_NATIVE_SAMPLER_SELECTED_TINY_V1`  
**Selected arm:** `XYZ_FPS`

## Result

The current vertex-index sampler is ineligible:

- permutation invariant: **false**;
- rare-group hit recall: `0.90625`;
- mean spatial cover radius: `0.149976`;
- worst spatial cover radius: `0.372825`.

Geometry-native farthest-point sampling (`XYZ_FPS`) passed every hard eligibility gate:

- permutation invariant on every synthetic surface family: **true**;
- rare-group hit recall: `1.0`;
- mean spatial cover radius: `0.075740`;
- worst spatial cover radius: `0.088903`;
- uniform-surface worst cover radius: `0.088903`.

Normal-aware FPS variants also passed eligibility and reduced mean normal-mismatch p95, but paid a spatial-coverage penalty. Under the preregistered selection rule the simpler `XYZ_FPS` arm is selected as the geometry-native baseline.

## Interpretation

The witness sampler must be a function of geometry, not extracted vertex-array ordering. The selected V3.1 candidate is deterministic XYZ farthest-point sampling at the admitted witness budget.

Normal-aware / curvature-aware stratification remains eligible for a later interaction court if downstream evidence demonstrates that angular coverage matters enough to justify its spatial penalty. It is not selected by this isolated court.

## Carry-forward

Replace index-stride sampling in the V3.1 candidate with deterministic `XYZ_FPS`, preserving deterministic tie-breaking and exact witness-budget accounting. Extraction × witness interaction remains required before final assembly because changing extraction density/topology changes the candidate point set presented to the sampler.

## Authority

No ProductRevision, ModelRelease, Truth139, FIT8, LOFO or genericity authority changes.
