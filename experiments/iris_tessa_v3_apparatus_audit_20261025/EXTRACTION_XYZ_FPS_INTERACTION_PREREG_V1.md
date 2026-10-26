# Extraction x XYZ-FPS Interaction Court V1 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** architecture-search only.

## Question

After selecting deterministic XYZ farthest-point sampling for surface witnesses, is residual loss of thin/low-area structure primarily owned by extraction observability or by the witness sampler? What extraction-policy form should V3.1 carry before real morphology statistics exist?

## Stress atlas

Use three analytic topology-observability families already established by the extraction court:

- sphere-like isolated support;
- thin cylindrical support;
- parallel-sheet/interface gap.

Test TESSA-normalized feature scales around the previously measured grid thresholds across extraction resolutions `24,32,48,64`, over deterministic sub-cell phase samples.

A feature is admitted only when the extraction lattice has sign/occupancy evidence under the corresponding analytic family criterion. Features that are not admitted are removed *before* witness sampling.

For every admitted scene, construct the same deterministic surface point atlas (dominant body + qualified rare supports) and run geometry-native XYZ-FPS at witness budgets `64` and `128`.

## Measurements

For every resolution / witness budget:

1. extraction admission recall over true qualified supports;
2. conditional XYZ-FPS support recall among admitted supports;
3. end-to-end support recall over all true supports;
4. permutation invariance;
5. spatial coverage radius on admitted geometry.

## Gates

The court is valid only if XYZ-FPS remains permutation invariant and conditional support recall is `>= 0.99` at budget 64.

Interpretation:

- if conditional sampler recall passes while end-to-end recall tracks extraction admission, the failure owner is extraction, not sampling;
- if budget 128 materially rescues supports missed by budget 64 *after admission*, witness budget remains an interaction owner;
- if fixed resolution materially changes admission, V3.1 must not hard-wire current res24 as a generic inference guarantee.

## Selection rule

This synthetic court cannot choose a final production resolution without a real normalized morphology census.

It may select the **policy form**:

`OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION + XYZ_FPS`

when:

- res24 misses at least one qualified stress stratum;
- higher tested resolutions monotonically recover it;
- XYZ-FPS conditional recall remains >=0.99;
- no sampler-budget increase can recover features absent from extraction.

Otherwise extraction policy remains open.

## Non-claims

No real morphology distribution, product mesh density, Truth Builder, ModelRelease, FIT8 or genericity claim is authorized by this court.
