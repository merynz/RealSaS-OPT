# RealSaS N1D — Adaptive Hypothesis Retention V1

**Date:** 2026-08-19  
**Verdict:** `FAIL__NO_COMPACT_ADAPTIVE_CONTRACT_PASSES__KEEP_F16_REFERENCE`  
**Status:** `TRUTH_OPEN_DEVELOPMENT_RESULT__NOT_QUALIFICATION`  
**Representation Contract V2 commit:** `9fa5484f162ba863f4522e4e5afb851c8fedabe2`  
**Prereg commit:** `3953c7c34540e3fc37386fd8c11a39fad0505d52`  
**Frozen Stage-B authority remains:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Question

Can observation-only descriptor/multiview uncertainty compress the demonstrated safe `K=16` hypothesis reference while preserving hard-tail coverage?

The preregistered primary coverage gates were:

```text
pooled primary-2x >= 0.975
worst family primary-2x >= 0.94
8/8 families >= 0.90
best-worst gap <= 7 pp
```

Efficiency gates were:

```text
mean retained K <= 10
median K <= 8
fraction K=16 <= 0.35
coarse32 carrier fraction <= 0.10
```

## Results

| Arm | Pooled 2x | Worst family | Families >=.90 | Gap | Mean K | Median K | K16 frac | Coverage | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| F4 | .93902 | .84746 | 6/8 | 15.25pp | 4.00 | 4 | 0 | FAIL | PASS |
| F8 | .95528 | .88136 | 6/8 | 11.86pp | 8.00 | 8 | 0 | FAIL | PASS |
| F16 | **.98171** | **.94915** | **8/8** | **5.08pp** | 16.00 | 16 | 1.00 | **PASS** | FAIL |
| A1 descriptor adaptive | .96951 | .89831 | 7/8 | 10.17pp | 9.38 | 8 | .331 | FAIL | **PASS** |
| A2 descriptor + geometry escalation | .97358 | .91525 | **8/8** | 8.47pp | 11.93 | 16 | .566 | FAIL | FAIL |

### Fixed K=8

`K=8` is not a safe compact substitute for the top16 reference. Historical hard-tail families remain below the preregistered floor:

- `11032 = 0.88136`
- `13203 = 0.88525`

### A1 — descriptor uncertainty only

A1 satisfies the efficiency gates but misses hard-tail coverage. In particular:

- pooled `.96951 < .975`
- worst `11032=.89831 < .94`
- only `7/8` families reach `.90`

Therefore descriptor score margin/entropy alone does not reliably identify every carrier that requires a wider geometric hypothesis set.

### A2 — descriptor + geometric escalation

Adding current multiview geometry diagnostics improves coverage:

- pooled `.96951 -> .97358`
- `11032 .89831 -> .91525`
- all `8/8` families reach `.90`

but it still fails the hard-tail gates and expands too aggressively:

- worst `.91525 < .94`
- gap `8.47pp > 7pp`
- mean K `11.93 > 10`
- median K `16 > 8`
- K16 fraction `.566 > .35`

The current raw reprojection-q75 trigger is therefore simultaneously **under-sensitive to the decisive 11032 tail and over-sensitive on many otherwise safe carriers**.

## Scientific interpretation

This is **not** a reversal of Representation Sufficiency Battery V1. The fixed top16 representation remains the demonstrated safe open-development reference, and the previous oracle ceiling remains strong.

The failure is narrower:

> **Adaptive compression V1 is not solved. The current uncertainty policy cannot yet tell, cheaply and family-robustly, when the sufficient hypothesis space may be collapsed from top16 to top8/top4.**

The result also strengthens one design lesson: raw family-global pixel reprojection thresholds are a poor universal uncertainty currency. The same q75-style geometry trigger expands many easy cases yet misses enough of the hard family to fail the worst-family gate.

## Next required diagnostic

Before any `p_active + log_amp` training, run a truth-open **expansion-need separability audit**.

Define the evaluation-only label:

```text
NEEDS_EXPANSION = F8 misses primary-2x AND F16 contains primary-2x
```

Truth is used only for this diagnostic label. Candidate trigger features must remain observation-only. Test, family-disjoint, at least:

```text
margin4 / margin8
entropy16
F4/F8 reprojection residuals
relative reprojection improvement F4->F8
selected endpoint shift F4->F8
normalized hypothesis-count/support
H width / multimodality
cross-view candidate disagreement
```

The aim is to identify a self-normalized uncertainty signal that detects top16-rescuable hard cases without expanding most easy carriers. Only after that signal is identified may Adaptive H V2 be preregistered.

## Training boundary

Because the prereg explicitly required a bounded H contract to be frozen first:

- small factor-head **execution is paused**;
- large/end-to-end training remains forbidden;
- top16 remains the development safety reference, not a permanent product constant.

## Reproducibility

Canonical compact result: `ADAPTIVE_HYPOTHESIS_RETENTION_V1_RESULT.json`  
Canonical source: `adaptive_hypothesis_retention_v1.py`

Local exact result SHA-256:
`39574fd299be0876bf947d0a7b1ea7f08d824f6ae678f4a9acda187f550a5533`

Local primary source SHA-256:
`0390f472cb3b1a893e6241fde22f2278185a1fe4d37efa61622804e2e997e445`

Execution-only optimized helper SHA-256:
`0f4e275ce27f4327610956e66752c1626b81953f3713ed60b179220512dacb67`

The optimized helper changed execution cost only (cached view-pair pseudoinverses / projection / memoization); the preregistered mathematics and decision rules were unchanged.
