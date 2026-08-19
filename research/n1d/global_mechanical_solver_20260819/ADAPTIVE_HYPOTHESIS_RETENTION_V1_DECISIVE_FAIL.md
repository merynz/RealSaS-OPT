# RealSaS N1D — Adaptive Hypothesis Retention V1 — Decisive Preregistered Failure

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DEVELOPMENT__PREREG_CANDIDATE_A2_FALSIFIED__NOT_QUALIFICATION`  
**Parent prereg commit:** `3953c7c34540e3fc37386fd8c11a39fad0505d52`  
**Representation Contract V2 commit:** `9fa5484f162ba863f4522e4e5afb851c8fedabe2`

## Decision boundary

The preregistered hard-tail acceptance rule required, among other gates:

```text
worst-family primary-2x containment >= 0.94
```

The experiment does **not** need all eight held-out folds to reject a candidate arm once one held-out family violates this hard gate.

## Known arm results before A2

### F8 fixed K=8

Broad 8-family e00 open-development panel:

- pooled primary-2x containment: approximately `0.9553`
- held-out/hard family `11032`: `0.8814`
- held-out/hard family `13203`: `0.8852`
- only `6/8` families at or above `0.90`

Therefore F8 failed the preregistered coverage gates.

### A1 descriptor-only adaptive retention

Exact recovered result:

- pooled primary-2x containment: `0.9695121951219512`
- worst family `11032`: `53/59 = 0.8983050847457628`
- families >=0.90: `7/8`
- best-worst gap: `10.1695 pp`
- mean retained K/view: `9.37649`
- median K: `8`
- K16 fraction: `0.33135`

A1 met the nominal K-efficiency envelope but failed coverage.

## A2 decisive hard-tail witness

A2 starts from A1 and adds bounded geometric escalation using leave-one-family-out observation-only calibration of reprojection error / pair support.

On held-out family `11032`:

```text
A1 containment = 53/59 = 0.8983050847
A2 containment = 54/59 = 0.9152542373
F16 containment = 56/59 = 0.9491525424
```

Thus A2 rescues only **1 of the 3 additional carriers** needed to reach the demonstrated F16 hard-tail ceiling on this family.

Because `0.9152542373 < 0.94`, the preregistered A2 candidate contract is already **falsified** regardless of the remaining held-out folds.

This is a decisive candidate-policy failure, not evidence against Representation Contract V2. The fixed F16 reference remains hard-tail sufficient on this panel, and the representation sufficiency battery remains unchanged.

## Scientific localization

The existing q75-style carrier escalation signal is informative but insufficient:

- it moves `11032` in the correct direction (`53 -> 54` contained carriers);
- it fails to identify two additional carriers whose target-near hypotheses are recoverable under F16;
- prior partial execution also showed aggressive K16 usage on some families, so the policy trends toward being both **late on some hard carriers** and **over-expansive on some easy/medium carriers**.

Therefore the next experiment must not tune q75/q95 thresholds post hoc. It must diagnose, on truth-open development data, which observation-only carrier diagnostics separate:

```text
A1 miss + F16 rescue
from
A1 contained
```

Candidate signals to audit include:

- per-view margin4 / margin8 tails;
- entropy16 / multimodality;
- disagreement of candidate-supported pair triangulations;
- reprojection residual distribution, not only its minimum mean;
- cross-view support count for coherent H clusters;
- H dispersion / feasible-set width;
- cycle / leave-one-view-out consistency;
- triangulation conditioning.

The desired next policy is a fail-closed bounded expansion rule driven by a small set of causally useful observable diagnostics. No learned p_active/log_amp head should start until this bounded H retention contract is frozen or F16 is explicitly accepted as the temporary frozen contract.

## Canonical decision

```text
F8 = FAIL
A1 = FAIL
A2 = FAIL (decisive held-out 11032 witness)
F16 = KEEP AS DEVELOPMENT REFERENCE
ADAPTIVE_COMPRESSION = NOT SOLVED
REPRESENTATION_V2 = SURVIVES
BIG_TRAINING = NOT AUTHORIZED
```

Frozen Stage-B qualification authority remains unchanged: `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`.
