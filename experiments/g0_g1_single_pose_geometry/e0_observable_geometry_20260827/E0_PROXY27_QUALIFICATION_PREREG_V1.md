# RealSaS E0 — Frozen Proxy27 Qualification Prereg V1

**Date:** 2026-08-28  
**Status:** `FROZEN_BEFORE_PROXY27_DOWNSTREAM_OPEN__ONE_SHOT_READY__DEV32_CLOSED`

## Question

Does the exact observable A×8 common-frame substrate preserve enough rigging-relevant information under the already-frozen matched Arachne/Geppetto information-isolation consumers when evaluated on the independent truth-capable Proxy32 intersection?

## Population

The population is fixed to the existing metadata-only truth-capable intersection in `E0_PROXY32_TRUTH_INTERSECTION_V1.json`:

- count: **27**;
- ordered-ID SHA-256: `b9120bbd3f603cee6bf80110e170309b9fe8b135ae56fbbd6d8d1bda4fc11817`;
- intersection bytes SHA-256: `fcbdd90d585a3845b8f799fbd1ba6dff526d4714aa581a7f6d301da86d2ecbd9`;
- eligibility: `capabilities.iris && capabilities.geppetto && capabilities.arachne` only;
- no downstream metric was inspected to define membership.

No family may be added or removed after downstream evaluation opens.

## Frozen substrate arms

```text
D0 = 512-slot area-uniform full-mesh reference + oracle observation provenance
D1 = 512-anchor exact visible-union substrate + oracle physical persistence
D2 = same observable substrate + frozen teacher-free MUTUAL_P003 persistence
```

The exact V1.2 builder source is reused (`e0_downstream_proxy_v1.py`, SHA-256 `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`) with camera geometry source SHA-256 `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40` and SurfaceBuilder SHA-256 `83819fc8869ff26fbfc928fbeda09e59115a0554c94cbddc623b97d1c790c1cb`.

The existing 437 FIT/calibration packs are **not rebuilt**. Qualification constructs only 27 new Proxy packs in a separate run directory and records source geometry/raster hashes plus compact-pack hashes.

## Frozen consumers

No training or checkpoint selection occurs in qualification. The six sealed V1.3 checkpoints are loaded byte-for-byte:

- Arachne D0 `e21e9e7f0a65b1c18a4db0327c0bead708cceea6740b9b9bf9946f0a169b80ce`
- Arachne D1 `fedc543d6ac3f1ea067e9ce4fccbaa3c3522a428bc89dc21ee0165f0b9fd088c`
- Arachne D2 `72898a62f23c55aa82047f7bc4b39be787abb97d14f9a3fb973d59b2b5689745`
- Geppetto D0 `ff993aeea8f92d2e0ae70d40605e4089f416c7de14f4e682983ec3e0af6d4cde`
- Geppetto D1 `f1dec9dd8ea623c36c1de1ac31fd9bcba76a510af460088680ab0f38abf75a6d`
- Geppetto D2 `f8c6146fc3ad81146ced01805b9be454ad194b86db3a9ab3d72d7b9eb3747b65`

Checkpoint calibration-result authority SHA-256: `82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`.

The exact calibration evaluator source is reused, SHA-256 `7fc3021726df64069ba44be42f946db25929372497f39f392898477b374e2ca4`.

## Target-availability fail-closed rule

The frozen population remains all 27 families. Before any downstream metric is evaluated, all 27 compact packs are built and audited for symmetric Arachne target availability:

```text
count(D0_skin_valid)>0 AND count(D1_skin_valid)>0 AND count(D2_skin_valid)>0
```

If any member fails, qualification stops with `BLOCKED_TARGET_AVAILABILITY__NO_DOWNSTREAM_METRICS_EVALUATED`. The family is **not** silently excluded and no PASS/FAIL metric is computed. This is an apparatus/target-availability block, not a scientific non-inferiority failure.

## Frozen qualification rule

Primary lower-is-better metrics:

- Arachne CE;
- Arachne influence displacement;
- Geppetto joint mean.

For each primary metric all must hold:

```text
D1/D0 <= 1.05
D2/D1 <= 1.05
D2/D0 <= 1.05
```

Geppetto `family_p95` uses the same three comparisons at `<=1.10`.

Catastrophe vetoes:

```text
PCK@0.05(D2) >= PCK@0.05(D0) - 0.10
PCK@0.08(D2) >= PCK@0.08(D0) - 0.10
```

`ALL_PRIMARY_AND_TAIL_AND_PCK_VETOES_MUST_PASS`.

No diagnostic result, expectation record, per-family anecdote, or post-hoc confidence interval may rescue or veto this rule.

## Non-binding expectation

`E0_PROXY27_NONBINDING_EXPECTATION_V1.json` is frozen before opening (SHA-256 `5c9cbecad2496c8145515f31a04f0bd8dbce85a0078e47e25d1acf8b04725b04`). It is descriptive only and is explicitly forbidden from influencing PASS/FAIL.

## Pre-Proxy32 diagnostic boundary

Diagnostic result SHA-256: `906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d`.

Its interpretation caveats are frozen separately. The diagnostic cannot change the population, margins, substrate, checkpoint set or qualification outcome.

## One-shot / resume policy

- The final result is written only after the frozen evaluation finishes.
- If a sealed result already exists, the notebook refuses a second open.
- An execution failure before a final result exists may be resumed only with the same notebook/source/checkpoint hashes; no retune or code change based on partial outcomes is allowed.

## Firewalls

- Proxy27 downstream: **CLOSED until this notebook executes**.
- DEV32: **CLOSED**.
- no product-domain filtering;
- no IRIS learner;
- no consumer training;
- no checkpoint replacement;
- no change to `MUTUAL_P003`;
- no change to margins;
- N-B3 remains a separate post-E0 experiment.
