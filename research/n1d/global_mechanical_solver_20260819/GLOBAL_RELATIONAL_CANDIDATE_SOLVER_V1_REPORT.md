# RealSaS N1D — Global Relational Candidate Solver V1 — Preflight Result

**Date:** 2026-08-19  
**Verdict:** `INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`  
**Qualification:** unchanged; truth-open development only.  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Preregistered boundary

`GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_PREREG.md` freezes M128 before endpoint results are inspected. The prereg requires candidate-mode preflight, on mapping-reliable carriers:

- pooled primary-2x containment >= `0.970`
- worst-family primary-2x containment >= `0.900`
- 8/8 families >= `0.90`
- best-worst gap <= `0.10`

If M128 fails this preflight, the prereg decision is `INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL` and the global solver must not be interpreted or M128 changed post hoc.

## Source parity guard

Before interpreting M128, the replay reconstructs the frozen F16 H and compares per-family primary-2x containment against the canonical Representation Sufficiency Battery broad 8-family result.

Exact parity was obtained on all 8 families and the mapping-reliable denominator is exactly `492/512`:

- 09908: `1.0000000000`
- 11032: `0.9491525424`
- 12772: `1.0000000000`
- 13203: `0.9508196721`
- 14404: `0.9682539683`
- 14702: `1.0000000000`
- 14758: `0.9838709677`
- 15290: `1.0000000000`

Therefore the M128 preflight is evaluated on a parity-validated frozen F16 reconstruction rather than a source-drifted replay.

## M128 preflight

M128 is observation-only: retain best 32 candidates by frozen unary reprojection score, then deterministic XYZ farthest-point fill to 128 from the deduplicated F16 H.

Result on the 492 mapping-reliable carriers:

- pooled primary-2x containment: `0.9695121951`
- worst family: `11032 = 0.9152542373`
- families >= 0.90: `8/8`
- best-worst gap: `0.0847457627`

Per family primary-2x:

- 09908: `1.0000000000`
- 11032: `0.9152542373`
- 12772: `1.0000000000`
- 13203: `0.9344262295`
- 14404: `0.9682539683`
- 14702: `1.0000000000`
- 14758: `0.9838709677`
- 15290: `0.9508196721`

Strict-1x:

- 09908: `0.9682539683`
- 11032: `0.7796610169`
- 12772: `0.9833333333`
- 13203: `0.8524590164`
- 14404: `0.8888888889`
- 14702: `0.9841269841`
- 14758: `0.9516129032`
- 15290: `0.7704918033`

## Decision

Three of the four preflight conditions pass, but pooled primary-2x containment is:

`0.9695121951 < 0.9700000000`

This is a preregistered FAIL even though the miss is numerically very small. No threshold rounding is used.

Therefore:

`INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`

The ICM `G_REL_DIS` solver arm was **not run/interpreted**. This result says nothing negative about the already-frozen `R_REL_DIS` relation signal. It says the preregistered M128 compression is not authorized as the candidate domain for that solver test.

## What is forbidden now

- do not change M128 after seeing this result and call it V1;
- do not relax `0.970` to the rounded displayed value;
- do not run the V1 solver and interpret endpoint metrics anyway;
- do not reopen standalone amplitude/vector heads;
- do not touch sealed21/external10;
- do not change frozen Stage-B qualification.

A new candidate-domain/solver experiment, if pursued, must be separately preregistered before its endpoint results are inspected. The frozen F16 H itself remains the supported XYZ authority.

## Provenance

- checkpoint SHA256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- execution source SHA256: `c293568554513e62c3aeedfd81cfb891cfcf52947690bfec1871dc41407ba54e`
- preflight JSON SHA256: `7414d44d662f82e6ad0c1d43b46f4f4cffcfa387ce77efcc05e15609467f32d8`
- result JSON SHA256: `9c5415ed12cdad3b77f50fcbb4c7027964e063a37410d87ba7853dddd6bda399`
