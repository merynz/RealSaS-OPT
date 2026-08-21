# RealSaS Rank2 G2 — Reciprocal Cycle Closure V1

**Date:** 2026-08-21  
**State:** `QUESTION_FROZEN__G2_TRUTH_CLOSED`

## Parent objective

This experiment is governed by `canonical/MODEL1_2P5D_MECHANICAL_SURFACE_CONTRACT.md`.

Model-1 is constructing an observation-grounded, functionally sufficient 2.5D mechanical surface from paired 8+8 raster views. The purpose of this line is not literal 3D reconstruction and not teacher-latent recovery. The frontend must expose enough observation-native evidence to distinguish mechanically relevant surface proposals reliably before downstream skeleton/skinning compilation.

## Parent treatment

G1 established that explicit multi-view geometry is a useful proposal-level discriminator when added to the frozen descriptor proposal surface while preserving V5 global relational `R`.

The next controlled question is whether **reciprocal descriptor association expressed as cycle-back error** supplies additional independent observation-native discrimination beyond G1.

## Scientific question

Given the exact G1 proposal support, known cameras, frozen descriptor checkpoint, G1 geometry construction, V5 full-pair `R`, matching cardinality and solver semantics, does adding a same-view B→A reciprocal descriptor cycle factor improve proposal selection on a separate truth-closed surface?

For carrier `i`, view `v`, and a frozen B proposal `q_B`:

1. sample the frozen B descriptor at `q_B`;
2. search that descriptor against the Pose-A foreground descriptor field in the same view using the exact canonical coarse-to-fine descriptor-search semantics;
3. obtain the reverse association `q_hat_A`;
4. compare it with the known observation-native Pose-A carrier projection `q_A = XY_A[v,i]`;
5. define the new raw evidence as

```text
cycle_px(i,v,q_B) = || q_hat_A - q_A ||_2
```

The new factor is the stable within-row rank of `cycle_px` ascending.

## Controlled change

G1 proposal cost:

```text
G1 = 1/2 descriptor_rank + 1/2 geometry_rank
```

G2 proposal cost:

```text
G2 = 1/3 descriptor_rank + 1/3 geometry_rank + 1/3 reciprocal_cycle_rank
```

No weight is fitted. Equal weighting is fixed before any G2 truth access.

The reciprocal match similarity may be persisted only as diagnostic provenance for the reverse lookup. It is **not** a second G2 scoring factor.

Everything else is frozen:

- proposal top-k support and forward descriptor-search semantics;
- N1D checkpoint;
- calibrated 8-view camera contract;
- G1 triangulation/reprojection/conditioning construction;
- V5 full-pair relational `R` formula;
- maximum-real-matching cardinality `K`;
- 16 deterministic restarts;
- strict-descent move/swap/real-abstain exchange solver;
- no F/D factor;
- no proposal creation or pruning;
- no camera fitting;
- no teacher field in forward inference.

## G2 prospective surface

The separately frozen G2 confirmation families are:

```text
10287, 15521, 16638, 15284,
13015, 14703, 16414, 16528,
16036, 15226, 14315, 12949,
14405, 11447, 13843, 14885
```

The experiment uses the canonical Rank2 episode contract used by the G1 line. Evaluation sidecar semantics for this G2 run remain CLOSED until source, tests, evidence, baseline G1 selections, G2 selections and preregistration are frozen.

`sealed21 = CLOSED`  
`external10 = CLOSED`

## Claim boundary

A G2 PASS would establish that reciprocal/cycle evidence is an additional useful deterministic frontend primitive for the 2.5D mechanical-surface construction. It would **not** establish learned generalization of a new network module.

A G2 FAIL does not authorize same-surface retuning. It would mean the reciprocal-cycle treatment as formulated is not promoted; the next treatment must be preregistered on a new surface.
