# RealSaS N1D — Global Relational Candidate Solver V1 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_SOLVER_TEST__RELATION_FROZEN`  
**Frozen geometry:** `BOUNDED_H_RESEARCH_CONTRACT_V1.md`  
**Frozen pair relation:** `R_REL_DIS` from Local Differential Relation Separability V1  
**Qualification:** unchanged; this is not a blind qualification.

## Purpose

Local Differential Relation Separability V1 showed that the evaluator-only correct H candidate pair has extremely low `R_REL_DIS` energy percentile across all eight open-development families, including the historical hard tails.

V1 now asks a separate question:

> With `R_REL_DIS` frozen, can a deterministic global candidate solver convert that pairwise evidence into materially better per-carrier endpoint selections than the observation-only unary selector?

**The relation formula, graph definition, candidate mode count, pair/unary normalization, pair weight and optimizer below are all frozen before solver endpoint results are inspected.** No solver/relationship co-tuning is allowed.

## Panel / truth boundary

Open-development e00 families:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Use the frozen checkpoint, observation-derived Problem-A carriers and frozen F16 H construction.

Truth is used only for candidate-mode containment preflight and endpoint evaluation after solver predictions are frozen. Truth never enters candidate compression, unary scores, graph construction, relation energies or optimization.

## Observation-only candidate modes M128

For every carrier independently:

1. construct frozen F16 H;
2. deduplicate candidate XYZ after rounding each coordinate to `1e-5`;
3. compute observation-only unary reprojection score:

```text
U_raw(i,h) = mean over Pose-A-usable views v of
             min_{q in frozen per-view top16 descriptor pixels}
             || project(h,v) - q ||_2
```

4. keep the best **32** unique candidates sorted by `(U_raw,x,y,z)`;
5. fill to **128** by deterministic XYZ farthest-point selection from remaining unique H candidates;
6. tie order: larger diversity distance, then lower U_raw, then lexicographic XYZ;
7. if H has fewer than 128 unique candidates, keep all.

This mode set is **M128**.

## Candidate-mode preflight gate

Before solver endpoint results are interpreted, M128 must satisfy on mapping-reliable carriers:

```text
pooled primary-2x containment >= 0.970
worst-family primary-2x containment >= 0.900
families primary-2x >=0.90 = 8/8
best-worst gap <= 10 percentage points
```

Report strict-1x. If this fails, verdict is `INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`; do not change M after solver results.

## Graph

Build the same deterministic undirected graph as the relation audit, but without truth gating:

1. every P_A carrier points to its four nearest other carriers;
2. symmetrize;
3. keep R_REL_DIS edge only when there are >=2 common Pose-A-visible views with finite raw DIS evidence.

The solver never sees truth-active labels. Later compiler integration will combine this with separately supported `p_active`.

## Frozen unary energy

Robust normalize within node:

```text
U(i,h) = [U_raw(i,h)-median(U_raw)] / max(IQR(U_raw),1e-6)
```

## Frozen pair relation R_REL_DIS

For edge `(i,j)` and pair `(h_i,h_j)`:

```text
candidate_relative_motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed_relative_motion(v) = DIS_i(v)-DIS_j(v)

R_raw = median_v ||candidate_relative_motion-observed_relative_motion||_2
```

Robust normalize over the full M_i x M_j pair matrix:

```text
R = [R_raw-median(R_raw)] / max(IQR(R_raw),1e-6)
```

No fusion, transport, delta3D, rigidity or learned relation in V1.

## Frozen global objective

For kept relation graph degree `deg_i`:

```text
w_ij = 1/deg_i + 1/deg_j

E(x) = sum_i U(i,x_i)
     + sum_(i,j) w_ij R(i,j,x_i,x_j)
```

No free lambda.

## Deterministic ICM V1

Initialization:

```text
x_i = argmin_h U(i,h)
```

Tie: lowest candidate index.

Maximum **20** sweeps. Odd sweeps visit `0->63`, even sweeps `63->0`.

At node i choose candidate minimizing:

```text
U(i,h) + sum_{j neighbor i} w_ij R(i,j,h,x_j)
```

Tie: lowest candidate index. Stop when a full sweep changes no node. No restart, randomization, annealing, lambda tuning or post-hoc iteration choice.

## Arms

- `U_ONLY`: M128 unary argmin before ICM.
- `G_REL_DIS`: final ICM configuration.

Report sweeps, changed-node counts and final objective.

## Primary endpoint evaluation

Primary evaluator-only set:

- mapping-reliable;
- truth-active (`||truth_P_B-truth_P_A|| > .005`);
- full F16 H truth-contained primary-2x;
- M128 truth-contained primary-2x.

The solver itself never sees these truth labels.

Report pooled/per-family:

```text
contain_1x
contain_2x
median normalized endpoint error
p90 normalized endpoint error
```

Hard tails: `11032`, `13203`, `15290`.

Also report all mapping-reliable M128-contained carriers as secondary safety panel.

## Absolute promotion gates for G_REL_DIS

```text
pooled contain_2x >= 0.75
worst-family contain_2x >= 0.60
11032 contain_2x >= 0.60
13203 contain_2x >= 0.60
15290 contain_2x >= 0.60
pooled contain_1x >= 0.50
worst-family contain_1x >= 0.35
pooled median normalized endpoint error <= 1.00
best-worst contain_2x gap <= 0.30
```

## Causal improvement gates vs U_ONLY

Required:

```text
pooled contain_2x gain >= +0.08
```

and either:

```text
worst-family contain_2x gain >= +0.08
```

or:

```text
at least 2 of {11032,13203,15290} improve contain_2x by >= +0.10
and none of the three degrades by >0.05
```

Secondary all-reliable M128-contained safety:

```text
G_REL_DIS pooled contain_2x must not be lower than U_ONLY by >0.03
```

## Decision

If M128 preflight passes and G_REL_DIS passes absolute + causal + safety gates:

`GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_PASS`

Then freeze solver mechanics before unchanged downstream GFDR evaluation with separately handled p_active/silence semantics.

If M128 passes but solver fails: `RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL`; diagnose optimization/objective/candidate interaction in a separately preregistered solver experiment, without reopening standalone amplitude/vector heads.

If M128 fails: `INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`.

## Forbidden

- changing M128 after endpoint results;
- changing pair/unary normalization;
- fitting/tuning lambda;
- swapping in R_REL_FUSION post hoc;
- relation learning;
- backbone/head training;
- truth-active gating inside solver;
- free XYZ;
- Stage-B requalification;
- large/end-to-end training.
