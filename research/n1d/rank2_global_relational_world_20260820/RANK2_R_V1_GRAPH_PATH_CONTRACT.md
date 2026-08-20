# RealSaS N1D Rank-2 Global Relational World — V1 Graph-Path Address Contract

Date: 2026-08-20
Parent: V0 CLOSED/FAIL AS FORMULATION (`RANK2_R_V0_DECISION.md`)
Panel: exact frozen R6.3 e01 observable panel.

## Result-driven change from V0

V0 proved that raw cross-pose 3D pair-vector equality is the wrong invariance under articulation: it produced stable unique optimization yet materially degraded mechanics in 20/23 hard-tail witnesses.

V1 keeps the user's semantic invariant — a carrier is located by its address in a global relational structure — but replaces raw extrinsic pair equality by an **intrinsic graph-path address**.

This is motivated independently by the historical structured-R ontology, which included a multi-view consensus observation graph and graph-path address to observation-derived anchors.

## Frozen candidate population

Identical to V0:
- exact R6.3 `H_i` sets, 64 carriers/family;
- same observable G reduction and top-domain rule;
- no teacher-created candidate;
- sealed21/external10 closed.

## Observation graph

Construct one deterministic Pose-A carrier graph from `P_A` only:
- undirected symmetrized 8-nearest-neighbour graph;
- edge length = Euclidean Pose-A carrier distance;
- topology is frozen before candidate optimization.

No Pose-B teacher geometry enters graph construction.

## Anchors

Select 20 carriers with the sharpest retained `H_i` support using the same observation-only covariance spread statistic as V0, deterministic ties by G rank then carrier index. The anchor candidate realization is the retained candidate nearest the frozen observable baseline `P_B` for that anchor. Anchors are fixed during V1 optimization.

Twenty anchors are used because historical structured-R operationalization used a broad anchor set; V1 does not require exact historical implementation parity.

## Relational address

For Pose A, compute shortest-path distances from every carrier to every anchor over the frozen graph using Pose-A edge lengths.

For a candidate Pose-B world, keep the same graph topology but replace each edge length by Euclidean distance between its selected B candidate endpoints. Compute the corresponding shortest-path distance matrix to the same anchors.

Normalize both address matrices by the median finite non-zero Pose-A graph-path distance. V1 relational loss is robust Huber error between the full `64 x 20` A and B graph-path address matrices.

This removes global translation/rotation and does not penalize rigid articulation orientation. It measures whether the selected B realization preserves the carrier's intrinsic relational address in the surface graph.

## G/R scale calibration

No fixed truth-tuned lambda is used. Before optimization, starting from the observation-only min-G world:
1. compute the positive one-coordinate change distribution in global G unary energy over all retained alternatives;
2. compute the positive one-coordinate change distribution in graph-path R energy over the same alternatives;
3. set `G_scale` and `R_scale` to the respective robust median positive changes, with deterministic epsilon floors.

Optimize

`E_V1 = E_G / G_scale + E_R_graphpath / R_scale`.

This gives G and R equal typical one-coordinate influence using observable evidence only.

## Optimization

- deterministic coordinate descent over non-anchor carriers;
- 12 deterministic observation-only multi-starts;
- maximum 30 sweeps;
- ties choose lower retained-candidate index.

For every coordinate candidate, the exact full graph-path R energy is recomputed. No local proxy is substituted for the final objective.

## Pre-truth outputs

Per family:
- graph edges and anchors;
- G/R perturbation scales;
- selected original H index for all 64 carriers;
- objective decomposition;
- restart agreement;
- local exact counterfactual margin under the full objective;
- baseline descriptive graph-path R energy.

## Evaluator

After source + observable selection are committed, evaluate with the already-frozen V0 truth evaluator protocol adapted only to read the V1 selected candidate indices. Population must exactly match the R6.3 23 hard-tail carriers.

Primary open-dev comparisons:
- hard-tail material mechanical improvement/degradation vs frozen R6.3 baseline;
- median truth-relative GFDR-V2 composite delta;
- geometry improvement vs baseline;
- reliable+active safety degradation.

## Interpretation

V1 tests whether the set-valued Rank-2 ambiguity can be collapsed by a global **intrinsic address** rather than an extrinsic pose-equality constraint. Failure falsifies this graph/path operationalization, not the general R concept. A successful e01 result remains open-development evidence and requires a later frozen episode/family qualification before product claims.
