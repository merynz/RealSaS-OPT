# RealSaS N1D — Local Differential Relation Separability V1 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_RELATIONAL_DIAGNOSTIC__NO_SOLVER_TUNING`  
**Parent result:** `FAIL__SINGLE_CARRIER_MOTION_VECTOR_INSUFFICIENT`  
**Frozen geometry authority:** F16 Bounded H Research Contract V1

## Question

Single-carrier scalar amplitude, H-relative amplitude, candidate-specific amplitude and candidate-specific full-vector motion all fail family-robust hard-tail selection.

V1 tests the next representation level:

> Does the correct feasible configuration become distinguishable through **carrier-to-carrier local differential/mechanical relations**, even when neither carrier can be resolved reliably in isolation?

This is a separability audit only. It does not fit or tune a graph solver.

## Carrier graph

For each family, reconstruct the frozen observation-derived 64 Pose-A carriers `P_A`.

Build a deterministic undirected graph:

1. for each carrier, find its four nearest other carriers in Euclidean `P_A` distance;
2. symmetrize the directed 4-NN edges;
3. keep an edge for the primary audit only when both endpoints are mapping-reliable, both F16 H sets contain their truth endpoints under the existing primary-2x criterion, and **at least one endpoint is truth-active** (`truth_amp > .005`).

Truth-active is evaluator-only in this diagnostic; the intended inference analogue is the already-supported `p_active` factor.

Report active-active and active-inactive edge strata separately.

## Oracle candidates — evaluator only

For each retained carrier `i`:

```text
h_i* = argmin_{h in H_i} ||h - truth_P_B(i)||
```

Truth is used only to identify the oracle pair for percentile evaluation. Every relation energy below is observation/geometry-native.

## Alternative candidate-pair set

Do not enumerate the full H_i x H_j product.

For each H_i independently, construct an observation-only deterministic 32-mode sample:

1. deduplicate endpoints after rounding XYZ to `1e-5`;
2. seed with the lexicographically smallest XYZ endpoint;
3. greedily add the endpoint with maximum minimum Euclidean distance to the selected set;
4. stop at 32 or when all unique endpoints are selected.

For an edge `(i,j)`, evaluate the Cartesian product of the two mode samples (<=1024 candidate pairs). Truth does not affect mode sampling.

The oracle pair energy is computed separately and compared with the sampled alternative energy distribution.

## Relation energies

Let:

```text
d_i(h_i) = h_i - P_A_i
```

and `L_ij = ||P_A_i-P_A_j||`.

### R_DISP

Local displacement coherence:

```text
E = ||d_i(h_i) - d_j(h_j)|| / max(L_ij, eps)
```

Lower is more locally coherent.

### R_RIGID

Local metric/rigidity residual:

```text
E = | ||h_i-h_j|| - L_ij | / max(L_ij, eps)
```

This tests whether the candidate pair preserves the local rest metric.

### R_REL_DIS

For each common Pose-A-visible view `v`:

```text
candidate_relative_motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed_relative_motion(v) = DIS_i(v) - DIS_j(v)
```

Energy is median native-pixel L2 residual across common views. Require >=2 common views.

### R_REL_TRANSPORT

Same relative-motion energy using frozen learned `transport_offset_srcA` converted to native pixels.

### R_REL_DELTA3D

Same relative-motion energy using per-view projected `delta_point_map_srcA` motion.

### R_REL_FUSION

For each edge and relation term among:

```text
R_RIGID
R_REL_DIS
R_REL_TRANSPORT
R_REL_DELTA3D
```

robust-normalize sampled-pair energies by the sampled-pair median/IQR. Apply the same normalization to the oracle pair energy, then take the median across terms. No fitted weights.

`R_DISP` is intentionally excluded from fusion because articulated local motion can produce real displacement differences across nearby carriers; it remains diagnostic.

## Primary metric — oracle pair energy percentile

For an edge and arm:

```text
percentile =
  (# sampled alternative pairs with lower energy
   + .5 * # exact-energy ties) / # finite sampled pairs
```

Lower is better. A value near `.5` is uninformative; near `0` means the oracle pair is strongly preferred by the relation.

Report per family and pooled:

```text
median oracle percentile
fraction oracle percentile <= .25
fraction oracle percentile <= .10
```

Also report active-active and active-inactive strata.

## Support gates

A relation arm is considered sufficiently separable to justify building a graph/global solver only if all hold:

```text
pooled median oracle percentile <= .25
worst-family median oracle percentile <= .40
pooled fraction percentile <=.25 >= .60
worst-family fraction percentile <=.25 >= .45
11032 median percentile <= .40
13203 median percentile <= .40
15290 median percentile <= .40
```

If a stratum has >=20 pooled edges:

```text
active-active median percentile <= .35
active-inactive median percentile <= .35
```

## Decision

1. If one or more observation-relative arms pass, prefer the simplest single passing arm unless R_REL_FUSION improves worst-family median percentile by >=.05.
2. If only R_RIGID passes, support a geometry-only local mechanical factor but do not claim the motion front door is solved.
3. If no arm passes, pairwise local relation alone is insufficient; move to a richer local Jacobian/patch relation or multi-carrier learned relation representation before any graph solver.
4. If an arm passes, the next step is a separately preregistered discrete/global solver using frozen H and the supported relation; do not tune solver and relation simultaneously.

## Boundaries

- no model/head training;
- no truth-conditioned relation energy;
- no free XYZ;
- no solver fitting/tuning in V1;
- no Stage-B requalification;
- no large/end-to-end training.
