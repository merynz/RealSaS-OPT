# RealSaS N1D — Round-2 Help-vs-Harm Localization V1 — Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DIAGNOSTIC_PREREG__NO_SOLVER_CHANGE`

## Parent result

Parent P1: `SECOND_ROUND_NOT_MATERIAL__FACTOR_OR_OBJECTIVE_INSUFFICIENCY_REMAINS`, prereg commit `85267a10f0cc53451ee9101b1d21338366aa4260`.

P1 preserves full M256×M256 pair structure and adds exactly one cavity round. Round 2 improves aggregate primary contain2 `.77778 -> .81609` and worst-family `.40 -> .52`, but effects are heterogeneous:

```text
11032 +.12000
14404 +.18750
12772 -.09091
```

This diagnostic asks why the exact same propagation rule helps some nodes/families and harms others, before any further solver change.

## Frozen state

Exactly reproduce P1:

- F16/M256 candidate domains/order;
- U_raw and V2 robust-z calibration;
- graph/edge eligibility;
- raw R_REL_DIS and robust-z calibration;
- degree weights;
- synchronous cavity round-1 and round-2 messages;
- P1 endpoint selections.

No new solver arm is permitted.

## Validity guards

Before interpreting any diagnostic reproduce:

```text
round1 candidate-index parity      512/512
primary n                          261
round1 primary contain2            .7777777778
round2 primary contain2            .8160919540
round1 worst-family contain2       .4000000000
round2 worst-family contain2       .5200000000
round2 12772 contain2              .7272727273
round2 11032 contain2              .5200000000
```

Any mismatch invalidates the diagnostic until parity is repaired without changing this preregistration.

## Evaluator-only outcome

For each primary node:

```text
delta_err_i = norm_error_round2 - norm_error_round1
```

Negative is improvement; positive is harm.

For a nonparametric help/harm ranking audit define:

```text
HELP if delta_err <= -.25
HARM if delta_err >= +.25
NEUTRAL otherwise
```

The `.25` threshold is fixed before inspection and is one quarter of evaluator local scale. HELP/HARM labels and `delta_err` are evaluator-only.

## Observation-native diagnostic D1 — MESSAGE_AMBIGUITY

Use incoming **round-1** directed messages to node i. For each incident message vector `m_e(k)`:

```text
IQR_e = q75(m_e)-q25(m_e)
amb_e = fraction_k [ m_e(k) <= min(m_e) + .25 * max(IQR_e,1e-6) ]
```

Node score:

```text
D1_i = median_e amb_e
```

Higher means the incoming pair message leaves a larger fraction of M256 candidates near its optimum; expected sign for harm is positive.

## Observation-native diagnostic D2 — TRIANGLE_OVERLAP

Let `N(i)` be the frozen graph neighbors. Enumerate directed two-hop cavity paths `q -> j -> i` with `j in N(i)` and `q in N(j), q != i`.

```text
D2_i = fraction of such paths where q is also in N(i)
```

These are paths that close a graph triangle and can carry correlated evidence back into i through a different neighbor. If no eligible two-hop paths exist, define zero. Higher is expected to increase loop double-counting risk.

## Observation-native diagnostic D3 — MESSAGE_VOTE_DISAGREEMENT

For each incoming round-1 message `m_{j->i}`, choose its lowest-cost central candidate:

```text
q_e = argmin_k m_{j->i}(k)
P_e = M_i[q_e]
```

Compute median pairwise distance among `{P_e}` and normalize by the median pairwise distance among all M256 candidates of node i:

```text
D3_i = median_pair_dist(P_e) / (median_pair_dist(M_i) + 1e-9)
```

For degree <2 define zero. Higher means incident messages vote for geometrically incompatible regions of the feasible candidate set.

## Observation-native diagnostic D4 — TRIANGLE_REL_FLOW_CLOSURE

For every graph edge `(a,b)`, fit one 3D relative displacement `g_ab` by ordinary least squares from observed relative DIS over the edge's frozen common visible views using the known orthographic right/up projection rows.

Orient `g_ba=-g_ab`.

For every graph triangle `(i,j,k)` containing node i, compute:

```text
closure = ||g_ij + g_jk + g_ki||
          / (||g_ij||+||g_jk||+||g_ki||+1e-9)
```

Node score `D4_i` is the median closure over its incident triangles; zero if no triangle exists. Higher means pairwise observed relative motions are less mutually compatible as a local 3D cycle.

## Scoring

No feature is fitted or combined.

For each D1-D4 report:

1. Spearman correlation between score and evaluator `delta_err` over all 261 primary nodes;
2. AUROC for HARM vs HELP after dropping NEUTRAL nodes, with higher score predicting HARM;
3. median score in HELP / NEUTRAL / HARM;
4. per-family median score and median `delta_err`;
5. the score percentile of family 12772 relative to the eight family medians.

If fewer than 20 total HELP+HARM nodes exist, AUROC support for all diagnostics is automatically false but descriptive values are still reported.

## Frozen support gate per diagnostic

A diagnostic is `SUPPORTED` only if all hold:

```text
Spearman rho(score, delta_err) >= +.20
HARM-vs-HELP AUROC            >= .65
HARM median score > HELP median score
HELP+HARM n                   >= 20
```

No absolute-value correlation, sign flip, score inversion or threshold relaxation is allowed.

## Classification

- D1 supported -> `PAIR_MESSAGE_AMBIGUITY_SUPPORTED`
- D2 supported -> `TRIANGLE_LOOP_OVERLAP_SUPPORTED`
- D3 supported -> `INCIDENT_MESSAGE_CONFLICT_SUPPORTED`
- D4 supported -> `REL_FLOW_CYCLE_INCONSISTENCY_SUPPORTED`

If two or more diagnostics are supported:

`MIXED_PROPAGATION_HETEROGENEITY_SUPPORTED`

If exactly one is supported, use its corresponding classification.

If none are supported:

`NO_SIMPLE_ROUND2_HELP_HARM_LOCALIZER_V1`

If validity fails:

`INVALID_P1_PARITY_FAIL`

## Boundaries

This audit cannot:

- alter messages or endpoints;
- suppress triangles/edges;
- downweight ambiguous factors;
- combine D1-D4 after results;
- fit a classifier;
- tune thresholds;
- test round 3+;
- change U/R weights, graph, R_REL_DIS, M256 or F16;
- access sealed21/external10;
- authorize large/end-to-end training.

A supported diagnostic may motivate exactly one separately preregistered causal intervention. If none is supported, no simple observation-native gating of round-2 propagation is authorized from this panel.
