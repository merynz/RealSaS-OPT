# RealSaS N1D — Observation-Native Edge Reliability / Cross-Consensus Audit V1 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `NO_SIMPLE_OBS_EDGE_RELIABILITY_SIGNAL_V1`  
**Qualification:** truth-open diagnostic only; no solver intervention authorized.  
**Prereg commit:** `f08f788301eef559abf245da1e703bd52b50d2c4`

## Parent parity

All parent guards reproduce exactly before predictor interpretation:

```text
primary n                         261   PASS
bad nodes                          46   PASS
pair-sum pooled contain2         .8544061302681992 PASS
pair-sum worst-family contain2   .6800000000000000 PASS
pooled median oracle rank        .072265625 PASS
worst-family median oracle rank  .201171875 PASS
LOEO rescue <=.25                .45652173913043476 PASS
```

The evaluator damaging-edge label is used only for scoring. Every predictor uses frozen U_ONLY neighbor states and observation-native evidence only.

## S1 — RANK_VECTOR_DISAGREE

```text
top1 hit                    .26087   FAIL < .35
top2 hit                    .50000   FAIL < .60
median normalized rank      .45000   FAIL > .35
mean normalized rank        .44565   FAIL > .40
families mean rank <=.50     7/8     PASS
```

The incident edge whose conditional candidate-rank vector most disagrees with the incident median is not a reliable damaging-edge predictor.

## S2 — ENDPOINT_VOTE_OUTLIER

```text
top1 hit                    .30435   FAIL < .35
top2 hit                    .56522   FAIL < .60
median normalized rank      .33333   PASS
mean normalized rank        .45036   FAIL > .40
families mean rank <=.50     6/8     PASS
```

This is the strongest of the three simple signals but still misses three independent frozen gates. It is not supported for solver use.

## S3 — REL_FLOW_VIEW_INCONSISTENCY

```text
top1 hit                    .19565   FAIL
top2 hit                    .39130   FAIL
median normalized rank      .66667   FAIL
mean normalized rank        .57391   FAIL
families mean rank <=.50     4/8     FAIL
```

Cross-view relative-flow inconsistency does not identify the damaging incident edge under this contract.

## Decision

No preregistered observation-only predictor passes all support gates.

Canonical decision:

`NO_SIMPLE_OBS_EDGE_RELIABILITY_SIGNAL_V1`

## Scientific interpretation

The previous mixed multi-edge failure cannot be reduced to a simple per-edge reliability ranking using:

1. candidate-rank disagreement against incident consensus;
2. geometric outlier distance of an edge's independently preferred endpoint;
3. cross-view inconsistency of observed relative flow.

This materially weakens the hypothesis that the global solver can be repaired by assigning each existing R_REL_DIS edge one simple independent confidence score and then trimming/downweighting low-confidence edges.

The remaining evidence points toward **higher-order relational context**: which edge is useful depends on the joint configuration/context of several neighboring carriers, rather than a stable scalar property of that edge alone.

The next safe diagnostic should test higher-order structures already implicit in the observations—most directly triangle/cycle consistency or small local factor motifs—before any graph, relation or optimizer modification.

## Boundaries

Not authorized:

- edge deletion/downweighting from S1/S2/S3;
- combining the three scores post hoc;
- learned edge-confidence training on this panel;
- lambda/threshold sweeps;
- graph or R_REL_DIS changes;
- optimizer/restart changes;
- sealed21/external10 access;
- large/end-to-end retraining.

## Provenance

- prereg commit: `f08f788301eef559abf245da1e703bd52b50d2c4`
- execution source SHA256: `a3e88e970278a30ef8897729087a337dfbf39552a9bcacea4ea0d386c50e50f4`
- full local result SHA256: `a07dfe3ea10ed5723a941f95bd632cdfb93594811f01373b8c41b073dfd52b4a`
