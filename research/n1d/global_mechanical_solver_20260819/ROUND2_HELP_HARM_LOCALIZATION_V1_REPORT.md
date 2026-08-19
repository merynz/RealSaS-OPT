# RealSaS N1D — Round-2 Help-vs-Harm Localization V1 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `NO_SIMPLE_ROUND2_HELP_HARM_LOCALIZER_V1`  
**Prereg commit:** `0bad17f775f8e8e67cce1b1cefaebe1a7a54e2b2`  
**Qualification:** truth-open diagnostic only; no solver intervention authorized.

## Validity / parent P1 parity

The family-level execution reproduces canonical P1 exactly:

```text
primary n                         261
round1 vs canonical P0 indices   512/512
round1 vs canonical P1 indices   512/512
round2 vs canonical P1 indices   512/512
round1 pooled contain2           .7777777778
round2 pooled contain2           .8160919540
round1 worst-family contain2     .4000000000
round2 worst-family contain2     .5200000000
12772 round2 contain2            .7272727273
11032 round2 contain2            .5200000000
```

Therefore the diagnostic is valid.

## Evaluator-only help/harm labels

For each primary node:

```text
delta_err = norm_error_round2 - norm_error_round1
HELP    if delta_err <= -.25
HARM    if delta_err >= +.25
NEUTRAL otherwise
```

Observed strata:

```text
HELP       29
NEUTRAL   215
HARM       17
HELP+HARM  46
```

Contain2 transitions:

```text
round1 bad -> round2 good   14
round1 good -> round2 bad    4
```

Truth is used only for these outcome labels / endpoint evaluation. D1-D4 are observation-native.

## D1 — MESSAGE_AMBIGUITY

Higher score means a larger fraction of M256 candidates remain near the optimum of incoming round-1 pair messages.

```text
Spearman rho(score, delta_err)   .03903   FAIL < .20
HARM-vs-HELP AUROC               .63590   FAIL < .65
HELP median                      .08203
NEUTRAL median                   .09375
HARM median                      .15234
```

The HARM median is higher, but the effect is not broad/monotonic enough to satisfy the frozen support gate.

Family 12772 median score percentile among the eight family medians: `.4375`.

## D2 — TRIANGLE_OVERLAP

D2 is the fraction of two-hop cavity paths `q -> j -> i` whose source q is also a direct neighbor of i, i.e. a graph-triangle overlap path.

```text
Spearman rho                    -.02690   FAIL
HARM-vs-HELP AUROC               .50000   FAIL
HELP median                      .42857
NEUTRAL median                   .42105
HARM median                      .41379
```

Simple triangle-path prevalence does not explain round-2 harm. Family 12772 is actually near the low end of triangle overlap (`.0625` percentile among family medians), contradicting a simple “12772 suffers because it has more triangles” explanation.

## D3 — MESSAGE_VOTE_DISAGREEMENT

D3 measures geometric dispersion among the M256 endpoints independently preferred by incoming round-1 messages.

```text
Spearman rho                    -.06006   FAIL
HARM-vs-HELP AUROC               .37323   FAIL
HELP median                      .68204
NEUTRAL median                   .44392
HARM median                      .54149
```

The expected positive relation is absent. Simple incident-message geometric disagreement is not a supported harm localizer.

## D4 — TRIANGLE_REL_FLOW_CLOSURE

For every graph edge, an observation-only 3D relative displacement is fit from relative DIS. Triangle closure measures normalized failure of those fitted relative displacements to close around local graph cycles.

```text
Spearman rho(score, delta_err)   .09091   FAIL < .20
HARM-vs-HELP AUROC               .68154   PASS >= .65
HELP median                      .07952
NEUTRAL median                   .14558
HARM median                      .18323
```

D4 is the strongest directional clue: HARM nodes tend to have worse relative-flow cycle closure and its HELP/HARM ranking is good. However the preregistered support gate requires **both** AUROC and pooled monotonic Spearman support. The pooled rho is only `.091`, so D4 is not promoted.

Family 12772 has relatively high D4 (`.8125` percentile among family medians), while 11032 is low; this is descriptive evidence consistent with some cycle-quality contribution, but not sufficient under the frozen gate.

## Preregistered decision

No diagnostic passes all support gates:

```text
D1 MESSAGE_AMBIGUITY       FAIL
D2 TRIANGLE_OVERLAP        FAIL
D3 MESSAGE_DISAGREEMENT    FAIL
D4 REL_FLOW_CYCLE_CLOSURE  FAIL
```

Canonical verdict:

`NO_SIMPLE_ROUND2_HELP_HARM_LOCALIZER_V1`

## Scientific interpretation

The round-2 heterogeneity cannot be reduced to any one of the four tested simple observation-native scalar explanations.

The strongest residual clue is **relative-flow cycle consistency**: it separates the 29 HELP vs 17 HARM nodes reasonably well (`AUROC .682`) and 12772 has high family-level cycle-closure error, but the signal is not monotonic enough over the full primary population for promotion.

Combined with the parent chain, this means:

1. full candidate-to-candidate pair structure is useful;
2. one and two cavity rounds produce real causal gains;
3. additional depth is heterogeneous;
4. neither simple edge reliability, symmetric incident summaries, raw triangle density, message ambiguity, message-vote spread, nor simple cycle-closure scalar is sufficient to decide when propagation should be trusted.

The remaining defect is therefore more likely **structured factor-context interaction** than a single scalar gate. The next move should not be a round-depth sweep or another hand-made threshold. A safe next diagnostic should test whether the existing pair factor itself is missing a higher-order invariant that only becomes visible on triples/local motifs, while preserving M256 as XYZ authority.

## Boundaries

Not authorized:

- gating round 2 with D1-D4;
- suppressing triangle edges;
- changing graph topology;
- testing round 3+;
- damping/softmin/lambda sweeps;
- retuning R_REL_DIS on this same panel;
- fitting a learned help/harm classifier from these four scores;
- M256/F16 changes;
- free XYZ;
- sealed21/external10 access;
- large/end-to-end retraining.

## Provenance

Execution used a family-level exact-parity worker to avoid repeatedly rebuilding large M256 pair matrices under the environment execution limit. This changes execution batching only, not scientific semantics.

- prereg commit: `0bad17f775f8e8e67cce1b1cefaebe1a7a54e2b2`
- diagnostic semantics source SHA256: `86e246bf4d04cdbf59060b7b33265bbe1c1ded4e5a3d23a94a2f26e42d9535af`
- family worker SHA256: `c8150fce49fdd9070d21c356b6bcb8c66e586277d8f78344c0b40a9eb44daada`
- aggregate combiner SHA256: `4905c32ba0e93fd71dbe607490d7f31b2afd1001201fc49254d66cc5117c1966`
- full local result SHA256: `dfa77f0438fba12aaffe4700732b572afb41a33505294a7e76fa45f784da6572`
