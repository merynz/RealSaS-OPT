# RealSaS-OPT — Current State

**Date:** 2026-08-18

## Active line

IRIS-SEES post-N1D continuation. Product objective is mechanical sufficiency, not exact teacher latent reconstruction. IRIS-SEES measures observation-native evidence; deterministic geometry/mechanics and the compiler derive downstream structured quantities.

## N1D baseline

Canonical N1D remains an informative FAIL as originally sealed. Important preserved signal:
- cross-pose descriptor top1 ≈ 0.8904;
- Pose-B is genuinely used;
- activity/amplitude ranking is learned more strongly than world-vector direction;
- near-zero silence and exact same-pose/swap invariants are preserved.

The original blocker was not absence of observable persistence evidence. N1D's mechanics path failed to consume the descriptor evidence correctly.

## Post-N1D causal localization

Closed dead ends include longer training, simple output calibration/fusion, hard current-vs-Problem-A selectors, raw DIS ordinary LS/LMedS 3D lift, forward/backward cycle weighting, and early singleton collapse of sparse candidate sets.

Critical localization:
- calibrated-camera 3D inversion is well-conditioned;
- a minority of large raw-DIS correspondence errors caused DIS-centered local search to erase correct candidates;
- fixed ±8 px DIS-centered search contained the true endpoint only ~85.34% of visible-view cases;
- conditional on the endpoint being in-window, the frozen N1D descriptor retained it in top-4 ~84.79% of the time.

## Authorized correspondence / world-response route

Frozen treatment:

```text
Pose A/B rasters
 -> Problem-A A visual-hull carriers
 -> frozen N1D persistent descriptor
 -> global observable B-foreground search
 -> top-k candidate set per view
 -> calibrated multiview rank-3 3D hypothesis solve
 -> frozen N1D activity/silence contract
```

16-family development:
- flow/zero **0.7409 → 0.4411**
- weighted direction **0.5839 → 0.8994**
- direction non-regress **14/16**
- non-abstain **99.90%**
- false activation **41 → 39**
- **5/5 gates PASS**

Previously untouched four-family e00 qualification (`10763,11214,12907,14714`):
- flow/zero **0.7439 → 0.6716**
- weighted direction **0.5476 → 0.7403**
- direction non-regress **3/4**
- coverage **100%**
- false activation **26 → 23**
- **5/5 gates PASS**

This route is authorized on open development as the post-N1D correspondence/world-response treatment.

## Causal downstream mechanics closure

### Exact-observation-geometry causal replay

Only response was replaced; exact observation geometry/normals/visibility were frozen for evaluator isolation.

Global-foreground response:
- D tangent error **0.148326**
- F activity **0.843860**
- F kernel **0.793054**
- R differential **0.104815**
- G direction **0.880993**
- G line / diag **0.087942**
- **9/9 canonical N1D GFDR-V2 gates PASS**

Frozen-current response under the same exact geometry had G line / diag **0.154159**, confirming that the old G-line failure was caused by correspondence/world-response rather than the downstream GFDR-V2 operators.

### Full observable front door

Prediction uses:

```text
Problem-A X / V_A
+ frozen N1D descriptor and normal heads
+ global B-foreground correspondence search
+ deterministic multiview 3D response
+ deterministic B visual-hull V_B
-> GFDRV2.ObservableMechanics.v4
```

20-family e00 open-development result:
- D tangent error **0.148326**
- F activity **0.843860**
- F kernel **0.813066**
- R differential **0.105003**
- G direction **0.890067**
- G line / diag **0.084111**
- **9/9 canonical N1D GFDR-V2 gates PASS**

Primitive front-door estimates are not exact (median family diagnostics: P_A error/diag ~0.0274, P_B ~0.03935, N_A ~26.8°, N_B ~29.6°), yet downstream mechanics passes. This supports:

```text
mechanical sufficiency > exact latent reconstruction
```

The old N1D point-map hard gates are therefore not automatically product blockers for this repaired hybrid route; geometry authority is supplied by the deterministic Problem-A front door rather than the failed N1D point-map head.

## Active next gate

The remaining open-development risk is **intervention generalization**. Current closure evidence is e00-focused.

Next required qualification: freeze the exact current route and evaluate previously unopened `e01..e07` interventions without retuning. Only after a multi-intervention open-dev PASS may sealed21/external10 be considered.

**sealed21 = CLOSED**  
**external10 = CLOSED**  
**product/model-2 handoff = NOT YET AUTHORIZED**

## Data / workspace policy

- GitHub: canonical code, prereg, compact results, reports, experiment history.
- Drive: heavy corpus/cache/checkpoint/proof-pack depot.
- Library: transient active diagnostics and working artifacts.

Never silently redefine structural GFDR semantics to fit an implementation. N1D `GFDRV2.ObservableMechanics.v4` diagnostics are downstream mechanics diagnostics, not a replacement definition of the older structural GFDR ontology.
