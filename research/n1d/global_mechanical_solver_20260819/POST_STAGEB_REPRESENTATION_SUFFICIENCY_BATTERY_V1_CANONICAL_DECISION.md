# RealSaS N1D — Representation Sufficiency Battery V1 — Canonical Decision

**Date:** 2026-08-19  
**Decision:** `REPRESENTATION_SUFFICIENCY_SUPPORTED_FOR_REVISED_SET_VALUED_FACTORIZED_CONTRACT`  
**Qualification status:** unchanged; frozen Stage-B remains `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`  
**sealed21 / external10:** `CLOSED`

## Scope of this decision

This is a **truth-open development sufficiency decision**, not a product pass and not a blind generalization qualification. It answers the narrower question that motivated the battery:

> Do the paired raster observations and proposed mechanical representation appear to contain enough information to support hard-tail generalization, or are there families for which the required mechanical state is absent from the representation even under an oracle constrained to that representation?

The answer is now:

> **The current fixed-top4 representation contract is insufficient, but the revised set-valued factorized representation is supported by the battery.** The hard-tail failures found in this battery are attributable to premature hypothesis truncation, feature fusion/calibration, and the need for an explicit motion-state factor—not to a demonstrated absence of the required raster-derived information.

## Preregistered battery

Parent prereg commit: `b87e222fdf7bf4c66a62314c88ee9e155a576c73`.

Overall support required:

1. Test A not RED;
2. Test B not RED;
3. Test C not RED for the representation contract being considered;
4. Test D shows family-disjoint learnable signal without repeated family collapse.

The representation contract was causally revised after Test C proved the old fixed-top4 contract RED. The old RED remains an immutable finding; the revised contract is judged separately.

## Test C — hard-tail feasible-set coverage

### Old/current fixed-top4 contract: **RED**

Broad 8-family e00 panel, `492/512` reliably mapped carriers:

- pooled primary 2x H-containment: `0.93902`
- worst family `11032`: `0.84746`
- `13203`: `0.86885`
- only `6/8` families >= `0.90`
- best–worst gap: `15.25 pp`

Failure localization showed that target-near B evidence was frequently absent from retained top4 per-view candidates before H construction.

### Same frozen evidence, set-valued retention: **GREEN**

Counterfactual was preregistered before breadth results. Keep coarse top8 and the **same exact refined descriptor pixel pool**, but retain final top16 rather than top4.

- pooled primary 2x H-containment: `483/492 = 0.98171`
- worst family: `0.94915`
- `8/8` families >= `0.90`
- best–worst gap: `5.08 pp`
- pooled strict 1x containment: `0.94919`

Hard tails recover without changing model weights or adding a new observable channel:

- `11032: 0.84746 -> 0.94915`
- `13203: 0.86885 -> 0.95082`
- `15290: 0.90164 -> 1.00000`

A coarse-search expansion top8->top32 adds almost no gain; only `14758` changes primary containment (`0.98387 -> 1.00000`). Dominant failure mechanism is therefore **premature final hypothesis truncation**.

**Decision:** fixed-top4 H is forbidden; revised set-valued H survives Test C.

## Test A — representation-conditioned oracle: **GREEN**

On the same broad 8-family e00 panel, truth may choose only the best endpoint already present inside revised top16 `H_i`; it may not inject free XYZ truth.

Aggregate median GFDR:

- F activity: **0.94033**
- F kernel: **0.93813**
- D tangent error: **0.15481**
- R differential error: **0.11745**
- G direction: **0.96670**
- G line: **0.04074**

All **6/6 principal gates PASS**. Tail values remain above gate rather than collapsing: minimum family F activity `0.69840`, minimum family G direction `0.74245`, maximum family G line `0.06935`; no H is empty.

**Decision:** once target-near hypotheses are preserved, the revised representation has a strong mechanical oracle ceiling across the tested hard families.

## Test D — family-disjoint learnability: **SUPPORTED WITH CALIBRATION/FUSION WARNING**

### Episode-level population

Canonical open-development population: `232` episodes / `29` families.

- p_active proxy AUROC: `0.97411`
- multifeature log-amplitude LOFO Spearman: `0.84524`
- naive multifeature fusion collapses held-out `12832` to `-0.10714`
- but a single frozen raster-derived amplitude scalar (`predicted_flow_mean`) gives `12832 = +0.78571`, aggregate `0.87594`, median family `0.85714`, minimum family `0.50`, **29/29 evaluable families >=0.50**, zero negative families.

Thus the `12832` collapse is a fusion failure, not missing amplitude-order information.

### Carrier-level p_active

Using only frozen carrier current log-amplitude, strict 8-fold leave-one-family-out logistic probe:

- overall AUROC: **0.92948**
- minimum family AUROC: **0.89247**
- no repeated family ordering collapse.

A fixed probability threshold is not uniformly calibrated (for example `12772` AUROC `0.99679` but balanced accuracy at 0.5 `0.625`). Therefore `p_active` needs explicit calibration/confidence/coverage semantics.

**Decision:** family-disjoint motion-state and amplitude-order signal exists; the unresolved problem is tail-safe factorization/fusion/calibration rather than demonstrated signal absence.

## Test B — carrier collision audit: **AMBER, localized to planned p_active**

Carrier representation distance includes revised H summaries, descriptor Z, P_A/N_A/V_A, frozen current differential/log-amplitude and candidate uncertainty summaries. Nearest neighbors are cross-family only.

Closest 5% tail:

- material collision fraction: `0.14815`
- motion-state mismatch: `0.14815`
- both-moving amplitude ratio >3x: **0**
- both-moving direction cosine <0.5: **0**

Closest 10% is similar (`0.15385`), again exclusively motion-state mismatches.

Under the prereg rules this is **AMBER**, not RED. The collision type is exactly the explicit `p_active` factor already proposed but not instantiated in the collision vector as a learned/calibrated field. Independent carrier LOFO shows the underlying raster-derived current signal can rank this factor family-disjoint (overall AUROC `0.92948`, minimum family `0.89247`).

Witness inspection also shows the frozen current amplitude separates the moving/non-moving members of the closest collision pairs; the composite 37-D distance dilutes this single authority rather than proving the factor is unobservable.

**Decision:** no new unresolved amplitude/direction equivalence-class collision was found. Motion-state factor remains a calibration/head task.

## Final representation decision

The representation supported by this battery is:

```text
P_A
P_B_geom / bounded set-valued H_i
N_A, N_B
V_A, V_B
Z
p_active
log_amp
dir
U_pred
+ typed U_obs:
  reprojection_error
  view_support
  triangulation_condition
  feasible_set_width
  match_margin
  candidate_multimodality
  cycle_error
```

Authority is explicitly separated:

```text
bounded set-valued H_i              -> physical XYZ authority
p_active                             -> absolute motion/non-motion authority
log_amp                              -> conditional magnitude/ranking evidence
dir                                  -> direction evidence
U / margin / multimodality / support -> retain hypotheses, expand search, abstain
compiler/global solver               -> final collapse after global consistency
```

### Forbidden by the battery

- fixed early top4 candidate collapse;
- treating one fused scalar/vector head as simultaneous activity, amplitude and direction authority;
- free XYZ motion head overriding feasible geometry;
- interpreting aggregate success as tail sufficiency without family/hard-tail coverage;
- immediately scaling to large training because a median metric passes.

## What this decision authorizes

The battery now authorizes the **small next learning step**, not large end-to-end architecture training:

1. freeze a bounded uncertainty-aware set-valued candidate contract, with top16 retention as the demonstrated development reference and adaptive expansion for low-margin/multimodal cases;
2. train/evaluate the small strict family/episode-disjoint `p_active + log_amp` factorized head with explicit ranking loss, motion-state supervision, calibration/coverage and no free XYZ authority;
3. keep `dir` and geometry authority separated;
4. after the revised system is frozen, use a **new untouched preregistered panel** for qualification.

**Large/new architecture training remains not authorized until the bounded set contract and small factorized head are frozen.**

## Scientific answer to the original concern

The concern was justified: majority success did hide real hard-tail insufficiency. But the battery localized it.

The evidence does **not** currently support “our paired raster representation fundamentally cannot generalize.” It supports the more specific statement:

> **We were collapsing a sufficiently informative raster-derived hypothesis space too early, and sometimes fusing separable mechanical factors in a way that destroys tail information.**

Keeping the representation set-valued and factorizing motion-state, conditional amplitude and direction removes the demonstrated hard-tail failure mechanisms on the truth-open development evidence examined here.

That is sufficiency support, not a guarantee. Final generalization authority still belongs to a new untouched qualification panel.
