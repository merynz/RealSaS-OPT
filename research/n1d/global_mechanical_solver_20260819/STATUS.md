# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_SUFFICIENCY_SUPPORTED_FOR_REVISED_SET_VALUED_FACTORIZED_CONTRACT__SMALL_HEAD_AUTHORIZED__BIG_TRAINING_NOT_AUTHORIZED`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; its primary F-activity was `0.433160 < 0.50` while the other principal geometry/mechanics metrics passed.
- Stage-B truth is truth-open development evidence only and cannot be reused as blind qualification.
- sealed21 / external10 remain CLOSED.
- Any future blind qualification requires a **new untouched preregistered panel**.

## Representation sufficiency battery V1

Canonical decision: `REPRESENTATION_SUFFICIENCY_SUPPORTED_FOR_REVISED_SET_VALUED_FACTORIZED_CONTRACT`.

This is a truth-open scientific sufficiency decision, **not** a product or qualification PASS.

### Test C — current compact candidate contract

The current fixed-top4 global-foreground `H_i` contract is **RED** on the broad 8-family e00 panel.

Across `492/512` reliably mapped carriers:

- pooled primary 2x containment: `0.93902`
- `11032`: `0.84746`
- `13203`: `0.86885`
- families >=0.90: `6/8`
- best–worst gap: `15.25 pp`

Failure localization shows the dominant miss occurs before downstream H selection: target-near descriptor evidence is discarded by final top4 retention.

**Fixed early top4 collapse is forbidden by this evidence.**

### Test C' — revised set-valued contract

Preregistered retention-only counterfactual changes only final retention `top4 -> top16`; coarse top8, refined pixel pool, model and descriptor remain frozen.

Broad 8-family result:

- pooled primary 2x containment: `483/492 = 0.98171`
- worst family: `0.94915`
- families >=0.90: `8/8`
- best–worst gap: `5.08 pp`
- pooled strict 1x containment: `0.94919`

Hard tails recover without new model evidence:

- `11032: 0.84746 -> 0.94915`
- `13203: 0.86885 -> 0.95082`
- `15290: 0.90164 -> 1.00000`

Broader coarse search (`8 -> 32`, final16) adds little; only `14758` gains primary coverage (`0.98387 -> 1.00000`). Dominant issue is premature truncation, with occasional coarse-beam narrowness secondary.

**Revised set-valued Test C is GREEN on 8/8 e00 families.**

### Test A — representation-conditioned oracle

Oracle may select only an endpoint already present inside revised top16 `H_i`; no free truth XYZ is injected.

Broad 8-family median GFDR:

- F activity `0.94033`
- F kernel `0.93813`
- D `0.15481`
- R `0.11745`
- G direction `0.96670`
- G line `0.04074`

**6/6 principal gates PASS.** Tail values remain above gate: minimum family F activity `0.69840`, minimum family G direction `0.74245`, maximum family G line `0.06935`; no H is empty.

**Test A = GREEN on revised 8-family e00 panel.**

### Test B — carrier collision audit

Representation distance uses revised H summaries + descriptor Z + P_A/N_A/V_A + frozen current differential/log-amplitude + descriptor margin/spread/entropy/support. Cross-family nearest-neighbor only.

Closest 5%:

- material collision fraction `0.14815`
- motion-state mismatch `0.14815`
- both-moving amplitude ratio >3x: `0`
- both-moving direction cosine <0.5: `0`

Closest 10% is similar (`0.15385`) and again entirely motion-state mismatch.

Under the prereg rules **Test B = AMBER, not RED**. No new unresolved amplitude/direction equivalence-class collision appears. The remaining class is the already-planned explicit `p_active` factor.

### Test D — family-disjoint learnability

Episode-level canonical open-development population: `232` episodes / `29` families.

- p_active proxy AUROC `0.97411`
- multifeature active log-amp LOFO Spearman `0.84524`
- naive multifeature fusion can collapse a held-out family (`12832 = -0.10714`)
- but frozen raw amplitude evidence has `12832 = +0.78571`, aggregate `0.87594`, median family `0.85714`, minimum family `0.50`, **29/29 >=0.50**, no negative family.

Carrier-level p_active LOFO on the 8-family panel, using only frozen current log-amplitude:

- overall AUROC `0.92948`
- minimum family AUROC `0.89247`
- no repeated family ordering collapse.

Fixed probability threshold calibration is nonuniform; `p_active` therefore requires explicit calibration/confidence/coverage semantics.

## Final supported representation

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

Authority:

```text
bounded set-valued H_i              -> physical XYZ authority
p_active                             -> absolute motion/non-motion authority
log_amp                              -> conditional magnitude/ranking evidence
dir                                  -> direction evidence
U / margin / multimodality / support -> bounded retention, adaptive search, abstention
compiler/global solver               -> final collapse after global consistency
```

## What the battery falsified

- fixed early top4 collapse;
- aggregate success as proof of tail sufficiency;
- unrestricted multifeature fusion as automatically beneficial;
- one head simultaneously owning activity, amplitude, direction and XYZ;
- the need to invent a free XYZ motion head to solve the observed tail.

## What the battery supports

The paired raster/frozen descriptor evidence tested here contains the required hard-tail geometry and motion-order information **when hypotheses are preserved rather than prematurely collapsed**. The user’s concern about majority-success hiding hard-tail failure was correct; the failure is real but localized.

The current evidence supports this statement:

> The representation class is sufficiently informative on the tested truth-open development panels, provided it remains set-valued and factorized. The demonstrated failures come from hypothesis truncation, fusion and calibration, not from a demonstrated absence of raster information.

This is sufficiency support, not a guarantee of unseen-domain generalization.

## Next authorized work

1. Freeze a **bounded uncertainty-aware set-valued candidate contract**. Top16 is the demonstrated safe development reference, not necessarily the final fixed K. Use margin/multimodality/view-support to retain/expand only where ambiguity requires it.
2. Train/evaluate the **small strict family/episode-disjoint `p_active + log_amp` head** with explicit amplitude ranking loss, motion-state supervision and calibration/coverage. No free XYZ authority.
3. Keep direction and geometry authority separated.
4. Only after the revised contract + small factor head are frozen should a new untouched preregistered qualification panel be opened.

**Large/end-to-end architecture training remains NOT AUTHORIZED at this point.**

## Canonical decision artifact

- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_CANONICAL_DECISION.md`

Supporting artifacts include the prereg, broad Test-C result/report, Test-A oracle result, Test-B collision result/addendum, carrier p_active LOFO result, and exact replay/reproducer scripts in this research directory.
