# RealSaS N1D — Representation Sufficiency Battery V1 — Interim

**Date:** 2026-08-19  
**Status:** `INTERIM_AMBER__REPRESENTATION_SUFFICIENCY_NOT_YET_SUPPORTED`  
**Authority:** truth-open development only; frozen Stage-B remains `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`; sealed21/external10 remain CLOSED.

## Why this checkpoint exists

The battery was preregistered before this population analysis because strong aggregate metrics can hide a family/hard-tail collapse. This checkpoint records the first broad family-disjoint evidence before current global-foreground candidate containment is measured.

## Population

Canonical N1D open-development export:

- 232 episodes
- 29 families × 8 episodes
- 135 finite, 64 strong, 33 near-zero
- source `N1D_CANONICAL_DEV_PER_EPISODE.json`
- source SHA-256 `37d81db0733f6fbf433a944aa0797550f3beee15ff65377ecbcfada1064f4326`

Only inference-available episode summaries were admitted to the tiny proxy probes: `predicted_flow_mean`, `moved_fraction_005`, transport gate/edge-mass/entropy/peak summaries, and `persistent_count`. Truth fields are targets/evaluation only.

## Test D — strict 29-family LOFO proxy

A fixed global StandardScaler + balanced logistic probe for active/silent and a fixed StandardScaler + Ridge(alpha=1) active-only log-amplitude probe were trained leave-one-family-out.

Aggregate results are strong:

- p_active balanced accuracy: **0.88937**
- p_active AUROC: **0.97411**
- active log-amp Spearman: **0.84524**
- median held-out-family log-amp Spearman: **0.92857**

But a real tail appears: held-out **family 12832** gets multifeature log-amplitude Spearman **-0.10714**.

Crucially, this is *not* evidence that the raster-derived representation has lost amplitude ordering in 12832. The single frozen scalar `predicted_flow_mean` has **+0.78571** Spearman with true amplitude inside 12832. Across all 29 families, this raw scalar has:

- aggregate active Spearman **0.87594**
- median family Spearman **0.85714**
- minimum family Spearman **0.50**
- **29/29 families >= 0.50**
- **0 negative families**

Therefore the 12832 failure localizes to naive cross-family feature fusion/calibration, not to absence of amplitude-order information in the frozen observable substrate.

This directly supports explicit pairwise/listwise ranking supervision and cautions against unconstrained feature mixing for `log_amp`.

### p_active tail

Ordering is also much better than fixed-threshold calibration:

- 12832: AUROC **1.0**, but balanced accuracy at 0.5 only **0.50** — perfect ordering, wrong global calibration.
- 12907: AUROC **0.9333**, balanced accuracy **0.6667**.
- 13203: AUROC **0.80**, balanced accuracy **0.70**.

Thus a distinct `p_active` factor is supported, but it must carry calibrated confidence/coverage semantics; one universal post-hoc threshold cannot be assumed safe from this proxy.

## Test B — episode-summary collision proxy

Using robust-IQR scaling on the same inference-safe summaries and restricting nearest-neighbor search to other families:

- closest 5% cross-family neighbors: active/silent mismatch **7.69%**;
- closest 10%: active/silent mismatch **16.67%**.

This is an AMBER warning, not an irreducible collision verdict. The proxy representation intentionally lacks several fields in the proposed final representation: explicit learned `p_active`, carrier-level candidate basin geometry, feasible-set width, match margin, multimodality and other typed uncertainty channels. The observed collisions are therefore exactly the class those planned factors are intended to disambiguate.

## Test C — historical hard-tail warning

The old epoch10 descriptor-consensus top8 route had mean true-match containment **0.89944**, but family tails were severe:

- 11032: **0.78689**
- 15290: **0.78107**
- best two families: about **0.964–0.974**
- bottom-two vs top-two gap: **18.47 percentage points**

Under the new preregistered Test-C rule this old route would be RED. It is **not the current representation**, because the later Stage-A/B route uses descriptor-global-foreground multiview 3D candidate solving and all-view refit. Therefore this historical failure is a warning that current-route containment must be measured, not a current RED decision.

## Interim decision

- Test A: **AMBER** — aggregate/oracle/factorization ceilings are strong, but full current feasible-basin hard-tail coverage is not yet measured.
- Test B: **AMBER** — coarse cross-family collisions exist but are potentially resolved by already-planned `p_active` and typed-U/candidate fields.
- Test C: **AMBER** — old route has a real RED tail; current global-foreground candidate containment remains to be measured.
- Test D: **PARTIAL PASS / TAIL WARNING** — broad family-disjoint amplitude/activity signal exists, but naive multifeature fusion/calibration can collapse a held-out family.

**Overall:** `REPRESENTATION_SUFFICIENCY_NOT_YET_SUPPORTED__BIG_TRAINING_NOT_AUTHORIZED`.

The evidence currently favors **representation signal exists, tail-safe factorization/normalization/fusion is the unresolved problem** over **the observations fundamentally contain no usable signal**. That distinction is not final until current candidate-basin containment and carrier-level collision tests close.

## Next

Replay the **current descriptor-global-foreground candidate generator**, not the historical descriptor route, on a broad family-disjoint panel and persist:

1. per-family truth/target containment in feasible candidate set / basin;
2. difficulty-stratified containment;
3. representation-conditioned feasible-basin oracle ceiling;
4. carrier-level collision witnesses with `p_active/log_amp/dir/U_obs` included.

Only after those are non-RED should a larger learned architecture be authorized.
