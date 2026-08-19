# RealSaS N1D — Post-Stage-B Representation Sufficiency Battery V1 Prereg

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DEVELOPMENT_PREREG__NOT_QUALIFICATION`  
**Frozen Stage-B authority remains:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`  
**sealed21 / external10:** `CLOSED`

## Goal

Determine whether the proposed representation

```text
P_A, P_B_geom / candidate basin H_i,
N_A, N_B,
V_A, V_B,
Z,
p_active,
log_amp,
dir,
U_pred + typed U_obs
```

contains enough information to support family-disjoint mechanical inference, or whether some hard-tail cases are genuine representation collisions that cannot be repaired by a larger selector/training run.

This battery is explicitly designed to separate:

```text
representation insufficiency
vs
learned selector / decoder / optimization failure
```

No result from this truth-open development panel can rewrite the frozen Stage-B qualification.

## Test A — representation-conditioned oracle ceiling

The oracle is allowed to choose/score only among states already expressible by the proposed representation: frozen/common-world geometry and feasible candidate/mechanical basin, plus factorized activity/amplitude/direction channels. It may use truth only to choose the best representable state for ceiling measurement; it may not inject a free XYZ endpoint outside the representation.

Report unchanged GFDR metrics and, where carrier-level candidates are available, truth endpoint containment inside the representable basin.

Interpretation:

- **GREEN:** representable oracle closes all six principal GFDR quality gates and no hard-tail stratum shows a structural basin-coverage collapse.
- **RED:** even the representation-conditioned oracle cannot reach the needed mechanical state on a material hard-tail subset.
- **AMBER:** aggregate ceiling is strong but candidate/basin evidence is too incomplete to exclude tail insufficiency.

## Test B — representation collision audit

Search for pairs/groups with very similar observable representation but materially different required mechanical outcomes.

Preferred carrier-level representation distance uses only inference-available fields (normalized by robust training/development scales): geometry/candidate-basin summaries, visibility/support, normals, descriptor/current evidence, uncertainty/conditioning terms, and factorized motion evidence. Truth is used only to measure target separation after nearest-neighbor retrieval.

A collision witness is flagged when a pair lies in the closest representation-distance tail yet differs materially in one or more of:

- active vs silent state;
- normalized amplitude order;
- required direction/mechanical axis;
- feasible endpoint/basin identity.

Cross-family witnesses are prioritized. A repeated cross-family collision family is stronger evidence than a single pair.

Interpretation:

- **GREEN:** nearest representation neighbors preserve target state except for bounded/noisy differences.
- **RED:** repeated low-distance / high-target-distance cross-family collisions exist that cannot be resolved by any field already in the representation.
- **AMBER:** collisions exist but can be explained by a missing field already planned (`p_active`, `log_amp`, typed U_obs) or by unavailable diagnostics.

## Test C — hard-tail candidate/geometry containment

Do not report only aggregate containment. Stratify by difficulty using frozen baseline error / activity-ranking failure and report at minimum easy/median/hard or quartiles.

For every stratum with candidate data, measure whether the truth endpoint/mechanical solution is contained or closely approximated by the feasible candidate/basin representation. Also report direction support and activity/amplitude evidence quality separately.

Tail alarm rule:

- **RED:** worst/hard stratum loses >=15 percentage points of containment relative to the easier strata, or falls below 0.90 containment when enough carriers exist for the estimate to be meaningful.
- **GREEN:** hard-tail containment remains >=0.90 and within 15 points of the easier strata.
- **AMBER:** insufficient candidate persistence/sample size prevents a reliable tail estimate.

This rule is diagnostic, not a product acceptance threshold.

## Test D — strict family-disjoint tiny probe

Use the smallest practical probe on frozen inference-available representation features. No image-backbone retuning and no family leakage.

Primary targets:

```text
p_active  -> activity/silence classification
log_amp   -> conditional amplitude/rank prediction on active support
```

Optional direction probe is allowed only as a sanity check because direction is already strong in V11.

Evaluation is leave-one-family-out (LOFO) where the data supports it. Hyperparameters are fixed globally; they are not retuned per held-out family.

Development decision reference values:

- conditional amplitude: median held-out Spearman >= 0.50 is considered useful signal, matching the existing F-activity quality scale;
- `p_active`: report balanced accuracy / AUROC and per-family false-active/false-silent behavior; no single aggregate may hide a family collapse;
- any family with amplitude correlation near zero/negative or severe activity collapse remains a hard-tail warning even if the median passes.

This is a sufficiency diagnostic, not a new qualification.

## Overall decision

`REPRESENTATION_SUFFICIENCY_SUPPORTED` requires:

1. Test A not RED;
2. Test B not RED;
3. Test C not RED;
4. Test D shows family-disjoint learnable signal without a repeated family collapse.

If A is strong but B/C/D fail, the correct conclusion is **not** “train a bigger model.” The failing witness must be localized to a missing observable/evidence channel, a non-singleton ambiguity that should remain a hypothesis set, or a genuine domain/corpus gap.

If a failure is caused by information absent from the raster observations themselves, the compiler must receive uncertainty/hypotheses rather than forced exact truth.

## Existing evidence admitted before execution

The battery begins with already-canonical evidence, not as a blind test:

- Stage-B frozen F activity `0.433160` with the other principal geometry/mechanics metrics passing;
- direction-fixed truth-rank ceiling closes all six principal quality gates while preserving the frozen magnitude multiset;
- candidate-anchored current-amplitude composition raises primary F activity to `0.68718` with 6/6 principal quality metrics passing;
- route-local current-amplitude correlation is heterogeneous across Stage-B episodes, including strong (`~0.92`) and weak (`~0.18–0.41`) cases;
- normalized amplitude alone does not preserve absolute silence, motivating a distinct `p_active` channel.

The purpose of the battery is therefore specifically to test **tail sufficiency/generalization**, not to re-prove those aggregate results.
