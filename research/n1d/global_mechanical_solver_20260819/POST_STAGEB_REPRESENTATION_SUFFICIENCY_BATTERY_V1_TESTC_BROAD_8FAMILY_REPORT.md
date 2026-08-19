# Representation Sufficiency Battery V1 — Test C Broad 8-Family Report

**Date:** 2026-08-19  
**Status:** `CURRENT_TOP4_RED__REVISED_SET_VALUED_TOP16_GREEN_ON_8FAMILY_E00_PANEL__TRUTH_OPEN_DEVELOPMENT_ONLY`

## Question

The sufficiency battery asked whether historical hard-tail failures mean the raster-derived representation itself lacks the information required to generalize, or whether information is present but discarded by the candidate/hypothesis representation before the compiler can use it.

The current production-like global-foreground route was replayed on the eight-family e00 panel:

`9908, 11032, 12772, 13203, 14404, 14702, 14758, 15290`.

The evaluator/mapping contract was preregistered before current-H results (`5dc6989b2026a2db7194130242243b0ab6902b09`). Candidate-breadth arms were preregistered before breadth results (`85139f87da7d510d581535d73e066f16a325a1cd`). Model weights, raster inputs and descriptor embeddings are frozen.

## Result — current fixed top4 is genuinely insufficient

Across `492/512` reliably mapped carriers:

- pooled primary 2x containment: **0.93902** (`462/492`)
- worst family: **11032 = 0.84746**
- second RED family: **13203 = 0.86885**
- families >=0.90: **6/8**
- best–worst gap: **15.25 percentage points**
- pooled strict 1x containment: **0.82724**

This triggers the preregistered Test-C RED rule. The current fixed-top4 `H_i` contract is not hard-tail sufficient.

## Causal counterfactual — preserve the same evidence, retain more hypotheses

Retention-only arm:

```text
coarse top8: unchanged
refined pixel pool: unchanged
final per-view retained candidates: 4 -> 16
model / descriptor / raster evidence: unchanged
```

Results:

- pooled primary 2x containment: **0.98171** (`483/492`)
- worst family: **0.94915**
- families >=0.90: **8/8**
- best–worst gap: **5.08 percentage points**
- pooled strict 1x containment: **0.94919**

Per-family primary 2x:

| Family | top4 | same-pool top16 |
|---|---:|---:|
| 9908 | 0.98413 | **1.00000** |
| 11032 | 0.84746 | **0.94915** |
| 12772 | 1.00000 | **1.00000** |
| 13203 | 0.86885 | **0.95082** |
| 14404 | 0.92063 | **0.96825** |
| 14702 | 1.00000 | **1.00000** |
| 14758 | 0.98387 | **0.98387** |
| 15290 | 0.90164 | **1.00000** |

The two current-top4 RED families (`11032`, `13203`) both cross the preregistered 0.90 line using **the same frozen refined descriptor evidence**. `15290`, an historical hard tail, moves to 1.0.

## Broader coarse search

A second arm expands coarse descriptor search from top8 to top32 while keeping final top16.

- pooled primary 2x: **0.98374** (`484/492`)
- worst family remains **0.94915**
- only `14758` gains primary coverage over retention-only: `0.98387 -> 1.00000`

Therefore the dominant failure is not lack of descriptor signal or insufficient coarse search. It is **premature final candidate truncation**. A small secondary coarse-beam issue exists and can be handled adaptively rather than by globally exploding search.

## Scientific interpretation

This result validates the user's concern in a precise way:

1. **Yes, a compact representation can look good on the majority and still be genuinely insufficient on hard tails.** Fixed top4 did exactly that.
2. **No, the tested hard-tail failure does not imply the paired raster observations/frozen descriptor lack the needed information.** On the same hard families, simply preserving more already-present hypotheses recovers the target-near feasible basin.
3. The correct representation contract is therefore more set-valued:

```text
P_B_geom / H_i = bounded hypothesis set, not early single/top4 collapse
p_active        = absolute motion/silence evidence
log_amp         = conditional magnitude/ranking evidence
dir             = direction evidence
U               = margin/multimodality/view support/conditioning used for pruning and abstention
```

4. The compiler should collapse the set only after global mechanical and deformation-consistency reasoning. Early descriptor pruning must be uncertainty-aware.

## Decision

- **Current fixed-top4 Test C:** `RED` and remains an immutable development finding.
- **Revised retention-top16 Test C on this 8-family panel:** `GREEN` under the preregistered hard-tail rule.
- **Overall representation sufficiency:** **NOT YET PASS**. Test A (representation-conditioned feasible-basin oracle over the revised set) and carrier-level Test B (collision audit with explicit factorized evidence/U) remain open.
- **Large training:** still **NOT AUTHORIZED**.

## Next

Design a bounded uncertainty-aware set-valued contract using top16 as a demonstrated safe development reference, rather than adopting fixed 16 blindly. Candidate count should expand when margins/multimodality/view support indicate ambiguity and remain compact when evidence is decisive. Then rerun Test A and carrier-level Test B on that frozen contract before training the small family-disjoint `p_active + log_amp` head.
