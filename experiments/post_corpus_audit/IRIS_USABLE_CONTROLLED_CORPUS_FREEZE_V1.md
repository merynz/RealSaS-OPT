# IRIS Usable Controlled Corpus Freeze V1

**Date:** 2026-08-23

## Decision

Stop post-corpus geometry repair work after Stage-B6 and return to the controlled IRIS research path. Stage-B7 publication/rerender is intentionally **DEFERRED**, not cancelled.

## Why

Stage-B6 froze 56 SHA-verified repaired geometry artifacts without mutating the canonical corpus. This gives a safe resume point if/when we later choose to publish those repairs and rerender their A-pass observations.

The immediate research objective is to close whether IRIS can recover persistent observable surface geometry under the strongest controlled conditions. Continuing to spend time recovering the final ~1.4% of the corpus is not on the critical path for that question.

## Immediate IRIS training/evaluation subset

Canonical selected assets: 3993.

Exclude from the immediate controlled IRIS subset:
- 56 Stage-B6 repair assets, because their current canonical A-pass renders still correspond to the pre-repair geometry authority;
- 6 active non-Basis shape-key quarantine assets;
- 1 all-8 blank observation asset.

Therefore the immediate controlled IRIS subset is:

`3993 - 56 - 6 - 1 = 3930 assets`

This is the authoritative usable count until Stage-B7 atomically publishes repaired geometry + matching A-pass rerenders.

## Frozen deferred state

- 56 repaired geometry artifacts are frozen and SHA-verified in Stage-B6 repair staging.
- 6 active shape-key assets remain quarantined pending separate rest-state qualification.
- 1 all-8 blank asset remains observation-ineligible.
- No canonical corpus mutation occurred in Stage-B6.
- Stage-B7 is deferred.

## Next scientific step

Resume the controlled IRIS critical path on the 3930 clean assets:
1. exact representation / information ceiling on legal observable truth (P, geometric N, visibility/support, persistence);
2. finalize the minimal IRIS neural + deterministic evidence architecture from that result;
3. train/evaluate the controlled baseline;
4. only after controlled success, expand camera jitter / unknown-camera and appearance-domain robustness.

Do not reopen the 56-asset repair thread unless it blocks a measured IRIS result or the corpus is being prepared for a final production retrain.
