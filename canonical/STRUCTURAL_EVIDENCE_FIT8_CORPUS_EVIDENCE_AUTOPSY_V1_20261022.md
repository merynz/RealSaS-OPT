# FIT8 Corpus / Evidence Autopsy V1

## Why this experiment exists

The sealed FIT8 teacher-forced directional census selected `B_ROLE_FREE_PRESENTATION` (5 B wins, 2 A wins, 1 tie), but the effect is strongly morphology-dependent. The current result therefore cannot be interpreted as a clean measurement of the broader shared-scene hypothesis alone.

The B arm is conditioned by a presentation-memory artifact built after the original FIT8 corpus was designed. That artifact has a fixed per-view quota and exposes only a narrow subset of the eventual shared-scene relation program. In particular, pair-front and global visibility are active, while occlusion, deformation coherence, and attachment persistence remain abstentions. The experiment may therefore measure a mixture of:

1. useful presentation information,
2. quality and coverage of the current direct-surface extractor,
3. fixed-budget compression / sampling effects,
4. TESSA utilization of that evidence.

This autopsy is a post-hoc mechanism diagnostic. It does not revise the sealed A/B result and cannot claim qualification, generalization, or causal proof.

## Phase 1 — corpus/evidence census

For every FIT8 subject and every view, compute raw query opportunity statistics before the 8192-token memory compression:

- foreground ray count,
- direct-surface hit-ray count,
- multi-hit ray count,
- total surface hits,
- oriented pair-row count,
- hit-ray fraction,
- hits per hit ray,
- pair rows per multi-hit ray,
- pair sampling fraction under the current 384-pair/view quota,
- ray-hit sampling fraction under the current 384-ray-hit/view quota,
- view-to-view variation.

Also audit the materialized presentation memory itself:

- total and per-kind token counts,
- validity fractions for front, global visibility, source RGBA, alpha boundary, appearance boundary, source support,
- validity fractions for currently abstaining relation families.

Aggregate subject-level descriptors and compare them with the already sealed `B-vs-A` teacher-forced effect size. Pearson and Spearman correlations are descriptive only because n=8.

## Phase 2 — deterministic counterfactual sampling stability

Use the same raw presentation-query pool and the same frozen B checkpoint. Rebuild contract-valid fixed-budget presentation memories for Spider, T-Rex, Horse, and GreenBlob with deterministic alternative seeds 1701–1704 while preserving the same token budget and relation semantics. Evaluate every topology window teacher-forced exactly as in the census.

The key question is whether the sign and magnitude of `B-vs-A` survive valid resampling.

- Stable B winners across seeds support a real information/utilization effect.
- Sign flips or large variance indicate sampler/corpus sensitivity.
- Stable negative-transfer subjects indicate that random sampling alone is insufficient to explain the harm; extractor quality, evidence relevance, or decoder utilization becomes the next target.

## Guardrails

- zero optimizer steps,
- no checkpoint mutation,
- no threshold relaxation,
- no qualification claim,
- no unseen/generalization claim,
- no claim that the full shared-scene hypothesis has been tested.

The purpose is to locate the confound before spending another full training budget.
