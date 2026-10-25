# Historical FIT8 Presentation-Evidence Reconciliation V1

**Audit date:** 2026-10-25  
**Evidence date:** 2026-10-22  
**Authority:** historical diagnostic only; no V3 qualification, no Truth139/FIT8 authorization.

## Source

Read-only Google Drive artifact discovered during the V3 apparatus audit:

- folder: `fit8_corpus_evidence_autopsy_v1_20261022`
- result file ID: `1KBYqMfAqBiudXRWi-Kmt2aWd1HL0wrYB`
- log file ID: `1YUEfQWcI9zcM_6-x_dUAvVbvqqE2KK4e`
- experiment id: `STRUCTURAL-EVIDENCE-FIT8-CORPUS-EVIDENCE-AUTOPSY-V1`
- execution source SHA: `b013d64d00c00408d20e11c407a100316fdc2b84`
- training execution source SHA recorded by artifact: `eb11004e4d8c7fba6f6ba95fa8eacaa0c24fcb3d`
- parent census result SHA256: `01836f85f17bc7308e35c9ed08acc523904ec1cfbcc51b0caf33e6c7cb9d4df8`
- presentation-memory manifest SHA256: `21434efbfd409504d11bd883f35041089dbe7329e829c37446eb21c59b19b9f6`
- query manifest SHA256: `9bc67c6457918e43d88a2ff1158d0e8102875c9a3cf8db24788ac457237c6695`
- artifact status: `COMPLETE_POST_HOC_DIAGNOSTIC__NO_QUALIFICATION`
- interpretation scope: `POST_HOC_CORPUS_EXTRACTOR_SAMPLER_MECHANISM_DIAGNOSTIC_ONLY`

The artifact explicitly reports `qualification_pass_claimed=false`, `shared_scene_hypothesis_proven=false`, no optimizer/backward steps, and no parameter mutation. This audit preserves those boundaries.

## Eight-subject directional signal

The historical A/B was null presentation (`A`) versus role-free presentation evidence (`B`). Relative total-objective improvement of B versus A:

| Subject | B relative improvement vs A | Historical census vote |
|---|---:|---|
| Bat | +0.101819 | B_ROLE_FREE_PRESENTATION |
| George | -0.007350 | TIE |
| GreenBlob | -0.087882 | A_NULL_PRESENTATION |
| Horse | -0.068264 | A_NULL_PRESENTATION |
| Shark | +0.145933 | B_ROLE_FREE_PRESENTATION |
| Snake | +0.233191 | B_ROLE_FREE_PRESENTATION |
| Spider | +0.324275 | B_ROLE_FREE_PRESENTATION |
| T-Rex | +0.364990 | B_ROLE_FREE_PRESENTATION |

This is heterogeneous, morphology-dependent evidence rather than a universal presentation win.

## Resampling stability

A second historical phase re-sampled the teacher-window mechanism for four diagnostic subjects over seeds `1701..1704`. Original memory reproduction passed for all four.

- Spider: `STABLE_POSITIVE`, resampled B-vs-A gain approximately `+0.324275` in every seed.
- T-Rex: `STABLE_POSITIVE`, approximately `+0.364990` in every seed.
- Horse: `STABLE_NEGATIVE`, approximately `-0.068264` in every seed.
- GreenBlob: `STABLE_NEGATIVE`, approximately `-0.087881` in every seed.

Therefore the sign heterogeneity was not explained by that historical sampler seed alone.

## Critical incompleteness of the presentation evidence

Across all eight subjects in the historical subject summary:

- `deformation_coherence_valid_fraction_mean = 0.0`
- `attachment_persistence_valid_fraction_mean = 0.0`
- `occlusion_valid_fraction_mean = 0.0`

The remaining presentation evidence was also sparse/partial by construction. Typical subject means were approximately:

- source RGBA validity: `0.25`
- source support validity: `0.25`
- front validity: `0.375`
- global visibility validity: `0.375`

Hence the historical negative subjects do **not** falsify the intended causal-presentation hypothesis. They show that the then-available presentation package could help some morphology and hurt others while key causal relation families were entirely absent.

## Relation to current V3 audit

This historical evidence raises the priority of two current audit questions:

1. **TESSA fusion placement / addressability.** Current V3 independently compresses scene, surface and presentation before concatenation. If a sparse presentation stream loses exact endpoint bindings during compression, useful evidence can become morphology-dependent noise.
2. **Truth completeness / objective geometry.** A presentation stream missing deformation, attachment and qualified occlusion cannot fairly test the causal Truth Foundry hypothesis.

The clean synthetic V2 endpoint-addressability court already established that current independent compression did not meet its preregistered tiny accessibility threshold. The separately preregistered fusion-placement court is therefore the correct next mechanism test.

## Scientific interpretation

The combined evidence supports neither of these premature conclusions:

- `presentation evidence is universally beneficial`; or
- `presentation evidence hurts some morphologies and should be removed`.

The defensible current interpretation is:

> role-free presentation evidence carries useful structural signal on real diverse subjects, but the historical evidence package was incomplete and its downstream compression/fusion path may not preserve endpoint-addressable relations robustly across morphology.

A fair test of the intended hypothesis requires both a qualified causal truth package and an apparatus that demonstrably preserves typed endpoint relations.

## Authority

This record is diagnostic provenance only. It does not authorize:

- V3 architecture replacement;
- Truth139/FIT8 training;
- ModelRelease promotion;
- product activation;
- a genericity or hypothesis-pass claim.
