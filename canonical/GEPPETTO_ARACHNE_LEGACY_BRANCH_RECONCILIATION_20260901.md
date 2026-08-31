# RealSaS — Geppetto/Arachne Legacy Branch Reconciliation — 2026-09-01

**Branch audited:** `geppetto-arachne-v0-1-20260830`  
**Current authority:** `main`  
**Status:** `RECONCILED__NO_BLIND_MERGE__ONE_ORPHAN_AUTHORITY_RESTORED__SELECTIVE_PORT_ONLY`

## Executive verdict

The legacy branch is not a continuation branch and must not be merged wholesale.

At audit time it was `12` commits ahead of and `39` commits behind canonical `main`, with `27` files unique to the old branch. Those files mix four materially different classes:

1. still-valid teacher/evaluation machinery worth selectively porting;
2. useful but authority-sensitive substrate features requiring requalification;
3. unsealed neural/reference scaffolds that may inform R6 design but cannot become canonical by inheritance;
4. corpus/input-quality apparatus superseded by later canonical image-integrity/semantic-triage work.

One genuine cross-branch orphan was found: current canonical visual-triage authority refers to `IMAGE_SEMANTIC_CHARACTER_GATE_PREREG_V1.md`, but that frozen label contract existed only on the legacy branch. Its exact text has therefore been restored to `canonical/IMAGE_SEMANTIC_CHARACTER_GATE_PREREG_V1.md` on `main` through this reconciliation branch. The legacy source blob was `36aef9ae6ecd543a1f467709e1608f9225b4459b`.

No learned architecture, model checkpoint, optimizer authorization, corpus membership, or solver authority is changed by this reconciliation.

## Classification vocabulary

- `RESTORE_AUTHORITY_COPY` — current mainline still depends on the frozen authority; copy to main with provenance.
- `PORT_SELECTIVELY` — implementation concept remains valid and should be moved under current R6 naming/contracts with tests.
- `REQUALIFY_BEFORE_PORT` — useful code exists, but its information/authority assumptions must be checked against current product contracts first.
- `REFERENCE_SCAFFOLD_ONLY` — may inform candidate design; not architecture authority and must not be silently revived.
- `HISTORICAL_MEASUREMENT_EVIDENCE` — important measured facts/policies remain useful, but later gates supersede its continuation semantics.
- `SUPERSEDED_CURRENT_APPARATUS` — later canonical apparatus/closure replaces it for current execution; retain branch history, do not port as active code.

## File-by-file reconciliation

| Legacy-only file | Classification | Current decision / rationale |
|---|---|---|
| `experiments/geppetto_arachne_v0_1_20260830/IMAGE_SEMANTIC_CHARACTER_GATE_PREREG_V1.md` | `RESTORE_AUTHORITY_COPY` | Frozen PASS/FAIL/AMBIGUOUS semantic label contract is still referenced by current visual-triage prereg. Restored to canonical main. |
| `experiments/geppetto_arachne_v0_1_20260830/teacher_projection_v1.py` | `PORT_SELECTIVELY` | One anonymous control per deform bone; nearest-deform-ancestor helper skip; multi-root preservation; teacher tail/index provenance only; no canonical IDs. Still compatible with R6 teacher/evaluator truth boundary. Port under a current training/evaluation namespace, not product runtime. |
| `experiments/geppetto_arachne_v0_1_20260830/contracts_v0_1.py` | `REQUALIFY_BEFORE_PORT` | Teacher projection dataclasses are reusable. The 27D token/`VolumeState` contract mixes surface and interior evidence and is no longer automatically authoritative after the partial-observed-surface R6 boundary. Split before reuse. |
| `experiments/geppetto_arachne_v0_1_20260830/substrate_adapter_v1.py` | `REQUALIFY_BEFORE_PORT` | `surface_tokens_from_rigging_surface_v1` and root-safe point/control geometry are promising. Interior sampling, typed volume states, and point-to-control volume path evidence must be re-audited so they cannot imply hidden completion or a second geometry authority. |
| `experiments/geppetto_arachne_v0_1_20260830/geppetto_g0_1.py` | `REFERENCE_SCAFFOLD_ONLY` | Unsealed KNN-attention + autoregressive GRU scaffold. Its proposal/Compiler authority split remains sound, but R6 deliberately leaves exact architecture/loss unsealed. No direct promotion. |
| `prospective/geppetto_arachne_20260830/GEPPETTO_ARACHNE_ARCHITECTURE_V0_1.md` | `REFERENCE_SCAFFOLD_ONLY` | Useful historical design decomposition; superseded by R0-R5 external-reference audit + R6 causal protocol + MWB contracts as architecture authority. |
| `experiments/geppetto_arachne_v0_1_20260830/V0_1_FOUNDATION_PRECOMMIT_REPORT.md` | `HISTORICAL_MEASUREMENT_EVIDENCE` | Precommit state/provenance context only; not a current seal. |
| `experiments/geppetto_arachne_v0_1_20260830/SKELETON_PROJECTION_CORPUS_AUDIT_PREREG_V1.md` | `HISTORICAL_MEASUREMENT_EVIDENCE` | Frozen read-only FIT measurement design remains valuable provenance for helper/multi-root/control-count facts. |
| `experiments/geppetto_arachne_v0_1_20260830/run_skeleton_projection_corpus_audit_v1.py` | `PORT_SELECTIVELY` | Reproducibility utility for the historical measurement and future corpus revisions; port only with exact teacher-projection contract and current split firewall. |
| `experiments/geppetto_arachne_v0_1_20260830/skeleton_projection_corpus_audit_v1.py` | `PORT_SELECTIVELY` | Same as runner: useful data-plane audit logic, not product inference. |
| `experiments/geppetto_arachne_v0_1_20260830/test_skeleton_projection_corpus_audit_v1.py` | `PORT_SELECTIVELY` | Preserve when the audit implementation is ported. |
| `experiments/geppetto_arachne_v0_1_20260830/POST_AUDIT_POLICY_FREEZE_V1.md` | `HISTORICAL_MEASUREMENT_EVIDENCE` | Important observed facts: 2914/2914 projection audit pass; deform controls `3/39/80/129.87/328`; 17 assets >160; multi-root `164/2914`; helper skip `11/2914`; non-deform skin mass essentially absent except one asset. The old “C0 apparatus next” continuation is superseded by later clean-C0 + R6 gates. |
| `experiments/geppetto_arachne_v0_1_20260830/build_c0_admission_manifests_v1.py` | `HISTORICAL_MEASUREMENT_EVIDENCE` | Encodes old structural C0 membership (`2897` Geppetto / `2527` Arachne) before later objective image and semantic clean-C0 filtering. Do not use as final current membership generator. |
| `experiments/geppetto_arachne_v0_1_20260830/policy_v0_1.py` | `REFERENCE_SCAFFOLD_ONLY` | Old candidate policy constants may document assumptions but cannot override current R6/MWB/clean-C0 authority. |
| `experiments/geppetto_arachne_v0_1_20260830/test_repo_slice_v0_1.py` | `REFERENCE_SCAFFOLD_ONLY` | Tests the old slice, not current architecture. Recreate narrow tests only for selectively ported components. |
| `experiments/geppetto_arachne_v0_1_20260830/CHARACTER_RENDER_PURITY_AUDIT_PREREG_V1.md` | `SUPERSEDED_CURRENT_APPARATUS` | Later native 1024 image-integrity closure and current visual anomaly triage are current authority. Historical prereg remains branch evidence only. |
| `experiments/geppetto_arachne_v0_1_20260830/CHARACTER_RENDER_PURITY_AUDIT_PREREG_AMENDMENT_V1_1.md` | `SUPERSEDED_CURRENT_APPARATUS` | Same. |
| `experiments/geppetto_arachne_v0_1_20260830/CHARACTER_RENDER_PURITY_AUDIT_APPARATUS_CORRECTION_V1_2.md` | `SUPERSEDED_CURRENT_APPARATUS` | Same; correction history remains provenance but is not active apparatus. |
| `experiments/geppetto_arachne_v0_1_20260830/CHARACTER_RENDER_USABILITY_POLICY_V1.md` | `HISTORICAL_MEASUREMENT_EVIDENCE` | Earlier objective rendering policy contributed to the 2874 objective-pass population; current execution follows later image-integrity + semantic triage authority. |
| `experiments/geppetto_arachne_v0_1_20260830/character_render_purity_audit_v1.py` | `SUPERSEDED_CURRENT_APPARATUS` | Current native image integrity/triage code supersedes execution role. |
| `experiments/geppetto_arachne_v0_1_20260830/character_render_purity_audit_v1_2.py` | `SUPERSEDED_CURRENT_APPARATUS` | Same. |
| `experiments/geppetto_arachne_v0_1_20260830/run_character_render_purity_audit_v1.py` | `SUPERSEDED_CURRENT_APPARATUS` | Same. |
| `experiments/geppetto_arachne_v0_1_20260830/run_character_render_purity_audit_v1_2.py` | `SUPERSEDED_CURRENT_APPARATUS` | Same. |
| `experiments/geppetto_arachne_v0_1_20260830/test_character_render_purity_audit_v1.py` | `SUPERSEDED_CURRENT_APPARATUS` | Same. |
| `experiments/geppetto_arachne_v0_1_20260830/test_character_render_purity_audit_v1_2.py` | `SUPERSEDED_CURRENT_APPARATUS` | Same. |
| `experiments/geppetto_arachne_v0_1_20260830/build_clean_character_gate_v1.py` | `SUPERSEDED_CURRENT_APPARATUS` | Do not emit final membership from this old gate; current semantic review must use the restored frozen label authority plus current visual-triage outputs. |
| `experiments/geppetto_arachne_v0_1_20260830/test_build_clean_character_gate_v1.py` | `SUPERSEDED_CURRENT_APPARATUS` | Test belongs to superseded membership builder, not current final clean-C0 path. |

All 27 unique paths are accounted for above.

## Preserved historical quantitative evidence

The old complete skeleton-projection audit/policy remains useful when sizing or stratifying future R6 candidates, without becoming a hidden architecture choice:

- complete Geppetto FIT audit: `2914 / 2914` hard-contract PASS;
- deform-control min/p50/p95/p99/max: `3 / 39 / 80 / 129.87 / 328`;
- `>160` controls: `17 / 2914` (`0.583%`);
- multi-root assets: `164 / 2914` (`5.63%`), maximum projected roots `19`;
- assets with helper skip: `11 / 2914`, maximum skip chain `1`;
- exact zero-length deform bones: `0`;
- Arachne-capable scope: `2540`;
- positive non-deform skin mass: one asset, corpus fraction approximately `4.748e-08`.

These facts are candidate-design/stratification evidence only. They do not authorize a fixed 160-control product limit or old G0.1 architecture.

## Orphan repair performed

`canonical/CORPUS_VISUAL_ANOMALY_TRIAGE_PREREG_20260831.md` depends on the semantic labels and review firewall frozen in `IMAGE_SEMANTIC_CHARACTER_GATE_PREREG_V1.md`.

Before this reconciliation, that file was absent from canonical `main` and survived only in the legacy branch. The authority text has now been restored to:

`canonical/IMAGE_SEMANTIC_CHARACTER_GATE_PREREG_V1.md`

This is an authority-location repair, not a semantic amendment.

## Selective port plan

The next permitted port is deliberately narrow:

1. extract `SkeletonTeacherProjectionV1` and its minimal teacher-only dataclasses into a current R6 training/evaluation namespace;
2. carry deterministic parent-forest validation, nearest-deform-ancestor policy, multi-root preservation, BFS audit serialization and skin-column provenance;
3. port its focused tests;
4. do **not** port the old 27D surface+interior token contract wholesale;
5. separately specify `GeppettoConditioningAdapter` from current `RiggingSurfaceIR` before deciding whether any old surface-token helper survives;
6. keep `GeppettoG01` only as an explicitly unsealed reference candidate until the independent R6 apparatus is frozen.

## Branch disposition

`geppetto-arachne-v0-1-20260830` remains historical provenance. It should not receive new continuation commits and should not be merged wholesale into `main`.

Canonical continuation remains `main`.
