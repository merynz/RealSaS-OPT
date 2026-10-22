# Frozen-B Causal Utilization Ablation V1 — Rationale

The parent FIT8 teacher census selected `B_ROLE_FREE_PRESENTATION`, while the subsequent corpus/evidence autopsy showed two facts that must be separated:

1. the B effect is stable to deterministic resampling of the same raw presentation-query pool for Spider/T-Rex (positive) and Horse/GreenBlob (negative); and
2. the current FIT8 direct-surface corpus is unusually clean with respect to tie-safe abstention: pair-front qualification and unique-front/global-visibility validity are effectively saturated.

The next question is therefore not whether random fixed-budget sampling caused the A/B split. That explanation is no longer supported. The question is whether the frozen B model actually uses the semantic values and spatial alignment of the current presentation evidence, or whether most of the B effect can survive with only presentation-token geometry/layout/branch conditioning.

This diagnostic changes no checkpoint and executes no optimization. It evaluates all topology windows for all FIT8 subjects under controlled interventions on the already materialized B presentation payload.

Important semantic distinction:

- `VALUE_ZERO_KEEP_VALID` is **not** an absence intervention. A typed probability/value of zero with validity still asserted is meaningful evidence. It tests dependence on numeric semantic values while preserving typed channel presence.
- `SEMANTICS_ABSTAIN` zeros both each semantic value and its paired validity bit while preserving token layout, structural anchors, camera, kind, and top-level token mask. This is the cleaner geometry/layout-only presentation control.
- Single-channel minus variants also zero value and validity together.
- `SHUFFLED_WITHIN_KIND` preserves per-kind semantic marginals while destroying their alignment with the original structural anchors/camera locations.

Interpretation is strictly frozen-model behavioral utilization. It does not establish unseen generalization, qualification, or world-level causal truth.
