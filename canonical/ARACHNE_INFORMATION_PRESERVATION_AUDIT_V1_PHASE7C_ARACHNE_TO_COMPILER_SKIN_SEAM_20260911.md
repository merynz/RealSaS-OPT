# RealSaS — Arachne Information-Preservation Audit V1 — Phase 7C Arachne→Compiler Skin Seam

**Date:** 2026-09-11  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY__NO_A1_IMPLEMENTATION_OR_RUNNING_A0_CHANGE`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`

## 0. Question

If V7-native A1 eventually predicts the skin field successfully, does the current Arachne→Compiler→mesh path preserve the learned semantic result cleanly, or does another model/Compiler transition erase/re-infer information?

## 1. Current public proposal contract is intentionally narrow

`SkinInfluenceProposal` contains only:

- `surface_id`;
- `canonical_joint_id`;
- scalar `weight`.

`SkinProposalIR` adds:

- exact surface binding hash;
- exact skeleton binding hash;
- model provenance string;
- metadata.

There is currently no typed per-row/per-influence field uncertainty, confidence, boundary ambiguity or token evidence in the public skin proposal schema.

This is **not presently a defect** because the current Arachne scientific contract requires semantic weight proposal, not calibrated weight uncertainty. It becomes a seam only if a future A1 explicitly learns/promotes such evidence.

## 2. Compiler skin qualification is owner-clean

Current `qualify_skin()` performs exactly the kind of deterministic qualification allowed by ownership:

- exact surface/skeleton lineage checks;
- reference validity;
- duplicate-pair rejection;
- finite checks;
- material-negative rejection;
- tiny-negative bounded clipping accounting;
- every surface must have a row;
- optional bounded top-k sparsification;
- simplex normalization;
- per-row and aggregate L1 correction budgets;
- explicit discarded-mass and correction accounting;
- fail closed on zero rows or repair budget excess.

It does **not** use geometry to invent new semantic weights, choose new joints, smooth a bad learned field, or invoke BBW as hidden repair.

The final `QualifiedSkinIR` retains:

- exact admitted normalized influence rows;
- simplex residual before qualification;
- exact correction L1;
- qualification report;
- exact surface/skeleton lineage;
- skin lineage hash committing proposal + report + admitted rows.

**Verdict:** `PRESERVED_SEMANTIC_WEIGHTS_WITH_BOUNDED_NUMERICAL_QUALIFICATION`.

This boundary is a second positive control, analogous to promoted GSA→Geppetto tensorization: ownership is clean and repair is measurable.

## 3. Important nuance: current skin qualifier can change numerical rows, but not silently

The qualifier does normalize submitted rows and may optionally sparsify them. Therefore Arachne output is not necessarily bit-identical to `QualifiedSkinIR`.

However the exact end-to-end row difference is measured as `correction_l1` against the submitted proposal row, and both per-row and aggregate correction budgets are enforced. Sparsification discarded mass is separately reported.

For A1 scientific evaluation, metrics must therefore distinguish:

1. raw Arachne proposal W*;
2. Compiler-qualified W;
3. mesh-bound B;
4. deformation consequence.

A raw model PASS must not be retroactively attributed to Compiler repair, and a qualification failure must not be hidden by evaluating only W*.

## 4. Surface skin→editable mesh transfer is also owner-clean

The current MWB2 binder maps qualified surface skin to each qualified mesh vertex using the mesh vertex's exact `SurfaceSupportBinding` convex coefficients:

`w_raw(v,j) = sum_s a(v,s) * w(s,j)`.

It records those source support coefficients in each `QualifiedMeshSkinRow`, measures numerical residual/correction, and uses extremely small bounded repair budgets. The report explicitly declares `semantic_skin_synthesis=False`.

Therefore the binder interpolates already-qualified semantics; it does not become a second skin predictor.

**Verdict:** `PRESERVED_BY_DECLARED_CONVEX_TRANSFER`.

This is particularly important for future Arachne evaluation: a visually good final mesh deformation should remain traceable back to the exact qualified surface skin rows that produced it.

## 5. What is currently *not* preserved because it does not yet have a public semantic contract

If future V7-native A1 produces additional meaningful evidence such as:

- row confidence;
- per-influence confidence;
- calibrated skin-field uncertainty;
- blend-boundary ambiguity;
- token-set uncertainty;
- explicit abstention/UNKNOWN evidence;

the current `SkinProposalIR -> QualifiedSkinIR -> QualifiedMeshSkinIR` schemas have no first-class typed carrier for it.

Do **not** preemptively add those fields merely because a neural network could output them. Phase 3 already showed the danger of overnaming an auxiliary scalar as “uncertainty.”

Rule:

> Add a public skin-evidence sidecar only after the quantity has a supervised/calibrated semantic contract and a demonstrated downstream use (qualification, owner-routed repair, proof attribution or user editing).

Until then, logits/tokens/attention remain model-private or experiment telemetry.

## 6. UNKNOWN / abstention question

Current `SkinProposalIR` normal path assumes Arachne submits influences for every surface row; `qualify_skin()` fails closed when a surface has no influences or zero mass. This is safe, but there is no explicit per-row `UNKNOWN` token/state in the skin proposal schema.

That is acceptable for current FIT1 only if the learned product path is required to either produce a legal row or fail the whole candidate before qualification.

Before broad product/unseen sealing, one design question should be revisited:

> Should Arachne be able to emit a typed row-level abstention/uncertainty state that routes back to the Arachne owner rather than representing uncertainty as an arbitrary low-mass weight row?

No answer is authorized here. The audit simply records that current fail-close is candidate-level, not typed row-level abstention.

## 7. Binding implications for V7-native A1

The future A1 implementation should preserve four explicit stages in metrics and artifacts:

```text
A1 rich S+G predictor
    -> raw field tokens Z_hat
    -> frozen A0 decoder
    -> raw SkinProposalIR W*
    -> Compiler qualify_skin
    -> QualifiedSkinIR W
    -> deterministic mesh-weight binder
    -> QualifiedMeshSkinIR B
    -> deformation/proof
```

Do not evaluate only final deformation and lose model causality. At minimum record:

- raw W* row metrics;
- qualification correction/rejection metrics;
- qualified W row metrics;
- mesh transfer correction metrics;
- deformation consequence.

This gives a causal answer to “who fixed or broke the result?” at every model/Compiler transition.

## 8. Phase-7C verdict

- **GREEN:** current Arachne scalar weight → Compiler skin qualification seam is ownership-clean.
- **GREEN:** qualified surface skin → mesh skin transfer is deterministic, lineage-bound and semantically non-generative.
- **INFO:** bounded numerical normalization/sparsification can change rows, but exact correction/discard accounting exists.
- **P2 FUTURE:** no typed calibrated skin-confidence/uncertainty/row-abstention sidecar exists; do not add until semantics justify it.
- **P0 remains upstream of this seam:** the material information-loss problem remains the old `S+Qualified G -> A1` consumer compression found in Phase 5.

**No running A0 mutation, A1 implementation, schema extension or product PASS is authorized by this phase.**
