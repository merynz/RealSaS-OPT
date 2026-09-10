# RealSaS — Arachne Information-Preservation Audit V1 — Phase 6 Qualified-Skeleton Evidence Sidecar Design

**Date:** 2026-09-11  
**Status:** `DESIGN_ONLY__NO_SCHEMA_IMPLEMENTATION__NO_A1_FEATURE_ADMISSION__NO_TRAINING_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Continuation authority:** `main/CURRENT_STATE.md`

## 0. Question

Can product-legal evidence about the **accepted** skeleton survive Geppetto→Compiler qualification without turning Arachne into a second root/parent/skeleton authority?

Answer: **yes, but only through an explicitly non-authoritative, qualified evidence sidecar.** Raw proposal alternatives must not be handed to Arachne as if canonical selection were still open.

This document defines the safest candidate contract. It does not implement it.

## 1. Existing authority boundary

Today `qualify_skeleton()`:

1. validates proposal/surface lineage and support references;
2. maps proposal joints/edges to canonical-graph candidates;
3. sends root score, joint confidence and edge score/confidence to the exact graph optimizer;
4. requires PASS + proven optimality;
5. mints canonical joint IDs only after the solve;
6. emits `QualifiedJoint(position,parent,support_surface_ids,source_proposal_id)`.

Therefore:

- accepted position/root/parent/tree/canonical IDs are Compiler authority;
- raw rejected parent/root alternatives are proposal evidence already consumed by that authority;
- Arachne may condition on the accepted state but may not reopen the solve.

## 2. Why a sidecar is preferable to bloating `QualifiedJoint`

`QualifiedJoint` should remain the clean canonical mechanical state. Evidence annotations have different semantics and calibration lifecycles. A separate typed object makes three rules explicit:

- the sidecar can be absent without changing skeleton identity;
- no sidecar value can change canonical joint/parent/root state;
- Arachne feature admission can be independently preregistered/ablated.

Conceptual type only:

```text
QualifiedSkeletonEvidenceIR
  source_surface_hash
  source_proposal_hash
  qualified_skeleton_hash
  joint_evidence[]
  edge_evidence[]
  root_evidence
  calibration_manifest
  evidence_lineage_hash
```

## 3. Candidate joint evidence

### 3.1 `position_sigma_normalized`

**Producer:** Geppetto selected proposal joint metadata.  
**Training semantics:** heteroscedastic position NLL.  
**Current Mage:** non-constant.  
**Authority:** evidence about proposal locus uncertainty; does not alter qualified position.

Candidate sidecar field:

```text
canonical_joint_id
source_proposal_id
position_sigma_normalized: Vec3
source_semantics: GEPPETTO_HETEROSCEDASTIC_POSITION_NLL
calibration_status: UNSEEN_CALIBRATION_NOT_ESTABLISHED
```

**Ruling:** `PRESERVE_CANDIDATE__NEURAL_ADMISSION_BLOCKED_PENDING_CALIBRATION/ABLATION`.

### 3.2 proposal joint confidence

Current proposal confidence is derived as:

`existence_probability * exp(-mean(position_sigma))`.

Compiler already consumes it as node confidence. On Mage it has narrow dynamic range.

Candidate field may be retained for audit/provenance:

`source_joint_confidence01`.

**Ruling:** optional sidecar evidence; **not a first-baseline A1 feature** unless it adds value beyond sigma.

### 3.3 mechanical salience

Current FIT1 target semantics make salience essentially all-positive and final Mage values saturate near one.

**Ruling:** **exclude from sidecar baseline contract.** It may remain in original proposal provenance but should not be promoted as qualified downstream evidence under current science.

### 3.4 support anchors

Exact `support_surface_ids` already survive in `QualifiedJoint` and are sufficient to preserve the admitted sparse anchor identity.

**Ruling:** do not duplicate them in the sidecar. Any future per-anchor confidence would require a separately calibrated contract; current top-k identity alone remains sparse positive mechanical evidence, not skin ownership.

## 4. Candidate selected-edge evidence

Raw all-pair Geppetto edge logits must not be exposed as a second parent-selection surface.

If Arachne later benefits from structural certainty, Compiler may emit evidence **only about the accepted relation** and clearly local diagnostics around it.

Conceptual per-selected-edge record:

```text
child_canonical_joint_id
parent_canonical_joint_id
source_selected_edge_id
source_selected_edge_score01
source_selected_edge_confidence01
local_best_rejected_parent_score01
local_selected_margin01
margin_semantics: LOCAL_PROPOSAL_EDGE_MARGIN_NOT_GLOBAL_TREE_CONFIDENCE
```

Important limitation:

`selected score - best rejected local parent score` is **not** the global exact-solver optimality margin. The global optimizer can couple root/tree legality across children. Therefore the sidecar must not call this quantity `tree_confidence`.

If a true global structural robustness score is ever needed, it requires a separate Compiler-side sensitivity/second-best-solve definition.

**Ruling:** designable, but **not required for first A1 baseline**.

## 5. Candidate selected-root evidence

Analogous safe local record:

```text
root_canonical_joint_id
source_root_score01
best_rejected_root_score01
local_root_margin01
margin_semantics: LOCAL_PROPOSAL_ROOT_MARGIN_NOT_GLOBAL_SOLVER_CONFIDENCE
```

Again, canonical root is immutable. The score is evidence about the accepted proposal candidate, not permission to choose another root.

**Ruling:** audit/provenance candidate; low priority for A1 unless an ablation shows value.

## 6. Explicitly forbidden sidecar contents

The following are prohibited from the baseline qualified evidence contract:

- raw all-pair parent logits as an Arachne decision surface;
- raw root alternatives without accepted-state qualification semantics;
- hidden Geppetto control states;
- raw surface-attention tensors;
- diffusion training loss;
- generation index as semantic identity;
- saturated FIT1 salience magnitude as if calibrated;
- teacher control IDs or teacher-derived product-inference selectors;
- source proposal/hash strings as neural embeddings.

The original proposal artifact may remain in provenance storage. That does not make its raw alternatives legal Arachne authority.

## 7. Sidecar invariants

Any future implementation must satisfy all of the following:

1. `qualified_skeleton_hash` exact match is mandatory.
2. Every sidecar joint key must resolve to one existing canonical joint.
3. Every selected-edge record must exactly match the accepted `parent_canonical_id`.
4. No sidecar field may modify or repair position, root, parent, support anchors or canonical IDs.
5. Raw IDs/hashes are lineage only and are not learner features.
6. Missing sidecar evidence must be representable as `UNKNOWN/ABSENT`, never silently zero-imputed as confidence.
7. Calibration semantics must be explicit per field.
8. Candidate A1 code must be able to run with the sidecar disabled for causal ablation.
9. Permuting canonical serialization while preserving graph/identity correspondence must not alter physical predictions.

## 8. Recommended first A1 policy

Phase 6 does **not** recommend making the sidecar mandatory for the first predictor.

First A1 architecture should be able to consume:

`rich RiggingSurfaceIR + canonical QualifiedSkeletonIR[V2]`

without any sidecar.

Then sidecar channels can be introduced one at a time:

1. `position_sigma_normalized` first, because it has the strongest actual training semantics;
2. selected-edge/root local margins only if a structural-ambiguity analysis motivates them;
3. joint confidence only if it provides incremental value beyond sigma.

This preserves identifiability and prevents the A1 baseline from inheriting every upstream model score by default.

## 9. Phase-6 verdict

- **No need to reopen Compiler ownership.**
- **No need to force all Geppetto evidence into `QualifiedJoint`.**
- A small non-authoritative `QualifiedSkeletonEvidenceIR` is a valid future seam if/when calibrated evidence proves useful.
- The only current high-quality sidecar candidate is Geppetto position sigma; even that remains feature-admission blocked until calibration/ablation.
- Raw parent/root alternatives stay behind the Compiler boundary.

## 10. Next phase

Phase 7: recover and classify rigid-attachment/assembly coverage for the Mage staff/shield and inspect the current extraction→assembly type seam. This remains separate from A0 codec science.

Phase 8 remains blocked on completion of the running A0 token arms.

**No code/schema implementation, A1 training, feature admission or product claim is authorized by this Phase-6 design.**
