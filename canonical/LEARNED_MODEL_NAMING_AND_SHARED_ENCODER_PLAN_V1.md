# RealSaS — Learned Model Naming and Shared Encoder Plan V1

**Date:** 2026-10-04  
**Status:** `CANONICAL_NAMES_SELECTED__ALIAS_FIRST_MIGRATION__SHARED_ENCODER_COURT_AUTHORIZED`

## Naming law

Each learned model name must satisfy the IRIS rule:

1. it reads naturally as a name;
2. every letter comes from a concise description of the model's actual function;
3. the expansion remains technically true as the implementation evolves;
4. external-paper names do not become canonical RealSaS model identity.

## Canonical learned-model names

### IRIS

Existing perception/evidence model name remains unchanged.

### ATLAS

**A**rticulation **T**opology & **L**ocus **A**utoregressive **S**ynthesis

Canonical role:

```text
RiggingSurfaceIR
 -> articulation/control locus evidence
 -> topology / parent / root evidence
 -> endogenous cardinality / stop evidence
 -> SkeletonProposalIR / control-basis evidence
 -> Compiler qualification
```

ATLAS does not own final canonical root/tree/IDs or final mechanically-optimal cardinality.

Legacy historical name: `Geppetto`.

### MIRA

**M**echanical **I**nfluence & **R**ig **A**ttachment

Canonical role:

```text
RiggingSurfaceIR + QualifiedSkeletonIR
 -> point/control mechanical relation reasoning
 -> dense influence / skin-weight proposal
 -> deformation-relevant attachment evidence
 -> SkinProposalIR
 -> Compiler qualification
```

MIRA does not own legal final skinning, sparsification or product authority.

Legacy historical name: `Arachne`.

## Canonical chain

```text
IRIS -> deterministic geometry/substrate
     -> ATLAS -> Compiler skeleton/control-basis qualification
     -> MIRA  -> Compiler skin/deformation qualification
     -> runtime
```

Short form:

`IRIS -> ATLAS -> MIRA -> Compiler`

## Migration policy

Do not perform blind repository-wide string replacement.

Migration is alias-first and atomic:

1. freeze semantic-name map;
2. add canonical package/class/architecture aliases;
3. update new docs/tests/workflows to ATLAS/MIRA;
4. preserve legacy imports and checkpoint/provenance readers;
5. rebind checkpoint metadata without changing tensors;
6. run replay/contracts;
7. remove legacy aliases only after proven safe.

Historical documents retain their historical names.

## Shared surface encoder question

ATLAS and MIRA remain separate semantic models even if they later share parameters.

The initially shareable region is only the **pre-skeleton, surface-only representation**:

```text
RiggingSurfaceIR
  -> SharedSurfaceEncoder?
       -> surface memory
          -> ATLAS-specific autoregressive articulation/topology stack
          -> MIRA-specific QualifiedSkeletonIR + pair/deformation stack
```

MIRA's skeleton-conditioned, pair-geometry, tree and skin/deformation layers are not shared by default.

## Experiment ladder

### E0 — separate baseline

Current ATLAS and MIRA encoders/checkpoints remain separate. Record:
- parameter count;
- input field overlap;
- surface-memory widths;
- ATLAS mechanical metrics;
- MIRA deformation metrics;
- runtime and peak memory.

### E1 — shared initialization, separate checkpoints

Initialize both task-specific surface encoders from one common encoder, then fine-tune independently.

Question: does a common geometric prior help without parameter sharing or gradient coupling?

### E2 — shared encoder, ATLAS-owned gradients

One encoder instance is reused; only ATLAS/rig loss updates it. MIRA consumes the frozen shared surface memory plus its qualified-skeleton-specific stack.

Question: can runtime/parameter reuse be gained without allowing skin supervision to alter rig representation?

### E3 — true multitask shared encoder

Both ATLAS and MIRA losses update the common surface encoder. Heads and typed IR boundaries remain separate.

This arm is allowed only after E2. It requires explicit negative-transfer checks. Skin/teacher information may affect shared parameters through training gradients, but may never appear as an ATLAS inference input or bypass the Compiler skeleton boundary.

## Required decision metrics

Every arm must report:

- ATLAS locus/topology proposal quality and Compiler-qualified mechanical result;
- MIRA final qualified skin/deformation behavior, not latent loss only;
- motion-probe consequences;
- parameter count;
- peak memory;
- inference wall time;
- surface-order/permutation stability;
- negative transfer in both directions;
- exact same RiggingSurfaceIR evidence contract.

Promote sharing only if it is Pareto-noninferior on both downstream mechanical tasks and materially improves fit/generalization stability, parameter count, runtime/memory or data efficiency.

A smaller model that worsens either qualified rig mechanics or qualified skin/deformation is not a win.

## Current authorization

- source-level encoder compatibility audit: authorized now;
- frozen-representation / no-training court: authorized now;
- short CPU/GPU diagnostic court: authorized now;
- new A100 fit: blocked until the shared-encoder interface and causal arm are preregistered.

