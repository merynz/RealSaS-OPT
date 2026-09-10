# RealSaS Learned Models

`models/` is the semantic home of the **current learned mainline**. A reader should not have to search dated experiment folders to discover what neural/learned subsystems RealSaS currently uses.

Models emit evidence or proposals. The Compiler remains the sole owner of canonical IDs, admissibility, qualified product state, proof binding and export authority.

## Current V4 learned stack

| Model / layer | Current package | Current role | Output boundary / status |
|---|---|---|---|
| IRIS V2 foundation/evidence | `models/iris/v2/` | promoted multi-view observation/foundation evidence package | observation/foundation evidence used by current IRIS lineage where applicable |
| IRIS V3 scene-first signed head | `models/iris/v3/` | promoted current Mage FIT1 signed-geometry composition | learned signed/support/uncertainty evidence; deterministic GSA owns `RiggingSurfaceIR` |
| Geppetto Reference-Strength V1 | `models/geppetto/reference_strength_v1/` | current FIT1-frozen anonymous skeleton/control proposal | `SkeletonProposalIR`; Mage FIT1 terminal PASS; generalization not claimed |
| Geppetto V2 | `models/geppetto/v2/` | prior mainline/research provenance | superseded for the current FIT1-frozen Geppetto formulation |
| SkinFieldCodec V1 | `models/skin_field_codec/v1/` | base learned continuous skin-field representation/shared decoder | current promoted base source; 278M V7 A0 remains research-only |
| Arachne V2 | `models/arachne/v2/` | prior/current qualified-skeleton-conditioned A1 scaffold | `SkinProposalIR`; **not** a current V7-native A1 FIT1 promotion |

IRIS V2 and V3 are **layers of one learned subsystem ownership envelope**, not a fifth learned subsystem. The V2 package preserves the promoted observation/foundation evidence surface; the later V3 scene-first signed field is the current promoted Mage signed-geometry witness used before deterministic GSA assembly.

Current Geppetto source authority is the separately promoted reference-strength package, not `models/geppetto/v2/`. The exact promotion and evidence chain are recorded in `canonical/GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md` and `canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`.

There are still four learned subsystem responsibilities in V4: IRIS, Geppetto, SkinFieldCodec and Arachne. Versioned packages/layers do not create extra semantic owners.

## What belongs with a model

A model version package may contain the tightly coupled current implementation surface:

- model/inference implementation;
- deterministic input conditioning owned by that model boundary;
- checkpoint compatibility and hash authority;
- current base loss/train step where promoted;
- current scientific evaluation where promoted;
- narrow compatibility shims needed to preserve audited/frozen source semantics.

This repo does not require artificial `src/`, `training/`, and `evaluation/` nesting when doing so would force needless rewrites of a tightly coupled audited package. **Semantic ownership and discoverability matter more than decorative depth.** A later refactor may subpackage internals after parity coverage exists.

## What does not belong with a model

- Compiler canonicalization/qualification;
- `ObservationEvidenceIR -> RiggingSurfaceIR` assembly;
- analytic world-point construction and deterministic local geometry authority;
- canonical graph/root/tree authority;
- skin legality/sparsification/qualification;
- MWB2, appearance compilation, preset motion qualification;
- proof, causal owner attribution, repair, export or runtime;
- one-off ablations, overfit probes, capacity diagnostics and post-failure remediation experiments unless separately promoted into the base contract.

## Learned vs deterministic

A learned module can execute deterministically for fixed weights, inputs and deterministic kernels. That does not make it a deterministic solver. SkinFieldCodec is the clearest example: it is an `nn.Module` with learned encoder/decoder parameters, even though a frozen checkpoint can produce repeatable output.

## Research upgrade lifecycle

```text
models/      = best currently authorized learned implementation
experiments/ = candidates, falsification, ablations, diagnostics

experiment proves a better mechanism
  -> audit / closure
  -> explicit promotion/refreeze transaction
  -> replace or amend model mainline
  -> regression / E2E
  -> old implementation remains recoverable through git + provenance
```

A scientific PASS does not promote itself. Do not keep two implementations as simultaneously current merely to preserve history; instead preserve the superseded one explicitly as provenance/compatibility source.

## Import firewall

- `experiments/` may import `models/`.
- current `models/` should not permanently depend on dated experiment semantic owners.
- a frozen promoted package may use a narrowly audited semantic-home import rebind to preserve exact historical bytes; such a compatibility seam must be documented and regression-gated.
- cross-model dependencies must resolve through the other model's current package or an explicitly frozen interface.
- models may construct typed Compiler proposal/evidence objects, but cannot claim Compiler qualification authority.

## Current scientific boundary

Geppetto is FIT1-frozen. Arachne is not.

The active learned-skinning science is A0 SkinFieldCodec representation/decode on the separate V7 research branch. If A0 closes, the actual V7 latent/decode interface must be frozen before a separately sized/preregistered V7-native A1 is trained. The old `models/arachne/v2/` scaffold is not evidence that this A1 has already been solved.

## Audit authority

See:

- `restoration/MODEL_SOURCE_OWNERSHIP_AUDIT_V1_20260903.md` for the original promoted source blobs/ownership audit;
- `models/iris/README.md` for current V2/V3 layered IRIS authority;
- `models/geppetto/reference_strength_v1/README.md` for the current frozen Geppetto formulation;
- `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` for current mechanism status.
