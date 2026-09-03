# RealSaS Learned Models

`models/` is the semantic home of the **current learned mainline**. A reader should not have to search dated experiment folders to discover what neural/learned subsystems RealSaS currently uses.

Models emit evidence or proposals. The Compiler remains the sole owner of canonical IDs, admissibility, qualified product state, proof binding and export authority.

## Current V4 learned stack

| Model | Current package | Production role | Output boundary |
|---|---|---|---|
| IRIS | `models/iris/v2/` | multi-view RGB observation evidence | learned depth/support/uncertainty -> observation evidence |
| Geppetto | `models/geppetto/v2/` | anonymous multimodal skeleton/control proposal | `SkeletonProposalIR` |
| SkinFieldCodec | `models/skin_field_codec/v1/` | learned continuous skin-field representation/shared decoder | latent field + dense proposal weights |
| Arachne | `models/arachne/v2/` | qualified-skeleton-conditioned skin proposal | `SkinProposalIR` |

Current inference code for all four is now visible in `models/`. Current base training/evaluation code is also kept with the corresponding version package when its ownership/firewall audit is closed.

Repository-wide neural-source review found older IRIS lines and Geppetto/Arachne V1 alternatives, but no fifth learned subsystem authorized by the current V4 architecture. Those older learned implementations remain research/provenance rather than parallel current mainline.

## What belongs with a model

A model version package may contain the tightly coupled current implementation surface:

- model/inference implementation;
- deterministic input conditioning owned by that model boundary;
- checkpoint compatibility;
- current base loss/train step;
- current scientific evaluation;
- narrow compatibility shims needed to point old relative imports at another model's canonical current package.

This repo does not require artificial `src/`, `training/`, and `evaluation/` nesting when doing so would force needless rewrites of a tightly coupled audited package. **Semantic ownership and discoverability matter more than decorative depth.** A later refactor may subpackage internals after parity coverage exists.

## What does not belong with a model

- Compiler canonicalization/qualification;
- `ObservationEvidenceIR -> RiggingSurfaceIR` assembly;
- analytic world-point construction and deterministic local geometry authority;
- canonical graph/root/tree authority;
- skin legality/sparsification/qualification;
- MWB2, appearance compilation, preset motion qualification;
- proof, causal owner attribution, repair, export or runtime;
- one-off ablations, overfit probes, capacity diagnostics and post-failure remediation experiments unless promoted into the base contract.

## Learned vs deterministic

A learned module can execute deterministically for fixed weights, inputs and deterministic kernels. That does not make it a deterministic solver. SkinFieldCodec is the clearest example: it is an `nn.Module` with learned encoder/decoder parameters, even though a frozen checkpoint can produce repeatable output.

## Research upgrade lifecycle

```text
models/      = best currently authorized learned implementation
experiments/ = candidates, falsification, ablations, diagnostics

experiment proves a better mechanism
  -> audit / closure
  -> replace or amend model mainline
  -> regression / E2E
  -> old implementation remains recoverable through git + provenance
```

Do not keep two current implementations merely to preserve history.

## Import firewall

- `experiments/` may import `models/`.
- current `models/` must not permanently import dated `experiments.*` implementations.
- cross-model dependencies must resolve through the other model's current package, not through the shared historical R6 experiment directory.
- models may construct typed Compiler proposal/evidence objects, but cannot claim Compiler qualification authority.

## Audit authority

See `restoration/MODEL_SOURCE_OWNERSHIP_AUDIT_V1_20260903.md` for exact source blobs, compatibility splits and experiment-only dispositions.
