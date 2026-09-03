# RealSaS Learned Models

`models/` is the production semantic home for RealSaS-owned learned components.

This directory exists so a reader can inspect each learned subsystem without searching dated experiment folders. It does **not** grant product authority to model outputs: models emit evidence or proposals; the Compiler remains the sole owner of canonical IDs, admissibility, product state, qualification, proof binding and export authority.

## Current canonical learned stack

| Model | Production role | Canonical output boundary | Model-local index |
|---|---|---|---|
| `iris/` | Multi-view RGB observation evidence learner | `ObservationEvidenceIR`-compatible learned evidence; learned geometric authority ends at depth/support/uncertainty | `models/iris/README.md` |
| `geppetto/` | Anonymous multimodal skeleton/control proposal learner | `SkeletonProposalIR` only | `models/geppetto/README.md` |
| `skin_field_codec/` | Continuous skin-field representation learner; shared decoder used by Arachne | latent field + dense decoded proposal weights; never qualification authority | `models/skin_field_codec/README.md` |
| `arachne/` | Qualified-skeleton-conditioned skin proposal learner | `SkinProposalIR` only | `models/arachne/README.md` |

This inventory is the **currently canonical learned stack proven by the V4 architecture and current source**. The restoration audit must still search the repository and historical authorities for any additional learned component before declaring the model inventory closed.

## Not models

The following are deterministic/compiler/runtime mechanisms and must not be moved here merely because they operate near learned outputs:

- `GeometricSubstrateAssembler`, analytic `P = O + dF`, local geometry qualification;
- canonical graph synthesis and skeleton qualification;
- skin legality/simplex/sparsification/qualification;
- MWB2 mesh/discretization and mesh-skin binding;
- appearance binding and visual completion qualification;
- deterministic preset motion and motion qualification;
- proof, owner attribution, bounded repair, export and runtime consumers.

## Required local layout

Each model converges to the same readable shape after source-promotion audit:

```text
models/<model>/
  README.md
  src/                 # inference-capable model implementation
  training/            # losses, targets, teacher-lane adapters, train entry points
  evaluation/          # eval/metrics/checkpoint compatibility
  tests/               # source-contract and model-local regressions
```

A subdirectory is created only when audited code is actually promoted into it. We do not create duplicate executable copies just to make the tree look complete.

## Promotion firewall

Current candidate implementations still live under dated `experiments/` packages. They are **not** bulk-moved.

Every file receives one disposition before promotion:

- `PROMOTE_MODEL_CORE`
- `PROMOTE_MODEL_TRAINING`
- `PROMOTE_MODEL_EVALUATION`
- `COMPILER_OWNED`
- `EXPERIMENT_ONLY`
- `ARCHIVAL_OR_SUPERSEDED`

Promotion requires dependency review, truth/firewall review, and a regression plan. Historical or experiment files remain as provenance after promotion; production imports must eventually resolve through `models/`, `compiler/`, or `runtime/`, never through a dated experiment package.

## Dependency direction

Allowed conceptual direction:

```text
models -> Compiler proposal/IR contracts
Compiler -> consumes model outputs through typed boundaries
runtime -> consumes proof-gated Compiler export only
```

The Compiler must never import a learned implementation in order to manufacture product truth. A model may construct typed proposal objects, but canonicalization and qualification remain Compiler-owned.

## Restoration status

Physical source promotion is intentionally blocked until `restoration/MODEL_SOURCE_OWNERSHIP_AUDIT_V1_20260903.md` closes the current + historical inventory and file-level dispositions. This is a source-organization gate, **not** a new architecture refreeze.