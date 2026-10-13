# Transition to a generic product and system repository

Date: 2026-10-13. User-authorized direction; implementation readiness remains in
the existing current-state and Platform contracts.

Evolve the existing system by promoting verified common behavior. Keep one main,
one Go control plane, one Python semantic owner and one C++ qualified consumer.
Fit8 is the next diversity/admission program; it does not replace those owners
or manufacture a second pipeline, scheduler, Registry or experiment ledger.

## Concrete delivery order

| Milestone | Exit evidence |
|---|---|
| Diverse source admission | The [eight pinned subjects](../../canonical/FIT8_DIVERSITY_SELECTION_V1.json) each pass source/import/pose/scene and first-render gates; failures remain visible |
| Shared Fit8 | Same model architecture, shared parameter sets and Compiler contracts; per-subject/module results and exact released-DAG Attempts; Knight regression retained |
| Common defect closure | Failure localized to an existing semantic owner, generic fix, exact dependency impact and matched affected-subject regression. Character names and joint indices never select production algorithms |
| LOFO/unseen | Presealed lineage-safe holdouts and historical-use audit; independent evidence, separate from fit success |
| Product construction | Real source input -> shipped inference -> Compiler qualification -> qualified ProductRevision on a clean host. External FIT receipts alone cannot substitute for the missing pretrained executor |
| Product operation | Qualified revision playback, portable dependency closure, resumable/idempotent work, cold compile and warm render timings, explicit supported-input envelope and failure behavior |

Source contracts, not human anatomy, define supported inputs. Pose, axes, units,
root/object placement, rigid attachments, materials and motion must travel through
typed artifact dependencies. The [Fit8 admission contract](../corpus/FIT8_ADMISSION.md)
puts those checks before fitting and again at the first compiled render.

## Repository discipline

- `models/`, `compiler/`, `platform/` and `runtime/` own shipped behavior. Each
  has its existing responsibility; promoted source does not import dated experiment
  code or consult a historical run-directory layout.
- `experiments/` and diagnostic tools produce hypotheses and scoped evidence.
  Promote a generic implementation through a short reviewed PR with its actual
  contracts, causal/regression evidence and source identity. Keep reproducible
  historical evidence at its indexed address.
- Subject examples, source locators and evaluation labels live in data/fixtures.
  Source-specific import adapters may explicitly handle format/version semantics;
  they must not conceal character-specific mechanics or appearance solutions.
- A normal new scientific/product result follows Go Attempts/EngineReleases or
  ProductRevisions. Historical direct scripts remain scoped reproducibility tools,
  never the default stateful product path.
- Version source, input, policy, checkpoint, camera and motion identities at their
  actual consumers. A fix invalidates the real dependent closure. One M remains
  each subject's coherent carrier; Knight's sealed M/G/W preservation is its
  regression boundary, not a ban on model or geometry development.
- Keep current entry documents and generated architecture discovery consistent.
  Retire an obsolete entrypoint only after migrating consumers and preserving its
  replay purpose. Cleanup proceeds with verified migrations, not bulk deletion.

Product promotion must close geometry, mechanics, appearance and presentation
together. A sound skeleton cannot compensate for missing limbs, broken grip,
coverage holes or incorrect overlap. Source/import admission, engineering CI,
teacher-relative fit, unseen evidence and product qualification retain their
distinct scopes under the existing qualification owner.

## Current open boundaries

The eight source identities are verified; first-render and truth admission are
pending. Shared Fit8, LOFO/unseen and product qualification are unproved. Full
pretrained input-to-output execution, recursive artifact portability and legacy
execution bypass enforcement remain existing system gaps. Current Knight demo
attachment/order candidates require their own canonical qualification.

This document orders that work. It does not change the current released graph,
register subjects, start training, relax proof thresholds or issue PASS. See
[execution lanes](EXECUTION_LANES.md), [Compiler/Platform ownership](adr/0002-compiler-platform-boundary.md)
and [Go/Python/C++ ownership](adr/0003-go-control-plane.md).
