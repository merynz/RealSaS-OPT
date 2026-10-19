# RealSaS-OPT Agent Entry Contract

## Mandatory current handoff — 2026-10-19

This repository must be resumable without conversational memory. **Current state and machine-readable experiment authority outrank historical recovery notes.**

### Read first

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`
4. `canonical/H_REP_P0_INPUT_BINDING_V1.json`
5. `canonical/H_REP_P0_PREREGISTRATION_V1.json`
6. `docs/research/IRIS_H_REP_P0_DECISION_LEDGER_20261019.md`
7. `docs/research/IRIS_SURF_A1_FROZEN_H_RESULT_20261019.json`
8. `canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json`
9. `docs/research/IRIS_TESSA_V2_DECISION_LEDGER_20261019.md`
10. `docs/research/IRIS_TESSA_V2_STRUCTURALIZATION_PLAN_20261019.md`
11. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
12. `SYSTEM_INDEX.md`
13. `docs/platform/EXECUTION_LANES.md`

### Continuity authority pointers

Current continuity files must be read and treated as current before historical predecessors:

- research experiment authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`
- current experiment registry: `canonical/EXPERIMENT_REGISTRY_V3.json`
- current scientific journal: `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
- active run ledger: `canonical/ACTIVE_RUN_V2.json`
- current pipeline plan: `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
- current product architecture authority: `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
- current appearance authority: `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`

Historical continuity files, Knight/V2/V5/V6 notes and old recovery branches are provenance/regression evidence. They do not override current authority.

If the user says only `continue / devam et`, continue the active work in `CURRENT_PLAN.md`. Do not revive an older branch, trainer, hidden semantic-owner program, independent visual mesh or superseded relation/surface experiment merely because historical files still exist.

## Authority model

Keep source, experiment and product authority separate:

```text
main              = merged product code truth
review branch     = temporary source collaboration
Artifact Registry = immutable artifact/dependency identity
Attempt           = durable research truth
ProductRevision   = durable production truth
Workflow Engine   = durable execution state
```

Repository source and green CI are **not** scientific PASS. A Drive path is not authority. Scientific authority requires the exact artifact hash plus typed receipt/court bound to the exact source and preregistration.

Completed IRIS-V8 is immutable model/history authority:

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- global step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`;
- shared `scene_core` sealed after M1.

Later source hardening does not rewrite those bytes or their scientific lineage.

Active review source is `research/iris-vnext-hybrid-tessa-v2-20261019` / PR #122. H-REP-P0 may be staged on a child source-gate branch while source review is incomplete; only an exact final green head may be used by an A100 worker.

## Current scientific program

The active experiment is **H-REP-P0**, opened because `SURF-A1-FROZEN-H` completed a real preregistered scientific FAIL after all apparatus gates passed.

SURF-A1 facts that must not drift:

- exact execution source `afc18debbc4d5858a4a632884b7cfd2141dcc35a`;
- A100 preflight passed; 64/64 hard raster views had precision/recall `1.0`;
- native R24 parity was exact on faces for all 8 subjects;
- all `2400` extraction-only optimizer steps completed;
- final independent R48 court status: `SCIENTIFIC_FAIL_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE`;
- A1 substantially improved native V8 C0 but missed locked global IoU, Chamfer, mask-gap, normal-alignment and per-subject p95 gates;
- next action is representation investigation, not threshold relaxation, SURF-A2 promotion or TESSA training.

Do **not** summarize this as “H contains no geometry.” H-REP-P0 exists specifically to separate representation decodability from mesh-extraction debt.

## H-REP-P0 law

Question:

> Is local surface geometry decodable from exact sealed V8 `H(x)` or the sealed raw field without FlexiCubes, SURF-A1 mesh extraction, TESSA or Compiler repair?

Four controls have exact equal parameter count and the same `387-D` input envelope:

```text
H_XYZ              sealed local 384-D H + XYZ
RAW_FIELD_XYZ      raw field in evidence channel 0 + zero padding + XYZ
XYZ_ONLY            all evidence channels zero + XYZ
SHUFFLED_H_XYZ     fixed within-subject permuted H + original XYZ
```

No subject ID, subject-specific trainable parameters or semantic-class routing is allowed.

Probe truth uses only sealed V4 geometry already admitted by independent courts:

- `teacher_surface_points_iris`;
- `mechanical_points`;
- `mechanical_normals`.

Nearest-point distance/offset is an explicit **point-cloud geometry proxy**, not exact SDF or occupancy truth. Teacher truth is target-only and must never enter the probe input.

V8 `scene_core` remains frozen and hash-checked. H-REP-P0 does not authorize carrier retraining merely because it exists or fails.

### H-REP outcome boundary

If H or raw field earns its entire preregistered held-out gate set:

```text
DESIGN_EXTRACTION_TRANSITION_ALTERNATIVE
DO_NOT_RETRAIN_H_YET
DO_NOT_TRAIN_TESSA
```

If neither earns its gate set:

```text
PREREGISTER_CARRIER_REPRESENTATION_RETRAINING_EXPERIMENT
DO_NOT_TRAIN_TESSA
```

Neither outcome directly authorizes SURF-A2.

## Runtime / notebook contract

Research notebooks are compute workers, not authority. Required operator UX:

```text
Open notebook
→ Run all
→ approve Google Drive exactly once
→ autonomous exact-hash execution
→ meaningful phase logs
→ resumable Drive checkpoints/receipts
→ independent final court
```

All new A100 notebooks must:

- use physically separated phase cells like the successful V8 worker;
- contain exactly one `google.colab.drive.mount('/content/drive', force_remount=False)`;
- never call `auth.authenticate_user()` or interactive rclone/gcloud auth;
- never scan full Drive with `os.walk`, `rglob` or broad recursive discovery for known inputs;
- use exact pre-resolved mounted paths plus hashes for scientific inputs;
- check exact source SHA and clean checkout;
- precompile before optimizer authorization;
- enforce the experiment’s explicit accelerator/memory gate;
- fail closed on source/input/prereg/checkpoint drift;
- forbid hidden reduced-capacity/resolution/subject fallback.

For H-REP-P0 specifically: exact A100-class accelerator with at least `39 GiB` CUDA memory is preregistered. Optimizer step 1 is forbidden until source, input binding, preregistration, V8 checkpoint, V4 prepared shards, probe dataset receipt, scene-core hash and equal probe parameter counts all pass.

## Single-carrier product rule

There is exactly one product mechanical carrier after candidate selection:

```text
candidate M*
  → static/mechanical courts
  → deterministic selection
  → ONE canonical M/G/W_M
  → Graphics / Motion / Runtime
```

No independent deformable visual mesh is allowed. Bounded 2D presentation support may exist only as an explicitly authorized sidecar driven by canonical mechanics; it may not hide missing 3D geometry or mint another mesh.

## IRIS / TESSA / Compiler boundary

```text
8-view observations
        ↓
IRIS continuous visual/geometric/mechanical evidence H
        ↓
qualified high-fidelity direct surface
        ↓
OPTIONAL TESSA structuralization / learned retopology
        ↓
Compiler qualification
        ↓
ONE canonical animation-ready M
```

TESSA is optional. It is not a second reconstructor and cannot repair a grossly wrong upstream witness. `TESSA-V2-B1` remains blocked until a genuinely qualified `SURF-A2-HIRES` direct arm exists.

> Compiler may enforce admissible truth and global consistency; Compiler may not compensate for model incapacity or fabricate unavailable truth.

Allowed Compiler work includes exact validation, IDs/hashes/addressing, source-bound deterministic derivation, global graph/constraint solving, candidate selection, exact contact/barycentric solves, bounded numerical sanitation and fail-closed/abstain behavior.

Forbidden Compiler work includes large topology/hole repair presented as model success, semantic skin transfer hiding carrier drift, invented hidden/amodal/presentation truth, an independent visual mesh or subject-specific production rescue branches.

## Truth law

> Unknown is a first-class result. Do not turn an unidentifiable proxy into training truth.

External papers/repos are design references, never authority. Keep evidence classes separate:

```text
CODE_EXACT        public code directly verified
PAPER_EXACT       paper statement/equation
REALSAS_INFERENCE our translation
UNAVAILABLE       exact implementation not public
```

StdGEN/StdGEN++ lessons may inform training discipline and representation transitions, but do not copy body/hair/cloth ontology as mandatory RealSaS inference classes.

## Fit8 law

Fit8 is a **shared-system qualification cohort**, not merely eight subjects and not an unseen-generalization claim. It forces the same models, contracts, Compiler layers and workflow across heterogeneous assets before future large unseen campaigns.

A subject failure is evidence about the generic system, not permission for a per-character production rescue branch.

## Graphics consumption rule

A learned or serialized field is not automatically behaviorally consumed. Use explicit states such as `BEHAVIORALLY_USED`, `TRANSPORT_ONLY`, or `UNCONSUMED`; avoid ambiguous “consumed”.

Read `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json` before making runtime claims.

## Topology rule

TESSA training loss is not topology qualification. No topology PASS exists until an independent held-out court measures actual face validity, component error, boundary loops/holes, non-manifold behavior where meaningful, canonical-surface deviation, source support and held-out stability.

## Quality / PASS discipline

Training completion is not PASS.

```text
MEASUREMENT_ONLY
OPTIMIZATION_DEBT
ELIGIBLE_FOR_COURT
COURT_PASS
```

`COURT_PASS` requires authoritative target, held-out metric, preregistered metric-specific threshold and required non-regression. Thresholds may not be changed after seeing results.

## Relation lane

Relation V10 is **parked, not cancelled**. Do not context-switch back to it while H-REP-P0 is unresolved. Preserve its separate Truth-V3/all-hit/ray-stack readiness boundary.

## Self-hosted execution safety

Known runner: `realsas-wsl-1660ti`, labels `[self-hosted, linux, x64, realsas]`.

Keep it offline by default. Historical stale-run Drive snapshots remain forensic evidence only unless explicitly requalified. If execution unexpectedly starts from a stale branch/head, stop it before persistence/promotion and record exact run/head/artifact impact.

## Platform-first execution

For stateful research/product work, read `docs/platform/EXECUTION_LANES.md` and use the Go Platform/`realsasctl` Attempt flow where supported.

Colab/A100 and self-hosted Actions are compute workers, not authority. Checkpoints/receipts become authoritative only after the owning scientific/Platform path validates and binds them.

## Source-work exit rule

Every meaningful architectural/training change must leave a recoverable record of:

1. exact source/commit;
2. owning module;
3. why it exists;
4. evidence/court;
5. what it proves;
6. what it does not prove;
7. what it replaces or falsifies;
8. next action.

Update `CURRENT_STATE.md`, `CURRENT_PLAN.md`, `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json` and the active decision ledger whenever interpretation changes. Keep the repo more current than the conversation, never the reverse.

## Claim discipline

A FIT/witness PASS is scoped to its exact subject/apparatus. Engineering CI is not unseen/generalization evidence. Near-zero training loss is not semantic authority. Serialization is not behavioral use. A visually incorrect puppet is not accepted merely because mechanics are valid.
