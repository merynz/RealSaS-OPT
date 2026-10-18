# IRIS vNext Post-V8 Quality Plan — 2026-10-18

Status: **ACTIVE RESEARCH PLAN / SOURCE HARDENING GREEN; EXPERIMENTAL QUALIFICATION IN PROGRESS**

Scope: IRIS/Fit8 post-V8 quality work. The layered-2D-asset program is intentionally out of scope.

Execution status is summarized in `CURRENT_PLAN.md`; decisions/falsifications live in `IRIS_VNEXT_DECISION_LEDGER_20261018.md`.

## Governing principles

1. **Do not rewrite a qualified shared carrier to make downstream losses easier.** After M1, `scene_core` remains sealed unless a dedicated falsification court authorizes reopening it.
2. **Training completion is not PASS.** Fixed stage length and decreasing training loss cannot substitute for a preregistered held-out court.
3. **Never relax thresholds after seeing the result.** A metric that misses a preregistered gate remains a failure or optimization debt.
4. **Do not normalize 0.1–0.2 residuals as “close enough.”** Historical ~0.05-scale expectations may guide comparable normalized metrics, but there is no global official threshold.
5. **Do not manufacture truth.** Unidentifiable owner/amodal/presentation targets must abstain rather than inherit convenient proxies.
6. **Separate optimizer LR from actual convergence.** Diagnose gradients, update norms, sampling and held-out curves before changing scalar LR.
7. **Preserve earned capabilities.** Later improvements require non-regression courts.
8. **Capacity changes are a last inference.** LoRA/adapters/shared-capacity changes require evidence that clean truth plus suitable specialist optimization still cannot be decoded from sealed H.

## Status vocabulary

- `MEASUREMENT_ONLY`
- `OPTIMIZATION_DEBT`
- `ELIGIBLE_FOR_COURT`
- `COURT_PASS`

No raw training metric becomes `COURT_PASS` by convention.

## Workstream A — Carrier seal and non-regression

**Status: HARDENED IN SOURCE / CI GREEN at `b38feee7...`, run `39126288809`**

Implemented:

- exact downstream V8 stage IDs in `quality_policy_v1.py`;
- unknown A/R/T/J aliases fail closed;
- stage definition requires zero post-M1 `scene_lr_scale` unless explicit reopening court;
- actual optimizer `scene_core` group LR is checked by `IRISVNextFit8TrainerV8._apply_stage_lrs()`;
- policy tests consume the real `vnext_fit8_policy_v5()` curriculum.

Historical V8 already had the M1 carrier seal and stage-level specialist isolation. V8's physical optimizer topology was `scene_core` + combined `evidence_heads` + `structure`. Post-V8 `optimization_v2.py` separates future mechanical/appearance/relation/structure groups for independent LR and gradient/update telemetry while keeping `scene_core=0` fail-closed. This is instrumentation/optimization hardening, not a reinterpretation of historical V8.

Remaining:

- preserve best **qualified held-out** checkpoint per specialist family once held-out courts exist;
- carrier reopening remains prohibited until a separate falsification court is designed and passed.

## Workstream B — Truth cleanup and abstention

### Boundary Causality V1

**Status: CONTRACT + TESTS IMPLEMENTED; POPULATED CORPUS PENDING**

Allowed causes:

- `GEOMETRIC_TERMINATION`
- `MATERIAL_BOUNDARY`
- `ALPHA_BOUNDARY`
- `OCCLUSION_VISIBILITY_BOUNDARY`
- `SOURCE_PRIMITIVE_COMPONENT_SEAM`
- `RESIDUAL_UNKNOWN`

`RESIDUAL_UNKNOWN` is explicit abstention and emits no target.

### Historical relation-target correction

**Status: AUDITED**

Two historical paths are distinct:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

The V8 cut metric therefore measures an **owner-or-material boundary proxy**, not pure `1 - same` and not authoritative drawable-cut truth.

### Graphics / Relation Truth V2

**Status: CONTRACT + LOSS + PRODUCER + SUPPORT AUDIT SOURCE + HEAD V2 IMPLEMENTED / CI GREEN; ACTUAL CORPUS MATERIALIZATION PENDING**

Implemented source:

- `models/iris/vnext/graphics_truth_v2.py` — value + evidence authority + provenance + `valid_for_training`;
- `models/iris/vnext/training_v2.py` — explicit validity-mask loss;
- `tools/iris_vnext/build_fit8_relation_truth_v2.py` — versioned sidecar/manifest producer with support statistics;
- `models/iris/vnext/relation_support_v2.py` — no-support/single-class/binary-support eligibility;
- `models/iris/vnext/heads_v2.py` — projected alpha/appearance boundaries moved to the camera-conditioned view branch; compatible V1 weights are migrated and new projected-boundary rows begin from neutral logits;
- `models/iris/vnext/optimization_v2.py` — specialist optimizer groups and convergence telemetry.

Historical V8 shards stay immutable. No repository evidence yet seals a materialized Fit8 Relation Truth V2 corpus as experiment authority.

Current authority facts:

- static `same_drawable`, `cut`, `attachment` abstain without independent authority;
- valid deterministic `occlusion_probability` rows currently have single-class positive support;
- single-class support is not eligible for a binary held-out PASS court;
- do not fabricate negative labels.

Exit criterion for Workstream B: materialized corpus rows include authority/provenance/validity, support counts, positive/negative balance and abstention rates per family; manifest and sidecars are bound to exact admitted inputs and source.

## Workstream C — Salvage measurable foundation evidence

**Status: SOURCE AUDIT COMPLETE; SELECTIVE PORTING/USE ONLY**

Foundation provenance: `agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`.

Selective source-defensible families:

- geometry/topology qualification;
- render↔geometry equivalence;
- glTF/FBX material/skin/topology provenance;
- static all-hit ray census;
- fixed-frame authored-motion visibility/order/mechanics;
- residual-boundary diagnostics;
- local continuation evidence/courts.

Exact paths are listed in `CURRENT_PLAN.md` and the decision ledger.

Do not import inferred drawable-owner/amodal/presentation/semantic-partition labels as authority.

Every salvaged producer must label authority as `RAW_AUTHORED`, `DETERMINISTIC_DERIVED`, `QUALIFIED_DERIVED`, or `DIAGNOSTIC_ONLY`.

## Workstream D — Specialist convergence / optimization

**Status: OPTIMIZER + BASIC TELEMETRY IMPLEMENTED / FOCUSED EXPERIMENT PENDING**

`optimization_v2.py` now provides disjoint complete groups:

```text
scene_core       LR = 0
mechanical_head  explicit LR
appearance_head  explicit LR
relation_head    explicit LR
structure        explicit LR
```

and captures per-group parameter L2, gradient L2, optional update L2 and update/parameter ratio. Experiments must preregister specialist LRs; no arbitrary defaults are invented.

Focused experiment must still add/record per family:

- train and held-out p50/p95;
- active-example count;
- positive/negative balance;
- abstention/trivial-negative fraction;
- group/LR and gradient/update telemetry;
- best-so-far held-out checkpoint;
- prior-family non-regression metrics.

Rules:

- diagnose authority, conditioning, sampling/support, gradients and updates before changing LR;
- preregister any stage extension;
- never reopen `scene_core` merely because a specialist converges slowly;
- target unidentifiability is solved by abstention/producer repair, not more steps.

## Workstream E — Compiler/runtime consumption audit

**Status: INITIAL MACHINE-READABLE LINEAGE COMPLETE; TWO CONSUMPTION DEBTS OPEN**

Audit file: `IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`.

Source-proven paths:

- same/cut/boundary → fragment clustering;
- front + occlusion → directed order edges;
- explicit order graph → native pixel precedence;
- visibility → native face inclusion/exclusion, with E2E behavioral proof.

Open debts:

- attachment is serialized through RuntimeGraphics/`graphics.bin` but the current native reference raster does not behaviorally use it;
- support-band pixels are serialized/validated but not applied by the reference renderer.

Exit criterion: every signal classified as `BEHAVIORALLY_USED`, `TRANSPORT_ONLY`, or `UNCONSUMED`, with an exact owning test/site.

## Workstream F — Independent TESSA topology court

**Status: NOT YET BUILT**

Training loss is insufficient. Build a held-out court measuring, where applicable:

- valid/degenerate faces;
- connected-component error;
- boundary-loop and hole behavior;
- manifold/non-manifold diagnostics without assuming all sources are closed;
- canonical-surface deviation;
- source-support coverage;
- stability under held-out views/contexts.

Only after this court identifies a concrete failure may StdGEN++-style hole/sign regularization, sparse high-resolution extraction or topology-specific specialist capacity be introduced. Every intervention needs an ablation and geometry non-regression.

## Workstream G — StdGEN++ transfer

**Status: FRESH RE-AUDIT COMPLETE; CONDITIONAL TRANSFERS RECORDED**

Pinned public sources:

- `hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766`;
- `TencentARC/InstantMesh@08822c5200a4bcc0e2c1a4573409ac10b9f6692c`.

Adopt as discipline now:

- stable shared carrier / constrained specialization;
- preserve relevant geometric/raster supervision through representation transitions;
- make decomposition truth identifiable in data;
- explicit topology/thin-structure courts;
- specialist branches only where evidence requires them.

Investigate after courts:

- evidence-specific LoRA/adapters;
- sparse coarse-to-fine high-resolution extraction;
- continuity-conditioned hole/sign regularization;
- local high-resolution geometry specialist;
- richer TESSA boundary/component summaries.

Do not copy fixed body/hair/cloth ontology or silent forced-SDF-sign repair as authority.

## Workstream H — Capacity decision court

**Status: POLICY DEFINED / NOT ELIGIBLE YET**

A specialist adapter/branch becomes eligible only if all are true:

1. target authority is defensible;
2. held-out sampling/class balance is adequate;
3. gradients/updates are healthy;
4. bounded specialist schedule search fails;
5. sealed-H probes show the signal is not reasonably decodable;
6. earlier capabilities remain non-regressed.

Modify the specialist first. Shared-carrier reopening is the highest-cost option.

## Self-hosted preparation safety event

Run `39121291171` / prepare #22 used stale PR head `90c9c2f...`, locally sealed a source-bound V4 preparation, then entered Drive persistence to `gdrive:RealSaS_Fit8_IRIS_vNext_20261016`. It was cancelled during that persistence step before round-trip SHA verification.

Policy consequence:

- remote preparation path is quarantined until explicit SHA/provenance re-audit;
- the stale local/GitHub artifact is forensic evidence only;
- current training must not consume the Drive path based on presence alone;
- self-hosted execution must be owned by the current review source before another preparation write.

## Execution order from current state

1. keep carrier/runtime seal + focused CI green;
2. keep stale remote preparation quarantined and reverify it;
3. establish a current-branch-owned safe self-hosted verification/preparation flow;
4. materialize Relation Truth V2 and Boundary Causality statistics;
5. seal real V8→GraphicsRelationHeadV2 migration receipt/checkpoint;
6. run focused frozen-carrier specialist optimization with independent telemetry;
7. close attachment/support-band runtime ownership audit;
8. build independent TESSA topology court;
9. synchronize draft PR #116 only after final green review head is coherent;
10. consider adapters/high-res extraction/hole regularization only from court evidence.

## Non-goals

- no layered-2D-asset program in this plan;
- no retroactive V8 PASS declaration;
- no fabricated global semantic ontology;
- no silent carrier unfreeze;
- no threshold relaxation to fit observed results;
- no broad 12k rerun before focused courts justify it.
