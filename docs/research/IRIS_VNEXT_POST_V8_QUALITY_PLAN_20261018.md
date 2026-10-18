# IRIS vNext Post-V8 Quality Plan — 2026-10-18

Status: **ACTIVE RESEARCH PLAN / IMPLEMENTATION IN PROGRESS**

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

**Status: HARDENED IN SOURCE; CI REQUIRED**

Implemented:

- exact downstream V8 stage IDs in `quality_policy_v1.py`;
- unknown A/R/T/J aliases fail closed;
- stage definition requires zero post-M1 `scene_lr_scale` unless explicit reopening court;
- actual optimizer `scene_core` group LR is checked by `IRISVNextFit8TrainerV8._apply_stage_lrs()`;
- policy tests consume the real `vnext_fit8_policy_v5()` curriculum.

Remaining:

- preserve best **qualified held-out** checkpoint per specialist family once held-out courts exist;
- carrier reopening remains prohibited until a separate falsification court is designed and passed.

## Workstream B — Truth cleanup and abstention

### Boundary Causality V1

**Status: CONTRACT + TESTS IMPLEMENTED; PRODUCER/CORPUS PENDING**

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

See `IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`.

### Graphics Truth V2

**Status: CONTRACT + EXPLICIT-VALIDITY LOSS + TESTS IMPLEMENTED; PRODUCER PENDING**

`graphics_truth_v2.py` requires each scalar target to carry:

- probability value;
- evidence authority;
- provenance;
- `valid_for_training`.

`training_v2.py` consumes explicit per-family/sample validity masks. An abstained family contributes no loss; an all-abstained batch fails closed.

Historical V8 shards stay immutable. V2 must be populated by a new source-audited producer before any focused training.

Exit criterion for Workstream B: new corpus rows include authority/provenance/validity, plus support counts, positive/negative balance and abstention rates per family.

## Workstream C — Salvage measurable foundation evidence

**Status: SOURCE AUDIT COMPLETE; SELECTIVE PORTING NEXT**

Port only evidence whose authority can be defended:

- source/provenance binding;
- render↔geometry equivalence;
- static all-hit ray census;
- fixed-frame dynamic visibility/order;
- material provenance where authored;
- rig/skin/deformation truth;
- component/boundary diagnostics;
- residual-boundary and local continuation evidence.

Do not import inferred drawable-owner/amodal/presentation labels as authority.

Every salvaged producer must label authority as `RAW_AUTHORED`, `DETERMINISTIC_DERIVED`, `QUALIFIED_DERIVED`, or `DIAGNOSTIC_ONLY`.

## Workstream D — Specialist convergence / optimization

**Status: POLICY DEFINED; INSTRUMENTATION PENDING**

Required telemetry per head/family:

- train and held-out p50/p95;
- gradient L2 norm;
- parameter-update L2 norm and update/parameter ratio;
- active-example count;
- positive/negative balance;
- trivial-negative fraction;
- LR and optimizer-group identity;
- best-so-far held-out checkpoint;
- prior-family non-regression metrics.

Rules:

- use head/family-specific optimizer groups where convergence differs materially;
- diagnose missing gradient, tiny updates, imbalance, trivial negatives and target ambiguity before changing LR;
- preregister any stage extension;
- never reopen `scene_core` merely because a specialist converges slowly;
- target unidentifiability is solved by abstention/producer repair, not more steps.

For comparable normalized residuals, `~0.05` remains a planning reference. `0.1–0.2` remains explicit debt unless a metric-specific preregistered court justifies otherwise.

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

Open surfaces are first-class.

Only after this court identifies a concrete failure may StdGEN++-style hole/sign regularization, sparse high-resolution extraction or topology-specific specialist capacity be introduced. Every intervention needs an ablation and geometry non-regression.

## Workstream G — StdGEN++ transfer

**Status: FRESH RE-AUDIT COMPLETE; CONDITIONAL TRANSFERS RECORDED**

See `IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`.

Adopt as discipline now:

- stable shared carrier / constrained specialization;
- preserve relevant geometric/raster supervision through mesh refinement;
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

## Execution order from current state

1. keep carrier/runtime seal + focused CI green;
2. consolidate vNext review into draft PR #116;
3. port identifiable foundation producers with explicit authority/provenance;
4. populate Graphics Truth V2 and Boundary Causality corpus statistics;
5. close attachment/support-band runtime ownership audit;
6. build independent TESSA topology court;
7. instrument specialist convergence and held-out checkpoint selection;
8. preregister metric-specific gates;
9. run narrow focused optimization experiments;
10. consider adapters/high-res extraction/hole regularization only from court evidence.

## Non-goals

- no layered-2D-asset program in this plan;
- no retroactive V8 PASS declaration;
- no fabricated global semantic ontology;
- no silent carrier unfreeze;
- no threshold relaxation to fit observed results;
- no broad 12k rerun before focused courts justify it.
