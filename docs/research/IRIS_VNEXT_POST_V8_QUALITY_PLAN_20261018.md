# IRIS vNext Post-V8 Quality Plan — 2026-10-18

Status: **ACTIVE RESEARCH PLAN**

Scope: IRIS/Fit8 post-V8 quality work. The layered-2D-asset program is intentionally out of scope for this plan.

## Governing principles

1. **Do not rewrite a qualified shared carrier to make downstream losses easier.** After M1, `scene_core` remains sealed unless a dedicated falsification court authorizes reopening it.
2. **Training completion is not PASS.** Fixed stage length and decreasing training loss cannot substitute for a preregistered held-out court.
3. **Never relax thresholds after seeing the result.** A metric that misses a preregistered gate remains a failure or optimization debt.
4. **Do not normalize 0.1–0.2 residuals as “close enough.”** Historical ~0.05-scale expectations may guide comparable normalized metrics, but there is no global official threshold yet.
5. **Do not manufacture truth.** Unidentifiable owner/amodal/presentation targets must abstain rather than inherit convenient geometric proxies.
6. **Separate optimizer learning rate from learning/convergence rate.** Slow convergence is diagnosed with gradients, update norms, sampling and held-out curves before scalar LR is changed.
7. **Preserve already-earned capabilities.** Every later-stage improvement is subject to non-regression courts on earlier qualified families.
8. **Capacity changes are a last inference, not the first reaction.** LoRA/adapters/new shared capacity require evidence that clean truth plus suitable optimization still cannot be decoded from the sealed carrier.

## Status vocabulary

- `MEASUREMENT_ONLY`: useful diagnostic; no preregistered pass gate.
- `OPTIMIZATION_DEBT`: identifiable target, but current held-out quality/convergence is inadequate.
- `ELIGIBLE_FOR_COURT`: target and metric are authoritative enough for a preregistered court.
- `COURT_PASS`: preregistered metric-specific gate passed without non-regression failure.

No raw training metric becomes `COURT_PASS` by convention.

## Workstream A — Carrier seal and non-regression

### A1. Make the V8 M1 seal a first-class contract

- Seal parameter prefix: `scene_core.`
- Downstream stages: A1/R1/T1/J1.
- Any attempt to optimize the sealed prefix downstream must fail closed.
- Reopening requires an explicit court ID, rationale and earlier-capability non-regression gates.

### A2. Preserve best qualified checkpoints

For each specialist family, retain the best checkpoint selected by held-out court score plus non-regression constraints; do not automatically promote the last step.

## Workstream B — Convergence/optimization program

The current V8 schedule is a baseline, not an optimized final curriculum.

### Required telemetry per head/family

At fixed evaluation intervals record:

- train and held-out loss/residual distribution (`p50`, `p95`, max where meaningful);
- gradient L2 norm;
- parameter-update L2 norm and update/parameter ratio;
- active-example count and positive/negative balance;
- trivial-negative fraction where applicable;
- learning rate and optimizer-group identity;
- best-so-far held-out checkpoint;
- non-regression metrics for already-qualified heads.

### Optimization rules

- Use head/family-specific optimizer groups instead of a blanket evidence LR when different heads exhibit different convergence behavior.
- Permit head-specific LR/schedules and stage extensions only after diagnostic evidence.
- Adaptive stage extension must be preregistered: extend only while held-out improvement remains meaningful and no non-regression gate is violated.
- Plateau with healthy gradients/updates is different from vanishing-gradient or under-sampled supervision; diagnose before changing LR.
- A target judged unidentifiable is removed/abstained, not trained for more steps.
- Boundary/cut heads may be isolated from already-converged relation heads if shared optimization causes interference.
- Never reopen `scene_core` merely because a specialist head converges slowly.

### Strict quality policy

No global numeric PASS gate is declared in this plan. Each metric family must preregister its own threshold and scale. For genuinely comparable normalized residuals, ~0.05 remains a historical engineering reference worth challenging the optimizer against; values in the 0.1–0.2 range remain explicit optimization debt unless a separately justified metric-specific court says otherwise.

## Workstream C — Salvage measurable StdGEN++-foundation evidence

Port only evidence whose authority can be defended from the current source assets:

- source/provenance binding;
- render-to-geometric equivalence;
- static all-hit ray census;
- fixed-frame dynamic visibility/order census;
- material provenance where authored;
- rig/skin/deformation truth;
- geometric component/boundary diagnostics;
- residual-boundary and continuation evidence.

Do not import inferred drawable-owner/amodal/presentation labels as authority.

Every salvaged producer must record provenance and distinguish `RAW_AUTHORED`, `DETERMINISTIC_DERIVED`, `QUALIFIED_DERIVED`, and diagnostic-only evidence where applicable.

## Workstream D — Boundary Causality Teacher V1

Replace the ambiguous monolithic boundary target with provable causes. A sample may contain multiple proven causes.

Allowed causes:

- `GEOMETRIC_TERMINATION`
- `MATERIAL_BOUNDARY`
- `ALPHA_BOUNDARY`
- `OCCLUSION_VISIBILITY_BOUNDARY`
- `SOURCE_PRIMITIVE_COMPONENT_SEAM`
- `RESIDUAL_UNKNOWN`

Rules:

- `RESIDUAL_UNKNOWN` is an explicit abstention, not a negative semantic label.
- No `drawable_owner`, hidden continuation, or intentional presentation override is inferred from these causes.
- Provenance for each positive cause is mandatory.
- Ambiguous evidence remains unknown instead of forced into the closest known cause.

Exit criterion: an audited boundary corpus with source-specific authority and class/support statistics, ready for preregistered held-out training courts.

## Workstream E — V8 evidence → compiler consumption audit

Audit at least:

- `front`
- `occlusion`
- `visibility`
- `same_drawable`
- `cut`
- `attachment`

For every signal, trace:

```text
DEFINED?
PRODUCED?
SERIALIZED?
CONSUMED?
BEHAVIORALLY_USED?
```

A learned head that is never behaviorally consumed cannot be credited with solving runtime presentation.

Exit criterion: a machine-readable lineage report with exact producer/serializer/consumer/use sites and fail-closed status for missing links.

## Workstream F — Independent TESSA topology court

Training loss alone is insufficient. Build a held-out court measuring, where applicable:

- valid/degenerate face counts;
- connected-component error;
- boundary-loop and hole behavior;
- manifold/non-manifold diagnostics without assuming all sources must be closed;
- canonical surface preservation / geometric deviation;
- source-support coverage;
- topology stability under selected held-out views/contexts.

Open surfaces are first-class; do not impose a universal closed-SDF ontology.

Only after this court identifies a concrete failure may StdGEN++-style hole/sign/topology regularization be introduced, with an ablation proving benefit and no carrier regression.

## Workstream G — Capacity decision court

Do **not** add generic LoRA/adapters/shared capacity merely because some V8 residuals are high.

A capacity change becomes eligible only if all are true:

1. target authority is defensible;
2. held-out sampling/class balance is adequate;
3. optimizer telemetry shows a genuine capacity/representation plateau rather than bad optimization;
4. specialist-head schedule search fails under preregistered bounds;
5. sealed-carrier probes show the required signal is not decodable with reasonable specialist capacity;
6. proposed change preserves earlier courts.

If the bottleneck is specialist capacity, modify the specialist first. Shared-carrier reopening remains the highest-cost option.

## Execution order

1. land carrier/quality policy contract;
2. land Boundary Causality V1 contract and tests;
3. port safe measurable census/boundary evidence from the foundation branch;
4. produce compiler-consumption lineage court;
5. produce independent topology court;
6. instrument head-specific convergence telemetry and held-out checkpoint selection;
7. preregister metric-specific gates;
8. run focused optimization experiments;
9. consider capacity changes only from court evidence.

## Non-goals

- no 2D layered-asset program in this plan;
- no retroactive V8 PASS declaration;
- no fabricated global semantic ontology;
- no silent carrier unfreeze;
- no threshold relaxation to fit observed results.
