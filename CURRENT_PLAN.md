# RealSaS-OPT — Current Plan

**Date:** 2026-10-21  
**Status:** **V2 TIE-SAFE EVIDENCE ADMITTED; INFERENCE-SAFE PRESENTATION MERGED; A/B PREREG LOCKED; EXECUTION SOURCE + COURTS MERGED; BUILD RUN-ALL NOTEBOOK THEN HAND OFF**  
**Production authority:** `main`  
**Long-term research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Active lane:** `SHARED_STRUCTURAL_EVIDENCE`  
**Active workstream:** `STRUCTURAL_EVIDENCE_AB_NOTEBOOK_BUILD_AND_HANDOFF`  
**Execution policy:** continue source/artifact assembly only until the notebook is complete. **Stop before notebook execution, optimizer steps or scientific court execution.**

## Objective

Test the actual shared-scene structural hypothesis without semantic shortcuts:

```text
8 views
  ↓
IRIS shared structural scene
  + qualified metric surface evidence
  ↓
geometry + graphics + mechanics evidence
  ↓
role-free topology structuralization
  ↓
Compiler qualification
  ↓
ONE persistent animation-ready canonical structure
  ↓
3D mechanics → source-preserved editable 2D presentation
```

The immediate question is narrower and preregistered:

> Does inference-admissible role-free presentation evidence improve learned topology structuralization and downstream presentation/motion behavior beyond the same scene+surface+mechanics evidence alone?

This experiment is Fit8 shared-system contract qualification. It is **not** unseen-generalization evidence and does not itself authorize product M.

## Closed prerequisites

### P1 — tie-safe V2 evidence — CLOSED

`canonical/STRUCTURAL_EVIDENCE_INPUT_ADMISSION_V2.json`

- 8 Fit8 subjects;
- 3,499,296 oriented pair rows;
- unique-min global visibility, tied minima ABSTAIN;
- pairwise equal-depth ties ABSTAIN;
- Drive round-trip verified;
- 0 optimizer / 0 backward;
- manifest SHA256 `9accf4d2ca85e90e23a3c7ed2a490329676582c5e6759b5d4a1df84df998bb1f`.

### P2 — inference-safe role-free presentation producer — CLOSED

PR #137 merge `b0a2ae68a7ef8652f0e319495def0304c1faa03e`.

B-arm presentation evidence comes from model-produced surface anchors plus admitted source observations. V2 all-hit truth is never direct inference conditioning.

### P3 — A/B preregistration — LOCKED

PR #138 merge `d9e7178a597e986cef2067e5d45210f5bafdfd52`.

`canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json`

A and B are matched in architecture, capacity, initialization, data authority, seed, optimizer, schedule, update budget and checkpoint policy. Only the presentation evidence value differs.

### P4 — execution source and courts — CLOSED

PR #139 merge / exact execution source:

`ebfe80fc9d0c43cc8075d3d38bdbc9231357b969`

All required source workflows were green before merge.

Execution components:

- `train_fit8_tessa_v3_structural_evidence_ab_v2.py` — exact A/B trainer + resume lineage;
- `preflight_structural_evidence_ab_a100_v1.py` — real forward/backward, 0 optimizer;
- `evaluate_structural_evidence_ab_free_running_v1.py` — strict final-step free-running both arms;
- `court_structural_evidence_ab_topology_surface_v1.py`;
- `court_structural_evidence_ab_static_presentation_v1.py`;
- `court_structural_evidence_ab_dynamic_motion_v1.py` — dynamic metrics + final locked A/B decision.

Metric definitions are frozen in:

`canonical/STRUCTURAL_EVIDENCE_AB_COURT_METRIC_DEFINITIONS_V1.json`.

Post-merge source binding is:

`canonical/STRUCTURAL_EVIDENCE_AB_EXECUTION_BINDING_V1.json`.

## Locked experiment

### Arm A — control

```text
A_NULL_PRESENTATION
scene memory + surface/mechanics memory + matched null presentation memory
→ TESSA V3
```

### Arm B — hypothesis

```text
B_ROLE_FREE_PRESENTATION
same scene memory + same surface/mechanics memory
+ inference-admissible role-free presentation memory
→ same TESSA V3
```

### Training contract

- seed: `26091991`;
- AdamW: lr `1e-4`, betas `(0.9,0.95)`, eps `1e-8`, wd `0.01`;
- clip norm `1.0`;
- BF16 autocast, FP32 parameters/optimizer state;
- exactly `2400` optimizer steps per arm;
- warmup `120`, then cosine to `1e-5`;
- checkpoint every `200` steps;
- Fit8 round-robin, `300` optimizer steps per subject;
- deterministic cyclic windows;
- no teacher-forced checkpoint selection;
- final-step checkpoint is evaluated;
- no free-running sampling.

## Locked court

### Gate 1 — topology/surface no regression

B must satisfy the locked face-economy, surface-distance, silhouette, manifold, degeneracy, boundary-chain, component-count and EOS requirements.

### Gate 2 — presentation primary

Five preregistered metrics:

1. `GLOBAL_VISIBILITY_ACCURACY`
2. `LOCAL_PARTIAL_ORDER_ACCURACY`
3. `POSE_DEPENDENT_ORDER_FLIP_ACCURACY`
4. `FRAGMENT_IDENTITY_STABILITY`
5. `SOURCE_ART_SUPPORT_PRESERVATION`

B requires macro absolute gain `>= 0.01` and no subject regression beyond `0.03`.

### Gate 3 — motion primary

- `OWNER_FRAGMENT_STABILITY_OVER_MOTION`: absolute gain `>=0.01`;
- `GAP_TEAR_RATE`: relative reduction `>=0.10`;
- `G3_DYNAMIC_QUALIFICATION`: must not regress;
- attachment persistence: ABSTAIN where explicit attachment authority is absent.

Teacher-authored motion is evaluation-only. It is applied identically to A/B candidate surfaces by exact closest-point barycentric binding; teacher rig/skin/motion never enters training or inference conditioning.

### Final rule

```text
B PASS = topology/surface gate PASS
         AND presentation primary PASS
         AND motion primary PASS
```

A locked FAIL is a valid experiment outcome. No metric/threshold/capacity changes are allowed after results are observed.

## Immediate implementation plan

### N1 — exact execution authority — COMPLETE IN THIS SOURCE WORKSTREAM

- bind PR #139 merge SHA;
- bind prereg + metric-definition identities;
- record all green source CI;
- advance `CURRENT_STATE`, `CURRENT_PLAN`, `AGENTS` and machine-readable ledger.

### N2 — build one fail-closed A100 run-all notebook — NEXT

The notebook must be orchestration only; scientific logic remains in repo source.

Required cells/stages:

1. mount/authenticate Drive and establish one output root;
2. clone repository and checkout exact execution SHA `ebfe80fc9d0c43cc8075d3d38bdbc9231357b969`;
3. verify preregistration and execution-binding identities;
4. hydrate exact Stage01/prepared/direct-surface/V2-evidence/direct-presentation/authored-teacher artifacts and verify SHA256;
5. require a physical NVIDIA A100; no reduced-capacity fallback;
6. run zero-optimizer preflight;
7. run Arm A exactly 2400 steps, with atomic resumability and hash-bound receipts;
8. run Arm B exactly 2400 steps under identical protocol;
9. run strict final-step free-running for both arms;
10. run topology/surface court;
11. run static presentation court;
12. run dynamic authored-motion court, which produces the final locked A/B result;
13. persist checkpoints, telemetry, receipts, courts and a final run manifest back to Drive;
14. print the exact output folder and final receipt hashes.

Infrastructure/integrity failures must stop the notebook. A valid scientific FAIL must be persisted as a completed result, not treated as an infrastructure exception.

### N3 — STOP AND HAND OFF

Once the notebook is built and audited, tell the user the notebook boundary has been reached. Do **not** run it automatically.

## Forbidden shortcuts

- V2 all-hit / teacher truth as B inference conditioning;
- teacher topology, rig or skin in inference forward;
- named semantics or material IDs as structural/owner authority;
- incomplete ray stack treated as global-visibility truth;
- different capacity, initialization, schedule or update budget between arms;
- reduced-capacity fallback;
- teacher-forced checkpoint selection;
- result-driven threshold relaxation;
- scientific PASS from CI/preflight;
- automatic H-GEO-P4 or SURF-A2 diversion.

## After the experiment

Only the locked result decides the next research move.

- If B passes: presentation evidence has causal support as a useful structural prior; next work can integrate the finding into the broader generic IRIS/TESSA path and then separately close Compiler/product mechanics.
- If B fails: inspect the preregistered per-metric/per-morphology evidence without relaxing the court; localize whether the failure is presentation evidence quality, topology consumption, geometry provider limitations or downstream mechanics.

Neither outcome alone proves unseen generalization.
