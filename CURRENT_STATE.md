# RealSaS-OPT — Current State

**Date:** 2026-10-18  
**Canonical production line:** `main` until reviewed vNext is merged  
**Active review source:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Latest verified focused source before this continuity repair:** `c22850ae4fd8ca31f315a5488633768f1bfa3fe6`  
**Verified focused workflows on that source:** completion audit `39132578331` + full vNext suite `39132578290` = `completed/success`  
**Review vehicle:** draft PR #116; keep it fast-forward synchronized only to green active-review heads  
**Active program:** **POST-V8 QUALITY / TRUTH / CONSUMPTION QUALIFICATION**  
**Scientific status:** **NO IRIS FIT8 SCIENTIFIC PASS**

## Authority separation

Do not collapse these into one notion of current:

1. `main` is merged production-code authority.
2. Completed V8 is an immutable architecture/measurement baseline bound to source `5083c9915533ec8ec6530c7d229c635887abfea7` plus exact checkpoint/receipt lineage.
3. The active post-V8 branch is review source only; it does not retroactively mutate V8.
4. Artifact Registry / Attempt / ProductRevision remain durable research/product authority. Git chronology and green CI are not scientific authority by themselves.

## Preserved V2 / product continuity

The post-V8 IRIS research program does **not** erase or replace the current V2 control-plane and product authority records already present on `main`.

- Current implementation-readiness pointer: `canonical/V2_IMPLEMENTATION_READINESS.json`.
- Current control-plane continuity includes the existing **43-stage dependency DAG** and gate id `PLATFORM_EXECUTION_LANES`.
- Pre-existing continuity state token: `CARRIER_NATIVE_FIT1_SCOPED_PASS__PLATFORM_LANES_IMPLEMENTED`.
- **Geometry**, **Mechanics**, and **Appearance** remain separate product-quality authority domains.
- `QualifiedMeshIR` remains the typed qualified-mesh authority entering the canonical carrier boundary after its own qualification.
- `SurfaceAddressing` remains the stable canonical-M addressing boundary; current implementation home is `compiler/realsas_compiler_core/surface_addressing_v1.py`.
- **Complete Appearance Authority** remains the source-preserving appearance authority downstream of canonical M; post-V8 learned Graphics evidence does not silently replace that product authority.

These continuity anchors are orthogonal to the active IRIS scientific program. Naming them here does **not** grant a new IRIS/Fit8 scientific PASS, absolute G3 PASS, unseen generalization, or product qualification.

## Training lineage and current invariant

- V5 falsified unsafe single-objective downstream rewriting of shared H.
- V6 showed cumulative retention can preserve Mechanics while Geometry/extraction still drift because post-M1 `scene_core` remained trainable.
- V7 repaired TESSA coordinate compatibility only; V6 optimizer/curriculum policy remained.
- V8 migrated exact V6 M1-end, sealed `scene_core`, completed step12000, and held Geometry/extraction invariant while downstream Appearance/relation/topology objectives remained learnable.

Current default:

```text
G / M establish shared carrier H
        ↓
M1 seal
        ↓
A / R / T / J specialist learning
        ↓
scene_core remains sealed
```

Historical V8 already had this carrier seal and stage-level specialist isolation. Its physical optimizer groups were `scene_core`, combined `evidence_heads` (mechanical + appearance + relations), and `structure`. Post-V8 `optimization_v2.py` does not invent the seal; it separates future `mechanical_head`, `appearance_head`, `relation_head`, and `structure` groups so LR, gradients and updates are independently attributable while `scene_core` stays LR=0 fail-closed.

V8 remains a **measurement baseline**, not a scientific Fit8 PASS.

## Verified post-V8 source stack

The focused post-V8 contracts on `c22850ae...` are green via completion audit `39132578331` and full vNext suite `39132578290`. They cover:

- exact canonical post-M1 stage IDs and fail-closed carrier guard;
- Truth V2 authority/provenance/validity contract and validity-aware loss;
- Relation Truth V2 producer source;
- relation support/class-balance eligibility;
- `GraphicsRelationHeadV2` conditioning and deterministic V1→V2 migration semantics;
- head-separated post-V8 specialist optimizer/telemetry;
- self-hosted workflow safety contract;
- single-carrier Compiler/runtime contracts;
- native Graphics ordering + RSS→native behavioral E2E;
- TESSA/decoder regression protection;
- current/historical continuity pointers in the agent handoff.

No specialist LR defaults are silently invented. Historical V8 optimizer/checkpoint provenance remains unchanged.

## Relation truth / conditioning state

V8 relation-boundary audit found projected alpha/appearance targets derived from a render view while `GraphicsRelationHeadV1.static_relations(Ha,Hb)` predicted them without camera conditioning. This is a supervision/conditioning mismatch, not merely an LR problem.

Implemented replacement source:

- `models/iris/vnext/graphics_truth_v2.py` — explicit authority/provenance/validity;
- `models/iris/vnext/training_v2.py` — explicit validity-mask loss;
- `tools/iris_vnext/build_fit8_relation_truth_v2.py` — versioned Relation Truth V2 sidecar/manifest producer;
- `models/iris/vnext/relation_support_v2.py` — support/class-balance court eligibility;
- `models/iris/vnext/heads_v2.py` — projected boundaries moved to the camera-conditioned view branch;
- `models/iris/vnext/optimization_v2.py` — head-separated focused optimizer and convergence telemetry.

Still **not established**:

- no actual Fit8 Relation Truth V2 corpus/manifest is sealed as experiment authority;
- no real V8 checkpoint→RelationHeadV2 migrated checkpoint/receipt is sealed;
- no focused post-V8 relation/boundary optimization experiment has run;
- no new scientific relation PASS exists.

Static `same_drawable`, `cut`, and `attachment` abstain when independent authority is absent. Current deterministic valid `occlusion_probability` rows are single-class positive support; this may remain training/measurement evidence but is not eligible for a binary held-out PASS court until authoritative negative support exists. Do not fabricate negatives.

## Boundary and foundation evidence

`BoundaryCausalityTeacherV1` allows only source-defensible causes:

```text
GEOMETRIC_TERMINATION
MATERIAL_BOUNDARY
ALPHA_BOUNDARY
OCCLUSION_VISIBILITY_BOUNDARY
SOURCE_PRIMITIVE_COMPONENT_SEAM
RESIDUAL_UNKNOWN
```

`RESIDUAL_UNKNOWN` is abstention.

Foundation research provenance remains:

`agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`

Selective salvage is limited to source-defensible geometry/topology provenance, material/skin provenance, static all-hit rays, fixed-frame authored-motion evidence, residual-boundary diagnostics and local-continuation courts. Exact paths are recorded in `CURRENT_PLAN.md` and the decision ledger. Authoritative drawable ownership, hidden/amodal support, intended 2D presentation override, global named semantic ontology, and inferred hidden continuation are not generic FIT8 truth.

## Compiler/runtime consumption

Audit: `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`.

Current source evidence supports:

- same/cut/boundary can affect Compiler fragment clustering;
- front+occlusion can create directed order edges;
- native explicit precedence can override contradictory physical depth;
- visibility changes final native RGBA behavior.

Consumption debt remains:

- attachment is serialized into `graphics.bin` but not behaviorally used by the current native reference raster;
- `support_band_pixels` are serialized/validated but not applied by the reference renderer.

Serialization is not behavioral use.

## TESSA/topology state

V8 showed topology-objective learnability from sealed H but did not establish an independent topology PASS. A held-out court must measure actual surface/topology outcomes: valid/degenerate faces, component error, boundary loops/holes, meaningful manifold diagnostics, canonical-surface deviation, source support and held-out stability.

Training loss is not topology qualification.

## StdGEN++ re-audit

Pinned public code inspected:

- `hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766`
- `TencentARC/InstantMesh@08822c5200a4bcc0e2c1a4573409ac10b9f6692c`

The inspected public StdGEN repository does not expose the exact full training loop. Paper-only optimizer/curriculum details remain `PAPER_EXACT`, not `CODE_EXACT`.

Transferable discipline: protect a useful carrier, constrain specialization, preserve earned supervision, require special data for hidden/internal truth, and consider coarse-to-fine extraction/topology regularization only after a held-out court identifies the relevant failure. Generic LoRA/adapters remain unauthorized until a clean-target capacity court fails.

## Self-hosted stale-run incident and safety hardening

Stale run `39121291171` / `iris-vnext-fit8-prepare-self-hosted-v1 #22` was queued on PR #116 head `research/iris-vnext-single-carrier-graphics@90c9c2f35ffc77aeea10cd99391bbe9a4d4a31c6`. When `realsas-wsl-1660ti` came online it locally sealed stale-source preparation and entered persistence to:

`gdrive:RealSaS_Fit8_IRIS_vNext_20261016`

The run was cancelled during Drive persistence before round-trip SHA verification. Forensic values from that stale run:

```text
TEACHER_BATCH_RECEIPT sha256 = 7ac1c66cc5f749e0f6f8cac43a9b79a135c8c45ed1138303d42c102f04c15d27
COHORT manifest value sha256 = 37a5fccbdcb2d98735291c7963e75e852e260e4d4eecfc2ef9f60ff8dbcb732b
PREPARATION_RECEIPT sha256 = fdd72430a9e6573b51b98eddf181d159cca08079654ceed08de92a25ae840082
source_commit = 90c9c2f35ffc77aeea10cd99391bbe9a4d4a31c6
```

Forensic GitHub artifact: `iris-vnext-fit8-preparation-39121291171`, artifact id `12137271848`.

The affected Drive path remains **QUARANTINED / UNVERIFIED**. Do not consume it based on file presence.

Safety hardening is source-implemented and focused-CI green:

- legacy `.github/workflows/iris_vnext_fit8_prepare_self_hosted_v1.yml` is retired: manual dispatch only, `ubuntu-latest`, no self-hosted runner, no `rclone`, typed failure;
- new `.github/workflows/iris_vnext_fit8_quarantine_audit_self_hosted_v2.yml` is manual-dispatch only and read-only;
- it is bound to the active review branch and an exact operator-supplied SHA equal to checked-out `GITHUB_SHA`;
- it only copies remote→local and uploads forensic evidence;
- `tools/iris_vnext/audit_fit8_preparation_snapshot_v1.py` classifies the snapshot but **never authorizes remote consumption**; even a current-source match remains non-authoritative until regeneration/full round-trip qualification.

The self-hosted runner remains offline until we deliberately dispatch that read-only audit. No self-hosted preparation writer is currently authorized.

## Quality / PASS discipline

```text
MEASUREMENT_ONLY
OPTIMIZATION_DEBT
ELIGIBLE_FOR_COURT
COURT_PASS
```

`COURT_PASS` requires authoritative target + held-out metric + preregistered metric-specific threshold + required non-regression. Historical `~0.05` is only planning context for genuinely comparable normalized metrics; `0.1–0.2` is not auto-success.

## Immediate next execution tranche

1. require this continuity repair to pass focused CI;
2. fast-forward PR #116 to that exact green head without force;
3. require the broad PR gate set to be green on that exact head;
4. deliberately run the read-only self-hosted quarantine audit and inspect its forensic artifact;
5. only after that, design a SHA-scoped candidate preparation writer; do not write the quarantined shared remote automatically;
6. materialize Relation Truth V2 sidecars/manifest on exact admitted Fit8 inputs and audit per-family support/class/abstention;
7. seal a real V8 relation-head state → `GraphicsRelationHeadV2` migration checkpoint/receipt;
8. preregister and run focused relation/boundary optimization with `scene_core` frozen and head-specific telemetry;
9. build the independent TESSA topology court;
10. close remaining evidence→Compiler→runtime behavioral-consumption gaps.

No broad 12k/A100 rerun is authorized merely because source hardening is green.

## Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
4. `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`
5. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
6. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
7. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
8. `AGENTS.md`
9. `SYSTEM_INDEX.md`
