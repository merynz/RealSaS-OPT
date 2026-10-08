# RealSaS Structural System Index

## 2026-10-07 canonical main

## Current continuation details

Current continuation authority is `canonical/CANONICAL_HANDOFF_20261007.md` plus `CURRENT_STATE.md`.

The active engineering program is **repo currentness -> Go platform enforcement -> performance optimization**. Runtime/presentation image diagnosis is intentionally deferred until those close.

> Navigation only. Historical recovery/runtime-audit documents are provenance and must not override current continuation authority.

| Product authority | Canonical home | Current rule |
|---|---|---|
| Observation | observation/camera authority | exact source evidence; source cardinality is separate from output-direction authority |
| Geometry / IRIS | `models/iris/` + geometry adapters | signed field/surface evidence; never RGB authority |
| Surface evidence / GSA | `compiler/realsas_compiler_core/substrate/` | compact mechanical/evidence substrate with provenance |
| Learned topology/geometry / TESSA | `models/tessa/` + Stage18 TESSA proposal bridge | learned proposal only; Compiler support-bind/repair/qualification required |
| Canonical mechanical carrier | Stage18/19 + `surface_addressing_v1.py` | Compiler-owned repaired/qualified static carrier; Stage19 is exact carrier authority |
| Rig / AXIS | `models/axis/` + learned mechanics adapters | AXIS V5.4.1 owns hard causal required parents + deterministic discrete XYZ; Compiler validates/materializes, no tree reselection on this path |
| Skin / MIRA | `models/mira/` + Stage32 carrier-native qualification | MIRA V5.5 predicts `W_M` directly on the exact Stage19 carrier basis |
| Appearance / CAA | `appearance_authority_v2.py` + appearance compile/bake/quality modules | total source-preserving art; source wins; proof-bound |
| Presentation | Stage37 qualified source-owned presentation | role-free editable presentation authority; no invented categorical identity |
| Motion | `motion_compile_v2.py` + proof | sealed mechanics; exact authored motion semantics where admitted |
| Runtime | `runtime_authority_v2.py` + C++ runtime/native renderer | package/playback/render hot path; no generative/corrective inference during ordinary render |
| Orchestration | Go platform + Python Engine activities | Platform owns durable time/state; Engine owns scientific meaning |
| Research state | Go `Attempt` | immutable research execution truth, not branch chronology |
| Production state | Go `ProductRevision` | qualified production truth; promotion is platform state, not Git merge semantics |
| Artifacts | Artifact Registry/store | immutable typed bytes + semantic dependencies |
| Closure | proof/qualification + product promotion | promotion eligibility must be explicit and fail-closed |

## Canonical code-line policy

`main` is the single canonical continuation line. Long-lived research/promotion/integration branches are not authority. A temporary review branch may exist briefly for code review, but research/product separation belongs to Platform state (`Attempt`, `ProductRevision`), not Git topology.

## Current mechanics boundary

The canonical mechanics chain is carrier-native:

```text
Stage19 M
  -> AXIS V5.4.1 G
  -> MIRA V5.5 W_M on the same M
  -> Compiler post-bind qualification
  -> Stage35 exact carrier mechanics
  -> Stage36 identity re-key/seal
```

No semantic skin transfer to another mechanical basis is canonical. Generic coincident-frame handling is post-bind and subject-free: a subtree with zero final mechanical influence may inherit its parent frame; an active coincident subtree fails closed and requires explicit AXIS orientation/tail evidence.

Knight FIT1 mechanics are closed relative to the sealed teacher/source motion envelope. Absolute universal G3, FIT8/FITK, unseen and product-generalization remain unclaimed.

### Canonical mesh domain

`compiler/realsas_compiler_core/surface_addressing_v1.py` provides SurfaceAddressing and the stable mechanical carrier addressing domain. Stage18/19 construct and statically qualify that domain; the mechanics path must preserve the same exact carrier basis through MIRA prediction and dynamic proof.

### Appearance / CAA

`compiler/realsas_compiler_core/appearance_authority_v2.py` owns complete appearance qualification. `compiler/realsas_compiler_core/runtime_authority_v2.py` governs qualified runtime consumption. Appearance remains source-preserving and may not silently become mechanical correction.

The 46-stage **dependency DAG** is the executable dependency authority.

## Fresh render baseline

`recover-and-verify-knight-frozen-canonical` run `37578120472` on `main@287eabef757cb2b1e6b532a4c2d0a71eb62cd324` rebuilt Stage23, requalified Stage24 and freshly rerendered IDLE/RUN/SLASH with exact reference hashes. Models were not reinferred. This proves regression reproducibility of the sealed artifact consumer path, not source-to-product compilation.

## Platform

The accepted implementation boundary remains:

```text
Go Platform  = durable product/research state, Artifact Registry metadata,
               Attempt/ProductRevision lifecycle, promotion, scheduling/workflows
Python Engine = compiler semantics, model inference/training, geometry/ML,
                proof logic and typed stage failures
C++ Runtime   = playback, deformation/render hot paths, package consumption
```

The current defect is **enforcement**, not absence: legacy workflows can still bypass the Go control plane via direct authority-root scripts. Closing that bypass is the next active platform program.

## Performance

The fresh regression run measured the dominant costs directly: renderer ~463.8 s, Stage21 CAA compile ~321.7 s, Stage24 qualification ~131.5 s. Optimization must attack measured dependency/rebuild and hot-path costs before additional visual-runtime research.

## Machine-readable ownership

Architecture ownership remains queryable from `platform/internal/architecture/registry.go` and `tools/realsas_architecture.py`. Every canonical stage must resolve to one Engine owner; Platform separately owns durable state/orchestration and Runtime owns hot playback/render paths.

## Platform execution entrypoints — 2026-10-08

- `docs/platform/EXECUTION_LANES.md`: development/product operations and limits.
- `platform/cmd/realsasctl`: Go operator commands.
- `tools/platform_release_snapshot.py`: source-only pinned stage/version request.
- `compiler/realsas_compiler_services/platform_worker/stage_inputs.py`: immutable CAS stage hydration.
- `canonical/PLATFORM_MAIN_AUDIT_20261008.json`: main and current evidence audit.
