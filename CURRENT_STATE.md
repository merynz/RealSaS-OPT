# RealSaS-OPT — Current State

**Date:** 2026-10-07  
**Canonical code line:** `main` only  
**Active program:** `REPO_CURRENTNESS__PLATFORM_ENFORCEMENT__PERFORMANCE_OPTIMIZATION`  
**Runtime visual diagnosis:** deferred until the active program closes

## Current canonical state

The recovered TESSA/MIRA mechanical-survivor promotion is now on canonical `main`.

The exact pre-program baseline was `main@287eabef757cb2b1e6b532a4c2d0a71eb62cd324` (merge PR #56). That baseline established:

- TESSA T1B as a learned topology/reconstruction proposal survivor, not final product authority;
- Compiler-owned static repair before Stage19;
- Knight static mechanical carrier survivor at 3653 vertices / 6928 faces, 0 frozen-policy violations, min angle 7.524628286°, max aspect 9.089329469°, with no threshold relaxation or Knight-specific exemption;
- Stage19 as the sole static carrier authority for the TESSA lane;
- MIRA teacher23 falsified and forbidden as promoted authority;
- MIRA RAW41 preserved as FIT1 research evidence only;
- TESSA Stage35 product mint fail-closed until a subject-free product-qualified carrier-field transport exists;
- actual IDLE/RUN/SLASH dynamic certification remains unclosed; no TESSA dynamic product PASS is claimed.

Canonical public model naming is `IRIS -> TESSA -> AXIS -> MIRA`. Legacy `GEPPETTO` / `ARACHNE` / `ATLAS` identifiers may remain where required for schema, checkpoint, artifact or historical compatibility; they are not the preferred public architecture names.

## Fresh canonical Knight regression render

Run `37578120472` executed on exact `main@287eabef757cb2b1e6b532a4c2d0a71eb62cd324` and completed PASS.

It rebuilt the sealed Stage23 mechanical-CAA state, requalified Stage24, reran the renderer, and matched the frozen GIF references exactly:

- IDLE `4c64437ba1f2ecaca8302a52c598f3ba7750a923fd50b90f7cafa3f8473386c3`
- RUN `64ad50eeb9ea55ce202ddc46728ffea9074419dbf855938840f69d06661806e3`
- SLASH `adc8e8cd0e2272a836e50f7a73585208074703ea5f3c9a549583d7a659fa2b85`

Artifact: `knight-frozen-canonical-recovered`, workflow artifact id `11464461500`.

This is an artifact-level reproducibility/regression render. It does **not** rerun IRIS/TESSA/AXIS/MIRA inference and it does **not** prove a fresh source-to-puppet product compile.

## Execution model

The intended system separation is now normative:

```text
Git/main          = code truth
Attempt           = research truth
ProductRevision   = production truth
Workflow Engine   = durable execution truth
Artifact Registry = immutable typed artifact identity/dependencies
Proof/Qualification = promotion eligibility
```

Long-lived research/promotion/integration branches must not be used as authority or product state. Temporary review branches are allowed only as short-lived source-code collaboration and must not become a competing continuation line.

Research execution should be dependency-aware and reuse exact qualified upstream artifacts. Product compile must run all inference/Compiler work required to construct a qualified ProductRevision unless an exact semantic cache hit is proven. Product render/playback consumes an already qualified ProductRevision and must not rerun model inference merely to render it.

## Platform enforcement gap

The Go control plane is already the intended owner of durable product/research state, Artifact Registry metadata, Attempt/ProductRevision lifecycle, promotion, dependency scheduling and durable workflows. However legacy GitHub Actions and direct `$REALSAS_AUTHORITY_ROOT` scripts still allow stateful work to bypass the platform.

This is the next platform blocker: the architecture exists, but normal execution is not yet forced through it.

Until platform-enforcement closure is complete, direct legacy workflows are permitted only for CI/reproducibility of already sealed evidence. They must not mint new normal product/research authority.

## Performance baseline

The fresh regression run exposed unacceptable latency even though it reused sealed upstream artifacts. Approximate measured work:

- Stage19 static qualification: 60.9 s
- Stage20 CAA preregistration: 43.8 s
- Stage21 CAA compile: 321.7 s
  - directional donor/local completion: 177.0 s
  - source projection/lock: 63.9 s
  - NPZ sealing: 33.7 s
- Stage22 seal: 3.1 s
- Stage23 bake: 32.9 s
- Stage24 qualification: 131.5 s
- renderer: 463.8 s

These measured costs, not speculative micro-optimizations, define the first optimization targets.

## Immediate execution priorities

Close in this order:

1. **Repository currentness closure** — all current entry/navigation/state documents must point to the same 2026-10-07 authority and priority order; historical files remain provenance only.
2. **Go platform enforcement** — make stateful research/product execution flow through Artifact/Attempt/ProductRevision/workflow authority instead of optional direct legacy paths.
3. **Performance optimization** — dependency-aware reuse, removal of redundant rebuild/qualification, native/batch hot paths and measured elimination of the Stage21/Stage24/render bottlenecks.
4. **Runtime/presentation visual diagnosis** — only after 1–3. If the Knight image still looks wrong, investigate presentation/runtime ownership then. Do not reopen already sealed static topology/geometry mechanically without new counter-evidence.

## Read first

1. `canonical/CANONICAL_HANDOFF_20261007.md`
2. `CURRENT_STATE.md`
3. `AGENTS.md`
4. `SYSTEM_INDEX.md`
5. `canonical/TESSA_MIRA_MECHANICAL_SURVIVOR_PROMOTION_LEDGER_V1_20261006.json`
6. `canonical/TESSA_MIRA_FINAL_DAG_PROMOTION_AUDIT_V1_20261007.json`
7. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
8. `docs/platform/adr/0002-compiler-platform-boundary.md`
9. `docs/platform/adr/0003-go-control-plane.md`
10. `canonical/TRUTH_CORPUS_V1_STATUS_20261004.md`

Older recovery/runtime-audit handoffs remain historical evidence and must not override this continuation authority.
