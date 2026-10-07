# RealSaS-OPT — Current State

**Date:** 2026-10-07  
**Canonical code line:** `main` only  
**Architecture:** RealSaS V2, 46-stage dependency DAG  
**Implementation readiness:** `canonical/V2_IMPLEMENTATION_READINESS.json`

## Current product architecture

Geometry, Mechanics and Appearance remain co-equal product authorities. `SurfaceAddressing` is the stable canonical mesh-domain/addressing authority and Complete Appearance Authority owns source-preserving visual appearance. Presentation remains first-class editable addressing authority.

Canonical public model naming is `IRIS -> TESSA -> AXIS -> MIRA`. Legacy `GEPPETTO` / `ARACHNE` / `ATLAS` identifiers may remain where required for schema, checkpoint, artifact or historical compatibility.

### Mechanics — latest developed FIT1 line

- Stage19 owns the exact frozen mechanical carrier `M`.
- AXIS V5.4.1 owns a hard causal required tree plus deterministic 256-bin XYZ; residual diffusion is not runtime position authority.
- Compiler validates/materializes AXIS required parents; it does not reselect the tree on this path.
- MIRA V5.5 predicts `W_M` directly on the exact Stage19 carrier basis.
- Stage32 qualifies carrier-native `W_M`; Stage34 derives a subject-free pre-bind provisional frame/envelope; Stage35 rederives and qualifies those frames against final `W_M` before proving the exact `(M,G,W_M)` carrier; Stage36 is identity re-key/seal only, not semantic skin transfer.
- Generic coincident-frame qualification is subject-free: zero-influence subtrees may inherit parent frame; active coincident subtrees fail closed and require explicit AXIS orientation/tail evidence.

Knight FIT1 mechanics are closed relative to the sealed source/teacher motion envelope. Absolute G3 is intentionally not promoted to a universal PASS because the sealed teacher itself lies outside the frozen absolute thresholds. No FIT8/FITK/unseen/product/generalization claim is made.

## Canonical mesh domain

`compiler/realsas_compiler_core/surface_addressing_v1.py` owns `SurfaceAddressing`; Stage18/19 own construction and static qualification of the mechanical carrier. The current mechanics chain preserves a single carrier basis across MIRA prediction, dynamic proof and runtime binding.

## Complete Appearance Authority

Appearance / CAA remains source-preserving and proof-bound under `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`. The mechanical mesh is not allowed to silently become a second appearance authority.

## Fresh canonical Knight regression render

Run `37578120472` executed on exact `main@287eabef757cb2b1e6b532a4c2d0a71eb62cd324` and completed PASS. It rebuilt sealed Stage23, requalified Stage24 and freshly rerendered IDLE/RUN/SLASH with exact frozen reference hashes.

This is artifact-level reproducibility. It reused sealed upstream artifacts; IRIS/TESSA/AXIS/MIRA were not reinferred and it is not a fresh source-to-puppet compile.

## Execution model

```text
Git/main            = code truth
Attempt             = research truth
ProductRevision     = production truth
Workflow Engine     = durable execution truth
Artifact Registry   = immutable typed artifact identity/dependencies
Proof/Qualification = promotion eligibility
```

Long-lived research/promotion/integration branches must not become product or scientific authority. Research Attempts may reuse exact qualified upstream artifacts by dependency identity. Product compile runs the required inference/Compiler construction path unless an exact semantic cache hit is proven. Runtime consumes qualified ProductRevision artifacts and must not silently re-solve mechanics or appearance.

## Platform enforcement gap

The Go control plane is the intended owner of durable product/research state, Artifact Registry metadata, Attempt/ProductRevision lifecycle, promotion, dependency scheduling and workflows. Legacy direct authority-root/GitHub Actions paths remain CI/reproducibility mechanisms and must not become a second authority.

## Performance baseline

Fresh regression telemetry remains the current optimization baseline:

- Stage19 static qualification: ~60.9 s
- Stage21 CAA compile: ~321.7 s
- Stage24 qualification: ~131.5 s
- renderer: ~463.8 s

These measured costs define optimization priorities after repository currentness is sealed.

## Immediate execution priorities

1. Close this carrier-native mechanics promotion onto canonical `main` with CI/governance green.
2. Keep Go platform enforcement as the next system blocker.
3. Optimize measured rebuild/CAA/render hot paths.
4. Return to presentation/runtime visual diagnosis only if the image remains wrong after 1–3.

## Read first

1. `canonical/CANONICAL_HANDOFF_20261007.md`
2. `CURRENT_STATE.md`
3. `AGENTS.md`
4. `SYSTEM_INDEX.md`
5. `canonical/V2_IMPLEMENTATION_READINESS.json`
6. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
7. `canonical/KNIGHT_AXIS541_MIRA55_CARRIER_NATIVE_PROMOTION_20261007.json`
8. `canonical/TRUTH_CORPUS_V1_STATUS_20261004.md`

Older recovery/runtime-audit documents remain provenance only and must not override this continuation authority.
