# IRIS-vNext Rendered / Mechanical Census — Phase 2

Date: 2026-10-15

Status: **RESEARCH EVIDENCE / NO FIT8 PASS / NO PRODUCT PASS**

This note separates three authority levels deliberately. Numbers from one level must not be silently promoted into another.

## 1. Exact-repository static FBX execution

The current static ray-census family exposes per-view all-hit stacks and complete directed geometric-component order graphs. The FBX path uses pinned ufbx import/proposal semantics and RealSaS polygon-triangulation qualification before static census admission.

At 128x128 canonical views, exact-repository execution already falsifies a universal `one geometric component -> one painter slot -> one per-view total order` assumption:

- Snake: 3 geometric components; mixed-order evidence in 7/8 canonical views; maximum distinct component stack 3.
- T-Rex: 1 component; component-order court is trivial and cannot support or refute a multi-component mechanism.
- Spider: 15 components; mixed-order evidence in 7/8 views; maximum distinct component stack 6.
- Shark: 1 component; component-order court is trivial.

Snake and Spider remain nontrivial under the 64 -> 128 -> 256 diagnostic resolution sweep. Therefore the failure is not explained by the 128 raster alone.

**Decision consequence:** a generic presentation system may not use one painter slot per whole geometric component as its sole visibility/order primitive.

This does **not** imply that a fragment DAG should replace geometric depth everywhere. Ordinary geometric visibility remains a separate mechanism.

## 2. Exact-selected-source glTF local replay — diagnostic only

The four glTF bytes available in the research workspace were checked against `canonical/FIT8_DIVERSITY_SELECTION_V1.json` source SHA-256 identities before local replay. The replay uses the same source-accessor/node-transform and render-to-geometric equivalence semantics as the repository producers, but is not persisted as an exact repository execution artifact here.

### Open-surface visible importance

Canonical-view front-visible surface-class fractions were swept at 64, 128 and 256 resolution.

Bat non-closed front-visible fraction:

- 64: OPEN ~15.43%, NONMANIFOLD ~11.66%, total non-closed ~27.09%.
- 128: OPEN ~15.47%, NONMANIFOLD ~11.80%, total non-closed ~27.27%.
- 256: OPEN ~15.60%, NONMANIFOLD ~11.72%, total non-closed ~27.32%.

George OPEN front-visible fraction:

- 64: ~12.40%.
- 128: ~11.79%.
- 256: ~11.77%.

The 128 -> 256 changes are small. The observed open/non-manifold support is therefore materially visible and is not plausibly dismissed as a low-resolution raster artifact.

**Decision consequence:** open surfaces remain first-class corpus topology. A universal closed-surface / globally signed SDF ontology is rejected for FIT8.

### Diagnostic static ordering

Local exact-selected-source replay also shows nontrivial component-order conflicts for Bat, GreenBlob and George, while Horse has one qualified geometric component and is therefore component-order-trivial. These rows remain **diagnostic** until the same source bytes are persisted through the exact repository producer execution path.

## 3. Dynamic authored-motion precheck — diagnostic only

The glTF subjects use LINEAR authored animation sampling over translation/rotation/scale channels in the selected sources inspected so far. A local precheck sampled all native clips at preregistered interior normalized times.

The static render-to-geometric quotient remained coincident under the sampled authored motion contexts in Bat, Horse, GreenBlob and George (`worst separation = 0` in that precheck).

Mechanical-tail observations identify Horse as a strong smooth-deformation stress subject, while George contains localized area-ratio outliers despite calmer p99 conditioning. Raw extrema remain diagnostic; tail metrics are the intended court.

Presentation precheck found strong pose-conditioned order changes. George, especially `Death` near normalized time 0.875, exhibits many new and reversed directed component relations relative to rest. Bat and GreenBlob also exhibit pose-dependent cyclic/mixed component ordering.

These dynamic numbers are **not qualified evidence yet**. Their purpose is to choose stress contexts and validate the deterministic producer design.

## 4. Dynamic camera-contract correction

An early integrated dynamic producer reused the static ray producer directly per pose. Because the static producer solves its canonical frame from the current input vertices, that design would silently recenter/rescale each pose.

That is rejected.

The corrected dynamic contract is:

```text
rest geometry + all preregistered authored pose samples
        -> solve ONE subject-wide dynamic canonical frame
        -> freeze center / half-extent / eight yaw cameras
        -> raster rest and every sampled pose in that same frame
        -> compare ordered-visibility graphs
```

Per-pose recentering or rescaling is forbidden. The v1 integrated producer is historical diagnostic code; the current path is the fixed-frame v2 producer.

## 5. Current presentation interpretation

The static and dynamic evidence rejects two overly simple interpretations:

1. one whole geometric component as one painter slot;
2. a static component order that is computed once and carried unchanged through pose.

A further local face-level diagnostic shows that even one mesh triangle can be in front of and behind another component in different image regions. Therefore `one mesh face -> one painter slot` is not a universal representation of exact geometric visibility either.

The currently preferred decomposition is consequently:

```text
ordinary geometric visibility
    = exact posed geometry + depth/ray visibility

presentation override
    = sparse, qualified, view/pose-conditioned evidence only where authored 2D presentation
      is not explained by ordinary geometric visibility
```

This is a candidate architecture direction, **not a frozen CAO override primitive**. FIT8 3D source geometry can falsify insufficient depth/order abstractions, but cannot by itself prove the intentional 2D presentation override required by artist-authored sheets such as Knight. That final override court needs source-art / decomposition ownership truth.

## 6. Open leaves after Phase 2

Still open:

- semantic owner / material query capacity: diagnostic dominant-joint counts are not semantic-owner authority;
- one universal sparse-support pitch: sparse high-resolution support is strongly motivated, but no single pitch is frozen;
- exact CAO presentation override primitive and granularity;
- qualified 8/8 dynamic census, especially the four FBX subjects;
- exact-repository persisted glTF static/dynamic execution artifacts;
- full rendered/decomposition/amodal truth bundle.

Already measurement-backed:

- open surfaces are materially visible and first-class;
- whole-component painter slots are insufficient generically;
- presentation ordering changes with authored pose;
- per-pose camera recenter/rescale is forbidden for dynamic graph comparison;
- ordinary z/depth visibility and intentional presentation override must remain separate courts.
