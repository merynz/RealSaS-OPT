# RealSaS-OPT — Current State

**Date:** 2026-09-13  
**Canonical continuation branch for repair:** `repair/mage-full-subject-reclosure-20260912`  
**Status:** `MAGE_FIT2_GEPPETTO_TERMINAL_PASS__MESH_BASELINE_PRESERVING_PATCH_EXPERIMENT_ACTIVE__ARACHNE_AUTHORIZED_NOT_STARTED__PRODUCT_PASS_OPEN`

This file is the continuation authority inside the active repair branch. Repository-wide/main reconciliation remains a separate transaction and must not be inferred from this branch.

**Read first for exact chat continuation:** `canonical/DEHYDRATED_CONTEXT_20260913.md`  
**Primary reopening context:** `canonical/MAGE_FIT2_REAL_E2E_REOPENING_CONTEXT_20260912.md`  
**Pipeline refit authority:** `canonical/MAGE_FIT2_PIPELINE_REFIT_AUTHORITY_V1.json`  
**Mesh product reclosure audit:** `canonical/MAGE_FIT2_MESH_PRODUCT_RECLOSURE_AUDIT_20260912.md`  
**Current mesh prereg:** `canonical/FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_PREREG_20260913.md`  
**Historical FIT1 evidence:** preserved as historical/scoped evidence only where bound to superseded old-S lineage

## One-line state

`Corrected H1 and GSA8192 are closed. Fresh Geppetto FIT2 has now reached a verified 48/48 terminal PASS on corrected S with persisted 8-view evidence and an Arachne handoff. In parallel, mesh reclosure has moved from failed global uniform/adaptive families to a baseline-preserving local adaptive strategy: P1 already lifts all eight views to ~97–99% recall at 100% precision, but inherited baseline shape-quality outliers remain and P2/P3 are still running. Fresh Arachne is authorized by the Geppetto handoff and may proceed independently of the mesh run. PRODUCT_PASS remains open.`

## Corrected upstream authority — CLOSED

### H1 observable product surface

- source run: `20260912T074348Z`
- H1 checkpoint SHA-256: `76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5`
- product-clipped zero-surface SHA-256: `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`
- minimum V0..V7 alpha recall: `0.9511473445`
- minimum precision: `0.9806321480`
- minimum IoU: `0.9356283394`
- product inference input remains `8x RGBA + exact 8 orthographic cameras`
- teacher mesh at product inference: `false`

### GSA8192

- nodes: `8171`
- relations: `23656`
- observed/completed: `7391 / 780`
- geometry lineage hash: `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da`
- tensorization hash: `fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e`
- observation authority: `EXACT_PRODUCT_ALPHA_AND_SELF_VISIBILITY`

No old 950-node Mage surface lineage is current product authority.

## Geppetto FIT2 — VERIFIED TERMINAL PASS

Fresh Geppetto was trained from scratch on corrected GSA8192 under the frozen refit contract.

Exact result:

`REALSAS_MAGE_FULL_SUBJECT_RECLOSURE_20260912/MAGE_FIT2_PIPELINE_REFIT/GEPPETTO_REFIT/MAIN/GEPPETTO_FIT2_CORRECTED_SUBSTRATE_RESULT.json`

Verified closure:

- result status: `FIT2_GEPPETTO_TERMINAL_PASS`
- closure step: `12160`
- terminal streak: `48/48`
- final seeds: `[11,23,47,89]`
- every final seed: `22` joints, PASS, root accuracy `1.0`, parent accuracy `1.0`, unsupported joints `0`, illegal parents `0`, deform roots `1`
- teacher feedback during free eval: `false`
- historical Geppetto checkpoint loaded: `false`
- thresholds changed: `false`
- source/result repo head: `135e6def2ba7112b8471586115d11bf1b936677d`

Persisted evidence:

- checkpoint SHA-256: `2e1f35d196af6eea957bb6377bc0b0e1b045c20dc652d798b74b107198ae80e9`
- result SHA-256: `91a7d4cb45f0402a0980a59a548c0555137fddd288f508500b8fdee6ab99afb5`
- run-log SHA-256: `14377a0e4302e24fc000f18989c0311f717e3c46db3e90d70bcfbd66255694e6`
- 8-view skeleton contact-sheet SHA-256: `865b215d88bc13e730adfcc7377755c4e3a25aec4d6f89af849e37ef9f453eaa`
- evidence manifest status: `PASS`

`ARACHNE_FIT2_INPUT_HANDOFF.json` is persisted and states `AUTHORIZED_BY_GEPPETTO_FIT2_TERMINAL_PASS` with `arachne_refit_authorized=true`, matching corrected GSA/tensorization and exact Geppetto checkpoint hash.

Claim boundary: this closes the **Geppetto FIT2 stage**. It does not claim `PRODUCT_PASS`; the Geppetto result itself keeps `product_pass_claimed=false` and `promotion_authorized=false`.

## Mesh reclosure — ACTIVE BASELINE-PRESERVING PATCH EXPERIMENT

The mesh/product qualification contract remains implementation-level CLOSED PASS. The corrected real Mage mesh result is still under active scientific reclosure.

Previous global family outcomes:

- uniform legal-Steiner `n={2,3,4}`: falsified as final global product family;
- global adaptive boundary/quality Delaunay `B4_G16 -> B2_G12 -> B1_G8`: falsified as the sole global final solver because admitted triangles were healthy but whole-character coverage fell to about the 70% band;
- legal support/adaptive triangulation survives as a **local** repair/recovery operator.

Current preregistered experiment:

`BASELINE-PRESERVING ADAPTIVE PATCH CDT`

Source/preflight:

- runner: `experiments/mage_full_subject_reclosure_v1/run_fit2_baseline_preserving_adaptive_patch_cdt_v1.py`
- branch head/source preflight at experiment start: `c69ab78eac42ec89d2bed650fd9aa607d6f44a00`
- self-hosted preflight: run `34752599677` — PASS
- frozen treatment order: `P1_B2_G10 -> P2_B1_G8 -> P2_B1_G6`

Architecture:

> sealed historical CDT = coverage authority  
> adaptive support-derived CDT = local quality/recovery operator

The baseline is **not** assumed universally correct. Every baseline face is scanned against frozen raster shape-quality gates. Quality-fail faces become repair cavities; healthy faces are preserved; supported residual parents are recovered locally. Current experiment treats faces that already pass the frozen policy as sufficient/KEEP, not globally optimal.

### P1_B2_G10 observed partial result

| View | baseline recall | final recall | precision | largest hole | P1 |
|---|---:|---:|---:|---:|---|
| V0 | 92.589% | 97.230% | 100.000% | 0.997% | quality FAIL |
| V1 | 94.061% | 98.948% | 100.000% | 0.322% | PASS |
| V2 | 90.557% | 97.679% | 100.000% | 1.373% | quality FAIL |
| V3 | 94.130% | 98.339% | 100.000% | 0.645% | quality FAIL |
| V4 | 92.850% | 97.343% | 100.000% | 1.001% | quality FAIL |
| V5 | 93.799% | 98.863% | 100.000% | 0.395% | PASS |
| V6 | 90.758% | 97.502% | 100.000% | 1.374% | PASS |
| V7 | 94.366% | 98.230% | 100.000% | 0.625% | quality FAIL |

P1 validates the asymmetric hybrid for coverage. Precision is 100% in all views and holes are already below the frozen 1.5% limit. Remaining failures are inherited shape-quality outliers. P1 quality repair accepted `0` cavities, leaving `18` reported bad faces across V0/V2/V3/V4/V7.

`P2_B1_G8` / `P2_B1_G6`: **pending/running at this state snapshot**. Do not infer outcome.

If the current full frozen policy is later judged insufficient for deformation quality, first measure the full triangle-quality distribution and actual deformation strain/flip behavior, then preregister a separate quality-uplift experiment. Do not post-hoc reinterpret this run or assume every policy-passing baseline triangle is optimal.

## UI skeleton overflow — root cause closed at adapter level

The previously observed UI skeleton displacement was not evidence that Geppetto generated an invalid rig.

Old UI adapter behavior used support-centroid / nearest-surface raster anchors instead of the qualified directional joint pivot. This can displace the same correct skeleton by tens to >100 pixels. Correct static rendering must consume qualified directional joint binding; if unavailable, rig overlay is withheld. Animated rig overlay likewise requires qualified per-frame joint/control state or must be withheld.

## Current mandatory execution order

1. corrected H1 observable product surface — **DONE**
2. GSA8192 canonical lineage + evidence — **DONE**
3. fresh Geppetto FIT2 — **TERMINAL PASS VERIFIED**
4. Geppetto real V0..V7 skeleton evidence + hash manifest — **VERIFIED**
5. fresh Arachne FIT2 on corrected S + Geppetto handoff — **AUTHORIZED / NOT YET CLOSED**
6. QualifiedSkinIR + exact W proof — **PENDING**
7. baseline-preserving directional mesh experiment — **ACTIVE; P1 observed, P2/P3 pending**
8. strict final QualifiedEditableMeshIR + exact mesh-skin transfer — **PENDING**
9. qualified components + mechanical/dynamic deformation proof — **PENDING**
10. professional motion closure — **OPEN**
11. exact runtime/export identity reclosure — **OPEN**
12. only then consider `PRODUCT_PASS`
13. unseen/FIT8/LOFO remains after same-Mage FIT2 product reclosure

Arachne no longer needs to wait for the mesh experiment to finish; its authorization dependency was Geppetto stage closure, which is now verified. Final mesh-skin/component product closure still waits for both W and the final mesh decision.

## Claim boundary

Strongest current claim:

`Corrected H1/GSA8192 are closed; fresh Geppetto FIT2 has reached a verified terminal 48/48 stage PASS with complete persisted 8-view evidence and an authorized Arachne handoff. Mesh science has converged on a baseline-preserving local-adaptive architecture; P1 already demonstrates ~97–99% recall at 100% precision but the preregistered P2/P3 run is not yet final. Arachne, final mesh/skin mechanics, professional motion and exact runtime reclosure remain open. PRODUCT_PASS is not claimed.`

Not claimed: Arachne FIT2 PASS; final corrected FIT2 mesh PASS; final mesh-skin/component mechanics PASS; professional motion quality; exact runtime product closure; unseen-family generalization; FIT8/LOFO; `PRODUCT_PASS`; commercial production readiness.
