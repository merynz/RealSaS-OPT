# RealSaS-OPT — Current State

**Date:** 2026-09-12  
**Canonical continuation branch for repair:** `repair/mage-full-subject-reclosure-20260912`  
**Status:** `MAGE_FIT1_REAL_E2E_REOPENED__FIT2_FRESH_PIPELINE_REFIT_ACTIVE__GEPPETTO_RUNNING__MESH_COMPONENT_HARDENED__MOTION_RUNTIME_RECLOSURE_PENDING__PRODUCT_PASS_OPEN`

This file is the continuation authority inside the active repair branch. Repository-wide continuation authority is reconciled from `main/CURRENT_STATE.md`; this branch contains the executable repair/refit lineage.

**Primary current context:** `canonical/MAGE_FIT2_REAL_E2E_REOPENING_CONTEXT_20260912.md`  
**Pipeline refit authority:** `canonical/MAGE_FIT2_PIPELINE_REFIT_AUTHORITY_V1.json`  
**Reopening contract:** `canonical/MAGE_FULL_SUBJECT_REOPENING_AND_ATTACHMENT_CONTRACT_20260912.md`  
**Mesh/component prereg:** `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`  
**Historical FIT1 evidence:** preserved, scoped historical evidence only where bound to the superseded old-S lineage

**Current scientific module:** `Mage FIT2 fresh pipeline refit / Geppetto`  
**Active experiment gate:** `MAGE_FIT2_PIPELINE_REFIT`  
**Active branch:** `repair/mage-full-subject-reclosure-20260912`  
**Current substage:** `GEPPETTO_FRESH_REFIT`  
**Most recent closed gate:** `GSA8192_REAL_8VIEW_EVIDENCE`  
**Unseen-family generalization:** `BLOCKED UNTIL SAME-MAGE FIT2 RECLOSES`  
**Product PASS:** `NOT CLAIMED`

## One-line state

`The controlled Mage FIT1 rigging-core experiments closed locally, but the first real eight-view end-to-end product run produced visibly poor output and exposed upstream subject-scope loss, insufficient render/deformation mesh coverage, runtime mesh-identity drift, weak product proof scope and non-professional rotation-only motion. The old Mage product chain is therefore reopened. Corrected H1 observable authority and GSA8192 are sealed; a fresh-from-scratch Geppetto FIT2 refit is now running on corrected S, followed by fresh Arachne, exact directional mesh/skin/component closure, professional motion and exact runtime reclosure.`

## Why FIT1 was reopened

The real product stack was finally exercised rather than inferred from local scientific gates. That output was not remotely sufficient as an Automatic-Spine product witness.

The failure was not one bug:

1. **Upstream subject-scope loss.** Historical H1 geometry did not cover the same complete subject present in the eight product RGBA observations.
2. **Mesh coverage/authority weakness.** The old conservative MWB2 relation-complex covered only about `18.18%..32.31%` of source alpha. A separate exported-runtime bundle forensic audit showed an alpha-clipped barycentric demo mesh with roughly `64%..72%` source-alpha recall, proving a second mesh identity/coverage drift between locally reasoned mechanics and rendered runtime.
3. **Motion quality gap.** Current product motion is a rotation-only mechanical probe, not a professional idle/run animation system.
4. **Proof gap.** Local legality/finite-deformation PASS did not prove the exact exported puppet preserved the artwork or moved credibly.
5. **Runtime equivalence gap.** Product qualification did not yet enforce that exact qualified skeleton + skin + mesh identity survived unchanged into the runtime package.

The full causal snapshot is frozen in `canonical/MAGE_FIT2_REAL_E2E_REOPENING_CONTEXT_20260912.md`.

## Corrected upstream authority — CLOSED

### H1 observable product surface

Current H1 product authority is observation-bound rather than global-watertight-hidden-teacher authority:

- run id: `20260912T074348Z`;
- H1 checkpoint SHA-256: `76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5`;
- product-clipped zero-surface SHA-256: `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`;
- minimum V0..V7 alpha recall: `0.9511473445`;
- minimum precision: `0.9806321480`;
- minimum IoU: `0.9356283394`;
- product inference still uses only `8x RGBA + exact 8 orthographic cameras`;
- teacher mesh at product inference: `false`.

Historical global-sign/full-hidden-teacher residuals remain preserved diagnostics and are not rewritten into a PASS.

### GSA8192

Corrected H1 deterministically emits the current downstream substrate:

- nodes: `8171`;
- relations: `23656`;
- observed nodes: `7391`;
- completed nodes: `780`;
- geometry lineage hash: `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da`;
- tensorization hash: `fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e`;
- observation authority: `EXACT_PRODUCT_ALPHA_AND_SELF_VISIBILITY`;
- deterministic replay: PASS;
- real V0..V7 GSA evidence seal: PASS.

No old 950-node Mage surface lineage is current product authority.

## Active stage — fresh Geppetto FIT2

The active authority is `MAGE_FIT2_PIPELINE_REFIT_AUTHORITY_V1`.

Geppetto is being fitted **fresh from scratch** on the corrected GSA8192 substrate:

- runner: `experiments/mage_full_subject_reclosure_v1/run_geppetto_fit2_corrected_substrate_refit_v1.py`;
- mode: `FRESH_FROM_SCRATCH__NO_HISTORICAL_GEPPETTO_CHECKPOINT`;
- historical checkpoint SHA-256 `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30` is preserved but loading it is forbidden;
- target joints: `22`;
- max steps: `16384`;
- check every: `64`;
- terminal closure: `48/48` consecutive checks;
- evaluation seeds: `[11, 23, 47, 89]`;
- thresholds changed: `false`;
- CPU real-input preflight: PASS;
- actual optimizer run: active external GPU run per operator context;
- Geppetto FIT2 PASS claimed: `false` until sealed result/evidence exists.

Arachne may not start before Geppetto terminal PASS and its mandatory real V0..V7 skeleton evidence.

## Mesh / component hardening — IMPLEMENTED, NOT YET A FIT2 MESH RESULT

While Geppetto runs, the product closure was hardened independently:

- `9dd533eb5c48408faa92bcfb3469b7d32fd39368` — `Harden FIT2 mesh coverage and component closure`;
- `ebe7358dbdafbf65068dde91f3eba39b58b784b6` — `Run FIT2 mesh and component closure tests`.

The extended self-hosted `mage-full-subject-reclosure-contract` run at `ebe7358...` completed successfully on `realsas-wsl-1660ti`.

Current frozen product-mesh admission policy includes:

- source-alpha recall `>= 0.94`;
- precision `>= 0.995`;
- alpha IoU `>= 0.935`;
- largest uncovered connected region `<= 0.015` of foreground;
- each large alpha component recall `>= 0.90`;
- no degenerate/duplicate/non-manifold triangles;
- raster minimum angle / maximum aspect ratio gates;
- real V0..V7 observation + exact mesh overlay + uncovered heatmap evidence.

Component qualification now checks exact S/G/W lineage and mechanically verifies rigid-skinned ownership instead of trusting labels. Visible surface may not silently disappear from component accounting.

These implementation/contract gates are green. **No corrected FIT2 mesh product PASS exists yet** because fresh Geppetto/Arachne mechanics are not yet available.

## Supported Steiner vertices

`LOCAL_CONVEX_INTERPOLATION` is now a legal qualified mesh-vertex route.

For an inserted vertex:

`P(v) = Σ a_i P(S_i)` and `W(v) = Σ a_i W(S_i)`, with `a_i >= 0` and `Σ a_i = 1`.

Thus geometry placement and mechanical skin transfer share the same admitted surface-support authority. The typed path and regression test exist. Automatic arbitrary CDT quality-Steiner generation is still blocked until each inserted point can construct an exact legal local support simplex.

The desired product target is a **directional alpha-domain watertight deformation mesh**, not a hallucinated watertight 3D body: opaque artist pixels must belong to the view-local triangulated deformation domain; true transparent holes stay holes.

## Motion / runtime status

Motion remains an open first-class product problem.

The current lane is `ROTATION_ONLY_CURRENT_PRESET_V1`; it is retained as a mechanical deformation probe only. It is not evidence of professional idle/run quality.

Preferred future architecture under consideration:

`intent / artist clip / constraints -> MotionProposalIR -> deterministic Motion Compiler retarget + constraints -> QualifiedMotionIR -> exact S/G/W/M/B -> V0..V7 motion-quality proof -> editable curves / sprite bake / live puppet export`.

This is a proposed product architecture, not yet a promoted model closure.

Exact runtime reclosure is mandatory after mechanics + motion are ready. Runtime frame 0 and dynamic frames must preserve exact qualified mesh, mesh-skin and skeleton identity; no hidden re-triangulation, barycentric demo transfer or second runtime truth may be accepted.

## Current mandatory execution order

1. corrected H1 observable product surface — **DONE**;
2. GSA8192 canonical lineage + real V0..V7 evidence — **DONE**;
3. fresh Geppetto FIT2 — **RUNNING**;
4. new QualifiedSkeletonIR + real V0..V7 skeleton evidence — **PENDING**;
5. fresh Arachne FIT2 on corrected S+new G — **BLOCKED ON 3/4**;
6. new QualifiedSkinIR + real V0..V7 skin/deformation evidence — **PENDING**;
7. directional alpha-domain MWB2/CDT + supported Steiner refinement — **PENDING CURRENT S/G/W**;
8. QualifiedEditableMeshIR + QualifiedMeshSkinIR + qualified components/mechanical assembly — **PENDING**;
9. real V0..V7 dynamic deformation proof — **PENDING**;
10. professional motion closure — **OPEN**;
11. exact runtime/export reclosure — **OPEN**;
12. only then consider `PRODUCT_PASS`;
13. unseen/FIT8/LOFO remains after same-Mage FIT2 product reclosure, not before it.

## Claim boundary

Strongest current claim:

`RealSaS completed the historical Mage FIT1 model/Compiler investigations but the first real end-to-end product output exposed genuine end-to-end authority and quality gaps. The old Mage product chain is reopened. Corrected observation-bound H1 and GSA8192 are now closed, a fresh Geppetto FIT2 refit is active, and downstream Arachne/mesh/motion/runtime product closure remains pending.`

Not claimed: Geppetto FIT2 PASS; Arachne FIT2 PASS; corrected FIT2 mesh PASS; professional motion quality; exact runtime product closure; unseen-family generalization; FIT8/LOFO; `PRODUCT_PASS`; commercial production readiness.
