# RealSaS IRIS — 80-Check Post-Corpus Closure Matrix V1

**Date:** 2026-08-23
**Authority:** current post-corpus continuation matrix on `g0-g1/single-pose-geometry`.

This matrix replaces the informal “75–80 remaining items” backlog. A check is only closed when its evidence scope supports the claim; historical evidence is reused rather than rerun.

## Immediate scientific blockers before any new IRIS optimizer step

1. Gate 1: exhaustive non-empty/useful visual-coverage census and quarantine of blank/tiny observation assets.
2. Gate 2: selected-source mesh authority audit for evaluated `.blend`, triangulation, and normal authority.
3. Gate 3: appearance recoverability census; decide/build appearance-preserving B observation surface.
4. Gate 4: exact Representation Authority Study (`P`, `P+N`, `E_surface`, uncertainty-aware/set-valued `R`) on legal observation truth.
5. Gate 5: product-relevant exact information ceiling / SOI-2 before any impossibility claim.

## Critical corrections from the fast audit

- `primary_geometry.npz: missing=2398` is **not a corpus result**. The fast API script filtered primary-geometry files by `modifiedTime > 2026-08-22T23:00:00Z`; valid files written earlier were falsely excluded. Deep 64/64 loaded geometry, and a direct witness asset has physical `primary_geometry.npz` with a pre-cutoff timestamp.
- A real localized qualification failure was found: `asset_7da4136e89f94d4ce0d18337` is a 4-vertex / 2-face planar `.blend`, admitted as IRIS geometry-only but producing zero foreground and zero authority rows in all eight views. This requires observable-coverage qualification/quarantine, not global rerender.
- D1/G2/D2 are not reopened: D1 objective is directionally supported; G2 reciprocal-cycle is promoted; D2 global ranking is falsified while its local precision signal is retained.

## Gate 0

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 1 | Canonical selected-family census | **PASS** | CANONICAL_VARIANT_SELECTION: 3993 selected |  |
| 2 | Capability census (IRIS/Geppetto/Arachne) | **PASS** | 3993 / 3890 / 3414 |  |
| 3 | Deterministic split recomputation | **PASS** | 0 mismatches; FIT/TUNE/CAL/DEV/EXTERNAL = 2981/318/251/274/169 |  |
| 4 | Source-provider distribution | **WARN** | 3765 Objaverse, 191 Quaternius, 37 KayKit | Require source-stratified metrics |
| 5 | Canonical candidate/asset uniqueness | **PASS** | 4008 master variants = 4008 unique canonical IDs; 15 QA_ONLY excluded |  |
| 6 | Latest unresolved retrieval tail | **WARN** | 628 unresolved latest-state records (372 extraction, 256 materialization) | Do not conflate with current 3993 validity; revisit only if diversity insufficient |
| 7 | Interrupted consumer export isolation | **PASS** | Master corpus intact; consumer stage partial only | Do not resume old exporter |
| 8 | Gold-source acquisition coverage | **WARN** | KayKit/Khronos/Quaternius READY; Blender Studio + MPFB pending; RigXL fallback ENOSPC | Only reopen if domain/diversity study demands it |

## Gate 1

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 9 | Asset folder census | **PASS** | 3993/3993 folders |  |
| 10 | RENDER_COMPLETE marker census | **PASS** | 3993/3993; duplicates 0 |  |
| 11 | V0..V7 folder census | **PASS** | 31,944/31,944 |  |
| 12 | Required view-file census | **PASS** | cel/ink native+512, raster authority, camera: 0 missing/0 zero-size |  |
| 13 | Primary-geometry census | **AUDITOR_BUG** | Fast audit reported 2398 missing because of modifiedTime filter; deep 64/64 loaded and direct witness existed before filter cutoff | Rerun API census without time filter |
| 14 | Camera convention | **PASS** | Deep 512/512 TOP_LEFT, one flip, expected yaw |  |
| 15 | Raster-authority schema + pixel uniqueness/bounds | **PASS** | Deep 512/512 |  |
| 16 | Alpha ↔ authority exact parity | **PASS** | Deep 512/512 |  |
| 17 | Triangle/barycentric validity + common-frame reprojection | **PASS** | Deep 512/512; p95 view error ~2.1e-5 px, rare tiny tail outliers |  |
| 18 | Non-empty / useful visual coverage | **FAIL_LOCALIZED** | 1/64 deep asset blank in all 8 views: asset_7da413... | Exhaustive blank/tiny census; quarantine/IRIS-ineligible, no global rerender |

## Gate 2

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 19 | Canonical transform forward/inverse authority | **PASS** | Builder regression + deep finite geometry; exact transform design |  |
| 20 | Base mesh vs evaluated-depsgraph parity for .blend | **OPEN** | Extractor reads object.data, not evaluated depsgraph | Census .blend then compare only that subset |
| 21 | Triangulation parity / concave n-gon risk | **OPEN** | Extractor fan-triangulates polygons | Detect n-gons/concavity on affected sources; selective re-extract if consequential |
| 22 | Normal authority: smooth recomputed vs authored custom/split | **OPEN** | Builder recomputes area-weighted vertex normals | Compare source custom/split normals on stratified/affected subset |
| 23 | Degenerate/zero-normal qualification | **WARN** | Deep: 6/64 degenerate-tail assets; 6/64 zero-normal assets | Quantify full selected set; quarantine or typed-quality flag |
| 24 | Nonmanifold/disconnected component characterization | **PASS_AS_REQUIREMENT** | Deep: 57/64 multi-component; 13/64 nonmanifold; max components 3733 | SurfaceBuilder must be component/sheet-aware |
| 25 | Skin truth technical validity | **PASS_WITH_WARN** | Skin finite/nonnegative; Arachne admission threshold is zero-row fraction <=1%; sparse zero rows observed | Preserve per-asset quality flags; do not treat skin as IRIS truth |

## Gate 3

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 26 | Geometry-isolation render A retained as control | **PASS** | Current flat/cel pass has exact geometry authority | Keep permanently |
| 27 | Appearance information-loss identification | **PASS** | Current RGB strips source texture/material/markings by construction | Do not train final IRIS on A alone |
| 28 | Raw-source appearance recoverability census | **OPEN** | Main source file preserved; external dependency completeness unknown | Classify embedded / external-present / external-missing / no-appearance |
| 29 | Appearance-preserving sibling render B | **OPEN** | Not built yet | Render same geometry/camera/authority only after recoverability census |
| 30 | A vs B causal correspondence ablation | **OPEN** | Not run | Measure top-k/localization/hard-tail consequence |
| 31 | Stylized/product-domain augmentation C | **OPEN** | Not built yet | Use B/A truth-preserving stylization; IRIS sees style, downstream sees geometry |

## Gate 4

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 32 | P as core rigging substrate | **PASS** | S0-B/B2: substantial downstream signal |  |
| 33 | Direct N requirement for skinning/deformation | **PASS_CURRENT** | S0-B2: exact N retains material Arachne/deformation gain vs strongest deterministic N | Keep direct N candidate |
| 34 | D1 coarse identity objective | **PASS_DIRECTIONAL** | same-view +0.00624; reciprocal +0.00666; margin +0.01042 | Use as coarse/high-recall evidence |
| 35 | Reciprocal/cycle deterministic primitive | **PASS** | G2 4/4 gates; hit +5.09pp; regret -13.65%; 16/16 nonworse | Promote in F1 |
| 36 | D2 fine descriptor as global rank authority | **FAIL_CLOSED** | Only 1/4 primary gates; 3/29 family nonworse; mean distance/regret worse | Never use as global top1 authority |
| 37 | D2 fine descriptor as local retained-top-k evidence | **SUPPORTED_LOCAL** | PCK2 +0.142; oracle top4 +0.121; safety 6/6 | Use only local rerank/refine |
| 38 | GFDR global relational address R exact-geometry ceiling | **PASS_SCOPED** | Historical rank-3: G~95.6% -> G+R 100% on 9/9; oracle within-pose carrier caveat | Reimplement uncertainty-aware/set-valued; no hard noisy anchors |
| 39 | Continuous surface embedding E_surface / CSE-like authority | **OPEN** | Dense triangle+bary truth now enables it | Run exact representation study |
| 40 | Representation authority comparison incl remesh/symmetry/occlusion | **OPEN** | Not yet run on new 3993 corpus | Must close before final learner architecture |

## Gate 5

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 41 | Old hidden-source SOI scope correction | **PASS_CONCEPTUAL** | Old witness only proves hidden source distinctions can be non-identifiable | Do not use as product impossibility |
| 42 | Product-relevant geometry counterfactual worlds | **OPEN** | Need same/near-same A×8 with product-distinct downstream consequence | SOI-2 |
| 43 | Exact legal-observation representation ceiling | **OPEN** | Need exact P/N/E/R stack vs downstream consequence | Run before impossibility claim |
| 44 | Learner-vs-representation failure separation | **PASS_POLICY** | Research operating rule committed | Enforce on every fail |
| 45 | Strict abstain vs creative prior-completion policy | **PASS_POLICY** | OBSERVE>INFER>CONSTRAIN>PRIOR-COMPLETE>VERIFY defined | Wire later into compiler epistemic status |

## Gate 6

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 46 | Frozen G1 executable preflight | **PASS** | G1_EXECUTABLE_PREFLIGHT PASS; optimizer steps 0 | Preserve as causal baseline |
| 47 | Forbidden-truth leakage firewall | **PASS** | Pose-B/mechanics/rig truth forbidden; preflight leak=false |  |
| 48 | N1D warmstart migration | **PASS** | 141/141 destination tensors loaded |  |
| 49 | G1 128px baseline vs native 1024 product path | **WARN** | Frozen G1 input [1,8,4,128,128] | Do not mistake G1 for final frontend; new prereg needed for native/multiscale |
| 50 | F0 coarse/global persistence stage | **PARTIAL_EXISTING** | D1 coarse identity supported, but not rerun on new corpus/native path | Re-evaluate on new corpus |
| 51 | F1 + reciprocal/cycle stage | **PASS_MECHANISM** | G2 deterministic primitive proven on confirm16 | Integrate prospectively in new frontend |
| 52 | F2 D2 local rerank stage | **PASS_ROLE** | Global use falsified; local role preserved | Implement only inside retained top-k |
| 53 | D3 query-conditioned local matcher + U calibration | **OPEN_CONDITIONAL** | D2 leaves hard tail; D3 not run; U not compiler-grade calibrated | Only run after F0-F2 on new representation |

## Gate 7

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 54 | Deterministic local graph/neighbourhood | **PASS_CURRENT** | S0 evidence; no need learned topology by default |  |
| 55 | Component/sheet separation | **REQUIRED_OPEN** | Deep corpus complexity proves need; implementation not qualified | Build/qualify SurfaceBuilder component logic |
| 56 | Reprojection + reciprocal/cycle consistency | **PASS_PRIMITIVES** | Raster reprojection clean; G2 cycle positive | Integrate deterministic |
| 57 | Curvature/tangent/local-frame derivation | **OPEN** | Deterministic-first policy, not yet corpus-qualified | Derive after normal authority decision |
| 58 | Set-valued ambiguity envelope / visual hull separation | **OPEN** | Required for symmetry/occlusion; not implemented | Keep separate from canonical surface |
| 59 | No premature singleton collapse | **PASS_POLICY** | Frozen architecture/research rule | Maintain top-k/set-valued evidence until supported |

## Gate 8

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 60 | Exact P/N -> Geppetto ceiling | **PARTIAL** | S0 oracle-count information probes exist; full product skeleton topology not closed | Run full product exact-substrate ceiling |
| 61 | Exact dense P/N + fixed G -> Arachne ceiling | **PARTIAL** | S0 Arachne/deformation probes exist | Run on new corpus/full product subset |
| 62 | Predicted substrate -> Geppetto/Arachne consequence gap | **BLOCKED_BY_IRIS** | Requires trained frontend | Primary end-to-end IRIS criterion |
| 63 | RigAnything exact-vs-predicted P/N witness/noise ladder | **OPEN** | External feasibility established; local sensitivity calibration not run | Research-only calibration |
| 64 | M4/M5 compiler reuse/interface audit | **DEFER_UNTIL_IRIS** | Working compiler exists in Drive; GitHub branch not canonical | Inspect after substrate freezes |
| 65 | Deformation/motion-probe verification | **DEFER_UNTIL_IRIS** | Compiler proof chain exists historically | Rebind to new substrate |
| 66 | Commercial clean-room downstream path | **OPEN_PRODUCT** | RigAnything license research-only | Use only as witness/calibration, not shipped code/weights |

## Gate 9

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 67 | Observation-compatible geometry hypothesis representation | **OPEN** | Need bounded H/uncertainty modes |  |
| 68 | Geometry legality + sensitivity pruning | **OPEN** | No Cartesian hypothesis explosion |  |
| 69 | Bounded Geppetto rig modes per consequential geometry | **BLOCKED_BY_GATE8** | Requires downstream ceiling |  |
| 70 | Product-equivalence quotient over rig/deformation | **BLOCKED_BY_GATE8** | Need behavioral equivalence metrics |  |
| 71 | Commit vs strict abstain vs creative completion | **PASS_POLICY** | Decision policy concept frozen | Implement when quotient path exists |

## Gate 10

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 72 | Stylized artist-domain observation qualification | **OPEN** | Current A pass synthetic geometry-control only | Use B+C and real artist sheets |
| 73 | Camera/order/scale perturbation robustness | **OPEN** | Shipping contract controlled but artist deviations inevitable | Bound perturbation tests |
| 74 | Input qualification / fail-closed gates | **OPEN** | Need reject blank, bad framing, inconsistent views | Blank-asset finding directly feeds this |
| 75 | Source/domain-stratified evaluation | **REQUIRED** | 94.3% Objaverse selected | Every major metric by provider/domain |

## Gate 11

| # | Check | Status | Evidence | Next |
|---:|---|---|---|---|
| 76 | Failure-owner decision tree | **PASS_POLICY** | Raster→representation→learner→SurfaceBuilder→Geppetto→Arachne→deformation owner tree defined |  |
| 77 | Pretraining apparatus gates 0–3 closure | **IN_PROGRESS** | Gate1 mostly clean; blank census + Gate2/3 remain | Close before optimizer |
| 78 | Representation/info-ceiling gates 4–5 closure | **BLOCKED** | Exact representation study + SOI-2 remain | Next scientific blocker |
| 79 | Final IRIS training/qualification authorization | **BLOCKED** | No optimizer step until pretraining gates close | New prereg after winning representation/frontend |
| 80 | IRIS v1 freeze / handoff to downstream | **BLOCKED** | Requires predicted-vs-exact product consequence | End research loop when PASS/FIXABLE FAIL/TRUE LIMIT decision reached |

## Decision discipline

`APPARATUS/DATA` → `REPRESENTATION` → `TARGET AUTHORITY` → `LEARNER/SOLVER` → only then `GENUINE INFORMATION LIMIT`.

No negative learner result may be promoted to an impossibility claim while Gate 4–5 exact ceilings remain open. No positive result may be promoted without leakage/shortcut/source-stratification checks.
