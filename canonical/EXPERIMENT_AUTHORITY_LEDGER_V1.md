# RealSaS — Experiment Authority Ledger V1

**Updated:** 2026-09-12  
**Repository-wide continuation authority:** `CURRENT_STATE.md` on `main`  
**Active executable experiment branch:** `repair/mage-full-subject-reclosure-20260912`

This ledger distinguishes historical scientific evidence, current corrected-lineage experiments, product-contract closure and actual product-result closure.

## Current gate matrix

| Gate / experiment | Status | What it establishes | What it does not establish |
|---|---|---|---|
| `GEPPETTO_REFERENCE_STRENGTH_FIT1` | **CLOSED PASS / PROMOTED FIT1 / HISTORICAL-SCOPED** | historical Mage old-S 22-control skeleton proposal closed under the FIT1 gate | corrected-lineage Geppetto FIT2 PASS, unseen generalization, PRODUCT_PASS |
| `V7_C4_SKINTOKENS_FACE_BARYCENTRIC_BIASED_DENSE_SUPERVISION` | **CLOSED FAIL** | exact C4 sampling/objective family failed | all topology methods fail, A1 failure, product failure |
| `ARACHNE_A0_K4_REPRESENTATION_CLOSURE` | **CLOSED PASS / RESEARCH ORACLE** | historical Mage K4 continuous-field representation can reconstruct skin below p95 0.05 | current corrected-lineage Arachne or shipping decoder choice |
| `ARACHNE_A1_V4_FIT1` | **CLOSED FAIL / HISTORICAL** | old `S+G -> Z -> frozen A0 decoder` route missed p95 gate despite healthy deformation | V4 backbone/K4 lacks sufficient information |
| `ARACHNE_SUPPORT_SHAPING_CAUSAL_PROBE_V1` | **CLOSED DIAGNOSTIC** | perfect support alone leaves p95 ~0.0737; leakage is not sufficient explanation | direct readout outcome |
| `ARACHNE_DIRECT_NXJ_SIMPLEX_V1_8K` | **CLOSED HISTORICAL PARTIAL / HORIZON-LIMITED** | direct readout materially rescued H and Z but 8K horizon did not close gate | frozen representations are insufficient |
| `ARACHNE_DIRECT_NXJ_SIMPLEX_V2_16K` | **CLOSED PASS / CAUSAL COMPONENT EVIDENCE** | H and Z direct simplex both stably passed; minimal Z decoder seam preferred | unseen-family generalization or automatic product promotion |
| `ARACHNE_A1_V5_MINIMAL_K4_DIRECT_SIMPLEX_FIT1` | **CLOSED PASS / PROMOTED FIT1 / HISTORICAL-SCOPED** | historical old-S V5 product-candidate path reproduced Z PASS without A0 runtime decoder | current corrected-lineage Arachne FIT2, unseen generalization, PRODUCT_PASS |
| `GSA8192_REAL_8VIEW_EVIDENCE` | **CLOSED PASS / CURRENT CORRECTED STAGE-0 AUTHORITY** | corrected H1 emits sealed 8171-node / 23656-relation GSA lineage with real V0..V7 evidence | Geppetto FIT2 PASS, Arachne FIT2 PASS, mesh/product PASS |
| `FIT2_MESH_PRODUCT_RECLOSURE_CONTRACT_AND_IMPLEMENTATION` | **CLOSED PASS / IMPLEMENTATION ONLY** | strict product mesh admission independently rerasterizes the exact promoted mesh against exact observation authority and applies frozen coverage/topology gates fail-closed; self-hosted run `34716890157` passed 41 tests | corrected real Mage FIT2 mesh PASS, Arachne PASS, runtime PRODUCT_PASS |
| `MAGE_FIT2_PIPELINE_REFIT` | **ACTIVE / GEPPETTO FRESH REFIT RUNNING** | current same-Mage corrected product-reclosure experiment | nothing downstream may be called closed before its own sealed evidence |

## Current active FIT2 experiment

Authority: `canonical/MAGE_FIT2_PIPELINE_REFIT_AUTHORITY_V1.json`.

The active Geppetto run is fresh from scratch on corrected GSA8192:

- historical checkpoint loading forbidden;
- target joints `22`;
- `max_steps=16384`;
- `check_every=64`;
- terminal requirement `48/48` consecutive checks;
- evaluation seeds `[11,23,47,89]`;
- thresholds unchanged from the frozen inherited gate;
- PASS not claimed until terminal evidence + real V0..V7 skeleton evidence is sealed.

Fresh Arachne is blocked until that gate closes.

## Mesh product contract closure

Authority: `canonical/MAGE_FIT2_MESH_PRODUCT_RECLOSURE_AUDIT_20260912.md`.

The contract closure is deliberately separate from a real corrected mesh result.

Product admission now requires:

1. exact `ObservationRasterDomain` binding;
2. low-level legal CDT/surface-support qualification;
3. exact mesh raster reconstruction from qualified `SurfaceSupportBinding` + admitted S raster bindings;
4. independent rerasterization of the exact promoted mesh;
5. recomputation of recall / precision / IoU / connected-hole metrics;
6. independent topology/shape quality remeasurement;
7. frozen product policy evaluation;
8. observation hashes + final mesh lineage re-sealing.

`candidate.residual_report` is diagnostic only. `qualify_mwb2_observation_cdt_mesh(...)` is compatibility evidence only and cannot establish product closure.

Frozen per-view mesh product thresholds remain:

- recall `>= 0.94`;
- precision `>= 0.995`;
- IoU `>= 0.935`;
- largest uncovered 4-connected region `<= 0.015` of foreground;
- every alpha component occupying at least `0.0025` foreground fraction has recall `>= 0.90`;
- zero degenerate / duplicate / non-manifold faces;
- minimum raster angle `>= 0.25 deg`;
- maximum aspect ratio `<= 250`.

Self-hosted evidence: workflow `mage-full-subject-reclosure-contract`, run `34716890157`, head `b1dc7fca6b6497d97c7be727d66fd9c0b64c3268`, runner `realsas-wsl-1660ti`, static compile PASS, `41 passed in 4.36s`.

## Historical V5 terminal authority

Promotion commit: `03d9f87dbb7100a72293915cf682cbf338335a37`.

Historical FIT1 V5 closure:

- verdict `PASS__MINIMAL_K4_Z_DIRECT_SIMPLEX_V5_FIT1_CLOSURE`;
- GSA p95 `0.04237784981177733`;
- deformation ratio `0.019856400787830353`;
- articulated ratio `0.002780771814286709`;
- Compiler rows `950/950`, correction L1 `2.9468642839168442e-05`;
- A0 runtime model loaded `false`;
- no optimizer/backward/update in transplant closure;
- unseen-family claimed `false`.

It remains valid historical old-lineage evidence but is not current corrected Mage skin authority.

## Required next gate

Current next scientific gate:

`MAGE_FIT2_GEPPETTO_TERMINAL_CLOSURE`

If and only if fresh Geppetto closes with mandatory real V0..V7 skeleton evidence, proceed to fresh Arachne on corrected S+G. The mesh product contract is already implementation-level CLOSED PASS and waits for the new G/W lineage before a real directional mesh result can be qualified.

Unseen/FIT8/LOFO stays blocked until the same-Mage product chain closes.

## Binding rules

1. Source exists != mechanism tested.
2. PASS != promotion.
3. FIT1 or same-Mage FIT2 != unseen-family generalization.
4. A historical local PASS does not override a later real end-to-end contradiction.
5. Geppetto PASS != Arachne PASS.
6. A0 oracle != A1 shipping path.
7. A1 V4 decoder-path FAIL != backbone information failure after direct-readout evidence.
8. Compiler qualification cannot silently replace learned semantics.
9. Legal CDT admission != product mesh closure.
10. Candidate-reported coverage != product coverage evidence; exact promoted mesh must be independently rerasterized against exact observation authority.
11. Product-qualified mesh identity must survive unchanged into runtime/export; hidden retriangulation or a second mesh truth is forbidden.
12. Rotation-only motion proof != professional animation quality.
13. PRODUCT_PASS remains independent and exact end-to-end.
