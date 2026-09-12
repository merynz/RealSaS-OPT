# RealSaS — Experiment Authority Ledger V1

**Updated:** 2026-09-12  
**Continuation authority:** `CURRENT_STATE.md` on `main`

## Current gate matrix

| Gate / experiment | Status | What it establishes | What it does not establish |
|---|---|---|---|
| `GEPPETTO_REFERENCE_STRENGTH_FIT1` | **CLOSED PASS / PROMOTED / HISTORICAL SCOPED** | historical old-lineage Mage 22-control skeleton proposal closed under its FIT1 gate | current corrected-lineage Geppetto PASS, unseen generalization, PRODUCT_PASS |
| `V7_C4_SKINTOKENS_FACE_BARYCENTRIC_BIASED_DENSE_SUPERVISION` | **CLOSED FAIL** | exact C4 sampling/objective family failed | all topology methods fail, all Arachne formulations fail, product failure |
| `ARACHNE_A0_K4_REPRESENTATION_CLOSURE` | **CLOSED PASS / RESEARCH ORACLE** | historical K4 representation can reconstruct Mage skin below p95 0.05 | current corrected-lineage inverse inference or shipping decoder choice |
| `ARACHNE_A1_V4_FIT1` | **CLOSED FAIL / HISTORICAL** | old `S+G -> Z -> frozen A0 decoder` route missed p95 gate despite healthy deformation | V4 backbone/K4 lacks sufficient information |
| `ARACHNE_SUPPORT_SHAPING_CAUSAL_PROBE_V1` | **CLOSED DIAGNOSTIC** | perfect support alone left p95 ~0.0737; leakage was not sufficient explanation | corrected FIT2 outcome |
| `ARACHNE_DIRECT_NXJ_SIMPLEX_V1_8K` | **CLOSED HISTORICAL PARTIAL / HORIZON-LIMITED** | direct readout materially rescued H and Z but 8K did not close stable gate | corrected FIT2 sufficiency |
| `ARACHNE_DIRECT_NXJ_SIMPLEX_V2_16K` | **CLOSED PASS / CAUSAL COMPONENT EVIDENCE** | historical H and Z direct simplex both stably passed; minimal Z seam preferred | corrected-lineage Arachne PASS, unseen generalization, automatic promotion |
| `ARACHNE_A1_V5_MINIMAL_K4_DIRECT_SIMPLEX_FIT1` | **CLOSED PASS / PROMOTED FIT1 FROZEN / HISTORICAL SCOPED** | historical old-lineage V5 path reproduced the direct-simplex PASS without A0 runtime decoder | current corrected-lineage Arachne PASS, unseen generalization, full PRODUCT_PASS |
| `H1_OBSERVABLE_PRODUCT_SURFACE_FIT2` | **CLOSED PASS / CURRENT STAGE-0 AUTHORITY** | corrected eight-view observable Mage product surface passes alpha + first-hit geometry admission without teacher geometry at product inference | hidden full-teacher completeness, Geppetto/Arachne/product PASS |
| `GSA8192_REAL_8VIEW_EVIDENCE` | **CLOSED PASS / CURRENT STAGE-0 AUTHORITY** | corrected H1 deterministically emits sealed 8171-node/23656-relation GSA lineage with real V0..V7 evidence | Geppetto FIT2 PASS, Arachne FIT2 PASS, mesh/product PASS |
| `MAGE_FIT2_PIPELINE_REFIT` | **ACTIVE — GEPPETTO FRESH REFIT RUNNING** | current same-Mage contradiction-driven product reclosure program | any downstream PASS before exact stage evidence exists |
| `FIT2_MESH_COMPONENT_CLOSURE` | **PREREGISTERED + IMPLEMENTED/CONTRACT-TESTED / CORRECTED RESULT PENDING** | mesh coverage/quality, component mechanics and supported inserted-vertex gates exist and self-hosted tests are green | corrected FIT2 mesh PASS before fresh S/G/W exists |
| `PROFESSIONAL_MOTION_CLOSURE` | **OPEN / NOT YET PREREGISTERED AS EXECUTED GATE** | current rotation-only lane is explicitly insufficient as product motion | artist-quality idle/run or learned/deterministic motion product PASS |
| `EXACT_RUNTIME_PRODUCT_RECLOSURE` | **OPEN / REQUIRED** | exact qualified-state runtime equivalence is now a mandatory product invariant | current runtime PRODUCT_PASS |

## Why current authority changed

The historical FIT1 local model/Compiler experiments did close as recorded. The current product interpretation changed only after the first real eight-view product execution produced visibly poor output and exposed multiple independent gaps.

Exact forensic sequence:

1. full deformation-supported source geometry reproduced ~`99.46%..99.68%` source-alpha recall;
2. historical promoted signed zero-surface reproduced only ~`62.24%..67.01%`;
3. historical conservative relation-complex MWB2 reproduced only ~`18.18%..32.31%`;
4. exported-bundle forensics later exposed a separate noncanonical alpha-clipped barycentric runtime mesh and substantial missing visual domain;
5. current motion was rotation-only mechanical probing;
6. proof/runtime contracts were not sufficient to establish exact art-preserving product quality.

Therefore the correct experiment-state transition is **product-authority reopening**, not deletion/relabeling of the historical FIT1 experiments.

Primary causal context: `canonical/MAGE_FIT2_REAL_E2E_REOPENING_CONTEXT_20260912.md`.

## Corrected Stage 0 authority

Active branch: `repair/mage-full-subject-reclosure-20260912`.

### H1 observable product surface

- status `PASS_H1_OBSERVABLE_PRODUCT_SURFACE`;
- source run `20260912T074348Z`;
- checkpoint SHA-256 `76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5`;
- product-clipped zero-surface SHA-256 `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`;
- minimum alpha recall `0.9511473445`;
- minimum precision `0.9806321480`;
- minimum IoU `0.9356283394`;
- teacher mesh at product inference `false`.

Historical global-sign/full-hidden-teacher metrics remain preserved diagnostics.

### GSA8192

- status `PASS__REAL_8VIEW_GSA_EVIDENCE_SEALED`;
- nodes `8171`;
- relations `23656`;
- observed/completed nodes `7391 / 780`;
- geometry lineage `65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da`;
- tensorization `fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e`.

This closes only the corrected substrate stage.

## Active gate — MAGE_FIT2_PIPELINE_REFIT

Current substage: `GEPPETTO_FRESH_REFIT`.

Binding current authority:

- mode `FRESH_FROM_SCRATCH__NO_HISTORICAL_GEPPETTO_CHECKPOINT`;
- historical checkpoint load: forbidden;
- target joints `22`;
- max steps `16384`;
- check every `64`;
- terminal streak required `48/48`;
- seeds `[11,23,47,89]`;
- thresholds unchanged;
- real-input CPU preflight PASS;
- actual external GPU optimizer run active per operator context;
- Geppetto FIT2 PASS currently `false`.

The older `frozen replay first` sequence is superseded. Fresh Arachne follows only after new Geppetto terminal closure + real V0..V7 skeleton evidence.

## Mesh/component hardening authority

Prereg: active branch `canonical/FIT2_MESH_COMPONENT_CLOSURE_PREREG_20260912.md`.

Implementation commits:

- `9dd533eb5c48408faa92bcfb3469b7d32fd39368` — harden FIT2 mesh coverage and component closure;
- `ebe7358dbdafbf65068dde91f3eba39b58b784b6` — run FIT2 mesh/component closure tests.

The self-hosted `mage-full-subject-reclosure-contract` workflow at `ebe7358...` completed **SUCCESS** on `realsas-wsl-1660ti`.

This proves the current gates are implemented/tested. It does **not** establish corrected FIT2 mesh quality yet.

## Required current next gate

`MAGE_FIT2_GEPPETTO_TERMINAL_CLOSURE`

If the fresh run reaches its unchanged 48/48 terminal gate, the closure transaction must still include new QualifiedSkeletonIR authority and mandatory real V0..V7 skeleton evidence before fresh Arachne is authorized.

If it fails, diagnose within the frozen experiment boundary. Do not relax thresholds post-result and do not silently route semantics into the Compiler.

## Downstream order after Geppetto

`fresh Arachne FIT2`

-> new `QualifiedSkinIR` + real V0..V7 deformation evidence

-> directional alpha-domain MWB2/CDT + support-bound Steiner refinement

-> exact `QualifiedEditableMeshIR + QualifiedMeshSkinIR`

-> qualified component/mechanical assembly

-> real V0..V7 dynamic proof

-> professional motion closure

-> exact runtime/export reclosure

-> only then same-Mage `PRODUCT_PASS` consideration

-> then family-disjoint unseen/FIT8-LOFO.

## Binding rules

1. Source exists != mechanism tested.
2. PASS != promotion.
3. Historical FIT1 local PASS != current corrected-lineage product authority.
4. Real E2E contradiction can reopen product authority without rewriting historical experiment outcomes.
5. Same-Mage FIT2 != generalization.
6. Geppetto PASS != Arachne PASS != mesh/product PASS.
7. Compiler qualification cannot silently replace learned semantics.
8. Aggregate mesh legality cannot hide missing artist-visible alpha domain.
9. Supported inserted vertices require one shared geometry/skin support simplex.
10. Rotation-only preset motion is a mechanical probe, not professional motion closure.
11. Runtime/export may not create a second unqualified mesh/mechanics truth.
12. `PRODUCT_PASS` remains independent and exact-end-to-end.
