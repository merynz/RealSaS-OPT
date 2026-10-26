# IRIS/TESSA Research Decision Ledger V3

**Date:** 2026-10-26  
**Status:** `ACTIVE_CANONICAL_RESEARCH_MEMORY__FAST_PROBE_MODE`  
**Supersedes for working decisions:** `RESEARCH_DECISION_LEDGER_V2.md` and the decision-summary role of `OPEN_QUESTIONS.md`. Those files remain immutable provenance.  
**Authority:** architecture-search evidence only; no ProductRevision, ModelRelease, Truth139, FIT8, LOFO or genericity authority.

## Operating rule

Every architecture court is retained as:

`question -> preregistered treatments/controls -> exact run/report -> metric/gates -> verdict -> carry-forward -> unresolved interaction/dependency`.

A green workflow means execution only. A scientific verdict exists only when the emitted report satisfies its preregistered interpretation rules.

Cheap synthetic / analytic / bounded micro-learning courts use sealed GitHub workflow reports. Real dataset/checkpoint/training/expensive-compute/model-product authority returns to Platform `Experiment -> Attempt -> Artifact/CAS`.

---

# A. Deterministic repairs already required

## A1 — preserve authored alpha/support

`PROVEN_REPAIR_REQUIRED`.

RGBA -> white RGB is non-injective and can erase silhouette/support before scene construction. Discarding alpha is ineligible. Exact compact alpha feature contract is still being selected.

## A2 — typed coordinate-domain coherence

`PROVEN_REPAIR_REQUIRED`.

IRIS scene/query coordinates and TESSA surface/presentation/output coordinates use different numeric domains and previous Bridge positions disagreed with `align_corners=False` cell centers.

Carry forward:
- typed coordinate domains;
- endpoint query conversion `x_iris = 2*x_tessa`;
- scene positions at actual carrier cell centers;
- no learned unit conversion / half-cell correction.

## A3 — occlusion truth needs negatives/N-A/abstention

`PROVEN_REPAIR_REQUIRED`.

Positive-only same-ray validity-masked BCE is not a calibrated occlusion truth source. Truth Builder must expose qualified positive, negative, N-A and abstention support.

## A4 — first-hit visibility is not compositing truth

`PROVEN_REPAIR_REQUIRED`.

Opaque first-hit binary truth cannot represent partial alpha. Truth Builder must retain full ordered ray-stack causal quantities such as incoming transmittance and per-hit contribution `w_k = alpha_k * product_{j<k}(1-alpha_j)`.

## A5 — tangent supervision needs anisotropy validity

`PROVEN_REPAIR_REQUIRED`.

Principal tangent is undefined near isotropic stretch. Carry forward a separate anisotropy/validity gate or equivalent continuous weighting.

## A6 — relation algebra must be typed

`STRUCTURAL_FACT_ESTABLISHED__FINAL_HEADS_TRUTH_DEPENDENT`.

Current generic pair representation can express directed quantities but does not enforce known invariants. Each causal family must declare one of `SYMMETRIC`, `DIRECTED`, `COMPLEMENTARY_OR_TIE_AWARE_ORDER`, `RAY_GLOBAL` before final head/objective selection.

---

# B. Selected / retained observation and fusion mechanisms

## B1 — frozen DINOv2-S/14 remains the foundation baseline

`TREATMENT_SELECTED_TINY`.

Earlier DINO L2 patches carry materially more local information than final-only tokens. A matched same-token-count / same-adapter-budget court selected aligned `L2 + FINAL` evidence on the existing 37x37 lattice:

- final-only mean local recovery `0.864626`, min `0.816327`;
- dual L2+final mean/min local recovery `1.0/1.0`;
- final-feature cosine mean/min `0.997598/0.996222`.

Immediate DINOv3 adoption is not supported by the current local-detail evidence.

## B2 — current scene-query -> 2x carrier route retained

`MECHANISM_ESTABLISHED`.

The subcell-routing court showed distinct subcell states can pass perfectly when present in source tokens. Do not redesign the route merely to compensate for final-only DINO information loss.

## B3 — query-mediated cross-view fusion retained

`MECHANISM_ESTABLISHED`.

Current query-mediated and source-self-attention-first arms both solved the controlled cross-view correspondence task perfectly. Early O(source-token^2) self-attention showed no measurable benefit.

## B4 — camera geometry remains open after V1

`APPARATUS_INCONCLUSIVE__REFINE`.

`IRIS_CAMERA_BACKPROJECTION_RESULT_V1.md`, run `40312959634`:

- current factorized camera+UV mean accuracy `0.499829`;
- deterministic backprojection mean `0.499835`;
- firewall ~chance;
- raw task-construction control `1.0`.

V1 entangled correspondence geometry with learned four-view parity aggregation and lacked a learned positive control. No camera treatment is selected or rejected.

Next: V2 canonical-locus recovery under held-out yaw/orthographic scale, with a learned positive control and matched capacity. Cross-view fusion itself stays governed by B3.

## B5 — compact alpha residual remains open

Alpha is mandatory by A1. The pending matched court compares RGB-only dual-level tokens with patch-aligned compact alpha summaries/residuals without increasing the 37x37 token count or image-adapter parameter budget.

---

# C. Scene carrier / continuous query evidence

## C1 — Oracle-Z V1 was inconclusive

`APPARATUS_INCONCLUSIVE__SUPERSEDED_BY_V2`.

V1 could not distinguish carrier capacity from optimizer/shared-Q failure because even the dense volumetric control failed.

## C2 — Oracle-Z V2 validates the apparatus but does not yet select a surface carrier

`VALID_CAPACITY_CURVES__SURFACE_SELECTION_OPEN`.

`ORACLE_Z_REPRESENTATION_RESULT_V2.md`, run `40312802934`:

- independent DENSE3D calibration PASS;
- selected LR `1e-3`;
- calibration mean/worst nRMSE `0.073784/0.076905`;
- random-target firewall PASS;
- DENSE3D solved easy volumetric frequency-1 mean/worst `0.072361/0.075248`.

However both TRIPLANE and DENSE3D missed the preregistered surface-family gates. Therefore there is no credible triplane failure at any tested radius.

Diagnostics:
- explicit xyz best worst-seed gain `1.36%` < `20%` selection gate -> not selected;
- multires best thin worst-seed gain `4.15%` < `20%` selection gate -> not selected.

Carry forward:
- do not reject the current triplane baseline;
- do not add explicit xyz or multires from V2;
- before carrier selection, calibrate the surface target with a learned/direct-coordinate positive control and a surface-relevant easier ladder;
- only blame a carrier after a valid positive control solves the same target/difficulty.

---

# D. Surface / extraction evidence

## D1 — extraction resolution controls topology observability before FlexiCubes offsets

`MECHANISM_ESTABLISHED__NO_PRODUCTION_RESOLUTION_SELECTED`.

Guaranteed TESSA-domain observability scales:

| R | sphere-like diameter | thin-support diameter | separation gap |
|---:|---:|---:|---:|
| 24 | 0.072169 | 0.058926 | 0.041667 |
| 32 | 0.054127 | 0.044194 | 0.031250 |
| 48 | 0.036084 | 0.029463 | 0.020833 |
| 64 | 0.027063 | 0.022097 | 0.015625 |

Historical authored edge-length statistics are descriptive only and cannot substitute for real normalized local-thickness/gap statistics.

## D2 — geometry-native XYZ farthest-point witness sampling selected

`TREATMENT_SELECTED_TINY`.

`SURFACE_WITNESS_SAMPLER_RESULT_V1.md`, run `40312976720`.

Current index sampler:
- permutation invariant: false;
- rare-group recall `0.90625`;
- mean/worst cover radius `0.149976/0.372825`.

Selected `XYZ_FPS`:
- permutation invariant on every synthetic family: true;
- rare-group recall `1.0`;
- mean/worst cover radius `0.075740/0.088903`.

Normal-aware FPS variants are eligible but trade spatial coverage for angular coverage; not selected by this isolated court.

Carry forward `XYZ_FPS` with deterministic tie-breaking and exact quota accounting. Extraction x witness interaction is still mandatory.

---

# E. TESSA evidence binding / memory

## E1 — endpoint-local presentation evidence is bound before compression

`TREATMENT_SELECTED_TINY`.

Endpoint-bound late compression reached `1.0` across all five seeds while current late/unbound and generic joint/unbound stayed near `0.50`. The deterministic binder has zero trainable parameters.

Carry forward:
1. typed endpoint references;
2. deterministic coordinate-domain conversion;
3. evaluate `Q(Z,x_a)` / `Q(Z,x_b)`;
4. attach admissible endpoint-local scene/mechanical evidence before presentation compression;
5. TESSA remains proposal/structuralization, Compiler remains canonical authority.

## E2 — current local-window semantics measured, adequacy open

`SEMANTICS_MEASURED__CORPUS_DEPENDENCY_OPEN`.

Current local attention gives roughly `3841-4096` direct historical tokens. Historical FIT8 teacher-forced sequences fit the 131072 decode cap, but sequence length alone does not prove useful dependency span. Do not enlarge memory without a dependency-span court.

---

# F. Required pre-V3.1 courts still open

These are the currently testable architecture questions that must close before V3.1 freeze:

1. **Alpha residual contract** — select compact patch-aligned alpha/support treatment.
2. **Camera geometry V2** — isolate canonical coordinate recovery from aggregation capacity.
3. **Oracle-Z surface calibration / carrier V3** — learned direct-coordinate positive control, then matched carrier comparison on a validated surface task.
4. **Shared-scene degree Stage A** — independent vs late fusion vs shared-trunk/private vs fully shared vs shared+family-adapters under matched synthetic mechanisms and nuisance/interference tests.
5. **Extraction x witness interaction** — verify selected XYZ_FPS remains stable across admitted extraction densities/topology changes; choose only a provisional extraction policy supported by current non-Truth evidence.
6. **TESSA topology/dependency audit** — inspect/run existing free-running topology and useful-dependency-span courts before memory/grammar changes.
7. **Selected observation x carrier x binding interaction court** — ensure local winners remain compatible when assembled.
8. **Bounded joint tiny system** — final L5 falsification before architecture freeze.

Truth-dependent questions intentionally deferred to authored Truth Builder / L4:
- final causal relation quantities and loss geometry;
- real positive/negative/N-A sampling distribution;
- real multi-objective gradient conflict / surgery;
- final presentation response-vector/ray schema details;
- genericity/product claims.

---

# V3.1 freeze rule

A treatment enters V3.1 only if one of the following is true:

1. deterministic mathematics/source contract proves the current form invalid and the repair is authority-safe; or
2. a valid bounded court selects the treatment and the relevant interaction court does not overturn it; or
3. no challenger is selected after a valid falsification court, so the existing low-compute baseline is explicitly retained as a **provisional pre-Truth default**, not declared globally optimal.

After the required pre-V3.1 courts close, write one `IRIS_TESSA_V3_1_ARCHITECTURE_FREEZE.md`. Do not merge the audit branch wholesale. Extract implementation changes later as small reviewed PRs.
