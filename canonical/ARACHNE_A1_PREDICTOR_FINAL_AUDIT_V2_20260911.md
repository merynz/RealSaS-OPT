# RealSaS — Arachne A1 Predictor Final Audit V2

**Date:** 2026-09-11  
**Branch:** `exp/arachne-a1-v7-native-fit1-20260911`  
**Status:** `PRE_NOTEBOOK_PREDICTOR_AUDIT_CLOSED__V3_TREATMENT_SUPERSEDED_PRE_OPTIMIZER__V4_FIX1_IS_EXECUTION_AUTHORITY`  
**Product PASS:** not claimed.  
**Unseen/generalization PASS:** not claimed.

This record is the final design audit requested before producing the A1 FIT1 notebook. It explicitly re-checks every predictor concern raised during the Arachne information-preservation audit and subsequent A1 architecture discussion. The earlier V3 A1 source/prereg was never trained and is now superseded before any A1 optimizer construction.

## 1. Frozen parent is valid

The selected A0 K4 interface is immutable for A1:

- K4 model SHA-256: `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`
- A1 supervision bank SHA-256: `b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a`
- A0 closure result SHA-256: `14043d1f2b638d576e86e25568ffa80e931f84807fafe0135f3f328e027abd6c`
- final GSA p95: `0.04491063521144626`
- final GSA continuity deformation ratio: `0.018186409026384354`
- stable-last-3: PASS
- full-set K4 FP32 permutation delta: `5.7220458984375e-06 <= 1e-05`

Therefore A1 predicts an **unordered K=4 continuous latent set** and ordered latent regression remains forbidden as semantic authority.

## 2. Concern-by-concern closure

| Concern raised in audit/design discussion | V3 draft | V4 FIX1 resolution | Status |
|---|---|---|---|
| Old A1 was too small (~0.55M) and capacity could remain a confound | 54,031,920 params | same V4 topology static sweep: 88,451,201 @512; **138,053,153 @640 selected**; 198,650,817 @768. 640/10 gives head-dim 64 and substantial headroom without immediately jumping to ~199M | **CLOSED for FIT1 baseline** |
| Historical 20D surface / 8D joint compression silently discarded rich evidence | removed | V4 consumes the rich V3 conditioning boundary: exact P/N, view evidence, exact GSA graph, qualified tree, exact anchors, 10D pair geometry | **CLOSED** |
| Exact 176 joint↔surface anchor identities were reduced to constant support-count | exact matrix present | exact `support_anchor_matrix` enters both joint anchor context and attention bias; it is never a skin mask | **CLOSED** |
| Exact GSA topology was reduced to degree summaries | exact graph present | 4 exact-edge message-passing layers + global surface attention | **CLOSED** |
| Raster evidence was reduced to count | exact per-view raster present | shared view encoder consumes exact raster XY/support/validity | **CLOSED** |
| Absolute view-slot flattening could create slot-order shortcut | V3 still concatenated slot fields into 41D base | V4 removes all view-specific fields from flat base; per-view evidence goes only through shared encoder with camera-bound yaw Fourier code and permutation-invariant pooling | **CLOSED** |
| View permutation/camera binding was not tested | absent | mandatory view-binding permutation equivariance preflight, tolerance `2e-5` | **CLOSED** |
| Surface/joint permutation equivariance | present in V3 | retained; support-anchor and pair mappings remap with permutations | **CLOSED** |
| `pair_mask` existed but V3 did not use it as attention legality | validation only | V4 masks illegal surface↔joint pairs in both attention directions; this is a legality mask, not nearest-joint sparsification | **CLOSED** |
| Dense J×K reasoning should not be prematurely factorized | one-way dense token→surface | V4 uses full `J×K` token set and **4 dense bidirectional fusion rounds** | **CLOSED** |
| Surface should be able to update from joint-field context | absent | surface→joint-field and joint-field→surface are both explicit in every fusion round | **CLOSED** |
| Full qualified tree/root must be used without re-parenting | yes | 4 tree message-passing layers; topology remains immutable Compiler authority | **CLOSED** |
| Fixed bone vocabulary / semantic joint-ID shortcut | absent | still absent; no learned joint-ID embedding | **CLOSED** |
| Hard nearest-bone mask would damage ambiguous boundaries | absent | still absent | **CLOSED** |
| Old 10D point↔joint/parent-segment geometry was useful and should survive | yes | retained as dense relation bias | **CLOSED** |
| Frozen codec condition representation must not masquerade as new evidence | ambiguous design risk | condition tokens are generated only inside the frozen decoder lane; predictor forward signature cannot accept them | **CLOSED** |
| Teacher W / teacher latent / source mesh / source IDs could leak into predictor | firewall intended | runner inspects predictor signature; W remains objective/evaluation only; hidden/source geometry, component names, identity, rejected parent alternatives remain forbidden | **CLOSED** |
| Current IRIS `log_uncertainty` is not calibrated evidence | excluded | remains excluded | **CLOSED** |
| Geppetto salience magnitude is saturated/unhelpful on FIT1 | excluded | remains excluded | **CLOSED** |
| Geppetto position sigma has plausible semantics but calibration unknown | deferred | remains an explicit future sidecar ablation, not baseline evidence | **JUSTIFIED DEFER** |
| K4 decoder is set-like; ordered token-1↔token-1 loss would be false authority | V3 ordered loss already zero | V4 ordered latent alignment remains exactly zero | **CLOSED** |
| Independent scalar objectives did not directly train post-normalization joint competition | V3 added mean coupled row-L1 | V4 retains normalized coupled row-L1 | **CLOSED** |
| Hard-tail/blend boundaries were not explicitly protected | missing | V4 adds sample CVaR10 row-L1 (`0.5`) + teacher-only blend-boundary row-L1 (`0.5`) | **CLOSED** |
| `hold_deform`/deformation consequence had no direct gradient | V3 intentionally zero due bad probe | V4 introduces a new deterministic parent-relative articulated probe and a direct deformation ratio loss (`0.25`) | **CLOSED** |
| Old synthetic translation probe is physically weak and joint-index dependent | still continuity metric | retained only for A0 continuity reporting; never training authority | **CLOSED** |
| New deformation probe must be joint-permutation consistent | not available | probe uses only qualified positions/tree + fixed world axes; explicit permutation preflight tolerance `1e-6` | **CLOSED** |
| Compiler must not secretly repair A1 semantics | qualification present | A1 gate also bounds aggregate Compiler correction to `1e-4`; proposal remains dense and normalized before qualification | **CLOSED** |
| V4 proposal metadata must not inherit V3 architecture label | V4 core initially reused V3 helper | pre-optimizer `proposal_v4.py` + `run_arachne_a1_v4_fit1_fix1.py` bind the V4 architecture metadata before execution | **CLOSED** |

## 3. Capacity decision

The final predictor baseline is intentionally **not small**:

- model width: `640`
- heads: `10`
- head dimension: `64`
- exact trainable predictor parameters: **138,053,153**
- frozen codec/decoder parameters are outside this count

A static same-topology width sweep gives:

- width 512 / 8 heads: `88,451,201`
- width 640 / 10 heads: `138,053,153` ← selected
- width 768 / 12 heads: `198,650,817`

This does **not** claim 138M is globally optimal. It removes the obvious under-capacity confound from the first real V7-native A1 treatment while retaining a materially lighter alternative than ~199M. Future multi-family/unseen work may still justify a controlled capacity ablation using the exact same rich boundary.

## 4. View/dataflow correction found during final audit

The V3 draft still had one important hidden information-architecture flaw: although it added a shared view encoder, it also flattened 8 support bits + 16 raster coordinates + 8 raster-valid bits into the base surface MLP. That meant camera-bound view order could still leak through absolute slots.

V4 removes this path. The view-independent base contains only geometry/state and invariant support/raster fractions. Exact per-view support/raster evidence is processed by one shared view encoder together with the bound yaw code, then pooled across views. Consistently permuting view evidence and yaw bindings must leave predicted K4 tokens unchanged within tolerance.

## 5. Bidirectional fusion correction

The discussed baseline was **dense bidirectional joint-field reasoning** because Mage has only `22 × 4 = 88` joint-field tokens. V3 only implemented token→surface reads. V4 now performs four rounds of:

`joint-field tokens -> full surface memory -> updated joint-field tokens`

followed by:

`updated surface queries -> all joint-field tokens -> updated surface memory`.

Both directions receive the same legal 10D point↔joint/segment relation bias and exact sparse anchor evidence. No hard nearest-joint mask is introduced.

## 6. Deformation consequence correction

The old continuity probe assigned synthetic translations by serialized joint index. It remains useful only for historical A0 metric continuity.

The new training probe builds bind/pose transforms from:

- qualified joint world positions;
- exact accepted parent tree;
- geometry-derived bone axes;
- four small deterministic articulated rotations.

No joint ID/name or serialized row number determines the physical transform. A consistent joint permutation + parent remap must produce the same transforms after inverse permutation.

The V4 objective therefore has a real gradient path:

`S+G -> K4 latent set -> frozen decoder -> normalized W -> articulated LBS -> deformation consequence loss`.

This directly addresses the signal-connectivity concern found in the A0 holdout audit.

## 7. Objective authority

Exact V4 loss weights:

- BCE: `1.0`
- MSE: `0.1`
- Dice: `1.0`
- coupled normalized row-L1: `1.0`
- CVaR10 hard-tail row-L1: `0.5`
- blend-boundary row-L1: `0.5`
- articulated deformation ratio: `0.25`
- ordered latent alignment: `0.0`

Teacher W is used only to construct targets/losses/evaluation. It never enters predictor features.

## 8. Static mechanism checks completed before notebook

Local CPU mechanism checks on the exact V4 source shape contract (small-width test instance where necessary) produced:

- default V4 exact parameter count: `138,053,153`
- surface permutation max delta: approximately `1.31e-06`
- joint permutation max delta: approximately `1.19e-06`
- view-binding permutation max delta: approximately `1.37e-06`
- articulated-probe joint permutation delta: `0.0`
- behavior/hard-tail/articulated loss produced nonzero gradients in a synthetic backward smoke test
- all V4 Python sources compiled successfully before repository publication

These are implementation/mechanism checks only, not scientific FIT1 results.

## 9. Execution authority

Execution preregistration:

`canonical/ARACHNE_A1_V7_NATIVE_FIT1_PREREG_V4_FIX1_20260911.json`

Execution entry point:

`experiments/arachne_a1_v4_fit1/run_arachne_a1_v4_fit1_fix1.py`

The FIX1 wrapper exists solely because the V4 core initially reused a V3 proposal helper whose metadata string still named the V3 architecture. The wrapper replaces that helper **before the optimizer is constructed**; no scientific mechanism or loss changed after optimizer exposure because no A1 optimizer has yet run.

## 10. Remaining items that are deliberately not blockers for the A1 FIT1 notebook

- `position_sigma_normalized`: potential future calibrated sidecar ablation; not required for baseline.
- explicit rigid assembly/socket semantics: separate product/Compiler assembly lane; current skin truth can still represent rigid attachments kinematically.
- exact source-component names (hat/book/staff): evaluation-only diagnostics, forbidden predictor features.
- disjoint-surface holdout: diagnostic only, not A1 transition or FIT1 acceptance gate.
- unseen/family generalization: not part of Mage FIT1 and not claimed.
- global optimality of 138M predictor capacity: not claimed; only the obvious tiny-predictor confound is removed.

## 11. Final decision

`ARACHNE_A1_PREDICTOR_PRE_NOTEBOOK_AUDIT = CLOSED`

`A1_V3_EXECUTION = SUPERSEDED_PRE_OPTIMIZER`

`A1_V4_FIX1_EXECUTION = AUTHORIZED_FOR_MAGE_FIT1`

`NOTEBOOK_GENERATION = NOW_ALLOWED`

No known predictor-design issue from the pre-A1 audit/discussion remains silently unresolved in the first Mage FIT1 treatment. Any architecture, feature-boundary, objective or probe change after this record requires a new preregistration transaction.
