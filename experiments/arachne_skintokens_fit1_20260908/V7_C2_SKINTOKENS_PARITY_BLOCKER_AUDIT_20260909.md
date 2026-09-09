# Arachne Mage A0 FIT1 — V7-C2 SkinTokens Parity / Blocker Audit

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Scope: diagnosis only. No architecture/training/output-contract change is authorized by this note.

## Why this audit exists

After V7 C0 long-horizon and C1/C1B/C2, the remaining error is concentrated in blend rows. C2 importance correction improves the result but does not close FIT1. Before introducing a new custom blend-ratio objective, compare the current RealSaS path against the upstream SkinTokens implementation and paper at the exact stage implicated by the residual failure.

## Current RealSaS evidence

C2 treatment final / best:
- best p95 row-L1: `0.21809207734976085` at step 320
- final p95 row-L1: `0.22197738558673968`
- final deformation-error ratio: `0.08492327481508255`
- final dominant accuracy: `0.9946466809421841`
- final top-3 inclusion of teacher dominant: `1.0`
- final p95-tail inactive predicted mass mean: `0.17833878206849968`
- final inactive predicted mass mean: `0.020908252814482105`
- final pure 1-joint p95: `0.0051647449339130725`
- final 2-joint p95: `0.2276252754342596`
- final 3+-joint p95: `0.2597313785928242`

C2 treatment causally improved matched control:
- p95: `0.29005521588480354 -> 0.21809207734976085`
- deformation: `0.10468220710754395 -> 0.08492327481508255`
- p95-tail inactive mass: `0.2287736541772101 -> 0.17833878206849968`

Therefore active-heavy proposal bias was real but is not the whole residual blocker.

C1 teacher-support oracle established an upper-bound localization:
- baseline p95 `0.3240030049430026 -> 0.0791619746438563`
- baseline deform `0.10528597980737686 -> 0.02997232973575592`
- deformation gate passes under exact teacher-support projection, while p95 still misses the `0.05` gate.

This means the residual is mixed: large inactive-support leakage plus a smaller within-correct-support blend-ratio error.

## Mage truth support cardinality

Exact bound cache `ARACHNE_MAGE_FS1_CONDITIONING_CACHE_V2.npz`, supervised rows only, threshold `>1e-8`:
- support size 1: 444 rows
- support size 2: 149 rows
- support size 3: 330 rows
- support size 4: 11 rows
- maximum support size: **4**

Thus a four-influence product contract does not discard any ground-truth support on Mage.

## Upstream SkinTokens facts relevant to this blocker

Upstream repository audited at commit `273b691d35989d71cd17ff2895fdc735097b92d1` (`VAST-AI-Research/SkinTokens`).

### 1. Reconstruction objective is not a missing custom blend-ratio objective

The SkinTokens paper describes the FSQ-CVAE reconstruction as BCE + small MSE + Dice. It explicitly motivates Dice for sparse positive regions and uses a hybrid sampling strategy. This is the same objective family already used by RealSaS V7.

Conclusion: there is currently no upstream evidence that a custom pairwise/relative blend-ratio auxiliary is the ingredient RealSaS is missing. Do not introduce one before exhausting direct upstream-parity differences.

### 2. Upstream dense sampling is boundary-aware, not exact-positive-only

File: `src/data/sampler.py`, `SamplerMix.sample_on_skin()`.

For one bone field upstream:
1. mark faces with any non-zero skin value;
2. if enabled, expand the mask to faces spatially near that support using `max_distance` and `rate_distance`;
3. resample mesh-surface points inside this support-plus-neighborhood mask;
4. carry interpolated skin values on those sampled points.

The paper describes this as mixing uniformly sampled mesh points with points sampled densely from regions with non-zero ground-truth weights.

RealSaS C0/C2 instead samples exact supervised rows using 50% active (`w_j > eps`) + 50% global. It does not create a geometric support-boundary neighborhood with fresh/interpolated local samples.

This is a direct structural difference at exactly the residual failure location: RealSaS must preserve weak true blend weights while suppressing nearby false inactive weights.

### 3. Upstream production export hard-caps influences to top 4 and renormalizes

Files:
- `demo.py`
- `src/rig_package/parser/bpy.py`

The official demo passes `group_per_vertex=4` for both direct export and transfer export. The exporter sorts each vertex's skin weights, retains the largest `group_per_vertex` entries, and renormalizes by the retained sum before writing vertex groups.

This is not the optional voxel-skin postprocess. It is on the normal export path.

RealSaS C2 final telemetry reports `compiler_total_sparsification_discarded_mass = 0.0`; therefore the current FIT1 gate evaluates all 22 normalized influences rather than upstream SkinTokens' production top-4 skin contract.

Because Mage truth has maximum support 4, this upstream rule is target-compatible on the current FIT1 character.

### 4. Upstream per-bone decoder itself is sigmoid scalar-field output

Files:
- `src/model/skin_vae/autoencoders/autoencoder_kl_tripo2.py`
- `src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py`

The decoder produces one scalar skin field per bone and applies sigmoid. The per-bone predictions are later assembled into a skin matrix. Thus the important direct mismatch is not that SkinTokens uses a native 22-way softmax/simplex head; it does not.

## Revised blocker classification

Previous provisional phrase `blend-ratio objective missing` is too strong and is withdrawn.

Current classification:

**`SUPPORT_BOUNDARY_CALIBRATION_PLUS_PRODUCT_SPARSIFICATION_CONTRACT_MISMATCH`**

with a secondary residual:

**`WITHIN_SUPPORT_BLEND_RATIO_ERROR_AFTER_PERFECT_SUPPORT`**

Evidence hierarchy:
1. ownership/geometry/ranking are essentially solved;
2. exact teacher-support removal makes deformation pass and cuts p95 strongly;
3. importance correction reduces inactive leakage but only partially;
4. upstream SkinTokens uses two directly relevant mechanisms that RealSaS has not matched: local support-neighborhood dense sampling and top-4 export sparsification;
5. upstream loss family is already essentially the one RealSaS uses, so inventing a new ratio loss is not currently justified.

## Next gate — one deterministic upstream-parity check only

Do **not** start C3 training yet.

Run a frozen, no-training replay on the exact C2 treatment final model:

`sigmoid scalar fields -> keep top 4 joints per row -> renormalize retained weights -> unchanged product metrics`

This is not a hyperparameter sweep:
- `K=4` is taken directly from upstream SkinTokens production export;
- Mage truth maximum support is exactly 4;
- no teacher support is used to choose the predicted top 4;
- no model weight, optimizer, architecture, sampler, or loss changes.

Required telemetry:
- mean / p95 / CVaR10 row-L1
- deformation-error ratio
- inactive predicted mass after top-4
- true-support recall inside predicted top-4, split by support size 1/2/3/4
- dominant/top3/variation preserved

Decision:
- If top-4 largely closes the leakage gap and/or reaches the product gates, the main blocker was partly a production-contract mismatch, not a missing learned objective. Any adoption must explicitly version RealSaS's max-influences product contract; do not silently weaken an existing gate.
- If top-4 reduces leakage but p95 settles near the teacher-support oracle floor (~0.08), support cleanup is effectively solved and the remaining blocker is genuine within-support blend calibration. Before adding a custom loss, compare/port upstream boundary-aware dense sampling.
- If top-4 remains near ~0.2 because weak true supports are displaced by false joints, the remaining learned blocker is support ordering specifically at blend boundaries. The next upstream-parity treatment is boundary-aware dense sampling, not an arbitrary objective change.

## Explicit non-decisions

- No C3 blend-ratio loss authorized.
- No 4k blind continuation authorized.
- No architecture/FSQ/token-count change authorized.
- No extra characters/pretraining justified by this residual.
- No threshold/top-k sweep authorized; only exact upstream `K=4` parity is justified.
