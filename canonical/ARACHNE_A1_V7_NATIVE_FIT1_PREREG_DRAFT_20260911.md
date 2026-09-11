# RealSaS — Arachne V7-Native A1 FIT1 — Architecture / Preregistration Draft

**Date:** 2026-09-11  
**Branch:** `exp/arachne-a1-v7-native-fit1-20260911`  
**Status:** `A0_K4_FROZEN__A1_SOURCE_AND_EXACT_PREREG_TRANSACTION_OPEN__NO_A1_OPTIMIZER_STEP_YET`  
**Parent audit decision:** `AUDIT_V1_PRE_A1_CLOSED__A0_MAGE_GSA_PASS__K4_MANUAL_INTERFACE_SELECTION`  
**Selected codec interface:** **4 continuous field tokens per joint**.  
**Frozen K4 checkpoint SHA-256:** `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`.  
**A1 supervision bank SHA-256:** `b255a75ae9ff42295547c5f023c63d4781ffd042f06c92a74745b9c7c715211a`.  
**Product PASS:** not claimed.  
**Unseen/generalization PASS:** not claimed.

This file begins the fresh V7-native A1 transaction required after A0 closure. It intentionally does not inherit the old A1 compressed 20D/8D consumer boundary as authority.

## 1. Scientific question

Can a product-time predictor infer the frozen A0 latent skin-field representation from legal shipping evidence only:

`RiggingSurfaceIR S + Compiler-qualified skeleton G -> per-joint K=4 field tokens -> frozen V7 decoder -> dense skin field W*`

on the same Mage FIT1 witness?

A1 is a prediction problem, not a representation-capacity problem. A0 has shown that the selected K=4 codec can represent/decode the Mage skin field at the FIT1 GSA gate with stable-last-3 closure.

## 2. Hard firewall

Predictor neural input may contain only product-time legal evidence derived from admitted observations and qualified state.

### Allowed baseline inputs

From `RiggingSurfaceIR`:

- compact surface position;
- derived normal + validity;
- exact 8-view support mask/state;
- exact per-view raster coordinates + validity where present;
- observed/completed validity state;
- exact admitted local GSA graph;
- legal relation metadata required by the current GSA contract;
- deterministic normalized geometry derived from the same fields.

From `QualifiedSkeletonIR`:

- canonical joint position;
- exact accepted parent graph/root;
- exact joint identity only as within-example structural indexing, never a cross-family semantic shortcut;
- exact sparse `support_surface_ids` mapping as positive mechanical-anchor evidence, not a skin mask;
- deform/assembly semantics that are already qualified and product-time available.

Derived deterministic relations:

- old-A1-style point↔joint / point↔parent-segment geometry remains allowed as an inductive bias;
- skeleton tree distance / parent-relative geometry may be derived from qualified G;
- camera/view-direction features may be derived from admitted exact cameras when bound by the IR contract.

### Forbidden predictor inputs

- teacher skin weights W;
- frozen-A0 teacher latent produced from W;
- hidden/source teacher mesh geometry not available to product inference;
- source bone IDs or source rig names;
- source object/component names such as `Mage_Hat`, `2H_Staff`, etc.;
- family/character identity shortcut;
- raw provenance hashes/filenames as neural features;
- outcome-selected metadata;
- raw Geppetto parent/root alternatives that compete with Compiler canonical authority;
- current IRIS V3 `log_uncertainty` legacy scalar as a baseline feature;
- saturated Geppetto mechanical-salience magnitude as a baseline feature.

Teacher W remains legal only in the **training objective / target lane**.

## 3. Frozen output contract

A1 must predict exactly the selected A0 latent/decode interface:

- joint count: current qualified G cardinality, 22 on Mage;
- field tokens per joint: **K=4**;
- latent channel width: **512**;
- condition-token count: **384**;
- V7 config hash: `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`;
- decoder/checkpoint SHA-256: `8a57d296c55298e941402c18d215ea5a1458863df604d3e088f4fb1d2292b5a7`;
- no FSQ / quantizer;
- no alternate direct-W head in the baseline treatment.

The predictor output is conceptually:

`Z_hat: [J, K=4, 512]`

and the frozen decoder produces scalar field logits/weights for requested geometry.

The K4 token-interface audit established FP32 full-set permutation invariance (`max abs logit delta = 5.7220458984375e-06 <= 1e-05`). Therefore token rank/order is **not semantic authority**. Ordered token-1 -> token-1 latent regression is forbidden as the primary objective.

## 4. Architecture direction — provisional baseline

A1 baseline is **non-autoregressive**.

Rationale: the qualified skeleton/tree already exists before A1; skin-field inference is a conditioned dense relation problem and does not require generating a causal skeleton/skin sequence.

### 4.1 Surface evidence encoder

Use a rich fieldwise tensorization derived from the successful promoted GSA->Geppetto principle rather than flattening S into a historical 20D summary.

The encoder must preserve:

- node fields separately;
- exact local graph connectivity;
- per-view/raster evidence;
- validity/observed-completed state.

A concrete implementation may combine local relation-aware message passing with global surface self-attention, but changing the legal information boundary is forbidden.

### 4.2 Skeleton encoder

Encode the exact qualified tree with:

- joint position/state;
- parent/root structure;
- tree-relative mechanical context;
- sparse support-anchor identity as a typed relation.

Compiler-selected topology is immutable input authority. The network does not re-parent or re-root the skeleton.

### 4.3 Joint-field token queries

Instantiate `J × K` conditioned field-token queries. On Mage this is only `22 × 4 = 88` joint-field tokens.

Because this cardinality is small, baseline fusion should use **dense joint-field reasoning**, not premature factorization.

The four token slots may use learned symmetry-breaking query prototypes, but their numerical slot order carries no semantic authority because the frozen decoder is full-set permutation invariant in FP32.

### 4.4 Dense surface↔joint-field fusion

Baseline should permit every joint-field token to attend to the complete encoded surface memory, with relation bias from legal point↔joint/segment geometry and exact support-anchor relations.

A limited refinement may model joint competition and shared boundaries without hard nearest-bone masking.

Do not impose a hard nearest-joint/top-k surface mask in the baseline.

### 4.5 Token projection

Project final joint-field states into the exact frozen V7 K=4 latent token space consumed by the A0 decoder.

No decoder retraining in baseline A1 FIT1 unless separately preregistered as an ablation.

## 5. Feature-admission policy

The audit established the rule:

`preserve at boundary != force neural consumption`.

Therefore the rich legal S+G boundary remains intact even if the first baseline excludes a candidate channel.

### Baseline ON

- surface position/normal/validity;
- exact support-view/raster evidence;
- exact GSA graph;
- qualified joint positions/tree;
- deterministic point↔joint/segment geometry;
- exact sparse joint↔surface support-anchor mapping as soft relation evidence.

### Baseline OFF / ablation-only

- Geppetto `position_sigma_normalized` sidecar until calibration/value is demonstrated;
- current IRIS legacy `log_uncertainty`;
- Geppetto salience probability;
- explicit source component labels;
- raw rejected parent/root alternatives.

## 6. Training target hierarchy — now constrained by K4 interface audit

Teacher W is never fed to the predictor, but may define training targets.

Baseline objective must be behavior-first:

1. **decoded scalar-field reconstruction** through the frozen K4 decoder;
2. **post-normalization row competition loss** over all qualified joints, so A1 receives gradient on the coupled skin row rather than independent marginals only;
3. **hard-tail / blend-boundary behavior** may be added prospectively if frozen before optimizer construction;
4. **ordered latent alignment is disabled** because K4 token rank is not semantic;
5. any future latent-set alignment must be permutation-aware and auxiliary only;
6. deformation consequence loss remains off until a parent-relative articulated, joint-permutation-consistent probe is separately frozen.

The current synthetic translation deformation probe remains evaluation/diagnostic only.

## 7. A1 FIT1 evaluation planes

A1 FIT1 must report separately:

- Mage GSA950 decoded skin-field p95;
- Mage GSA950 qualified/normalized row error;
- Mage GSA950 deformation consequence under the declared diagnostic probe;
- disjoint-surface transfer as a diagnostic co-metric, not an unseen-character claim;
- Compiler `qualify_skin()` legality/correction accounting;
- rigid-attachment regional diagnostics for head/hat/book/staff where product-legal evaluation masks are available;
- exact lineage hashes for S, G, frozen codec and A1 checkpoint.

A1 PASS does not imply PRODUCT_PASS.

## 8. Mandatory equivariance / shortcut checks

Before claiming A1 FIT1 closure:

- surface-node permutation equivariance/invariance as appropriate;
- joint-order permutation equivariance with canonical structure remapped consistently;
- view-order/camera binding consistency under allowed view permutation;
- support-anchor identity remapping consistency;
- teacher-W predictor-input firewall assertion;
- source-ID/component-name predictor-input firewall assertion.

## 9. Old A1 reuse policy

`models/arachne/v2/` is a **mechanism reference**, not the new authority.

Reusable ideas:

- surface memory;
- joint queries;
- parent context;
- explicit 10D point↔joint/segment geometry;
- frozen codec decode concept;
- consequence-aware training/evaluation idea.

Not reusable as final authority:

- historical 20D surface summary;
- historical 8D joint summary;
- support-count scalar replacing exact support-anchor identity;
- raster-count replacing exact per-view raster coordinates;
- old latent width/interface assumptions.

## 10. Parameter-count policy

No claim of minimal or optimal A1 capacity is made in FIT1.

The old ~0.55M predictor is known to be small and information-compressed, but the audit does not prove that raw parameter count was its dominant weakness. The new baseline must first remove the conditioning-boundary confound. A later controlled capacity ablation may compare parameter scales using the **same rich conditioning contract**.

## 11. Execution gate

### A0 interface

**CLOSED / FROZEN.**

Authority record: `canonical/ARACHNE_A0_V7_K4_CLOSURE_FREEZE_20260911.md`.

### A1 design / implementation work

**AUTHORIZED.**

### A1 optimizer training

The frozen K4 artifact prerequisite is satisfied. The remaining blocker is now only the exact A1 source/config/objective/evaluation preregistration transaction. No optimizer object may be constructed before that record binds the final implementation hashes and training constants.

## 12. Current transition statement

`A0 Mage/GSA representation gate is closed and frozen at K=4; larger token count did not establish a meaningful practical gain; K=4 is manually selected for parsimony/continuity; the K4 token set is semantically unordered under FP32 full-set permutation audit; the pre-A1 information audit is closed; V7-native A1 implementation is authorized; A1 optimizer remains fail-closed only until the exact V7-native A1 source/config/objective/evaluation preregistration is sealed.`
