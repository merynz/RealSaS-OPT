# RealSaS E0 — Downstream Adapter / OOD Contract V1

**Date:** 2026-08-27  
**Status:** `INTERFACE_AUDIT_FROZEN__PRETRAINED_CONSUMERS_DIAGNOSTIC_ONLY_UNTIL_PROVENANCE_CLEAN`  
**Scope:** Geppetto/Arachne downstream-consumer boundary for the E0 observable-surface study.

## 1. Why this contract exists

RealSaS IRIS emits an observation-supported surface, not automatically the same shape distribution consumed by existing 3D auto-riggers. E0 must separate three causes that otherwise alias:

1. **observable-surface information loss** — visible union lacks geometry a rigging task actually needs;
2. **SurfaceBuilder persistence loss** — observable geometry is sufficient, but deterministic cross-view persistence damages it;
3. **consumer OOD / adapter mismatch** — observable geometry is sufficient, but a downstream model trained on closed/full mesh samples expects a different coverage, sampling density or normalization distribution.

No pretrained downstream result is allowed to answer (1) by itself.

## 2. Frozen E0 geometry arms

All arms use the same family membership and downstream evaluator.

### E0-0 — full-mesh consumer-native ceiling

A closed/full source mesh is sampled through the target consumer's native preprocessing contract. This arm may use source mesh geometry because it is an upper-bound / distribution-reference control, not an IRIS product input.

Purpose: establish the downstream consumer's expected geometry distribution and the maximum attainable result under that distribution.

### E0-a — observable visible-union + oracle persistence

Only geometry actually supported by A×8 is admitted. Cross-view physical persistence is granted with teacher surface identity after the P-only anchor sampler has frozen the anchor rows.

Purpose: isolate the penalty of **coverage + raster-derived sampling + observable-only normalization** from persistence error.

### E0-b — observable visible-union + deterministic persistence

Same exact visible P and point budget as E0-a; persistence is built without teacher identity from deterministic SurfaceBuilder evidence.

Purpose: isolate persistence loss:

`E0-a -> E0-b`.

Therefore:

```text
E0-0 -> E0-a = observable-surface / distribution gap
E0-a -> E0-b = persistence / SurfaceBuilder gap
```

## 3. Primary information-sufficiency consumer

The **primary E0 causal gate** remains a from-scratch, matched-capacity research consumer derived from the S0 fixed-probe discipline.

Reason: a pretrained rigging model confounds substrate sufficiency with its training distribution and may have pretrained-data overlap with RealSaS panels.

The primary comparison keeps fixed:

- model capacity;
- initialization rule / seeds;
- optimizer and update budget;
- point count presented to the probe;
- target/evaluator;
- train/calibration/qualification membership;
- geometry-arm adapter output dimensionality.

Only the geometry arm changes.

### Primary interpretation

```text
scratch E0-0 PASS, scratch E0-a FAIL
    -> observable-only substrate loses downstream-required information

scratch E0-a PASS, scratch E0-b FAIL
    -> SurfaceBuilder persistence is the blocker

scratch E0-a ~= E0-0 and scratch E0-b ~= E0-a
    -> observable substrate + deterministic persistence are information-sufficient
```

## 4. Secondary consumer-OOD diagnostic

After primary sufficiency is resolved, a pretrained consumer may be evaluated in a separate diagnostic matrix:

```text
stock pretrained consumer      × E0-0 / E0-a / E0-b
observable-adapted consumer    × E0-a / E0-b   [only if legally/technically available]
```

Interpretation:

```text
scratch observable PASS
stock pretrained observable FAIL
adapted observable PASS
    -> consumer OOD / adapter mismatch

scratch observable FAIL
    -> do not rescue the substrate by finetuning a consumer and call it sufficient
```

A stock-vs-adapted comparison is therefore an **OOD diagnosis**, not the definition of information sufficiency.

## 5. RigAnything verified inference contract

Verified against the public 2026-08-27 upstream inference release:

- accepts a mesh (`.glb` / `.obj`) at the CLI;
- converts Blender-world points into its model frame by the released matrix, giving row-vector mapping approximately `(x,y,z) -> (x,z,-y)`;
- samples **1024 surface points** with `trimesh.sample.sample_surface_even`;
- uses the corresponding **face normals**, normalized to unit length;
- shape-tokenizer config declares `in_channels=6`, `n_points=1024`;
- computes normalization center from the sampled-point axis-aligned bbox: `(max(points)+min(points))/2`;
- subtracts that center from both sampled and full mesh points;
- divides by `max(abs(sampled_points))` and applies the same scale to full mesh points;
- additionally carries full mesh vertices + vertex normals for later rig/skinning use.

### Consequence for RealSaS

Feeding an A×8 visible union into the stock checkpoint changes at least:

- surface coverage;
- sampling density measure;
- normal distribution / back-face availability;
- bbox center and scale;
- full-pointcloud distribution.

That is a real OOD shift and must not be mistaken for substrate insufficiency.

### Legal/product status

The public RigAnything release is under the Adobe Research License, noncommercial research only. RealSaS source registry therefore retains RigAnything as `REJECT_PRODUCT_TRAINING`. Any use is research-diagnostic only unless rights change.

The public repository is inference-oriented; an official observable-only finetuning pipeline is not currently treated as available authority. Do not preregister “cheap finetune” as an executable product route without an independently implemented and legally admissible training path.

## 6. SkinTokens / TokenRig verified model boundary

Verified against the public 2026-08-27 upstream release:

- core `TokenRig.generate` consumes `vertices: [N,3]` and `normals: [N,3]`;
- it concatenates them into a 6D per-sample condition;
- mesh conditioning uses the same vertices/normals;
- an existing skeleton can be supplied through `skeleton_tokens` as the autoregressive prefix;
- dataset processing exposes transformed `sampled_vertices` and `sampled_normals` to the model;
- exact release-checkpoint sampler count / normalization transform remains checkpoint-transform-config dependent and is **not yet frozen in this document**.

### Consequence for RealSaS

SkinTokens is structurally easier to use as an Arachne-side diagnostic because Geppetto's skeleton can be supplied as a prefix while the skin decoder is conditioned on surface samples. This does not remove the need to match the checkpoint's sampler/normalization distribution.

### Training provenance

The recommended TokenRig checkpoint is publicly described as trained on:

- ArticulationXL 2.0 — 70%;
- VRoid Hub — 20%;
- ModelsResource — 10%;
- then GRPO refinement.

RealSaS master registry includes ArticulationXL2 and RigXL/VRoid-linked sources. Therefore pretrained TokenRig evaluations on those source lineages are **not eligible as pristine RealSaS holdout evidence** without exact source-identity exclusion.

## 7. Pretrained contamination firewall

RealSaS-native scratch training/evaluation splits remain valid for RealSaS models. The following restriction applies only when a pretrained external consumer is introduced.

Each asset evaluated with an external pretrained consumer receives one state per consumer:

- `PRETRAIN_CLEAN_PROVEN` — source identity is proven absent from external pretraining;
- `PRETRAIN_CONTAMINATED_PROVEN` — exact/source-linked identity is known present;
- `PRETRAIN_CONTAMINATION_UNKNOWN` — external training membership cannot be resolved.

Only `PRETRAIN_CLEAN_PROVEN` may support an external-consumer generalization claim.

`UNKNOWN` may be used for engineering/OOD diagnostics, never for a clean-holdout claim.

### RigAnything specific

The paper states training used RigNet plus **9,686 curated rigged Objaverse shapes**. The public inference repository/checkpoint does not currently expose an authoritative 9,686-ID membership list in the audited release. Thus an Objaverse-origin RealSaS asset is `PRETRAIN_CONTAMINATION_UNKNOWN`, not automatically contaminated and not clean.

### SkinTokens specific

Assets whose RealSaS provenance maps to ArticulationXL2, VRoid/RigXL, or ModelsResource lineage are contamination-risk by construction for the recommended pretrained checkpoint and cannot support a pristine pretrained-consumer holdout claim without exact split provenance proving exclusion.

## 8. Adapter output contract RealSaS must freeze

SurfaceBuilder must expose consumer adapters without changing canonical RiggingSurface authority:

```text
RiggingSurface (canonical, observation-grounded)
    |
    +-- Adapter/RigAnythingResearch
    |     xyz[1024,3]
    |     normal[1024,3]
    |     consumer frame transform
    |     consumer center/scale
    |     sampling provenance
    |     coverage mask / observable flag (side metadata; stock model does not consume it)
    |
    +-- Adapter/TokenRigResearch
          vertices[N,3]
          normals[N,3]
          consumer transform metadata
          optional skeleton_tokens prefix
          sampling provenance
```

Adapters are representations for consumers; they do not redefine IRIS truth.

## 9. E0 execution order amendment

```text
A. finish E0 persistence apparatus and calibration
B. construct E0-0 full-mesh consumer-native ceiling
C. run primary scratch matched probes on E0-0 / E0-a / E0-b
D. freeze E0 information-sufficiency decision
E. only then run pretrained stock-consumer OOD diagnostics on provenance-eligible assets
F. observable-adapted pretrained diagnostic only if training implementation + license + clean split are established
G. only after E0 closure proceed to correspondence C1/C2/C3
```

This order prevents a pretrained model's domain shift or memorization from rewriting the observable-substrate contract.
