# RealSaS — FIT1 Evidence Index — 2026-09-09

This file is the short audit path for anyone evaluating whether the current Mage FIT1 work is real, reproducible in lineage, and correctly scoped.

Machine-readable companion:

`canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`

Local consistency test:

```bash
python -m pytest -q tests/models/test_geppetto_fit1_evidence_manifest_v1.py
```

## Current claim status

| Subsystem / gate | Status | Strongest supported claim |
|---|---|---|
| IRIS/GSA upstream witness | **FROZEN INPUT AUTHORITY** | promoted scene-first signed IRIS V3 witness + deterministic GSA substrate are hash-bound downstream FIT1 inputs; teacher mesh is not used at inference |
| Geppetto | **FIT1 TERMINAL PASS / PROMOTED FROZEN SOURCE** | the reference-strength formulation fits and stably maintains the Mage 22-control mechanical core under the preregistered shipping-faithful gate |
| SkinFieldCodec A0 | **OPEN / EXPERIMENTAL** | representation/decode ceiling is still being solved; V7/C4 is research evidence only |
| Arachne A1 | **NOT AUTHORIZED / NOT PROMOTED** | no current V7-native learned `S+G -> latent -> W` model has been frozen |
| Full end-to-end FIT1 product | **NOT CLOSED** | no `PRODUCT_PASS` claim exists yet |

## IRIS/GSA upstream witness

Current Mage signed-geometry witness:

`models/iris/v3/PROMOTED_MAGE_FIT_WITNESS_V1.json`

Current signed-field source:

`models/iris/v3/scene_first_signed_v3.py`

IRIS V2 remains the promoted observation/foundation/evidence base package under `models/iris/v2/`; V3 is the later promoted scene-first signed-field composition used by this FIT1 witness. Deterministic GSA remains the owner of compaction, local geometry, provenance and final `RiggingSurfaceIR` assembly.

Frozen upstream identities and witness facts:

- IRIS checkpoint SHA-256: `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2`
- signed zero-surface SHA-256: `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`
- product inference inputs: `8x1024_RGBA + 8xORTHOGRAPHIC_CAMERA`
- teacher mesh used at inference: `false`
- CharacterGen runtime used: `false`
- zero-bracket rate: `0.9091796875`
- predicted-to-full-truth p95 normalized: `0.019385051340437998`
- visible-depth overall p95 normalized: `0.0464760137`
- max single-view visible-depth p95 normalized: `0.0719296063`
- minimum silhouette IoU: `0.946365`
- minimum silhouette recall: `0.971979`

These are Mage FIT witness/evaluation facts, not unseen-family generalization.

## Geppetto: exact proof chain

Scientific preregistration:

`experiments/geppetto_reference_strength_fullstack_v1/GEPPETTO_REFERENCE_STRENGTH_FIT1_PREREG_V1.md`

Frozen execution companions:

- `experiments/geppetto_reference_strength_fullstack_v1/GEPPETTO_REFERENCE_STRENGTH_APPARATUS_FREEZE_V1.md`
- `experiments/geppetto_reference_strength_fullstack_v1/GEPPETTO_REFERENCE_STRENGTH_LOSS_FREEZE_V1.md`
- `experiments/geppetto_reference_strength_fullstack_v1/run_geppetto_reference_strength_fit1_v1.py`

Scientific closure:

`canonical/GEPPETTO_REFERENCE_STRENGTH_FIT1_CLOSURE_20260908.md`

Promotion/refreeze decision:

`canonical/GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md`

Frozen source / seal lineage:

- optimizer/source commit: `f7be46f0a97df62a793ebf91b22297c894854f39`
- seal commit: `ae0af0cd39dd2468a012ba21890a4fed2da7c4c9`
- promoted mainline source: `models/geppetto/reference_strength_v1/`
- architecture id: `RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion.DeterministicViewDirection.v1`

Terminal result:

- closure optimizer step: `14080`
- required/observed terminal streak: `48/48` full structural checks
- terminal stability span: `3072` optimizer steps
- qualified controls: `22`
- qualified deform roots: `1`
- free-running teacher feedback: `false`
- diffusion evaluation seeds: `11, 23, 47, 89`

Artifact hashes:

- target content SHA-256: `0b5a25c877116de60b710b7bb2a7848f30988e1622e2eda8084cad21c8ca23c9`
- qualified skeleton IR SHA-256: `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`
- final checkpoint SHA-256: `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30`
- final result JSON SHA-256: `728f5b5fe9e98865dd38c907ef19a741c57606f0e15557a40095d81144dc2045`
- signed zero-surface SHA-256: `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`
- upstream IRIS checkpoint SHA-256: `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2`

## External run bytes / artifact locators

Large result bytes are stored outside Git and hash-bound in the machine evidence manifest. Current Google Drive locators:

- scientific contract folder: `12KG1v1WQ_1o9KAns7SobftJvSzZl1Fs8`
- sealed package manifest: `1XyD8wgvhkGAU7nSWdsQWtJCeMchjTzeW`
- seal document: `1ZnUdt1gt6tyVQ9mTGT2fDtRkYoYg9xkX`
- run folder: `1ZTm47BL6ByfHhN8HdRExxzZCC7kSndkZ`
- final checkpoint: `1ZCbqD52-5JhsHEUxEdxs99oRHgg0l01-` — `46,014,477` bytes — SHA-256 `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30`
- final result JSON: `1w64sr58t4JNctknp9KZPsxqyXxyL5jED` — `1,921,262` bytes — SHA-256 `728f5b5fe9e98865dd38c907ef19a741c57606f0e15557a40095d81144dc2045`
- final QualifiedSkeletonIR: `1N-UsUnbqfxUUCVUYFHVsvL18ieDX3lCM` — `16,604` bytes — SHA-256 `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`
- final SkeletonProposalIR: `1hs160Md_zpH3FYPKfTjvAPx1ywLvNysH` — `246,381` bytes — SHA-256 `91147822d1e24c2d5d8600eea5c60543eb1c0584ad2fca9baca952fe788ded26`
- final diffusion seed reports: `1hxVWR_0TNEKrQRGJDHITgbMfnNSkZ3Ff` — `1,176,731` bytes — SHA-256 `606a66137d34cb362ea58793228b729ccd56274fd6acfc99102ff42cc428526c`

Drive IDs are locators, not proof by themselves. The authority is the combination of exact source/prereg/result lineage plus content hashes; a reviewer with artifact access can independently download and hash the bytes.

Canonical audit/demo visuals are also locator-bound in the manifest, while the old perspective 3D debug image is explicitly retired as scientific visualization evidence.

## What Geppetto actually computes

Current frozen learned route:

`8 raster views + exact cameras -> IRIS signed evidence -> deterministic GSA/RiggingSurfaceIR -> lossless Geppetto surface tensorization -> full-surface/reference-strength Geppetto -> SkeletonProposalIR -> Compiler exact graph qualification -> QualifiedSkeletonIR`

The frozen Geppetto contains direct full-surface evidence encoding, exact GSA-relation message passing, global transformer memory, prediction-only causal recurrence, per-step full-surface cross-attention, conditional residual diffusion, native STOP/count, soft internal parent feedback, and all-pairs final parent evidence. Canonical IDs and the final legal tree are not neural outputs; the Compiler owns those decisions.

## Arachne: current honest state

No Arachne source is promoted by the Geppetto transaction.

Current research branch:

`exp/arachne-skintokens-cleanroom-fit1-20260908`

Current experiment scope:

**Mage A0 FIT1 SkinFieldCodec only.**

Current V7 apparatus:

- architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
- parameters: `278,010,880`
- no FSQ in the active learning path
- current C4 prereg: `experiments/arachne_skintokens_fit1_20260908/V7_C4_SKINTOKENS_FACE_BARYCENTRIC_BIASED_DENSE_SUPERVISION_PREREG_20260909.md`
- C3 point-cloud/importance-corrected boundary treatment did not close A0
- A1 remains unauthorized until A0 closes and the V7-facing latent/decode interface is frozen

The older mainline `models/arachne/v2/` source is a prior scaffold/current source inventory item, not evidence that the present V7-native A1 has been trained or promoted.

## Non-claims

FIT1 is a controlled same-witness scientific/product-capability gate. It is **not** unseen-character or unseen-family generalization.

A Geppetto FIT1 PASS is not Arachne PASS. Arachne A0 PASS would not automatically be A1 PASS. Full learned skinning closure would still require A1. Even that would not automatically be `PRODUCT_PASS`; an independent end-to-end product contract remains required.
