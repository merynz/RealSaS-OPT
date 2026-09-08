# RealSaS — Arachne / RigAnything Skinning Cross-Check V1 — 2026-09-08

**Status:** `CROSSCHECK_CLOSED__SKINTOKENS_REMAINS_PRIMARY__RIGANYTHING_DIRECT_PAIRWISE_ARM_SECONDARY`

## Scope

This cross-check is performed before the real Mage Arachne FIT1 optimizer path.

Frozen upstream reference:

`Isabella98Liu/RigAnything@d03cdb21dd134fa81df6b0947522469db3f78bd2`

License boundary: Adobe Research License / noncommercial research. No upstream RigAnything code is admitted into RealSaS product code. This file records only independently stated architectural/behavioral facts.

## 1. What RigAnything skinning actually does

RigAnything does **not** use a separate skin codec or a common skeleton/skin geometric decoder.

Its released model contains:

- a point tokenizer for surface `P+N`;
- a joint-position tokenizer;
- one hybrid autoregressive Transformer stack;
- joint fusion / parent modules;
- a small `skinning_mlp` with functional shape `2d -> d -> 1`.

During skeleton generation, the Transformer produces contextual next-joint state. After sampling the joint position, the released model fuses:

`generated transformer joint state + joint-index embedding + joint-position token`

into a contextual generated-joint representation used later by skinning.

After the skeleton terminates, skinning is evaluated as:

`contextual surface-point token H_l + contextual generated-joint token T_k -> pairwise skinning MLP -> raw influence logit a_lk`.

The released code concatenates every point token with every generated-joint token and applies the same pairwise MLP independently to each pair.

There is:

- no SkinTokens-style FSQ latent;
- no per-joint field codec;
- no dedicated skin Transformer;
- no final decoder shared with skeleton coordinates/topology.

## 2. What is shared between RigAnything skeleton and skinning

The important sharing is **representation/backbone sharing**, not final-decoder sharing.

- The same point tokenizer and Transformer family contextualize geometry used by skeleton generation and by the skinning route.
- Skinning consumes generated joint representations derived from the autoregressive skeleton process.
- Thus skin prediction has a direct causal dependency on the generated skeleton representation.

For full-mesh skin inference, the released wrapper retains the complete mesh vertices/normals. The model re-encodes those points in chunks through the point tokenizer + Transformer and pairs those point tokens with the already-generated joint tokens.

This is stronger coupling than the current split RealSaS baseline, but it is not evidence that such hidden shared features are *necessary* when a complete explicit `QualifiedSkeletonIR` is available.

## 3. Raw learned output versus released final skin

The neural skin head produces raw point/joint logits. The released inference wrapper then materially transforms them:

1. keep top-5 joint logits per vertex;
2. softmax across retained joints;
3. zero weights below `0.068`;
4. renormalize;
5. mesh-neighbor smoothing for `10` iterations with neighbor factor `0.35`;
6. nearest-neighbor transfer to duplicate/original GLB vertices;
7. renormalize again.

Therefore released final RigAnything skin quality is not attributable to the raw learned pairwise head alone.

The mesh-neighbor smoothing step also uses complete mesh topology unavailable as a hidden product dependency in the observation-grounded RealSaS Arachne contract.

## 4. Training evidence boundary

The paper describes end-to-end skin influence supervision, with per-surface-point softmax-normalized skin weights and cross-entropy-style skinning loss.

The frozen public repository does not expose a complete authoritative published-run training driver/data integration. Exact training recipe details beyond directly published facts remain `NOT_PUBLICLY_VERIFIABLE`.

## 5. Comparison with SkinTokens

### RigAnything

`surface representation + generated joint representation -> direct pairwise scalar logits -> deterministic top-k/softmax/cutoff/smoothing/transfer`.

Strength:

- simple direct skeleton-conditioned predictor;
- shared rig representation may transfer useful semantics into skin prediction.

Weakness for RealSaS clean attribution:

- no measured standalone skin representation ceiling;
- final result materially relies on topology/postprocess;
- complete-mesh skin route is richer than the shipping observation-grounded substrate.

### SkinTokens

`per-joint skin field + geometry -> compact latent/token state -> dedicated geometry-conditioned field decoder`.

Strength for RealSaS clean attribution:

- codec representation ceiling can be established before predictor training;
- same frozen decoder can be used for A0 teacher-codec and A1 predicted-latent paths;
- existing-skeleton skin-only inference is directly supported;
- learned field can be evaluated before Compiler/postprocess.

Therefore:

`PRIMARY_ARACHNE_REFERENCE = SKINTOKENS_FIELD_REPRESENTATION`

remains unchanged.

## 6. RealSaS architectural decisions after cross-check

### Common Geppetto/Arachne final decoder

`REJECTED`

Neither SkinTokens nor RigAnything requires a common final geometric decoder.

### Shared Geppetto/Arachne Transformer/backbone for Mage FIT1

`NOT_REQUIRED_FOR_FIRST_FIT1`

RigAnything provides positive evidence that shared rig representations are useful, but it does not prove they are necessary. The initial causal FIT1 must preserve clean attribution:

`RiggingSurfaceIR + QualifiedSkeletonIR -> Arachne -> frozen SkinFieldCodec decoder -> SkinProposalIR -> Compiler -> verified deformation`.

A future generalization/efficiency ablation may compare:

- split Arachne baseline;
- shared geometry encoder;
- Geppetto latent/context reuse;
- unified joint rig prior.

Those are not authorized dependencies for the first Mage FIT1.

### Direct pairwise point×joint head

`SECONDARY_CAUSAL_ARM`

If the SkinTokens-inspired latent-field predictor fails while the frozen codec itself passes A0, a RigAnything-style direct pairwise head is a justified independent alternative arm.

It must still consume only explicit RealSaS product information and must be evaluated before any topology-dependent smoothing.

### Topology smoothing

`FORBIDDEN_AS_HIDDEN_FIT1_RESCUE`

Compiler/deformation qualification may perform only separately frozen, bounded product-authorized repair. RigAnything's 10-iteration mesh-neighbor smoothing may not be introduced merely to rescue a failed Arachne predictor.

## 7. Final verdict

`SKINTOKENS_PRIMARY_DECISION_UNCHANGED`

`RIGANYTHING_SHARED_REPRESENTATION_EVIDENCE = RETAIN_FOR_FUTURE_ABLATION`

`RIGANYTHING_DIRECT_PAIRWISE_SKIN_HEAD = VALID_SECONDARY_SOLUTION_CLASS`

`COMMON_FINAL_RIG_SKIN_DECODER_REQUIRED = FALSE`

`MAGE_FIT1_NEXT_GATE = MULTIVIEW_TEACHER_W_BINDING`

No Arachne optimizer step is authorized by this cross-check alone.