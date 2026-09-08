# RealSaS — Arachne / SkinTokens Clean-Room Mapping V2 — 2026-09-08

**Status:** `REFERENCE_MAPPING_CLOSED__FIT1_DESIGN_NOT_YET_OPTIMIZER_AUTHORIZED`

## 0. Why V2 exists

The earlier 2026-08-31 clean-room audit correctly extracted SkinTokens as the primary external representation reference for Arachne, but Geppetto had not yet closed on the real Mage shipping witness.

That boundary condition has now changed:

- Geppetto Mage FIT1 is sealed `FIT1_TERMINAL_PASS`;
- the consumer skeleton input for Arachne is now a real Compiler-qualified `QualifiedSkeletonIR` with `22` controls and exactly one deform root;
- Arachne no longer needs any mechanism that owns skeleton cardinality, root, topology or canonical IDs.

SkinTokens upstream is unchanged from the frozen audited snapshot:

`VAST-AI-Research/SkinTokens@273b691d35989d71cd17ff2895fdc735097b92d1`.

This document therefore updates the architecture mapping, not the upstream facts.

## 1. What SkinTokens actually does

SkinTokens contains **two different learned levels** that must not be conflated.

### 1.1 Skin codec / field decoder

For each bone/joint influence field, an FSQ-CVAE compresses sparse skin weights conditioned on mesh geometry into a short discrete latent/token state.

Public-code decomposition:

`per-bone weight field + geometry -> skin encoder -> latent -> FSQ -> SkinTokens`

and

`SkinTokens + geometry condition -> dedicated skin-field decoder -> scalar influence field over queried mesh points`.

The public decoder ends in a sigmoid and therefore predicts each bone field independently in `[0,1]` before later assembly/normalization/postprocessing.

This decoder is a **skin decoder only**. It does not decode skeleton coordinates or hierarchy.

### 1.2 Unified autoregressive TokenRig model

TokenRig then uses a separate decoder-only causal Transformer (Qwen3-0.6B configuration) to generate one rig sequence:

`skeleton tokens -> skeleton delimiter/eos -> J * T_D skin tokens -> final eos`.

The same causal Transformer predicts both skeleton-token vocabulary and skin-token vocabulary. A deterministic logits processor switches the allowed vocabulary after the skeleton is complete and knows how many skin tokens are required from the generated skeleton cardinality.

After generation:

- skeleton tokens are decoded by the deterministic skeleton tokenizer/detokenizer;
- skin tokens are converted through FSQ codes and decoded by the **separate frozen FSQ-CVAE skin decoder**.

Therefore the accurate statement is:

> SkinTokens shares an **autoregressive token predictor**, not a final skeleton/skin geometric decoder.

### 1.3 Existing-skeleton path

The released inference path explicitly supports using an existing skeleton. In that mode the existing skeleton is serialized as the prefix and the autoregressive model continues with skin tokens.

Thus the SkinTokens solution class does **not** require the skin decoder to own skeleton generation.

## 2. Shared-vs-split architecture decision for RealSaS FIT1

### Decision A — common Geppetto/Arachne final decoder

`REJECTED_FOR_FIT1`

Reason:

- SkinTokens itself provides no such mechanism: skeleton detokenization and skin-field decoding are separate;
- Geppetto already has a sealed real FIT1 solution and final graph authority remains Compiler-owned;
- forcing a common final decoder would reopen solved ownership boundaries without external-reference justification;
- it would confound attribution in Arachne FIT1.

### Decision B — one shared causal Transformer for Geppetto + Arachne

`DEFERRED_FUTURE_JOINT_MODEL_ARM`

SkinTokens provides evidence that a shared autoregressive prior can model dependencies from the generated skeleton into skin tokens. It also claims benefits from unified rig modeling.

However this is **not required for Arachne FIT1**. The current fit experiment must first answer the narrower causal question:

> Given the already-qualified correct skeleton and admitted surface evidence, can a learned Arachne infer a valid deformation field?

A later generalization experiment may compare:

- split Geppetto/Arachne;
- shared surface backbone;
- shared causal rig prior / joint fine-tuning.

No such sharing is authorized before the split FIT1 baseline closes.

### Decision C — shared surface encoder

`OPTIONAL_LATER_ABLATION__NOT_REQUIRED_FOR_FIT1`

SkinTokens shares/globalizes mesh conditioning inside TokenRig, but the Arachne FIT1 experiment should preserve clean attribution. Arachne gets the complete product-level information available at its boundary through its own deterministic conditioning adapter.

A frozen Geppetto surface representation may later be tested as an efficiency/generalization arm, never as hidden required evidence for the initial Arachne claim.

### Decision D — same skin decoder for codec ceiling and predictor path

`MANDATORY`

This is the important decoder-sharing rule for Arachne.

The exact same qualified SkinFieldCodec decoder object/config must be used in:

`A0: teacher W -> codec encoder -> latent -> SAME DECODER -> W*`

and

`A1/FIT1: shipping conditioning -> Arachne predicted latent -> SAME FROZEN DECODER -> W_hat`.

No decoder substitution, retraining or special A1 reconstruction head may hide a representation seam.

## 3. A material representation difference that must be decided explicitly

SkinTokens reference factorization:

- each bone influence is an independent scalar field;
- decoder output is bounded per field by sigmoid;
- the dense `N x J` matrix is assembled afterward;
- downstream code can normalize skin rows and optionally transfer/smooth sampled skin onto the original mesh.

Current RealSaS `SkinFieldCodecV1` factorization:

- one latent per qualified joint;
- all joint fields are decoded jointly;
- row-softmax produces a simplex distribution directly inside the codec.

This is **not** a source violation, but it changes responsibility.

Compiler already owns legal skin simplex/reference policy. Therefore Arachne FIT1 must not quietly rely on an architectural simplex hard-constraint and then claim that Compiler validated an unconstrained influence proposal.

### Frozen design choice for the next candidate

`REFERENCE_STRENGTH_FIELD_MODE = INDEPENDENT_NONNEGATIVE_PER_JOINT_FIELDS`

Preferred clean factorization:

1. each joint latent produces independent nonnegative `[0,1]` influence evidence per admitted surface node;
2. Arachne emits these raw dense influence scores in `SkinProposalIR`;
3. raw proposal quality is evaluated before qualification;
4. Compiler owns bounded sparsification / simplex normalization / rejection;
5. Compiler correction magnitude is reported and must remain small.

The existing row-softmax `SkinFieldCodecV1` remains valid historical evidence that compact latent fields can represent skin, but it is not automatically frozen as the final Arachne FIT1 decoder merely because A0 synthetic capacity previously passed.

A controlled source-level comparison must decide whether to retain row-softmax or move to independent sigmoid fields **before optimizer step 1 of the final Mage FIT1 predictor run**.

## 4. Functional obligations retained from SkinTokens

Arachne FIT1 must retain these implementation-independent obligations:

1. geometry-conditioned per-qualified-joint influence fields;
2. explicit conditioning on the exact `QualifiedSkeletonIR` lineage;
3. compact field representation with a measured codec reconstruction ceiling;
4. active/sparse deformation regions must receive nontrivial supervision;
5. decoder must answer on every admitted `RiggingSurfaceIR` node;
6. no hidden full-mesh completion or teacher-only skin feature reaches inference;
7. prediction emits only `SkinProposalIR` / influence evidence;
8. Compiler owns final legal skin references/simplex policy;
9. final qualification includes deformation/motion-sensitive proof, not static weight error alone.

## 5. Mechanisms that remain reference-supported but not mandatory

The following SkinTokens choices are **not** frozen RealSaS requirements:

- FSQ specifically;
- exact codebook levels;
- exact number of SkinTokens per joint;
- Qwen3-0.6B;
- unified skeleton+skin autoregression;
- skeleton tokenizer grammar/order;
- nested-dropout exact schedule;
- exact BCE/MSE/Dice coefficients;
- GRPO or its exact reward recipe;
- full-mesh nearest-neighbor skin transfer;
- reference postprocessing.

## 6. Product boundary for Arachne FIT1

Frozen inference boundary:

`RiggingSurfaceIR + QualifiedSkeletonIR`

Deterministic adapter may use only information derivable from those objects, including:

### Surface lane

- admitted surface 3D positions;
- valid normals;
- exact GSA local geometric relations;
- support/view provenance;
- observed/completed status if present in admitted product IR;
- raster/view bindings if present in admitted product IR;
- deterministic normalization.

### Skeleton lane

- canonical joint IDs for deterministic serialization/binding only;
- qualified joint positions;
- exact parent indices/tree;
- root indicator;
- skeleton depth / deterministic graph-derived features;
- support-surface bindings if qualified and product-visible;
- skeleton lineage hash.

Forbidden at inference:

- source bone IDs/names as semantic hints;
- teacher skin weights;
- source mesh vertices/faces outside admitted product geometry;
- helper/IK controls removed from the 22-joint qualified core;
- teacher active-region masks;
- hidden full-mesh completion.

## 7. Training-only teacher lane

Mage has authoritative dense source skin truth. For Arachne training this may be projected offline onto the exact admitted product surface and the exact qualified 22-joint identity order.

Required target object:

`SkinFieldTeacherTarget(surface_ids, canonical_joint_ids, W_teacher)`

Required invariants:

- exact surface-ID alignment, never row-index assumption;
- exact source-bone -> qualified-joint provenance mapping sealed before optimizer;
- teacher rows finite/nonnegative and simplex;
- helper/IK/assembly-only source bones excluded unless explicitly mapped into an admitted qualified control;
- every admitted surface row has target coverage or is explicitly fail-closed;
- no teacher-only field becomes inference conditioning.

## 8. Arachne FIT1 experimental decomposition

The experiment must remain staged so a failure is interpretable.

### A0-MAGE — codec ceiling

Question:

> Can the frozen candidate skin-field representation reconstruct the Mage 22-joint teacher field on the exact admitted GSA surface and survive Compiler + deformation proof?

Input to encoder may include teacher W because this is a representation ceiling. Decoder sees only candidate latent + product conditioning.

PASS is required before predictor training.

### A1-MAGE — predictor ceiling / FIT1

Question:

> Can Arachne infer the needed per-joint field latent from only `RiggingSurfaceIR + QualifiedSkeletonIR` and reproduce a deformation-valid skin field on Mage?

Pipeline:

`shipping S + frozen qualified G`
` -> Arachne predictor`
` -> frozen A0-qualified decoder`
` -> raw dense influence proposal`
` -> SkinProposalIR`
` -> Compiler.qualify_skin`
` -> QualifiedSkinIR`
` -> verified deformation/motion probes`.

No teacher W is visible to the predictor at inference/evaluation.

## 9. Evaluation families required for FIT1

Static metrics alone are insufficient.

Required layers:

### Raw field fidelity

- row L1 / p95;
- active-support recall/precision or equivalent sparse-field measure;
- dominant-joint accuracy;
- nonfinite/negative checks;
- raw row-sum distribution if independent field decoder is used.

### Compiler qualification

- exact surface coverage;
- exact skeleton lineage;
- illegal refs = 0;
- correction magnitude reported;
- correction must remain bounded and too small to rescue a materially bad raw field.

### Deformation proof

Use verified LBS or the product-equivalent deformation operator with preregistered joint probes.

At minimum include:

- individual hinge/bend probes on limbs;
- parent-child compound rotations;
- symmetric left/right probes;
- torso/root motion;
- a combined stress pose;
- deformation error relative to teacher-skinned motion;
- finite/no-explosion checks;
- local smoothness / fold or stretch diagnostics appropriate to the admitted surface.

FIT1 success means the learned field behaves correctly under motion, not merely that W numerically resembles the teacher matrix.

## 10. Current historical RealSaS evidence and what it means

Historical A0 work already established that the larger shipping/default continuous `SkinFieldCodecV1` (`192 hidden / 64 latent / 3 encoder / 3 decoder`) could represent the frozen heterogeneous synthetic field panel when trained with cosine cooling. This falsified a generic representation-capacity bottleneck for that synthetic panel.

That result is **supporting evidence only** for Mage FIT1 because:

- it was not the real Mage admitted surface/22-joint target;
- the current codec uses row-softmax rather than the cleaner independent-field factorization of SkinTokens;
- it does not prove the Arachne predictor can infer the latent from product conditioning.

Historical ArachneCandidateV2 (`128 model dim / 2 surface encoder layers / 4 heads / 384 FF`) is likewise a candidate starting point, not a frozen final Mage FIT1 architecture.

## 11. Decisions frozen by this mapping

- `SKINTOKENS_PRIMARY_ARACHNE_REFERENCE = TRUE`
- `SKINTOKENS_UPSTREAM_COMMIT = 273b691d35989d71cd17ff2895fdc735097b92d1`
- `COMMON_FINAL_GEPPETTO_ARACHNE_DECODER = FALSE`
- `SHARED_GEPPETTO_ARACHNE_CAUSAL_LM_FOR_FIT1 = FALSE`
- `SHARED_CAUSAL_RIG_PRIOR_FUTURE_ARM = ALLOWED`
- `SAME_SKIN_DECODER_A0_A1 = REQUIRED`
- `QUALIFIED_SKELETON_INPUT = REQUIRED`
- `GEPPETTO_REOPEN_FOR_ARACHNE_FIT1 = FORBIDDEN`
- `TEACHER_W_AT_ARACHNE_INFERENCE = FORBIDDEN`
- `COMPILER_FINAL_SKIN_AUTHORITY = REQUIRED`
- `DEFORMATION_PROOF = REQUIRED`
- `GENERALIZATION_CLAIM = FALSE`
- `OPTIMIZER_STEP_1_AUTHORIZED = FALSE`

## 12. Immediate next gate before implementation freeze

Perform one source-only controlled design closure:

`D0_DECODER_FACTORISATION`

Compare, without Mage predictor training:

- Arm S: current joint-coupled row-softmax decoder;
- Arm I: independent per-joint nonnegative/sigmoid field decoder with Compiler-owned simplex normalization.

Both arms must use the same Mage teacher projection, surface/skeleton conditioning and deformation harness.

Decision criterion is not mean reconstruction alone. Prefer the arm that:

- reaches the required static/deformation ceiling;
- requires negligible Compiler correction;
- preserves sparse local influence structure;
- maintains clean learned-vs-Compiler ownership;
- avoids unnecessary coupling to joint count/order.

After `D0` closes, freeze the final codec/decoder and preregister `A0-MAGE`, then `A1-MAGE/FIT1`. No final Mage Arachne optimizer run before these seals.