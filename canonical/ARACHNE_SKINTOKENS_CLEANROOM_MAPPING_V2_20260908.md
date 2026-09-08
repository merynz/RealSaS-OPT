# RealSaS — Arachne / SkinTokens Clean-Room Mapping V2 — 2026-09-08

**Status:** `REFERENCE_MAPPING_CLOSED__MAGE_TARGET_PROJECTION_NEXT__FINAL_FIT1_OPTIMIZER_NOT_AUTHORIZED`

## 0. Updated boundary

The earlier 2026-08-31 clean-room audit remains valid. This V2 updates it after real Mage Geppetto FIT1 closure.

Frozen upstream reference:

`VAST-AI-Research/SkinTokens@273b691d35989d71cd17ff2895fdc735097b92d1`.

RealSaS upstream boundary is now concrete:

- real IRIS/GSA Mage substrate: `950` admitted nodes / `2813` local topology edges;
- Geppetto FIT1: `FIT1_TERMINAL_PASS`, closure step `14080`, streak `48/48`;
- Compiler-qualified Mage skeleton: `22` joints, exactly `1` deform root;
- Arachne is not allowed to reopen joint count/root/parent/canonical-ID authority.

Historical RealSaS Arachne evidence also remains authoritative:

- shipping `SkinFieldCodecV1` capacity: PASS on the preregistered heterogeneous synthetic panel with cosine cooling;
- shipping `ArachneCandidateV2 -> frozen codec -> Compiler -> verified LBS`: PASS on the same panel;
- observation-oracle U1 consumer gate: heterogeneous `3/3 PASS`;
- none of those results is a real Mage-family fit or generalization claim.

## 1. What SkinTokens actually does

SkinTokens contains two distinct learned levels.

### 1.1 Per-bone skin codec

For each bone/joint influence field:

`weight field + mesh geometry -> skin encoder -> latent -> FSQ -> SkinTokens`

and

`SkinTokens + mesh geometry condition -> dedicated skin-field decoder -> scalar field`.

The released skin decoder ends in a sigmoid, so each bone field is independently bounded in `[0,1]`. The assembled `N x J` skin matrix can then be row-normalized/postprocessed outside that decoder.

This decoder does **not** decode skeleton coordinates or topology.

### 1.2 Unified TokenRig causal model

TokenRig uses a separate decoder-only causal Transformer to predict one sequence:

`skeleton block -> skeleton delimiter/eos -> J * T_D skin tokens -> final eos`.

The same Transformer parameters predict both vocabulary regions. A deterministic logits processor switches vocabulary after the skeleton block and derives the required skin-token count from the already-generated skeleton cardinality.

After autoregressive generation:

- skeleton tokens are decoded by the deterministic skeleton tokenizer/detokenizer;
- skin tokens are mapped through FSQ and decoded by the **separate pretrained skin-field decoder**.

Therefore:

> SkinTokens shares an autoregressive token predictor, **not** a common final skeleton/skin geometric decoder.

### 1.3 Causal asymmetry matters

The public sequence is skeleton-first, skin-second.

At inference:

- skin-token prediction can attend to the complete generated skeleton;
- earlier skeleton-token prediction cannot attend to future skin tokens.

Thus the claimed skeleton/skin coupling comes from shared model parameters/training and later sequence-level RL rewards as well as the direct causal path `skeleton -> skin`; it is not a bidirectional common decoder.

### 1.4 Geometry is not fully shared either

The public implementation uses:

- a learned mesh encoder to condition the TokenRig autoregressive Transformer;
- a separate geometry-condition encoder inside the pretrained FSQ-CVAE used for skin reconstruction.

So even SkinTokens does not imply that rigging and skinning require one identical geometric backbone.

### 1.5 Existing-skeleton mode

The released runtime explicitly supports an existing skeleton. The skeleton is serialized as the prefix and TokenRig continues with skin tokens.

Therefore the external solution class directly supports the RealSaS authority split:

`Compiler-qualified skeleton -> learned skin prediction`.

## 2. RealSaS shared-vs-split decisions for Mage FIT1

### A. Common final Geppetto/Arachne decoder

`FALSE / REJECTED`

No such obligation exists in SkinTokens, and it would reopen a solved Geppetto boundary without evidence.

### B. One shared causal LM for Geppetto + Arachne

`NOT_FOR_FIT1__FUTURE_GENERALIZATION_ARM_ALLOWED`

A shared rig prior is scientifically interesting later, especially for joint training/reward refinement. It is unnecessary for the causal FIT1 question:

> Given a correct qualified skeleton, can Arachne infer deformation-valid skin from admitted product evidence?

### C. Shared surface encoder with Geppetto

`OPTIONAL_LATER_ABLATION`

No FIT1 dependency on hidden Geppetto features is permitted. Arachne must be solvable from its explicit product contract.

### D. Same skin decoder between codec ceiling and predictor run

`REQUIRED`

For a given Mage FIT1 attempt:

`A0: teacher W -> codec encoder -> latent -> DECODER D -> W*`

then freeze D and use:

`A1: product conditioning -> Arachne predicted latent -> SAME DECODER D -> W_hat`.

No special A1 decoder or decoder retraining may hide a representation seam.

## 3. Do not redesign the proven RealSaS codec merely to look more like SkinTokens

SkinTokens uses independent sigmoid per-bone fields. Current RealSaS `SkinFieldCodecV1` instead decodes joint latents jointly and applies row-softmax.

That is a real factorization difference, but **not** a clean-room failure. The previously frozen reference audit explicitly retained functional obligations rather than exact mechanisms, and the current row-softmax shipping codec has already passed:

- heterogeneous A0 capacity;
- A1 predictor-to-frozen-codec behavior;
- Compiler qualification;
- verified LBS;
- observation-oracle U1 small-panel sufficiency.

Therefore the Mage FIT1 baseline retains the existing shipping/default codec architecture:

- hidden dim `192`;
- latent dim `64`;
- encoder layers `3`;
- decoder layers `3`;
- architecture `RealSaS.SkinFieldCodec.v1`.

Independent sigmoid fields remain a **contingency ablation**, not a mandatory pre-FIT1 redesign. They may be opened only if Mage A0/A1 produces a causal failure implicating joint-coupled simplex decoding or if a later generalization experiment specifically tests factorization.

Compiler still owns legal lineage/reference/simplex admission even when a model happens to emit an already-simplex row; qualification authority does not require the proposal to be unconstrained.

## 4. Arachne predictor baseline

The previously closed shipping/default predictor remains the first real-Mage baseline:

- architecture: `RealSaS.ArachneCandidate.SegmentAwareJointField.v2`;
- model dim `128`;
- surface encoder layers `2`;
- attention heads `4`;
- feedforward dim `384`;
- frozen qualified shipping codec decoder downstream during A1.

Do not redesign this apparatus before the real Mage evidence tells us it is necessary.

## 5. Frozen Arachne inference boundary

`RiggingSurfaceIR + QualifiedSkeletonIR` only.

Allowed deterministic conditioning includes product-visible/derivable fields such as:

### Surface

- admitted 3D node positions;
- normals/normal validity;
- GSA local geometric relations;
- support-view provenance;
- observed/completed status;
- raster bindings;
- deterministic normalization.

### Skeleton

- qualified joint positions;
- exact parent tree;
- root indicator;
- deterministic depth/graph features;
- qualified support-surface bindings;
- canonical IDs only for stable binding/serialization;
- skeleton lineage hash.

Forbidden inference inputs:

- teacher skin weights;
- source bone names/IDs as semantic hints;
- helper/IK controls excluded from the qualified 22-joint core;
- source/full hidden mesh geometry outside admitted product S;
- teacher active-region masks;
- teacher projection confidence labels.

## 6. Training-only Mage skin truth lane

The dense source corpus contains `5321 x 41` skin weights, but only source columns `1..22` carry skin mass for the rendered Mage mechanical core. The authority render mesh contains `3348` vertices in the same world frame and is an exact coordinate subset of the dense source mesh.

Training must produce an explicit target:

`SkinFieldTeacherTarget(surface_ids, qualified_joint_ids, W_teacher_on_GSA)`.

This is teacher-only projection. It must never become product inference information.

Required invariants before A0 optimizer step 1:

1. render-authority vertices -> dense-skin source vertices mapped exactly and hash-sealed;
2. source skin columns `1..22` -> qualified 22 joint IDs mapped and hash-sealed;
3. GSA node -> teacher surface mapping is explicit, reproducible and measured;
4. no row-index assumptions; all final target rows bind to exact `surface_id`;
5. every target row finite/nonnegative/simplex after projection;
6. projection uncertainty/coverage is reported rather than silently hidden.

## 7. Mage target-projection design

A pure nearest-3D-triangle transfer is accepted only as a diagnostic/fallback, not yet as final target authority.

Reason: preliminary exact diagnostic on the current 950 GSA nodes gives approximately:

- mean GSA-to-teacher-surface distance `0.00937` world units;
- p95 `0.03069`;
- p99 `0.06861`;
- max `0.09476`;
- `98.3%` of nodes within `0.05`;
- completed-node p95 is worse than observed-node p95.

Those outliers are large enough that Euclidean nearest-surface transfer could cross semantically adjacent parts.

### Preferred teacher projection

For observed GSA nodes:

`support-view raster binding -> exact teacher camera z-buffer/rasterized triangle -> barycentric teacher skin sample`.

Use all available support views and record multi-view agreement. Robustly fuse only mutually consistent teacher samples.

For the 53 model-completed/no-support nodes:

use an explicitly flagged 3D nearest-triangle barycentric fallback, with distance/confidence telemetry.

Required projection telemetry:

- valid teacher views per node;
- cross-view skin disagreement;
- teacher depth/ray residual;
- nearest-3D surface distance;
- fallback count;
- source triangle IDs as teacher-only provenance;
- final row simplex residual;
- exact surface/joint binding hashes.

Final projection must fail closed if ambiguity exceeds the preregistered policy.

## 8. Experimental sequence

### P0 — `MAGE_SKIN_TARGET_PROJECTION`

Close and hash the real teacher target on the exact 950-node admitted Mage substrate and exact 22-joint qualified skeleton.

No learned optimizer.

### A0-MAGE — real codec representation ceiling

Question:

> Can the existing shipping/default SkinFieldCodec architecture reconstruct the projected real Mage 22-joint skin field and survive raw-W, Compiler and deformation checks?

Teacher W is allowed only in the codec encoder lane.

Use the already causally supported generic cosine optimizer protocol as the initial training protocol unless a Mage-specific pre-optimizer prereg explicitly changes it.

A0 PASS is mandatory before A1.

### A1-MAGE / Arachne FIT1

Freeze the A0-qualified Mage decoder.

Question:

> Can `ArachneCandidateV2` infer the required per-joint latents from only real `RiggingSurfaceIR + QualifiedSkeletonIR` and produce a deformation-valid skin field on Mage?

Path:

`real Mage S + frozen Geppetto Qualified G`
` -> ArachneCandidateV2`
` -> frozen Mage A0 decoder`
` -> raw W`
` -> SkinProposalIR`
` -> Compiler.qualify_skin`
` -> QualifiedSkinIR`
` -> verified deformation/motion proof`.

Teacher W is evaluation truth only during A1.

## 9. Evaluation obligations

### Raw learned field

- row-L1 distribution / p95;
- active influence fidelity;
- dominant-joint fidelity;
- finite/nonnegative checks;
- no hidden target-conditioned inputs.

### Compiler

- exact S/G lineage;
- exact surface coverage;
- illegal references `0`;
- bounded correction magnitude reported;
- raw W must independently be good enough that Compiler rescue cannot explain PASS.

### Deformation

At minimum preregister:

- individual limb bends;
- parent-child compound rotations;
- symmetric left/right probes;
- torso/root probe;
- combined stress pose;
- teacher-vs-predicted deformation error;
- finite/no-explosion;
- local stretch/fold/smoothness diagnostics on admitted geometry.

FIT1 authority is deformation behavior, not only matrix similarity.

## 10. Frozen decisions

- `SKINTOKENS_PRIMARY_ARACHNE_REFERENCE = TRUE`
- `SKINTOKENS_UPSTREAM_COMMIT = 273b691d35989d71cd17ff2895fdc735097b92d1`
- `SKINTOKENS_SHARED_FINAL_SKELETON_SKIN_DECODER = FALSE`
- `SKINTOKENS_SHARED_CAUSAL_TOKEN_PREDICTOR = TRUE`
- `SKINTOKENS_SEQUENCE_CAUSAL_ORDER = SKELETON_THEN_SKIN`
- `COMMON_FINAL_GEPPETTO_ARACHNE_DECODER = FALSE`
- `SHARED_GEPPETTO_ARACHNE_CAUSAL_MODEL_FOR_MAGE_FIT1 = FALSE`
- `SHARED_RIG_PRIOR_FUTURE_ARM = ALLOWED`
- `SAME_MAGE_SKIN_DECODER_A0_A1 = REQUIRED`
- `KEEP_PROVEN_REALSAS_SKINFIELD_CODEC_V1_FOR_INITIAL_MAGE_FIT1 = TRUE`
- `INDEPENDENT_SIGMOID_FIELD_REDESIGN_BEFORE_MAGE_EVIDENCE = FORBIDDEN`
- `QUALIFIED_GEPPETTO_SKELETON_IS_ARACHNE_INPUT_AUTHORITY = TRUE`
- `GEPPETTO_REOPEN_FOR_ARACHNE_FIT1 = FORBIDDEN`
- `COMPILER_FINAL_SKIN_AUTHORITY = TRUE`
- `DEFORMATION_PROOF = REQUIRED`
- `GENERALIZATION_CLAIM = FALSE`
- `NEXT_GATE = MAGE_SKIN_TARGET_PROJECTION`
- `FINAL_ARACHNE_FIT1_OPTIMIZER_AUTHORIZED = FALSE`
