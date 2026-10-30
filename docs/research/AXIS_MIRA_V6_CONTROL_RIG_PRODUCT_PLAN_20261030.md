# AXIS/MIRA V6 — control-ready rig product plan

Date: 2026-10-30  
Status: `ACTIVE_PRODUCT_REDESIGN__TRUTH139_UNSEEN_FIRST_PREREGISTERED__TRAINING_BLOCKED_ON_EXACT_TARGET_AND_SPLIT_SEAL`

## 0. Product decision

AXIS is no longer defined as a skeleton generator. The product Rig responsibility is a control-ready mechanical program:

`RigProgram = deform structure + local frames/DOF + animator controls + kinematic chains + constraints + attachment anchors`.

MIRA is no longer an independent downstream dense-weight regressor in the new learned architecture. MIRA becomes the discrete Skin representation/codec used by the AXIS causal generator. Product responsibilities remain separate (`Rig`, `Skin`) even when one learned causal run emits both proposal families.

Historical AXIS V5.4.1 and MIRA V5.5 are immutable regression baselines.

## 1. SkinTokens code-derived design

The VAST-AI-Research SkinTokens/TokenRig implementation provides three reusable representation ideas:

1. skeleton vocabulary and learned skin-code vocabulary are exposed to one causal transformer;
2. a grammar/logits processor constrains which token family may be emitted at each point of the sequence;
3. skin codes are decoded by a frozen FSQ-CVAE back into dense per-vertex influences.

RealSaS adopts the representation pattern, not TokenRig's product ontology. TokenRig's output remains skeleton + skin. RealSaS extends the learned sequence to:

`D, F, C, K, A, Z_skin`

where:

- `D`: deform graph/joints;
- `F`: local mechanical frames and DOF declarations;
- `C`: animator-facing controls;
- `K`: chains and constraint-program structure;
- `A`: attachment/socket anchors;
- `Z_skin`: discrete MIRA skin codes.

No named humanoid part taxonomy is required by the product representation.

## 2. Compiler anti-bottleneck rule

Compiler must not become a second learned rigging system.

Compiler may:

- validate graph references, acyclicity and root legality;
- validate nondegenerate numeric frames;
- validate declared chain contiguity;
- validate constraint/control/attachment references;
- validate exact Mesh/evidence/model/codec lineage;
- mint canonical IDs and hashes;
- validate decoded Skin coverage/simplex/bounds;
- fail closed or abstain when a required operator/capability is unavailable.

Compiler may not:

- reselect AXIS parents;
- synthesize missing controls;
- choose IK chains on behalf of AXIS;
- synthesize or rewrite constraints;
- search SkinTokens;
- substitute token IDs;
- repair learned local frames into a different rig;
- infer semantic body-part labels;
- execute the constraint solver while minting Rig;
- use subject-specific fallback rules.

Constraint **program** belongs to canonical Rig. Constraint **execution/solution** belongs to Mechanics.

## 3. Canonical identity decomposition

`CanonicalRigIR.v2` separates:

- `deform_structure_hash` — deform joints, rest geometry/local frames and deform-root semantics;
- `control_program_hash` — controls, chains, constraints and attachment anchors;
- `rig_lineage_hash` — full Rig identity.

Skin binds `CanonicalMesh + deform_structure_hash`, not the complete control program. Therefore a UI/controller-only edit does not force weight regeneration, while any deform/rest-structure mutation does.

## 4. Learned architecture

Target causal flow:

```text
IRIS/TESSA product Mesh + shared evidence
              |
              v
          AXIS V6
   shared causal transformer
              |
      +-------+---------+
      |                 |
RigProgramProposal   SkinTokenProposal
      |                 |
      v                 |
Compiler Rig court      |
      |                 |
CanonicalRig.v2         |
      |                 v
      |            MIRA V6 codec
      |                 |
      |            dense W proposal
      |                 |
      +---------> Compiler Skin court
                        |
                 CanonicalSkin.v2
```

MIRA's learned codec can be trained first as `W -> Z_skin -> W_hat`, then frozen for the first unified AXIS run. Later joint fine-tuning is a separate experiment, not silently enabled.

## 5. Sequence grammar

The initial grammar is structural rather than semantic:

```text
<BOS>
<DEFORM_GRAPH> ...
<LOCAL_FRAMES> ...
<CONTROLS> ...
<CHAINS> ...
<CONSTRAINTS> ...
<ATTACHMENTS> ...
<SKIN_CODES> ...
<EOS>
```

The model/runtime grammar owns token-family ordering and token-domain masking, following the useful TokenRig pattern. Compiler sees typed decoded proposals, not raw autoregressive logits, and does not perform beam-search repair.

## 6. Mechanics and Motion consequences

Mechanics consumes the exact canonical Rig/Skin and owns:

- control-program execution;
- IK/constraint residuals;
- frame observability under motion;
- joint-limit/chain stability;
- post-bind deformation validity;
- attachment/grip residuals;
- dynamic stretch/area/fold/contact courts.

Motion must migrate from canonical-joint editing authority to control-space editing authority:

`ControlTransformTrack -> RigProgram/Mechanics -> derived joint transforms -> Skin -> Mesh`.

Joint-space tracks remain admissible as derived bake/proof/debug artifacts and as the historical Knight regression witness; they are not the final editable-motion authority.

## 7. Knight regression

Before a generic checkpoint can be product-admitted, the new path must reproduce or improve the known Knight behavior:

- stable deform structure;
- usable control handles;
- at least the required limb IK/control behavior where proposed;
- sword/grip attachment through a typed anchor/socket;
- Run/Idle/Slash visible result no worse than the historical M+CAA witness;
- no regression in Skin/deformation proofs;
- no Compiler-generated hidden rig semantics.

Knight remains regression only and does not participate in Truth139 training or unseen selection.

## 8. Truth139 unseen-first protocol

The first generic AXIS/MIRA V6 checkpoint will use the same lineage-safe leakage-connected-component TRAIN/UNSEEN split being prepared for Truth139. No FIT8 or LOFO result may be used to tune this checkpoint before the sealed unseen evaluation.

Sequence:

1. seal Truth139 targets required by Rig/Skin truth extraction;
2. seal leakage-connected TRAIN/UNSEEN split and fixed TRAIN-internal validation rows;
3. freeze MIRA V6 codec configuration/codebook using TRAIN only;
4. train the unified AXIS V6 causal generator on TRAIN only;
5. select stopping/checkpoint using only fixed TRAIN-internal validation;
6. run exactly one preregistered UNSEEN evaluation family;
7. run Knight regression independently;
8. Compiler + Mechanics + Graphics + Motion + Proof consume the exact candidate checkpoint;
9. only an exact all-required-domain product court may promote the checkpoint toward product admission.

The unseen set is not used for architecture, threshold, token-count, codebook, reward-weight, beam-search or postprocess selection.

## 9. Product-ready does not mean auto-promoted

A checkpoint produced by this protocol is a **product candidate by construction**, because it was trained against the shipping v2 proposal contracts and evaluated on the preregistered unseen split. It still cannot mint its own ModelRelease channel or ProductRevision. Platform model admission and Compiler/Proof product qualification remain explicit transitions.

## 10. Implementation order

1. v2 Rig/Skin typed contracts and Compiler no-repair courts;
2. MIRA discrete skin codec + reconstruction/deformation court;
3. AXIS V6 grammar/tokenizer/shared causal model;
4. v2 Rig/Skin executors and Platform capability contract versions;
5. Mechanics control/constraint execution boundary;
6. control-space Motion IR/runtime;
7. Knight regression;
8. Truth139 unseen-first Platform campaign;
9. product proof and model/channel admission.

The active IRIS/TESSA Truth139 preparation is not rewritten by this branch. This work consumes its sealed Mesh/shared-evidence outputs downstream.
